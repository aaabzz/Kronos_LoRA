"""Non-blocking Kronos inference for the real-time market API."""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass
from datetime import datetime, time, timedelta
from typing import Any, Awaitable, Callable, Protocol
from zoneinfo import ZoneInfo

import pandas as pd

from api.lora_runtime import LoRADeployment, discover_lora_deployments, project_path


LOGGER = logging.getLogger("kronos.forecast")
CN_TZ = ZoneInfo("Asia/Shanghai")
PERIOD_PATTERN = re.compile(r"^(\d+)([mhd])$")
FORECAST_MODE_KEYS = ("greedy", "median", "mean")
class Predictor(Protocol):
    def predict(self, **kwargs: Any) -> pd.DataFrame: ...


@dataclass(frozen=True, slots=True)
class ForecastConfig:
    enabled: bool = True
    window: int = 512
    pred_len: int = 24
    max_context: int = 512
    model_id: str = "pretrained/Kronos-base"
    small_model_id: str = "pretrained/Kronos-small"
    tokenizer_id: str = "pretrained/Kronos-Tokenizer-base"
    device: str = "cpu"
    temperature: float = 1.0
    top_p: float = 0.9
    sample_count: int = 32
    top_k: int = 0
    seed: int = 100
    flat_tolerance: float = 0.0
    adapters_root: str = "adapters"
    lora_base_model_path: str = "pretrained/Kronos-base"
    lora_tokenizer_path: str = "pretrained/Kronos-Tokenizer-base"

    def validate_request(self, window: int, pred_len: int) -> None:
        if not 16 <= window <= self.max_context:
            raise ValueError(f"window must be between 16 and {self.max_context}")
        if not 1 <= pred_len <= 120:
            raise ValueError("pred_len must be between 1 and 120")

    @staticmethod
    def validate_sample_count(sample_count: int) -> None:
        if not 1 <= sample_count <= 256:
            raise ValueError("sample_count must be between 1 and 256")


def _period_delta(period: str) -> timedelta:
    match = PERIOD_PATTERN.fullmatch(period.strip().lower())
    if not match:
        raise ValueError(f"Unsupported forecast period: {period!r}")
    value, unit = int(match.group(1)), match.group(2)
    if unit == "m":
        return timedelta(minutes=value)
    if unit == "h":
        return timedelta(hours=value)
    return timedelta(days=value)


def _next_weekday(day: datetime, bar_delta: timedelta) -> datetime:
    candidate = day + timedelta(days=1)
    while candidate.weekday() >= 5:
        candidate += timedelta(days=1)
    # QMT labels intraday bars by end time: 5m starts at 09:35, 15m at 09:45.
    return candidate.replace(hour=9, minute=30, second=0, microsecond=0) + bar_delta


def future_timestamps(last_timestamp: pd.Timestamp, period: str, count: int) -> pd.DatetimeIndex:
    """Build future bar start times, respecting the mainland A-share lunch break."""

    delta = _period_delta(period)
    current = last_timestamp.to_pydatetime()
    if current.tzinfo is None:
        current = current.replace(tzinfo=CN_TZ)
    else:
        current = current.astimezone(CN_TZ)

    result: list[datetime] = []
    intraday = delta < timedelta(days=1)
    session_first_time = (
        current.replace(hour=9, minute=30, second=0, microsecond=0) + delta
    ).time()
    while len(result) < count:
        candidate = current + delta
        if intraday:
            if candidate.weekday() >= 5:
                candidate = _next_weekday(candidate, delta)
            elif time(11, 30) < candidate.time() <= time(13, 0):
                candidate = candidate.replace(
                    hour=13, minute=0, second=0, microsecond=0
                ) + delta
            elif candidate.time() > time(15, 0):
                candidate = _next_weekday(candidate, delta)
            elif candidate.time() < session_first_time:
                candidate = candidate.replace(
                    hour=9, minute=30, second=0, microsecond=0
                ) + delta
        elif candidate.weekday() >= 5:
            candidate = _next_weekday(candidate, timedelta(0)).replace(
                hour=current.hour, minute=current.minute
            )
        result.append(candidate)
        current = candidate
    return pd.DatetimeIndex(result)


class KronosForecastService:
    """Coalesce forecast requests and run the expensive model outside the event loop."""

    def __init__(
        self,
        config: ForecastConfig,
        publish: Callable[[dict[str, Any]], Awaitable[None]],
        predictor: Predictor | None = None,
    ) -> None:
        self.config = config
        self._publish = publish
        self._predictors: dict[str, Predictor] = {}
        self._runtimes: dict[str, tuple[Any, Any]] = {}
        self._injected_predictor = predictor is not None
        self._model_deployments = self._discover_models()
        if predictor is not None:
            if not any(self.is_lora_key(key) for key in self._model_deployments):
                self._model_deployments["lora:injected"] = None
            self._predictors.update(
                {key: predictor for key in self._model_deployments}
            )
        self._pending: dict[
            str, tuple[str, list[dict[str, Any]], str, int, int, str, tuple[str, ...], int]
        ] = {}
        self._work_available = asyncio.Event()
        self._task: asyncio.Task[None] | None = None
        self._stopping = False

    async def start(self) -> None:
        if not self.config.enabled:
            return
        self._stopping = False
        self._task = asyncio.create_task(self._worker(), name="kronos-forecast-worker")

    async def stop(self) -> None:
        self._stopping = True
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    def cancel_symbol(self, symbol: str) -> None:
        """Drop queued work for an unsubscribed symbol.

        A prediction already executing in ``to_thread`` cannot be forcefully
        stopped, so the API bridge also rejects its late result.
        """

        self._pending.pop(symbol, None)
        if not self._pending:
            self._work_available.clear()

    def submit(
        self, symbol: str, bars: list[dict[str, Any]], period: str,
        *, window: int | None = None, pred_len: int | None = None,
        use_lora: bool = False, model_key: str | None = None,
        modes: tuple[str, ...] | list[str] | None = None,
        sample_count: int | None = None,
    ) -> bool:
        if not self.config.enabled:
            return False
        use_window = window or self.config.window
        use_pred_len = pred_len or self.config.pred_len
        self.config.validate_request(use_window, use_pred_len)
        selected_model = model_key or (self.default_lora_key if use_lora else "base")
        selected_modes = self.normalize_modes(modes)
        selected_sample_count = (
            self.config.sample_count if sample_count is None else sample_count
        )
        self.config.validate_sample_count(selected_sample_count)
        if selected_model not in self.model_deployments:
            raise ValueError(f"Forecast model is unavailable: {selected_model}")
        if len(bars) < use_window:
            raise ValueError(f"{symbol} only has {len(bars)} bars; {use_window} required")
        item = (
            symbol, bars[-use_window:], period, use_window, use_pred_len,
            selected_model, selected_modes,
            selected_sample_count,
        )
        # Replace only this symbol's stale request. Other watched symbols retain
        # their own latest request and will be processed in insertion order.
        self._pending[symbol] = item
        self._work_available.set()
        return True

    @staticmethod
    def normalize_modes(modes: tuple[str, ...] | list[str] | None) -> tuple[str, ...]:
        selected = tuple(dict.fromkeys(modes or FORECAST_MODE_KEYS))
        invalid = [mode for mode in selected if mode not in FORECAST_MODE_KEYS]
        if invalid:
            raise ValueError(f"Unsupported forecast modes: {', '.join(invalid)}")
        if not selected:
            raise ValueError("At least one forecast mode must be selected")
        return selected

    @property
    def lora_available(self) -> bool:
        if self._injected_predictor:
            return True
        return any(
            self.is_lora_key(key) and deployment is not None
            for key, deployment in self._model_deployments.items()
        )

    @staticmethod
    def is_lora_key(model_key: str) -> bool:
        return model_key.startswith("lora:")

    @property
    def model_deployments(self) -> dict[str, LoRADeployment | None]:
        return dict(self._model_deployments)

    def _discover_models(self) -> dict[str, LoRADeployment | None]:
        # Keep ``base`` as the backward-compatible key for Kronos-base.
        # New bundles select their own base/tokenizer through deployment.json;
        # legacy bundles continue to use the configured global fallback paths.
        models: dict[str, LoRADeployment | None] = {"base": None, "small": None}
        for key, deployment in discover_lora_deployments(
            self.config.adapters_root,
            self.config.lora_base_model_path,
            self.config.lora_tokenizer_path,
        ):
            models[key] = deployment
        return models

    @property
    def default_lora_key(self) -> str:
        try:
            return next(
                key for key in self._model_deployments if self.is_lora_key(key)
            )
        except StopIteration as exc:
            raise ValueError("No deployable LoRA model was found") from exc

    @staticmethod
    def _model_source(value: str) -> str:
        """Resolve repository-local model directories without breaking remote IDs."""

        candidate = project_path(value)
        return str(candidate) if candidate.exists() else value

    @property
    def model_options(self) -> list[dict[str, Any]]:
        options = [
            {"key": "base", "name": "Kronos-base", "type": "base"},
            {"key": "small", "name": "Kronos-small", "type": "base"},
        ]
        options.extend(
            {
                "key": key,
                "name": deployment.weight_path.name,
                "directory": deployment.adapter_path.name,
                "type": "lora",
                # Expose the resolved identity for diagnostics and future UI
                # use without changing the existing front-end label.
                "base_model_key": (
                    deployment.metadata.get("base_model", {}).get("key")
                    if deployment.metadata is not None
                    else deployment.base_model_path.name.removeprefix("Kronos-").lower()
                ),
                "base_model_name": deployment.base_model_path.name,
            }
            for key, deployment in self.model_deployments.items()
            if key != "base" and deployment is not None
        )
        return options

    def _load_predictor(self, model_key: str) -> Predictor:
        if model_key not in self._predictors:
            from model import Kronos, KronosPredictor, KronosTokenizer

            deployment = self.model_deployments.get(model_key)
            if deployment is not None:
                from lora.config import QVLoRAConfig
                from lora.model_factory import load_qv_lora_adapter

                lora_config = QVLoRAConfig(
                    base_model_path=deployment.base_model_path,
                    tokenizer_path=deployment.tokenizer_path,
                    adapter_output_dir=deployment.adapter_path.parent,
                )
                tokenizer = KronosTokenizer.from_pretrained(
                    lora_config.tokenizer_path
                )
                peft_model = load_qv_lora_adapter(
                    lora_config, deployment.adapter_path
                )
                model = peft_model.merge_and_unload()
            else:
                tokenizer = KronosTokenizer.from_pretrained(
                    self._model_source(self.config.tokenizer_id)
                )
                model = Kronos.from_pretrained(
                    self._model_source(
                        self.config.small_model_id
                        if model_key == "small" else self.config.model_id
                    )
                )
            self._predictors[model_key] = KronosPredictor(
                model, tokenizer, device=self.config.device,
                max_context=self.config.max_context,
            )
        return self._predictors[model_key]

    def _load_runtime(self, model_key: str) -> tuple[Any, Any]:
        if model_key not in self._runtimes:
            from model import Kronos, KronosTokenizer

            deployment = self.model_deployments.get(model_key)
            if deployment is not None:
                from lora.config import QVLoRAConfig
                from lora.model_factory import load_qv_lora_adapter

                lora_config = QVLoRAConfig(
                    base_model_path=deployment.base_model_path,
                    tokenizer_path=deployment.tokenizer_path,
                    adapter_output_dir=deployment.adapter_path.parent,
                )
                tokenizer = KronosTokenizer.from_pretrained(
                    lora_config.tokenizer_path
                )
                model = load_qv_lora_adapter(
                    lora_config, deployment.adapter_path
                ).merge_and_unload()
            else:
                tokenizer = KronosTokenizer.from_pretrained(
                    self._model_source(self.config.tokenizer_id)
                )
                model = Kronos.from_pretrained(
                    self._model_source(
                        self.config.small_model_id
                        if model_key == "small" else self.config.model_id
                    )
                )
            tokenizer = tokenizer.to(self.config.device).eval()
            model = model.to(self.config.device).eval()
            self._runtimes[model_key] = (tokenizer, model)
        return self._runtimes[model_key]

    def _predict(
        self, symbol: str, bars: list[dict[str, Any]], period: str,
        window: int, pred_len: int, model_key: str, selected_modes: tuple[str, ...],
        sample_count: int,
    ) -> dict[str, Any]:
        frame = pd.DataFrame(bars)
        columns = ["open", "high", "low", "close", "volume", "amount"]
        missing = [column for column in columns if column not in frame]
        if missing:
            raise ValueError(f"Kronos input is missing columns: {', '.join(missing)}")
        x_timestamp = pd.to_datetime(frame["timestamp"], utc=True).dt.tz_convert(CN_TZ)
        y_timestamp = future_timestamps(x_timestamp.iloc[-1], period, pred_len)
        y_timestamp_series = pd.Series(y_timestamp)
        if self._injected_predictor:
            prediction = self._load_predictor(model_key).predict(
                df=frame[columns].astype(float), x_timestamp=x_timestamp,
                y_timestamp=y_timestamp_series, pred_len=pred_len,
                T=self.config.temperature, top_p=self.config.top_p,
                sample_count=sample_count, verbose=False,
            )
            mode_frames = {key: prediction for key in selected_modes}
            first_close = float(prediction.iloc[0]["close"])
            reference = float(frame.iloc[-1]["close"])
            probabilities = {
                "up": float(first_close > reference),
                "down": float(first_close < reference),
                "flat": float(first_close == reference),
                "sample_count": 1,
            }
        else:
            from kronos_prediction_modes import predict_modes

            tokenizer, model = self._load_runtime(model_key)
            result = predict_modes(
                frame[columns].astype(float), x_timestamp, y_timestamp_series,
                tokenizer=tokenizer, model=model, device=self.config.device,
                sample_count=sample_count,
                temperature=self.config.temperature, top_k=self.config.top_k,
                top_p=self.config.top_p, seed=self.config.seed,
                max_context=self.config.max_context,
                flat_tolerance=self.config.flat_tolerance,
                modes=selected_modes,
            )
            mode_frames = {
                "greedy": result.greedy,
                "median": result.sample_median,
                "mean": result.sample_mean,
            }
            mode_frames = {
                key: value for key, value in mode_frames.items()
                if key in selected_modes
            }
            probabilities = {
                "up": result.up_probability,
                "down": result.down_probability,
                "flat": result.flat_probability,
                "sample_count": int(result.sampled_paths.shape[0]),
            }

        def serialize(prediction: pd.DataFrame) -> list[dict[str, Any]]:
            output: list[dict[str, Any]] = []
            for timestamp, (_, row) in zip(y_timestamp, prediction.iterrows()):
                open_price = round(float(row["open"]), 2)
                close_price = round(float(row["close"]), 2)
                high_price = round(max(float(row["high"]), open_price, close_price), 2)
                low_price = round(min(float(row["low"]), open_price, close_price), 2)
                output.append({
                    "symbol": symbol, "period": period,
                    "timestamp_ms": int(timestamp.timestamp() * 1000),
                    "timestamp": timestamp.isoformat(), "open": open_price,
                    "high": high_price, "low": low_price, "close": close_price,
                    "volume": max(0, round(float(row["volume"]))),
                    "amount": max(0, round(float(row["amount"]))),
                })
            return output
        modes = {name: serialize(value) for name, value in mode_frames.items()}
        return {
            "type": "forecast", "symbol": symbol, "period": period,
            "window": window, "pred_len": pred_len,
            "bars": modes.get("greedy", modes[selected_modes[0]]), "modes": modes,
            "selected_modes": list(selected_modes),
            "sample_count": sample_count,
            "probabilities": probabilities,
            "model_key": model_key, "use_lora": self.is_lora_key(model_key),
            "based_on_timestamp": bars[-1]["timestamp"],
        }

    async def _worker(self) -> None:
        while not self._stopping:
            await self._work_available.wait()
            try:
                _, item = next(iter(self._pending.items()))
            except StopIteration:
                self._work_available.clear()
                continue
            symbol, bars, period, window, pred_len, model_key, selected_modes, sample_count = item
            self._pending.pop(symbol, None)
            if not self._pending:
                self._work_available.clear()
            await self._publish({
                "type": "forecast_status", "symbol": symbol,
                "state": "running", "window": window, "pred_len": pred_len,
                "model_key": model_key, "use_lora": self.is_lora_key(model_key),
                "selected_modes": list(selected_modes),
                "sample_count": sample_count,
            })
            try:
                event = await asyncio.to_thread(
                    self._predict, symbol, bars, period, window, pred_len,
                    model_key, selected_modes,
                    sample_count,
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                LOGGER.exception("Kronos forecast failed for %s", symbol)
                await self._publish({
                    "type": "forecast_status", "symbol": symbol,
                    "state": "error", "message": str(exc),
                    "window": window, "pred_len": pred_len,
                    "model_key": model_key, "use_lora": self.is_lora_key(model_key),
                    "selected_modes": list(selected_modes),
                    "sample_count": sample_count,
                })
            else:
                await self._publish(event)
