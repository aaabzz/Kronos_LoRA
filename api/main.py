"""FastAPI bridge for normalized QMT market-data events.

Run from the repository root with::

    python -m uvicorn api.main:app --host 127.0.0.1 --port 7000
"""

from __future__ import annotations

import asyncio
import logging
import os
import queue
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
from datetime import datetime, timedelta
from typing import Any, Protocol

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from api.forecasting import ForecastConfig, KronosForecastService, Predictor
from real_time.qmt_gateway import GatewayConfig, QmtGateway


LOGGER = logging.getLogger("kronos.api")


def _parse_modes(value: str) -> tuple[str, ...]:
    return tuple(mode.strip().lower() for mode in value.split(",") if mode.strip())


class EventGateway(Protocol):
    """Minimal gateway interface required by the FastAPI bridge."""

    @property
    def is_running(self) -> bool: ...

    def start(self) -> None: ...

    def stop(self) -> None: ...

    def get_event(self, timeout: float | None = None) -> dict[str, Any]: ...

    def ensure_symbol(self, symbol: str) -> int: ...

    def unsubscribe_symbol(self, symbol: str) -> bool: ...


class ConnectionManager:
    """Track WebSocket clients and fan out each market event once."""

    def __init__(self) -> None:
        self._clients: dict[WebSocket, str | None] = {}

    @property
    def connection_count(self) -> int:
        return len(self._clients)

    async def connect(
        self,
        websocket: WebSocket,
        *,
        symbol: str | None,
        initial_events: tuple[dict[str, Any], ...],
    ) -> None:
        await websocket.accept()
        for event in initial_events:
            await websocket.send_json(event)
        self._clients[websocket] = symbol

    def disconnect(self, websocket: WebSocket) -> None:
        self._clients.pop(websocket, None)

    def subscriber_count(self, symbol: str) -> int:
        return sum(1 for subscribed in self._clients.values() if subscribed == symbol)

    async def broadcast(self, event: dict[str, Any]) -> None:
        stale: list[WebSocket] = []
        for websocket, subscribed_symbol in tuple(self._clients.items()):
            event_symbol = event.get("symbol")
            if subscribed_symbol is not None and event_symbol not in {
                None,
                subscribed_symbol,
            }:
                continue
            try:
                await websocket.send_json(event)
            except Exception:
                LOGGER.debug("Removing disconnected WebSocket client", exc_info=True)
                stale.append(websocket)
        for websocket in stale:
            self.disconnect(websocket)


class MarketEventBridge:
    """Move blocking QMT queue events into FastAPI's async event loop."""

    def __init__(self, gateway: EventGateway, manager: ConnectionManager) -> None:
        self.gateway = gateway
        self.manager = manager
        self.latest_event: dict[str, Any] | None = None
        self.history_by_symbol: dict[str, dict[str, Any]] = {}
        self.latest_bar_by_symbol: dict[str, dict[str, Any]] = {}
        self.forecast_by_symbol: dict[str, dict[str, Any]] = {}
        self.closed_bars_by_symbol: dict[str, list[dict[str, Any]]] = {}
        self.forecast_parameters_by_symbol: dict[
            str, tuple[int, int, str, tuple[str, ...], int]
        ] = {}
        self.forgotten_symbols: set[str] = set()
        self.forecaster: KronosForecastService | None = None
        self._consumer_task: asyncio.Task[None] | None = None
        self._stopping = False

    async def start(self) -> None:
        await asyncio.to_thread(self.gateway.start)
        if self.forecaster is not None:
            await self.forecaster.start()
        self._stopping = False
        self._consumer_task = asyncio.create_task(
            self._consume_events(), name="qmt-fastapi-event-bridge"
        )

    async def stop(self) -> None:
        self._stopping = True
        if self.forecaster is not None:
            await self.forecaster.stop()
        await asyncio.to_thread(self.gateway.stop)
        if self._consumer_task is not None:
            self._consumer_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._consumer_task
            self._consumer_task = None

    async def _consume_events(self) -> None:
        while not self._stopping:
            try:
                event = await asyncio.to_thread(self.gateway.get_event, 0.5)
            except queue.Empty:
                continue
            except asyncio.CancelledError:
                raise
            except Exception:
                LOGGER.exception("Failed to read an event from QMT gateway")
                await asyncio.sleep(0.5)
                continue

            if self._is_suppressed(event):
                continue
            self.latest_event = event
            self._remember(event)
            await self.manager.broadcast(event)

    async def publish_forecast(self, event: dict[str, Any]) -> None:
        """Receive an event produced by the inference worker."""

        if self._is_suppressed(event):
            return
        self.latest_event = event
        self._remember(event)
        await self.manager.broadcast(event)

    def _remember(self, event: dict[str, Any]) -> None:
        symbol = event.get("symbol")
        if not isinstance(symbol, str):
            return
        if event.get("type") == "history_snapshot":
            self.forgotten_symbols.discard(symbol)
            self.history_by_symbol[symbol] = event
            self.closed_bars_by_symbol[symbol] = list(event.get("bars") or [])
            self._submit_default_forecast(symbol, event.get("period", "5m"))
        elif event.get("type") in {"bar_update", "bar_closed"}:
            self.latest_bar_by_symbol[symbol] = event
            if event.get("type") == "bar_closed" and isinstance(event.get("bar"), dict):
                bars = self.closed_bars_by_symbol.setdefault(symbol, [])
                bar = event["bar"]
                bars[:] = [item for item in bars if item.get("timestamp_ms") != bar.get("timestamp_ms")]
                bars.append(bar)
                bars.sort(key=lambda item: item.get("timestamp_ms", 0))
                history = self.history_by_symbol.get(symbol)
                history_limit = (
                    int(history.get("required_count", 0))
                    if history is not None
                    else 0
                )
                if history_limit <= 0 and self.forecaster is not None:
                    history_limit = self.forecaster.config.max_context
                history_limit = max(1, history_limit or 240)
                del bars[:-history_limit]
                if history is not None:
                    history["bars"] = list(bars)
                    history["count"] = len(bars)
                    history["history_complete"] = len(bars) >= history_limit
                    history["history_fresh"] = True
                self._submit_default_forecast(symbol, event.get("period", "5m"))
        elif event.get("type") == "forecast":
            self.forecast_by_symbol[symbol] = event

    def _submit_default_forecast(self, symbol: str, period: str) -> None:
        if self.forecaster is None:
            return
        window, pred_len, model_key, modes, sample_count = self.forecast_parameters_by_symbol.get(
            symbol,
            (
                self.forecaster.config.window,
                self.forecaster.config.pred_len,
                "base",
                ("greedy", "median", "mean"),
                self.forecaster.config.sample_count,
            ),
        )
        try:
            self.forecaster.submit(
                symbol,
                self.closed_bars_by_symbol[symbol],
                period,
                window=window,
                pred_len=pred_len,
                model_key=model_key,
                modes=modes,
                sample_count=sample_count,
            )
        except ValueError as exc:
            LOGGER.info("Not enough data to forecast %s yet: %s", symbol, exc)
            asyncio.create_task(self.publish_forecast({
                "type": "forecast_status", "symbol": symbol,
                "state": "insufficient_data", "message": str(exc),
                "window": window,
                "pred_len": pred_len,
                "model_key": model_key,
                "use_lora": self.forecaster.is_lora_key(model_key),
                "selected_modes": list(modes),
                "sample_count": sample_count,
            }))

    def request_forecast(
        self, symbol: str, window: int, pred_len: int, use_lora: bool = False,
        model_key: str | None = None, modes: tuple[str, ...] | None = None,
        sample_count: int | None = None,
    ) -> bool:
        if self.forecaster is None or not self.forecaster.config.enabled:
            raise RuntimeError("Kronos forecasting is disabled")
        history = self.history_by_symbol.get(symbol)
        if history is None:
            raise ValueError(f"No market history is available for {symbol}")
        accepted = self.forecaster.submit(
            symbol,
            self.closed_bars_by_symbol.get(symbol, []),
            history.get("period", "5m"),
            window=window,
            pred_len=pred_len,
            use_lora=use_lora, model_key=model_key,
            modes=modes,
            sample_count=sample_count,
        )
        if accepted:
            self.forecast_parameters_by_symbol[symbol] = (
                window, pred_len, model_key or (
                    self.forecaster.default_lora_key if use_lora else "base"
                ), self.forecaster.normalize_modes(modes),
                self.forecaster.config.sample_count
                if sample_count is None else sample_count,
            )
        return accepted

    def set_forecast_preferences(
        self, symbol: str, window: int, pred_len: int, model_key: str,
        modes: tuple[str, ...], sample_count: int,
    ) -> None:
        """Register per-symbol settings before QMT history triggers inference."""

        if self.forecaster is None:
            return
        self.forecaster.config.validate_request(window, pred_len)
        if model_key not in self.forecaster.model_deployments:
            raise ValueError(f"Forecast model is unavailable: {model_key}")
        normalized_modes = self.forecaster.normalize_modes(modes)
        self.forecaster.config.validate_sample_count(sample_count)
        preferences = (window, pred_len, model_key, normalized_modes, sample_count)
        changed = self.forecast_parameters_by_symbol.get(symbol) != preferences
        self.forecast_parameters_by_symbol[symbol] = preferences
        if symbol in self.history_by_symbol and (
            changed or symbol not in self.forecast_by_symbol
        ):
            self._submit_default_forecast(
                symbol, self.history_by_symbol[symbol].get("period", "5m")
            )

    def replay_events(self, symbol: str | None) -> tuple[dict[str, Any], ...]:
        symbols = (
            (symbol,)
            if symbol is not None
            else tuple(sorted(set(self.history_by_symbol) | set(self.latest_bar_by_symbol)))
        )
        events: list[dict[str, Any]] = []
        for item_symbol in symbols:
            history = self.history_by_symbol.get(item_symbol)
            latest = self.latest_bar_by_symbol.get(item_symbol)
            if history is not None:
                events.append(history)
            if latest is not None:
                events.append(latest)
            forecast = self.forecast_by_symbol.get(item_symbol)
            if forecast is not None:
                events.append(forecast)
        return tuple(events)

    def forget_symbol(self, symbol: str) -> None:
        self.forgotten_symbols.add(symbol)
        if self.forecaster is not None:
            self.forecaster.cancel_symbol(symbol)
        self.history_by_symbol.pop(symbol, None)
        self.latest_bar_by_symbol.pop(symbol, None)
        self.forecast_by_symbol.pop(symbol, None)
        self.closed_bars_by_symbol.pop(symbol, None)
        self.forecast_parameters_by_symbol.pop(symbol, None)

    def _is_suppressed(self, event: dict[str, Any]) -> bool:
        symbol = event.get("symbol")
        return (
            isinstance(symbol, str)
            and symbol in self.forgotten_symbols
            and event.get("type") != "history_snapshot"
        )


def _gateway_from_environment() -> QmtGateway:
    symbols = tuple(
        symbol.strip().upper()
        for symbol in os.getenv("QMT_SYMBOLS", "").split(",")
        if symbol.strip()
    )
    # Keep a 512-bar market-history window independently from the smaller
    # prediction context selected in the frontend.
    history_count = max(
        int(os.getenv("QMT_HISTORY_COUNT", "512")),
        int(os.getenv("KRONOS_WINDOW", "512")),
    )
    return QmtGateway(
        GatewayConfig(
            symbols=symbols,
            period=os.getenv("QMT_PERIOD", "5m"),
            history_count=history_count,
            history_start=os.getenv(
                "QMT_HISTORY_START",
                (datetime.now() - timedelta(days=60)).strftime("%Y%m%d"),
            ),
            history_end=os.getenv("QMT_HISTORY_END", ""),
            # A fresh MiniQMT cache may contain only today's bars. Downloading
            # makes the configured history and prediction windows available.
            download_history=os.getenv("QMT_DOWNLOAD_HISTORY", "1") == "1",
        )
    )


def _forecast_config_from_environment(*, enabled: bool) -> ForecastConfig:
    return ForecastConfig(
        enabled=enabled and os.getenv("KRONOS_FORECAST_ENABLED", "1") == "1",
        window=int(os.getenv("KRONOS_WINDOW", "512")),
        pred_len=int(os.getenv("KRONOS_PRED_LEN", "24")),
        max_context=int(os.getenv("KRONOS_MAX_CONTEXT", "512")),
        model_id=os.getenv("KRONOS_MODEL_ID", "pretrained/Kronos-base"),
        small_model_id=os.getenv(
            "KRONOS_SMALL_MODEL_ID", "pretrained/Kronos-small"
        ),
        tokenizer_id=os.getenv(
            "KRONOS_TOKENIZER_ID", "pretrained/Kronos-Tokenizer-base"
        ),
        device=os.getenv("KRONOS_DEVICE", "cpu"),
        sample_count=int(os.getenv("KRONOS_SAMPLE_COUNT", "32")),
        top_k=int(os.getenv("KRONOS_TOP_K", "0")),
        seed=int(os.getenv("KRONOS_SEED", "100")),
        flat_tolerance=float(os.getenv("KRONOS_FLAT_TOLERANCE", "0")),
        adapters_root=os.getenv("KRONOS_ADAPTERS_DIR", "adapters"),
        lora_base_model_path=os.getenv(
            "KRONOS_LORA_BASE_MODEL_PATH", "pretrained/Kronos-base"
        ),
        lora_tokenizer_path=os.getenv(
            "KRONOS_LORA_TOKENIZER_PATH", "pretrained/Kronos-Tokenizer-base"
        ),
    )


def create_app(
    gateway: EventGateway | None = None,
    *,
    predictor: Predictor | None = None,
    forecast_config: ForecastConfig | None = None,
) -> FastAPI:
    """Create the API; ``gateway`` injection keeps tests independent of QMT."""

    injected_gateway = gateway is not None
    market_gateway = gateway or _gateway_from_environment()
    manager = ConnectionManager()
    bridge = MarketEventBridge(market_gateway, manager)
    config = forecast_config or _forecast_config_from_environment(
        enabled=not injected_gateway or predictor is not None
    )
    bridge.forecaster = KronosForecastService(
        config, bridge.publish_forecast, predictor=predictor
    )

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        await bridge.start()
        try:
            yield
        finally:
            await bridge.stop()

    api = FastAPI(title="Kronos Real-Time Market API", lifespan=lifespan)
    api.add_middleware(
        CORSMiddleware,
        allow_origins=[
            origin.strip()
            for origin in os.getenv(
                "CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173"
            ).split(",")
            if origin.strip()
        ],
        allow_credentials=True,
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["*"],
    )

    api.state.market_bridge = bridge

    @api.get("/health")
    async def health() -> dict[str, Any]:
        return {
            "status": "ok" if market_gateway.is_running else "degraded",
            "gateway_running": market_gateway.is_running,
            "websocket_clients": manager.connection_count,
            "forecast_enabled": config.enabled,
            "forecast_window": config.window,
            "forecast_pred_len": config.pred_len,
            "lora_available": config.enabled and bridge.forecaster.lora_available,
        }

    @api.get("/api/forecast/options")
    async def forecast_options() -> dict[str, Any]:
        return {
            "lora_available": config.enabled and bridge.forecaster.lora_available,
            "lora_default": False,
            "models": bridge.forecaster.model_options,
            "default_model_key": "base",
        }

    @api.get("/api/events/latest")
    async def latest_event() -> dict[str, Any] | None:
        return bridge.latest_event

    @api.post("/api/forecast/{symbol}")
    async def request_forecast(
        symbol: str, window: int = 512, pred_len: int = 24,
        use_lora: bool = False, model_key: str | None = None,
        modes: str = "greedy,median,mean",
        sample_count: int = 32,
    ) -> dict[str, Any]:
        normalized = symbol.strip().upper()
        try:
            accepted = bridge.request_forecast(
                normalized, window, pred_len, use_lora, model_key,
                _parse_modes(modes),
                sample_count,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return {
            "symbol": normalized, "status": "queued" if accepted else "disabled",
            "window": window, "pred_len": pred_len,
            "model_key": model_key or (
                bridge.forecaster.default_lora_key if use_lora else "base"
            ),
            "use_lora": use_lora or bool(
                model_key and bridge.forecaster.is_lora_key(model_key)
            ),
            "selected_modes": list(_parse_modes(modes)),
            "sample_count": sample_count,
        }

    @api.post("/api/subscriptions/{symbol}")
    async def subscribe_symbol(
        symbol: str, window: int = 512, pred_len: int = 24,
        model_key: str = "base", modes: str = "greedy,median,mean",
        sample_count: int = 32,
    ) -> dict[str, Any]:
        normalized = symbol.strip().upper()
        try:
            bridge.set_forecast_preferences(
                normalized, window, pred_len, model_key, _parse_modes(modes),
                sample_count,
            )
            subscription_id = await asyncio.to_thread(
                market_gateway.ensure_symbol, normalized
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        return {
            "symbol": normalized,
            "subscription_id": subscription_id,
            "status": "subscribed",
        }

    @api.delete("/api/subscriptions/{symbol}")
    async def unsubscribe_symbol(symbol: str) -> dict[str, Any]:
        normalized = symbol.strip().upper()
        active_clients = manager.subscriber_count(normalized)
        if active_clients:
            raise HTTPException(
                status_code=409,
                detail=f"{normalized} still has {active_clients} active viewer(s)",
            )
        try:
            removed = await asyncio.to_thread(
                market_gateway.unsubscribe_symbol, normalized
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        bridge.forget_symbol(normalized)
        return {
            "symbol": normalized,
            "status": "unsubscribed" if removed else "not_subscribed",
        }

    @api.websocket("/ws/market")
    async def market_events(websocket: WebSocket) -> None:
        requested_symbol = websocket.query_params.get("symbol")
        symbol = requested_symbol.strip().upper() if requested_symbol else None
        if symbol is not None:
            try:
                await asyncio.to_thread(market_gateway.ensure_symbol, symbol)
            except (ValueError, RuntimeError):
                await websocket.close(code=1008, reason="QMT subscription failed")
                return
        await manager.connect(
            websocket,
            symbol=symbol,
            initial_events=bridge.replay_events(symbol),
        )
        try:
            # Receiving keeps disconnect detection independent of market flow.
            while True:
                await websocket.receive_text()
        except WebSocketDisconnect:
            pass
        finally:
            manager.disconnect(websocket)

    return api


app = create_app()
