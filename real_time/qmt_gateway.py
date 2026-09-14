"""QMT real-time market-data gateway for Kronos.

The gateway keeps XtQuant callbacks deliberately lightweight. Callback payloads
are copied into a bounded queue and normalized on a worker thread. Consumers can
read normalized events from :meth:`QmtGateway.get_event` without importing QMT.

Event types
-----------
``history_snapshot``
    Initial closed-bar history for one symbol.
``bar_update``
    The current, still-forming bar changed.
``bar_closed``
    A bar became final because a newer bar arrived.
``status``
    Lifecycle and diagnostic information.

This module collects market data only. Model inference must run in a separate
worker so a slow prediction can never block XtQuant's callback thread.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import queue
import re
import signal
import sys
import threading
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from decimal import Decimal, ROUND_HALF_UP
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

try:
    import pandas as pd
except ImportError as exc:  # pragma: no cover - environment error
    raise RuntimeError("pandas is required by qmt_gateway.py") from exc

try:
    from xtquant import xtdata
except ImportError as exc:  # pragma: no cover - environment error
    raise RuntimeError(
        "xtquant is required. Install it with: python -m pip install xtquant"
    ) from exc


LOGGER = logging.getLogger("qmt_gateway")
CN_TZ = ZoneInfo("Asia/Shanghai")
UTC = timezone.utc
STOP = object()
SYMBOL_PATTERN = re.compile(r"^[0-9A-Z]+\.(?:SZ|SH|BJ)$")


@dataclass(frozen=True, slots=True)
class GatewayConfig:
    """Runtime configuration for :class:`QmtGateway`."""

    symbols: tuple[str, ...]
    period: str = "5m"
    history_count: int = 512
    history_start: str = ""
    history_end: str = ""
    dividend_type: str = "none"
    fill_data: bool = False
    download_history: bool = True
    raw_queue_size: int = 4096
    event_queue_size: int = 4096
    jsonl_path: Path | None = None

    def __post_init__(self) -> None:
        normalized = tuple(dict.fromkeys(s.strip().upper() for s in self.symbols if s.strip()))
        object.__setattr__(self, "symbols", normalized)
        if self.history_count <= 0:
            raise ValueError("history_count must be greater than zero")
        if self.history_count > 512:
            raise ValueError("history_count cannot exceed 512")
        if self.raw_queue_size <= 0 or self.event_queue_size <= 0:
            raise ValueError("queue sizes must be greater than zero")
        if not self.period.strip():
            raise ValueError("period cannot be empty")


@dataclass(frozen=True, slots=True)
class MarketBar:
    """Normalized OHLCVA bar used by downstream Kronos workers."""

    symbol: str
    period: str
    timestamp_ms: int
    timestamp: str
    open: float
    high: float
    low: float
    close: float
    volume: int
    amount: int

    def validate(self) -> None:
        values = (self.open, self.high, self.low, self.close, self.volume, self.amount)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("bar contains a non-finite numeric value")
        if self.high < self.low:
            raise ValueError("bar high is lower than bar low")
        if self.high < max(self.open, self.close):
            raise ValueError("bar high is lower than open or close")
        if self.low > min(self.open, self.close):
            raise ValueError("bar low is higher than open or close")
        if self.volume < 0 or self.amount < 0:
            raise ValueError("bar volume or amount is negative")


class QmtGateway:
    """Collect, normalize, and publish QMT market-data events.

    Parameters
    ----------
    config:
        Gateway settings.
    event_listener:
        Optional callback invoked synchronously on the thread emitting each
        normalized event. It must return quickly. Consumers that may block
        should use :meth:`get_event` instead.
    xtdata_module:
        Dependency-injection hook used by tests. Defaults to ``xtquant.xtdata``.
    """

    def __init__(
        self,
        config: GatewayConfig,
        event_listener: Callable[[dict[str, Any]], None] | None = None,
        xtdata_module: Any = xtdata,
    ) -> None:
        self.config = config
        self._event_listener = event_listener
        self._xtdata = xtdata_module

        self._raw_queue: queue.Queue[Any] = queue.Queue(maxsize=config.raw_queue_size)
        self._event_queue: queue.Queue[dict[str, Any]] = queue.Queue(
            maxsize=config.event_queue_size
        )
        self._stop_event = threading.Event()
        self._processor_thread: threading.Thread | None = None
        self._subscription_ids: dict[str, int] = {}
        self._subscription_lock = threading.RLock()
        self._initializing_symbols: set[str] = set()
        self._pending_live_bars: dict[str, list[MarketBar]] = {}
        self._forming_bars: dict[str, MarketBar] = {}
        self._last_closed_timestamp: dict[str, int] = {}
        self._dropped_raw_messages = 0
        self._dropped_events = 0
        self._started = False
        self._jsonl_file: Any = None

    @property
    def is_running(self) -> bool:
        return self._started and not self._stop_event.is_set()

    @property
    def subscription_ids(self) -> dict[str, int]:
        return dict(self._subscription_ids)

    def start(self) -> None:
        """Start processing, bootstrap history, and subscribe to all symbols.

        This method does not call the blocking ``xtdata.run()``. Use
        :meth:`run_forever` for a complete command-line lifecycle.
        """

        if self._started:
            raise RuntimeError("QmtGateway has already been started")

        self._open_jsonl()
        self._processor_thread = threading.Thread(
            target=self._processor_loop,
            name="qmt-event-processor",
            daemon=True,
        )
        self._processor_thread.start()

        try:
            self._bootstrap_history()
            self._subscribe_all()
        except Exception:
            self.stop()
            raise

        self._started = True
        self._publish_status(
            "running",
            subscriptions=self.subscription_ids,
            symbols=list(self.config.symbols),
            period=self.config.period,
        )

    def run_forever(self) -> None:
        """Start the gateway and enter XtQuant's blocking event loop."""

        self.start()
        try:
            self._xtdata.run()
        except KeyboardInterrupt:
            LOGGER.info("Keyboard interrupt received")
        finally:
            self.stop()

    def stop(self) -> None:
        """Unsubscribe and stop the processor thread. Safe to call repeatedly."""

        if self._stop_event.is_set():
            return

        self._stop_event.set()

        with self._subscription_lock:
            subscriptions = list(self._subscription_ids.items())
            self._subscription_ids.clear()
            self._initializing_symbols.clear()
            self._pending_live_bars.clear()

        for symbol, subscription_id in subscriptions:
            try:
                self._xtdata.unsubscribe_quote(subscription_id)
            except Exception:
                LOGGER.exception(
                    "Failed to unsubscribe %s (subscription=%s)",
                    symbol,
                    subscription_id,
                )
        self._put_latest(self._raw_queue, STOP, count_drop=True)
        if self._processor_thread and self._processor_thread.is_alive():
            self._processor_thread.join(timeout=5)

        self._publish_status(
            "stopped",
            dropped_raw_messages=self._dropped_raw_messages,
            dropped_events=self._dropped_events,
        )
        self._close_jsonl()
        self._started = False

    def get_event(self, timeout: float | None = None) -> dict[str, Any]:
        """Return the next normalized event for a downstream consumer."""

        return self._event_queue.get(timeout=timeout)

    def iter_events(self, timeout: float = 1.0) -> Iterable[dict[str, Any]]:
        """Yield normalized events until the gateway has stopped and drained."""

        while self.is_running or not self._event_queue.empty():
            try:
                yield self.get_event(timeout=timeout)
            except queue.Empty:
                continue

    def ensure_symbol(self, symbol: str) -> int:
        """Validate and dynamically subscribe one symbol if necessary."""

        normalized = symbol.strip().upper()
        if not SYMBOL_PATTERN.fullmatch(normalized):
            raise ValueError(f"Invalid QMT symbol format: {symbol!r}")
        if not self.is_running:
            raise RuntimeError("QMT gateway must be running before adding symbols")

        with self._subscription_lock:
            existing = self._subscription_ids.get(normalized)
            if existing is not None:
                return existing

            detail_getter = getattr(self._xtdata, "get_instrument_detail", None)
            if callable(detail_getter) and not detail_getter(normalized):
                raise ValueError(f"Unknown QMT symbol: {normalized}")

            self._initializing_symbols.add(normalized)
            subscription_id: int | None = None
            try:
                subscription_id = self._subscribe_symbol(normalized)
                self._bootstrap_symbol(normalized)
            except Exception:
                if subscription_id is not None:
                    self._subscription_ids.pop(normalized, None)
                    try:
                        self._xtdata.unsubscribe_quote(subscription_id)
                    except Exception:
                        LOGGER.exception("Failed to roll back subscription for %s", normalized)
                raise
            finally:
                self._initializing_symbols.discard(normalized)

            pending = self._pending_live_bars.pop(normalized, [])
            for bar in self._deduplicate_bars(sorted(pending, key=lambda item: item.timestamp_ms)):
                self._ingest_live_bar(bar)

            self._publish_status(
                "symbol_subscribed",
                symbol=normalized,
                subscription_id=subscription_id,
            )
            return subscription_id

    def unsubscribe_symbol(self, symbol: str) -> bool:
        """Stop QMT updates for one dynamically selected symbol."""

        normalized = symbol.strip().upper()
        if not SYMBOL_PATTERN.fullmatch(normalized):
            raise ValueError(f"Invalid QMT symbol format: {symbol!r}")

        with self._subscription_lock:
            subscription_id = self._subscription_ids.pop(normalized, None)
            self._initializing_symbols.discard(normalized)
            self._pending_live_bars.pop(normalized, None)

        if subscription_id is None:
            return False

        self._xtdata.unsubscribe_quote(subscription_id)
        self._forming_bars.pop(normalized, None)
        self._last_closed_timestamp.pop(normalized, None)
        self._publish_status(
            "symbol_unsubscribed",
            symbol=normalized,
            subscription_id=subscription_id,
        )
        LOGGER.info(
            "Unsubscribed %s subscription=%s", normalized, subscription_id
        )
        return True

    def _subscribe_all(self) -> None:
        for symbol in self.config.symbols:
            self._subscribe_symbol(symbol)

    def _subscribe_symbol(self, symbol: str) -> int:
        subscription_id = self._xtdata.subscribe_quote(
            stock_code=symbol,
            period=self.config.period,
            start_time="",
            end_time="",
            count=0,
            callback=self._on_quote,
        )
        if not isinstance(subscription_id, int) or subscription_id < 0:
            raise RuntimeError(f"QMT subscription failed for {symbol}: {subscription_id!r}")
        self._subscription_ids[symbol] = subscription_id
        LOGGER.info(
            "Subscribed %s period=%s subscription=%s",
            symbol,
            self.config.period,
            subscription_id,
        )
        return subscription_id

    def _bootstrap_history(self) -> None:
        for symbol in self.config.symbols:
            self._bootstrap_symbol(symbol)

    def _bootstrap_symbol(self, symbol: str) -> None:
        required_count = self.config.history_count
        now = datetime.now(tz=CN_TZ)
        if self.config.download_history:
            download_start = self._history_download_start(required_count, now)
            LOGGER.info(
                "Synchronizing the latest %d QMT bars for %s from %s",
                required_count,
                symbol,
                download_start,
            )
            self._xtdata.download_history_data(
                stock_code=symbol,
                period=self.config.period,
                start_time=download_start,
                end_time=self.config.history_end,
            )

        bars = self._read_history(symbol, required_count + 1)
        closed_bars, forming_bar = self._split_closed_history(bars, now)
        history_fresh = self._history_is_fresh(closed_bars, now)

        closed_bars = closed_bars[-required_count:]
        self._forming_bars.pop(symbol, None)
        if forming_bar is not None:
            self._forming_bars[symbol] = forming_bar

        if len(closed_bars) < required_count:
            LOGGER.warning(
                "QMT returned only %d/%d closed bars for %s",
                len(closed_bars),
                required_count,
                symbol,
            )

        if not bars:
            LOGGER.warning("No QMT history returned for %s", symbol)
            self._emit(self._event(
                "history_snapshot", symbol=symbol, bars=[], count=0,
                required_count=required_count, history_complete=False,
                history_fresh=False,
            ))
            return

        if closed_bars:
            self._last_closed_timestamp[symbol] = closed_bars[-1].timestamp_ms

        self._emit(self._event(
            "history_snapshot",
            symbol=symbol,
            bars=[asdict(bar) for bar in closed_bars],
            count=len(closed_bars),
            required_count=required_count,
            history_complete=len(closed_bars) >= required_count,
            history_fresh=history_fresh,
        ))
        if forming_bar is not None:
            self._emit(self._event(
                "bar_update", symbol=symbol, bar=asdict(forming_bar),
                source="history_bootstrap",
            ))

        LOGGER.info(
            "Bootstrapped %s with %d/%d closed bars%s",
            symbol,
            len(closed_bars),
            required_count,
            " and one forming bar" if forming_bar is not None else "",
        )

    def _read_history(self, symbol: str, count: int) -> list[MarketBar]:
        payload = self._xtdata.get_market_data_ex(
            field_list=[],
            stock_list=[symbol],
            period=self.config.period,
            start_time=self.config.history_start,
            end_time=self.config.history_end,
            count=count,
            dividend_type=self.config.dividend_type,
            fill_data=self.config.fill_data,
        )
        raw_rows = payload.get(symbol) if isinstance(payload, Mapping) else None
        bars = sorted(
            self._normalize_rows(symbol, raw_rows),
            key=lambda bar: bar.timestamp_ms,
        )
        return self._deduplicate_bars(bars)

    def _split_closed_history(
        self, bars: list[MarketBar], now: datetime
    ) -> tuple[list[MarketBar], MarketBar | None]:
        if not bars or self._is_bar_closed(bars[-1], now):
            return bars, None
        return bars[:-1], bars[-1]

    def _history_download_start(self, minimum_count: int, now: datetime) -> str:
        """Choose a conservative date range able to cover the forecast window."""

        period = self.config.period.lower().strip()
        bars_per_day = 1
        if period.endswith("m") and period[:-1].isdigit():
            bars_per_day = max(1, 240 // int(period[:-1]))
        elif period.endswith("h") and period[:-1].isdigit():
            bars_per_day = max(1, 4 // int(period[:-1]))
        trading_days = math.ceil(minimum_count / bars_per_day)
        calendar_days = math.ceil(trading_days * 7 / 5) + 10
        calculated = (now - timedelta(days=calendar_days)).strftime("%Y%m%d")
        configured = self.config.history_start.strip()
        return min(configured, calculated) if configured else calculated

    def _history_is_fresh(self, closed_bars: list[MarketBar], now: datetime) -> bool:
        if not closed_bars:
            return False
        expected = self._expected_latest_closed_start(now)
        if expected is None:
            return True
        return closed_bars[-1].timestamp_ms >= int(expected.timestamp() * 1000)

    def _expected_latest_closed_start(self, now: datetime) -> datetime | None:
        """Return the newest bar start that should already be finalized."""

        period = self.config.period.lower().strip()

        def previous_weekday(day: datetime) -> datetime:
            candidate = day - timedelta(days=1)
            while candidate.weekday() >= 5:
                candidate -= timedelta(days=1)
            return candidate

        if period == "1d":
            market_close = now.replace(hour=15, minute=0, second=0, microsecond=0)
            day = now if now.weekday() < 5 and now >= market_close else previous_weekday(now)
            return day.replace(hour=0, minute=0, second=0, microsecond=0)
        if period.endswith("m") and period[:-1].isdigit():
            delta = timedelta(minutes=int(period[:-1]))
        elif period.endswith("h") and period[:-1].isdigit():
            delta = timedelta(hours=int(period[:-1]))
        else:
            return None

        if now.weekday() >= 5:
            prior = previous_weekday(now)
            return prior.replace(hour=15, minute=0, second=0, microsecond=0) - delta

        morning_start = now.replace(hour=9, minute=30, second=0, microsecond=0)
        morning_end = now.replace(hour=11, minute=30, second=0, microsecond=0)
        afternoon_start = now.replace(hour=13, minute=0, second=0, microsecond=0)
        market_close = now.replace(hour=15, minute=0, second=0, microsecond=0)

        if now < morning_start + delta:
            prior = previous_weekday(now)
            return prior.replace(hour=15, minute=0, second=0, microsecond=0) - delta
        if now <= morning_end:
            completed = int((now - morning_start) // delta)
            return morning_start + (completed - 1) * delta
        if now < afternoon_start + delta:
            return morning_end - delta
        if now <= market_close:
            completed = int((now - afternoon_start) // delta)
            return afternoon_start + (completed - 1) * delta
        return market_close - delta

    def _on_quote(self, payload: Any) -> None:
        """XtQuant callback. Never perform parsing, I/O, or inference here."""

        if self._stop_event.is_set():
            return
        self._put_latest(self._raw_queue, payload, count_drop=True)

    def _processor_loop(self) -> None:
        while not self._stop_event.is_set() or not self._raw_queue.empty():
            try:
                payload = self._raw_queue.get(timeout=0.5)
            except queue.Empty:
                self._close_due_forming_bars()
                continue

            if payload is STOP:
                break

            try:
                self._process_payload(payload)
            except Exception:
                LOGGER.exception("Failed to process QMT callback payload")
                self._publish_status("payload_error")
            finally:
                self._raw_queue.task_done()

    def _close_due_forming_bars(self) -> None:
        """Finalize bars whose period elapsed without a newer QMT bar.

        This is essential at 11:30 and 15:00, when no immediate next-session
        candle exists to trigger the normal ``newer_bar`` transition.
        """

        now = datetime.now(tz=CN_TZ)
        for symbol, bar in tuple(self._forming_bars.items()):
            if self._is_bar_closed(bar, now):
                self._close_forming_bar(symbol, reason="period_elapsed")

    def _process_payload(self, payload: Any) -> None:
        if not isinstance(payload, Mapping):
            raise TypeError(f"Unexpected QMT callback type: {type(payload)!r}")

        for raw_symbol, raw_rows in payload.items():
            symbol = str(raw_symbol).strip().upper()
            bars = sorted(
                self._normalize_rows(symbol, raw_rows),
                key=lambda bar: bar.timestamp_ms,
            )
            with self._subscription_lock:
                if symbol in self._initializing_symbols:
                    self._pending_live_bars.setdefault(symbol, []).extend(bars)
                    continue
            for bar in self._deduplicate_bars(bars):
                self._ingest_live_bar(bar)

    def _ingest_live_bar(self, bar: MarketBar) -> None:
        symbol = bar.symbol
        current = self._forming_bars.get(symbol)

        if current is None:
            last_closed = self._last_closed_timestamp.get(symbol, -1)
            if bar.timestamp_ms <= last_closed:
                return
            self._forming_bars[symbol] = bar
            self._emit(self._event("bar_update", symbol=symbol, bar=asdict(bar)))
            return

        if bar.timestamp_ms < current.timestamp_ms:
            LOGGER.debug(
                "Ignoring out-of-order bar %s %s; current=%s",
                symbol,
                bar.timestamp,
                current.timestamp,
            )
            return

        if bar.timestamp_ms == current.timestamp_ms:
            self._forming_bars[symbol] = bar
            self._emit(self._event("bar_update", symbol=symbol, bar=asdict(bar)))
            return

        self._close_forming_bar(symbol, reason="newer_bar")

        self._forming_bars[symbol] = bar
        self._emit(self._event("bar_update", symbol=symbol, bar=asdict(bar)))

    def _close_forming_bar(self, symbol: str, *, reason: str) -> MarketBar | None:
        """Close and publish the current bar exactly once for ``symbol``."""

        current = self._forming_bars.pop(symbol, None)
        if current is None:
            return None

        last_closed = self._last_closed_timestamp.get(symbol, -1)
        if current.timestamp_ms <= last_closed:
            return None

        self._last_closed_timestamp[symbol] = current.timestamp_ms
        self._emit(
            self._event(
                "bar_closed",
                symbol=symbol,
                bar=asdict(current),
                reason=reason,
            )
        )
        return current

    def _normalize_rows(self, symbol: str, raw_rows: Any) -> list[MarketBar]:
        rows = list(self._iter_rows(raw_rows))
        result: list[MarketBar] = []
        for row in rows:
            try:
                result.append(self._normalize_bar(symbol, row))
            except (KeyError, TypeError, ValueError):
                LOGGER.exception("Invalid QMT bar for %s: %r", symbol, row)
        return result

    @staticmethod
    def _iter_rows(raw_rows: Any) -> Iterable[Mapping[str, Any]]:
        if raw_rows is None:
            return
        if isinstance(raw_rows, pd.DataFrame):
            frame = raw_rows
            if "time" not in frame.columns and frame.index.name:
                frame = frame.reset_index()
            for row in frame.to_dict(orient="records"):
                yield row
            return
        if isinstance(raw_rows, Mapping):
            yield raw_rows
            return
        if isinstance(raw_rows, Sequence) and not isinstance(
            raw_rows, (str, bytes, bytearray)
        ):
            for row in raw_rows:
                if isinstance(row, Mapping):
                    yield row
            return
        raise TypeError(f"Unsupported QMT row container: {type(raw_rows)!r}")

    def _normalize_bar(self, symbol: str, row: Mapping[str, Any]) -> MarketBar:
        timestamp_value = self._first(row, "time", "timestamp", "datetime")
        timestamp = self._parse_timestamp(timestamp_value)

        bar = MarketBar(
            symbol=symbol,
            period=self.config.period,
            timestamp_ms=int(timestamp.timestamp() * 1000),
            timestamp=timestamp.isoformat(),
            open=self._round_half_up(self._number(row, "open"), decimals=2),
            high=self._round_half_up(self._number(row, "high"), decimals=2),
            low=self._round_half_up(self._number(row, "low"), decimals=2),
            close=self._round_half_up(self._number(row, "close"), decimals=2),
            volume=int(
                self._round_half_up(
                    self._number(row, "volume", "vol", default=0.0), decimals=0
                )
            ),
            amount=int(
                self._round_half_up(
                    self._number(row, "amount", "turnover", default=0.0), decimals=0
                )
            ),
        )
        bar.validate()
        return bar

    @staticmethod
    def _round_half_up(value: float, *, decimals: int) -> float:
        """Round decimal input conventionally instead of using banker's rounding."""

        if not math.isfinite(value):
            raise ValueError("cannot round a non-finite numeric value")
        quantum = Decimal("1").scaleb(-decimals)
        return float(Decimal(str(value)).quantize(quantum, rounding=ROUND_HALF_UP))

    @staticmethod
    def _first(row: Mapping[str, Any], *names: str) -> Any:
        for name in names:
            if name in row and row[name] is not None:
                return row[name]
        raise KeyError(f"Missing required field; expected one of {names}")

    @classmethod
    def _number(
        cls,
        row: Mapping[str, Any],
        *names: str,
        default: float | None = None,
    ) -> float:
        try:
            value = cls._first(row, *names)
        except KeyError:
            if default is None:
                raise
            return default
        if pd.isna(value):
            if default is not None:
                return default
            raise ValueError(f"Field {names} contains NaN")
        return float(value)

    @staticmethod
    def _parse_timestamp(value: Any) -> datetime:
        if isinstance(value, datetime):
            timestamp = value
            if timestamp.tzinfo is None:
                return timestamp.replace(tzinfo=CN_TZ)
            return timestamp.astimezone(CN_TZ)

        if isinstance(value, (int, float)) and not isinstance(value, bool):
            numeric = float(value)
            absolute = abs(numeric)
            if absolute >= 1e17:
                unit = "ns"
            elif absolute >= 1e14:
                unit = "us"
            elif absolute >= 1e11:
                unit = "ms"
            else:
                unit = "s"
            parsed = pd.to_datetime(numeric, unit=unit, utc=True)
            return parsed.to_pydatetime().astimezone(CN_TZ)

        parsed = pd.to_datetime(value, errors="raise")
        if isinstance(parsed, pd.DatetimeIndex):
            if len(parsed) != 1:
                raise ValueError("timestamp must be scalar")
            parsed = parsed[0]
        timestamp = parsed.to_pydatetime()
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=CN_TZ)
        return timestamp.astimezone(CN_TZ)

    @staticmethod
    def _deduplicate_bars(bars: list[MarketBar]) -> list[MarketBar]:
        by_timestamp: dict[int, MarketBar] = {}
        for bar in bars:
            by_timestamp[bar.timestamp_ms] = bar
        return [by_timestamp[key] for key in sorted(by_timestamp)]

    def _is_bar_closed(self, bar: MarketBar, now: datetime) -> bool:
        start = datetime.fromtimestamp(bar.timestamp_ms / 1000, tz=CN_TZ)
        period = self.config.period.lower().strip()

        if period.endswith("m") and period[:-1].isdigit():
            return start + timedelta(minutes=int(period[:-1])) <= now
        if period.endswith("h") and period[:-1].isdigit():
            return start + timedelta(hours=int(period[:-1])) <= now
        if period == "1d":
            market_close = start.replace(hour=15, minute=0, second=0, microsecond=0)
            return now >= market_close

        # Tick and unknown/custom periods are treated as closed observations.
        return True

    def _event(self, event_type: str, symbol: str | None = None, **data: Any) -> dict[str, Any]:
        event: dict[str, Any] = {
            "type": event_type,
            "period": self.config.period,
            "emitted_at": datetime.now(tz=UTC).isoformat(),
        }
        if symbol is not None:
            event["symbol"] = symbol
        event.update(data)
        return event

    def _publish_status(self, state: str, **data: Any) -> None:
        self._emit(self._event("status", state=state, **data))

    def _emit(self, event: dict[str, Any]) -> None:
        self._write_jsonl(event)

        if self._event_listener is not None:
            try:
                self._event_listener(event)
            except Exception:
                LOGGER.exception("QMT event listener failed")

        self._put_latest(self._event_queue, event, count_drop=False)

    def _put_latest(
        self,
        target: queue.Queue[Any],
        item: Any,
        *,
        count_drop: bool,
    ) -> None:
        try:
            target.put_nowait(item)
            return
        except queue.Full:
            pass

        try:
            target.get_nowait()
            target.task_done()
        except queue.Empty:
            pass

        if count_drop:
            self._dropped_raw_messages += 1
        else:
            self._dropped_events += 1

        try:
            target.put_nowait(item)
        except queue.Full:
            LOGGER.warning("Queue remained full after dropping its oldest item")

    def _open_jsonl(self) -> None:
        path = self.config.jsonl_path
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        self._jsonl_file = path.open("a", encoding="utf-8", buffering=1)

    def _write_jsonl(self, event: dict[str, Any]) -> None:
        if self._jsonl_file is None:
            return
        self._jsonl_file.write(
            json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n"
        )

    def _close_jsonl(self) -> None:
        if self._jsonl_file is not None:
            self._jsonl_file.close()
            self._jsonl_file = None


def print_bar(event: dict[str, Any]) -> None:
    """Print normalized real-time bars and log non-bar gateway events."""

    event_type = event.get("type")
    if event_type in {"bar_update", "bar_closed"}:
        bar = event["bar"]
        timestamp = datetime.fromisoformat(bar["timestamp"]).strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        print(
            f'{event_type} {bar["symbol"]} {timestamp} '
            f'O={bar["open"]:.2f} H={bar["high"]:.2f} '
            f'L={bar["low"]:.2f} C={bar["close"]:.2f} '
            f'V={bar["volume"]} A={bar["amount"]}',
            flush=True,
        )
    elif event_type == "history_snapshot":
        LOGGER.info("HISTORY %s count=%s", event.get("symbol"), event.get("count"))
    elif event_type == "status":
        LOGGER.info("STATUS %s", event.get("state"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Subscribe to QMT bars and publish normalized market events."
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        required=True,
        help="QMT symbols, for example 600519.SH 000001.SZ",
    )
    parser.add_argument("--period", default="5m", help="QMT period, default: 5m")
    parser.add_argument("--history-count", type=int, default=512)
    parser.add_argument("--history-start", default="")
    parser.add_argument("--history-end", default="")
    parser.add_argument("--dividend-type", default="none")
    parser.add_argument(
        "--fill-data",
        action="store_true",
        help="Allow QMT to fill missing bars (disabled by default)",
    )
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="Read QMT local cache without downloading history first",
    )
    parser.add_argument(
        "--jsonl",
        type=Path,
        default=None,
        help="Optional append-only JSONL event output",
    )
    parser.add_argument("--raw-queue-size", type=int, default=4096)
    parser.add_argument("--event-queue-size", type=int, default=4096)
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate configuration and XtQuant imports without connecting",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    config = GatewayConfig(
        symbols=tuple(args.symbols),
        period=args.period,
        history_count=args.history_count,
        history_start=args.history_start,
        history_end=args.history_end,
        dividend_type=args.dividend_type,
        fill_data=args.fill_data,
        download_history=not args.no_download,
        raw_queue_size=args.raw_queue_size,
        event_queue_size=args.event_queue_size,
        jsonl_path=args.jsonl,
    )

    if args.check:
        LOGGER.info(
            "Configuration is valid; XtQuant import succeeded: symbols=%s period=%s",
            config.symbols,
            config.period,
        )
        return 0

    gateway = QmtGateway(config=config, event_listener=print_bar)

    def stop_gateway(signum: int, _frame: Any) -> None:
        LOGGER.info("Signal %s received, stopping QMT gateway", signum)
        gateway.stop()
        # xtdata.run() is blocking. Raising here guarantees control returns to
        # run_forever(), whose finally block is intentionally idempotent.
        raise KeyboardInterrupt

    for signal_name in ("SIGINT", "SIGTERM"):
        signal_value = getattr(signal, signal_name, None)
        if signal_value is not None:
            signal.signal(signal_value, stop_gateway)

    try:
        gateway.run_forever()
    except Exception:
        LOGGER.exception("QMT gateway terminated with an error")
        gateway.stop()
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
