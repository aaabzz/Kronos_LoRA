"""Tests for the QMT real-time market-data gateway.

The deterministic test uses a fake XtData client, so it can run without a
MiniQMT login or an open market.  The live smoke test is opt-in:

    $env:RUN_QMT_LIVE_TEST = "1"
    $env:QMT_TEST_SYMBOL = "000001.SZ"
    python -m pytest tests/test_qmt_gateway.py -s
"""

from __future__ import annotations

import os
import queue
import time
from typing import Any, Callable

import pytest

from real_time.qmt_gateway import GatewayConfig, QmtGateway, print_bar


class FakeXtData:
    """Small in-memory replacement for the XtQuant API used by QmtGateway."""

    def __init__(self) -> None:
        self.callback: Callable[[dict[str, Any]], None] | None = None
        self.unsubscribed: list[int] = []
        self.subscription_symbols: list[str] = []

    def download_history_data(self, **_: Any) -> None:
        return None

    def get_market_data_ex(self, *, stock_list: list[str], **_: Any) -> dict[str, list[Any]]:
        return {stock_list[0]: []}

    def subscribe_quote(
        self,
        *,
        stock_code: str,
        callback: Callable[[dict[str, Any]], None],
        **_: Any,
    ) -> int:
        self.callback = callback
        self.subscription_symbols.append(stock_code)
        return 100 + len(self.subscription_symbols)

    def get_instrument_detail(self, symbol: str) -> dict[str, str] | None:
        if symbol == "999999.SZ":
            return None
        return {"InstrumentID": symbol}

    def unsubscribe_quote(self, subscription_id: int) -> None:
        self.unsubscribed.append(subscription_id)

    def emit(self, symbol: str, bar: dict[str, Any]) -> None:
        assert self.callback is not None, "gateway has not subscribed yet"
        self.callback({symbol: [bar]})


def _bar(timestamp_ms: int, close: float, volume: float) -> dict[str, Any]:
    return {
        "time": timestamp_ms,
        "open": 10.00,
        "high": max(10.05, close),
        "low": min(9.98, close),
        "close": close,
        "volume": volume,
        "amount": close * volume,
    }


def _next_market_event(gateway: QmtGateway, timeout: float = 2.0) -> dict[str, Any]:
    """Ignore lifecycle events and return the next bar event."""

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        remaining = max(0.01, deadline - time.monotonic())
        try:
            event = gateway.get_event(timeout=remaining)
        except queue.Empty:
            break
        if event["type"] in {"bar_update", "bar_closed"}:
            return event
    raise AssertionError("timed out waiting for a real-time bar event")


def test_history_window_defaults_to_512_and_rejects_larger_values() -> None:
    assert GatewayConfig(symbols=()).history_count == 512
    with pytest.raises(ValueError, match="cannot exceed 512"):
        GatewayConfig(symbols=(), history_count=513)


def test_gateway_starts_idle_until_first_dynamic_subscription() -> None:
    fake_xtdata = FakeXtData()
    gateway = QmtGateway(
        GatewayConfig(symbols=(), history_count=10, download_history=False),
        xtdata_module=fake_xtdata,
    )

    gateway.start()
    try:
        assert gateway.is_running is True
        assert gateway.subscription_ids == {}
        assert fake_xtdata.subscription_symbols == []

        subscription_id = gateway.ensure_symbol("000001.SZ")
        assert subscription_id == 101
        assert fake_xtdata.subscription_symbols == ["000001.SZ"]
    finally:
        gateway.stop()

    assert fake_xtdata.unsubscribed == [101]


def test_gateway_supplements_intraday_history_to_closed_window_size() -> None:
    class SupplementingXtData(FakeXtData):
        def __init__(self) -> None:
            super().__init__()
            self.downloaded = False
            self.download_calls: list[dict[str, Any]] = []
            self.requested_counts: list[int] = []

        def download_history_data(self, **kwargs: Any) -> None:
            self.downloaded = True
            self.download_calls.append(kwargs)

        def get_market_data_ex(
            self, *, stock_list: list[str], count: int, **_: Any
        ) -> dict[str, list[Any]]:
            self.requested_counts.append(count)
            available = 10 if self.downloaded else 4
            start = (
                int(time.time() * 1000) - (available - 1) * 5 * 60 * 1000 - 6 * 60 * 1000
                if self.downloaded
                else 1_754_271_000_000
            )
            return {
                stock_list[0]: [
                    _bar(start + index * 5 * 60 * 1000, 10.0 + index / 100, 1_000)
                    for index in range(available)
                ]
            }

    fake_xtdata = SupplementingXtData()
    gateway = QmtGateway(
        GatewayConfig(
            symbols=("000001.SZ",), history_count=10, download_history=True,
        ),
        xtdata_module=fake_xtdata,
    )

    gateway.start()
    try:
        history = gateway.get_event(timeout=1)
        assert history["type"] == "history_snapshot"
        assert history["count"] == 10
        assert history["required_count"] == 10
        assert history["history_complete"] is True
        assert isinstance(history["history_fresh"], bool)
        assert fake_xtdata.requested_counts == [11]
        assert len(fake_xtdata.download_calls) == 1
        assert fake_xtdata.download_calls[0]["start_time"]
    finally:
        gateway.stop()


def test_gateway_outputs_latest_realtime_bar(capsys: pytest.CaptureFixture[str]) -> None:
    """A changing 5m bar is emitted, and its latest state is the closed state."""

    symbol = "000001.SZ"
    first_5m = 1_754_271_000_000
    next_5m = first_5m + 5 * 60 * 1000
    fake_xtdata = FakeXtData()

    def print_latest(event: dict[str, Any]) -> None:
        if event["type"] in {"bar_update", "bar_closed"}:
            bar = event["bar"]
            print(
                f'{event["type"]} {bar["symbol"]} {bar["timestamp"]} '
                f'close={bar["close"]:.2f} volume={bar["volume"]:.0f}',
                flush=True,
            )

    gateway = QmtGateway(
        GatewayConfig(
            symbols=(symbol,),
            period="5m",
            history_count=10,
            download_history=False,
        ),
        event_listener=print_latest,
        xtdata_module=fake_xtdata,
    )

    gateway.start()
    try:
        fake_xtdata.emit(symbol, _bar(first_5m, close=10.01, volume=1_000))
        update_1 = _next_market_event(gateway)

        fake_xtdata.emit(symbol, _bar(first_5m, close=10.04, volume=8_000))
        update_2 = _next_market_event(gateway)

        fake_xtdata.emit(symbol, _bar(next_5m, close=10.05, volume=500))
        closed = _next_market_event(gateway)
        next_update = _next_market_event(gateway)
    finally:
        gateway.stop()

    assert update_1["type"] == "bar_update"
    assert update_1["bar"]["close"] == pytest.approx(10.01)
    assert update_2["type"] == "bar_update"
    assert update_2["bar"]["close"] == pytest.approx(10.04)

    # The first bar must close with its most recently received values.
    assert closed["type"] == "bar_closed"
    assert closed["reason"] == "newer_bar"
    assert closed["bar"]["timestamp_ms"] == first_5m
    assert closed["bar"]["close"] == pytest.approx(10.04)
    assert closed["bar"]["volume"] == pytest.approx(8_000)

    assert next_update["type"] == "bar_update"
    assert next_update["bar"]["timestamp_ms"] == next_5m
    assert next_update["bar"]["close"] == pytest.approx(10.05)
    assert fake_xtdata.unsubscribed == [101]

    output = capsys.readouterr().out
    assert "bar_update 000001.SZ" in output
    assert "close=10.04 volume=8000" in output
    assert "bar_closed 000001.SZ" in output
    assert "close=10.05 volume=500" in output


def test_gateway_closes_elapsed_bar_without_waiting_for_next_session_bar() -> None:
    symbol = "000001.SZ"
    fake_xtdata = FakeXtData()
    gateway = QmtGateway(
        GatewayConfig(
            symbols=(symbol,), period="5m", history_count=10,
            download_history=False,
        ),
        xtdata_module=fake_xtdata,
    )

    gateway.start()
    try:
        elapsed_timestamp = int(time.time() * 1000) - 6 * 60 * 1000
        fake_xtdata.emit(symbol, _bar(elapsed_timestamp, close=10.08, volume=1_200))
        update = _next_market_event(gateway)
        closed = _next_market_event(gateway)
    finally:
        gateway.stop()

    assert update["type"] == "bar_update"
    assert closed["type"] == "bar_closed"
    assert closed["reason"] == "period_elapsed"
    assert closed["bar"]["timestamp_ms"] == elapsed_timestamp


def test_normalization_rounds_prices_and_va(capsys: pytest.CaptureFixture[str]) -> None:
    fake_xtdata = FakeXtData()
    gateway = QmtGateway(
        GatewayConfig(
            symbols=("000001.SZ",),
            period="5m",
            history_count=10,
            download_history=False,
        ),
        event_listener=print_bar,
        xtdata_module=fake_xtdata,
    )

    gateway.start()
    try:
        fake_xtdata.emit(
            "000001.SZ",
            {
                "time": 1_754_271_000_000,
                "open": 10.126,
                "high": 10.987,
                "low": 9.874,
                "close": 10.555,
                "volume": 1_000.6,
                "amount": 10_555.7,
            },
        )
        event = _next_market_event(gateway)
    finally:
        gateway.stop()

    bar = event["bar"]
    assert bar["open"] == 10.13
    assert bar["high"] == 10.99
    assert bar["low"] == 9.87
    assert bar["close"] == 10.56
    assert bar["volume"] == 1_001
    assert bar["amount"] == 10_556
    assert isinstance(bar["volume"], int)
    assert isinstance(bar["amount"], int)

    output = capsys.readouterr().out
    assert "2025-08-04 09:30:00" in output
    assert "O=10.13 H=10.99 L=9.87 C=10.56 V=1001 A=10556" in output


def test_gateway_dynamically_subscribes_new_symbol() -> None:
    fake_xtdata = FakeXtData()
    gateway = QmtGateway(
        GatewayConfig(
            symbols=("000001.SZ",),
            period="5m",
            history_count=10,
            download_history=False,
        ),
        xtdata_module=fake_xtdata,
    )

    gateway.start()
    try:
        subscription_id = gateway.ensure_symbol("600519.sh")
        duplicate_id = gateway.ensure_symbol("600519.SH")
        assert subscription_id == 102
        assert duplicate_id == subscription_id
        assert fake_xtdata.subscription_symbols == ["000001.SZ", "600519.SH"]
        assert gateway.subscription_ids["600519.SH"] == 102

        with pytest.raises(ValueError, match="Unknown QMT symbol"):
            gateway.ensure_symbol("999999.SZ")
    finally:
        gateway.stop()

    assert fake_xtdata.unsubscribed == [101, 102]


def test_gateway_unsubscribes_dynamic_symbol() -> None:
    fake_xtdata = FakeXtData()
    gateway = QmtGateway(
        GatewayConfig(
            symbols=("000001.SZ",),
            period="5m",
            history_count=10,
            download_history=False,
        ),
        xtdata_module=fake_xtdata,
    )

    gateway.start()
    try:
        gateway.ensure_symbol("600519.SH")
        assert gateway.unsubscribe_symbol("600519.SH") is True
        assert gateway.unsubscribe_symbol("600519.SH") is False
        assert "600519.SH" not in gateway.subscription_ids
    finally:
        gateway.stop()

    assert fake_xtdata.unsubscribed == [102, 101]


@pytest.mark.skipif(
    os.getenv("RUN_QMT_LIVE_TEST") != "1",
    reason="set RUN_QMT_LIVE_TEST=1 while MiniQMT is logged in",
)
def test_live_qmt_prints_latest_bar() -> None:
    """Connect to a real MiniQMT session and print at least one live bar update."""

    symbol = os.getenv("QMT_TEST_SYMBOL", "000001.SZ").strip().upper()
    period = os.getenv("QMT_TEST_PERIOD", "5m").strip()
    timeout = float(os.getenv("QMT_TEST_TIMEOUT", "20"))

    gateway = QmtGateway(
        GatewayConfig(
            symbols=(symbol,),
            period=period,
            history_count=10,
            download_history=False,
        )
    )

    gateway.start()
    try:
        deadline = time.monotonic() + timeout
        latest_bar: dict[str, Any] | None = None
        collection_deadline: float | None = None
        while time.monotonic() < deadline:
            if collection_deadline is not None and time.monotonic() >= collection_deadline:
                break
            try:
                wait_until = collection_deadline if collection_deadline is not None else deadline
                event = gateway.get_event(timeout=min(1.0, max(0.01, wait_until - time.monotonic())))
            except queue.Empty:
                continue
            if event["type"] != "bar_update":
                continue
            # Bootstrap may expose the latest cached, still-forming bar.  The
            # live test specifically requires a later subscription callback.
            if event.get("source") == "history_bootstrap":
                continue

            bar = event["bar"]
            if latest_bar is None or bar["timestamp_ms"] >= latest_bar["timestamp_ms"]:
                latest_bar = bar
            # One callback can contain several incremental bars. Give the
            # worker a short window to drain that batch, then print the newest.
            if collection_deadline is None:
                collection_deadline = min(deadline, time.monotonic() + 1.0)

        if latest_bar is not None:
            bar = latest_bar
            print_bar({"type": "bar_update", "bar": bar})
            assert bar["symbol"] == symbol
            assert bar["period"] == period
            return
    finally:
        gateway.stop()

    pytest.fail(
        f"QMT did not push a {period} bar for {symbol} within {timeout:g}s; "
        "check MiniQMT login, quote permission, symbol, and trading hours"
    )
