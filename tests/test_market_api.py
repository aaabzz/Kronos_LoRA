from __future__ import annotations

import queue
import time
from datetime import datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
from fastapi.testclient import TestClient

from api.forecasting import ForecastConfig
from api.main import create_app


class FakeGateway:
    def __init__(self) -> None:
        self._events: queue.Queue[dict[str, Any]] = queue.Queue()
        self.is_running = False
        self.start_calls = 0
        self.stop_calls = 0
        self.subscriptions: dict[str, int] = {"000001.SZ": 101}
        self.unsubscribed: list[str] = []

    def start(self) -> None:
        self.start_calls += 1
        self.is_running = True

    def stop(self) -> None:
        self.stop_calls += 1
        self.is_running = False

    def get_event(self, timeout: float | None = None) -> dict[str, Any]:
        return self._events.get(timeout=timeout)

    def emit(self, event: dict[str, Any]) -> None:
        self._events.put(event)

    def ensure_symbol(self, symbol: str) -> int:
        normalized = symbol.strip().upper()
        if normalized == "999999.SZ":
            raise ValueError(f"Unknown QMT symbol: {normalized}")
        return self.subscriptions.setdefault(normalized, 100 + len(self.subscriptions) + 1)

    def unsubscribe_symbol(self, symbol: str) -> bool:
        normalized = symbol.strip().upper()
        removed = self.subscriptions.pop(normalized, None) is not None
        if removed:
            self.unsubscribed.append(normalized)
        return removed


class FakePredictor:
    def predict(self, **kwargs: Any) -> pd.DataFrame:
        count = kwargs["pred_len"]
        return pd.DataFrame({
            "open": [10.0] * count, "high": [10.2] * count,
            "low": [9.9] * count, "close": [10.1] * count,
            "volume": [1000] * count, "amount": [10100] * count,
        })


def test_health_and_gateway_lifecycle() -> None:
    gateway = FakeGateway()
    app = create_app(gateway)

    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {
            "status": "ok",
            "gateway_running": True,
            "websocket_clients": 0,
            "forecast_enabled": False,
            "forecast_window": 512,
            "forecast_pred_len": 24,
            "lora_available": False,
        }
        options = client.get("/api/forecast/options")
        assert options.status_code == 200
        assert options.json()["lora_available"] is False
        assert options.json()["lora_default"] is False
        assert gateway.start_calls == 1

    assert gateway.stop_calls == 1
    assert gateway.is_running is False


def test_websocket_broadcasts_qmt_event() -> None:
    gateway = FakeGateway()
    app = create_app(gateway)
    event = {
        "type": "bar_update",
        "period": "5m",
        "symbol": "000001.SZ",
        "bar": {
            "timestamp_ms": 1_754_271_000_000,
            "timestamp": "2025-08-04T09:30:00+08:00",
            "open": 10.13,
            "high": 10.99,
            "low": 9.87,
            "close": 10.56,
            "volume": 1001,
            "amount": 10556,
        },
    }

    with TestClient(app) as client:
        with client.websocket_connect("/ws/market") as websocket:
            gateway.emit(event)
            assert websocket.receive_json() == event

        latest = client.get("/api/events/latest")
        assert latest.status_code == 200
        assert latest.json() == event


def test_websocket_replays_cached_history_for_symbol() -> None:
    gateway = FakeGateway()
    app = create_app(gateway)
    history = {
        "type": "history_snapshot",
        "period": "5m",
        "symbol": "000001.SZ",
        "count": 1,
        "bars": [
            {
                "timestamp_ms": 1_754_271_000_000,
                "timestamp": "2025-08-04T09:30:00+08:00",
                "open": 10.0,
                "high": 10.2,
                "low": 9.9,
                "close": 10.1,
                "volume": 1000,
                "amount": 10100,
            }
        ],
    }

    with TestClient(app) as client:
        gateway.emit(history)
        for _ in range(100):
            if app.state.market_bridge.history_by_symbol:
                break
            import time

            time.sleep(0.01)

        with client.websocket_connect("/ws/market?symbol=000001.SZ") as websocket:
            assert websocket.receive_json() == history


def test_closed_bars_are_merged_into_replayable_history() -> None:
    gateway = FakeGateway()
    app = create_app(gateway)
    first_timestamp = 1_754_271_000_000
    history = {
        "type": "history_snapshot", "period": "5m", "symbol": "000001.SZ",
        "count": 1, "required_count": 3,
        "bars": [{
            "timestamp_ms": first_timestamp,
            "timestamp": "2025-08-04T09:30:00+08:00",
            "open": 10.0, "high": 10.2, "low": 9.9, "close": 10.1,
            "volume": 1000, "amount": 10100,
        }],
    }

    with TestClient(app):
        gateway.emit(history)
        for index in (1, 2):
            timestamp_ms = first_timestamp + index * 5 * 60 * 1000
            gateway.emit({
                "type": "bar_closed", "period": "5m", "symbol": "000001.SZ",
                "bar": {
                    "timestamp_ms": timestamp_ms,
                    "timestamp": f"2025-08-04T09:{30 + index * 5:02d}:00+08:00",
                    "open": 10.0 + index / 10, "high": 10.3 + index / 10,
                    "low": 9.9 + index / 10, "close": 10.1 + index / 10,
                    "volume": 1000 + index, "amount": 10100 + index,
                },
            })

        for _ in range(100):
            cached = app.state.market_bridge.history_by_symbol.get("000001.SZ")
            if cached and cached.get("count") == 3:
                break
            time.sleep(0.01)

        replay = app.state.market_bridge.replay_events("000001.SZ")
        assert replay[0]["type"] == "history_snapshot"
        assert replay[0]["count"] == 3
        assert [bar["timestamp_ms"] for bar in replay[0]["bars"]] == [
            first_timestamp,
            first_timestamp + 5 * 60 * 1000,
            first_timestamp + 10 * 60 * 1000,
        ]
        assert replay[0]["history_complete"] is True


def test_dynamic_subscription_endpoint() -> None:
    gateway = FakeGateway()
    app = create_app(gateway)

    with TestClient(app) as client:
        response = client.post("/api/subscriptions/600519.sh")
        assert response.status_code == 200
        assert response.json() == {
            "symbol": "600519.SH",
            "subscription_id": 102,
            "status": "subscribed",
        }

        invalid = client.post("/api/subscriptions/999999.SZ")
        assert invalid.status_code == 400
        assert invalid.json()["detail"] == "Unknown QMT symbol: 999999.SZ"


def test_dynamic_unsubscription_endpoint() -> None:
    gateway = FakeGateway()
    gateway.subscriptions["600519.SH"] = 102
    app = create_app(gateway)

    with TestClient(app) as client:
        response = client.delete("/api/subscriptions/600519.SH")
        assert response.status_code == 200
        assert response.json() == {
            "symbol": "600519.SH",
            "status": "unsubscribed",
        }
        assert gateway.unsubscribed == ["600519.SH"]

        duplicate = client.delete("/api/subscriptions/600519.SH")
        assert duplicate.status_code == 200
        assert duplicate.json()["status"] == "not_subscribed"


def test_unsubscription_discards_late_events_and_clears_cache() -> None:
    gateway = FakeGateway()
    app = create_app(gateway)
    history = {
        "type": "history_snapshot", "period": "5m", "symbol": "000001.SZ",
        "count": 1,
        "bars": [{
            "timestamp_ms": 1_754_271_000_000,
            "timestamp": "2025-08-04T09:35:00+08:00",
            "open": 10.0, "high": 10.2, "low": 9.9, "close": 10.1,
            "volume": 1000, "amount": 10100,
        }],
    }
    late_update = {
        "type": "bar_update", "period": "5m", "symbol": "000001.SZ",
        "bar": history["bars"][0],
    }

    with TestClient(app) as client:
        gateway.emit(history)
        for _ in range(100):
            if "000001.SZ" in app.state.market_bridge.history_by_symbol:
                break
            time.sleep(0.01)

        response = client.delete("/api/subscriptions/000001.SZ")
        assert response.status_code == 200
        gateway.emit(late_update)
        time.sleep(0.05)

        bridge = app.state.market_bridge
        assert "000001.SZ" not in bridge.history_by_symbol
        assert "000001.SZ" not in bridge.latest_bar_by_symbol
        assert "000001.SZ" in bridge.forgotten_symbols


def test_cannot_unsubscribe_symbol_with_active_viewer() -> None:
    gateway = FakeGateway()
    app = create_app(gateway)

    with TestClient(app) as client:
        with client.websocket_connect("/ws/market?symbol=000001.SZ"):
            response = client.delete("/api/subscriptions/000001.SZ")
            assert response.status_code == 409
            assert "active viewer" in response.json()["detail"]


def test_history_triggers_forecast_and_custom_window_endpoint() -> None:
    gateway = FakeGateway()
    app = create_app(
        gateway,
        predictor=FakePredictor(),
        forecast_config=ForecastConfig(window=16, pred_len=3),
    )
    start = datetime(2026, 8, 3, 9, 30, tzinfo=ZoneInfo("Asia/Shanghai"))
    bars = [
        {
            "timestamp": (start + timedelta(minutes=5 * index)).isoformat(),
            "timestamp_ms": int((start + timedelta(minutes=5 * index)).timestamp() * 1000),
            "open": 10.0, "high": 10.2, "low": 9.9, "close": 10.1,
            "volume": 1000, "amount": 10100,
        }
        for index in range(20)
    ]
    history = {
        "type": "history_snapshot", "symbol": "000001.SZ", "period": "5m",
        "count": len(bars), "bars": bars,
    }

    with TestClient(app) as client:
        gateway.emit(history)
        for _ in range(200):
            if app.state.market_bridge.forecast_by_symbol:
                break
            time.sleep(0.01)
        forecast = app.state.market_bridge.forecast_by_symbol["000001.SZ"]
        assert forecast["window"] == 16
        assert forecast["pred_len"] == 3

        response = client.post(
            "/api/forecast/000001.SZ?window=18&pred_len=2&use_lora=true"
        )
        assert response.status_code == 200
        assert response.json()["status"] == "queued"
        assert response.json()["use_lora"] is True

        for _ in range(200):
            forecast = app.state.market_bridge.forecast_by_symbol.get("000001.SZ")
            if forecast and forecast["window"] == 18:
                break
            time.sleep(0.01)
        assert forecast["window"] == 18
        assert forecast["pred_len"] == 2
        assert forecast["use_lora"] is True

        next_time = start + timedelta(minutes=5 * len(bars))
        gateway.emit({
            "type": "bar_closed", "symbol": "000001.SZ", "period": "5m",
            "bar": {
                "symbol": "000001.SZ", "period": "5m",
                "timestamp": next_time.isoformat(),
                "timestamp_ms": int(next_time.timestamp() * 1000),
                "open": 10.1, "high": 10.3, "low": 10.0, "close": 10.2,
                "volume": 1100, "amount": 11220,
            },
        })
        for _ in range(200):
            forecast = app.state.market_bridge.forecast_by_symbol.get("000001.SZ")
            if forecast and forecast["based_on_timestamp"] == next_time.isoformat():
                break
            time.sleep(0.01)
        assert forecast["window"] == 18
        assert forecast["pred_len"] == 2
        assert forecast["use_lora"] is True
        assert forecast["based_on_timestamp"] == next_time.isoformat()


def test_forecast_options_include_both_local_backbones() -> None:
    app = create_app(
        FakeGateway(),
        predictor=FakePredictor(),
        forecast_config=ForecastConfig(window=16, pred_len=3),
    )

    with TestClient(app) as client:
        response = client.get("/api/forecast/options")

    assert response.status_code == 200
    base_models = [
        (model["key"], model["name"])
        for model in response.json()["models"]
        if model["type"] == "base"
    ]
    assert base_models == [
        ("base", "Kronos-base"),
        ("small", "Kronos-small"),
    ]


def test_subscription_preferences_apply_to_first_history_forecast() -> None:
    gateway = FakeGateway()
    app = create_app(
        gateway,
        predictor=FakePredictor(),
        forecast_config=ForecastConfig(window=16, pred_len=3),
    )
    start = datetime(2026, 8, 3, 9, 30, tzinfo=ZoneInfo("Asia/Shanghai"))
    bars = [
        {
            "timestamp": (start + timedelta(minutes=5 * index)).isoformat(),
            "timestamp_ms": int((start + timedelta(minutes=5 * index)).timestamp() * 1000),
            "open": 10.0, "high": 10.2, "low": 9.9, "close": 10.1,
            "volume": 1000, "amount": 10100,
        }
        for index in range(18)
    ]

    with TestClient(app) as client:
        response = client.post(
            "/api/subscriptions/600519.SH"
            "?window=18&pred_len=2&model_key=lora%3AKronos-base-QV-LoRA"
            "&modes=median%2Cmean&sample_count=7"
        )
        assert response.status_code == 200
        gateway.emit({
            "type": "history_snapshot", "symbol": "600519.SH",
            "period": "5m", "count": len(bars), "bars": bars,
        })
        for _ in range(200):
            forecast = app.state.market_bridge.forecast_by_symbol.get("600519.SH")
            if forecast:
                break
            time.sleep(0.01)

        assert forecast["window"] == 18
        assert forecast["pred_len"] == 2
        assert forecast["model_key"] == "lora:Kronos-base-QV-LoRA"
        assert forecast["use_lora"] is True
        assert forecast["selected_modes"] == ["median", "mean"]
        assert set(forecast["modes"]) == {"median", "mean"}
        assert forecast["sample_count"] == 7
