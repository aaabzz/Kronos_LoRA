from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd

from api.forecasting import ForecastConfig, KronosForecastService, future_timestamps


CN_TZ = ZoneInfo("Asia/Shanghai")


class FakePredictor:
    def predict(self, **kwargs):
        count = kwargs["pred_len"]
        return pd.DataFrame(
            {
                "open": [10.001] * count,
                "high": [9.5] * count,
                "low": [10.5] * count,
                "close": [10.234] * count,
                "volume": [1000.6] * count,
                "amount": [10000.4] * count,
            }
        )


def make_bars(count: int) -> list[dict]:
    start = datetime(2026, 8, 3, 9, 30, tzinfo=CN_TZ)
    return [
        {
            "timestamp": (start + timedelta(minutes=5 * index)).isoformat(),
            "timestamp_ms": int((start + timedelta(minutes=5 * index)).timestamp() * 1000),
            "open": 10.0,
            "high": 10.3,
            "low": 9.9,
            "close": 10.2,
            "volume": 1000,
            "amount": 10000,
        }
        for index in range(count)
    ]


def test_future_timestamps_skip_lunch_and_weekend() -> None:
    lunch = future_timestamps(pd.Timestamp("2026-08-05T11:25:00+08:00"), "5m", 2)
    assert [item.strftime("%H:%M") for item in lunch] == ["11:30", "13:05"]

    weekend = future_timestamps(pd.Timestamp("2026-08-07T14:55:00+08:00"), "5m", 1)
    assert weekend[0].strftime("%Y-%m-%d %H:%M") == "2026-08-07 15:00"

    next_session = future_timestamps(pd.Timestamp("2026-08-07T15:00:00+08:00"), "5m", 1)
    assert next_session[0].strftime("%Y-%m-%d %H:%M") == "2026-08-10 09:35"


def test_forecast_service_emits_normalized_prediction() -> None:
    async def run() -> list[dict]:
        events: list[dict] = []

        async def publish(event: dict) -> None:
            events.append(event)

        service = KronosForecastService(
            ForecastConfig(window=16, pred_len=2), publish, predictor=FakePredictor()
        )
        await service.start()
        service.submit("000001.SZ", make_bars(16), "5m")
        for _ in range(100):
            if any(event["type"] == "forecast" for event in events):
                break
            await asyncio.sleep(0.01)
        await service.stop()
        return events

    events = asyncio.run(run())
    forecast = next(event for event in events if event["type"] == "forecast")
    assert forecast["window"] == 16
    assert forecast["pred_len"] == 2
    assert len(forecast["bars"]) == 2
    assert forecast["bars"][0]["open"] == 10.0
    assert forecast["bars"][0]["close"] == 10.23
    assert forecast["bars"][0]["high"] == 10.23
    assert forecast["bars"][0]["low"] == 10.0
    assert forecast["bars"][0]["volume"] == 1001
    assert set(forecast["modes"]) == {"greedy", "median", "mean"}
    assert forecast["modes"]["greedy"] == forecast["bars"]
    assert forecast["probabilities"] == {
        "up": 1.0, "down": 0.0, "flat": 0.0, "sample_count": 1,
    }
