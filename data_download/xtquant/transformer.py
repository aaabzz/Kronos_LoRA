"""Normalize and validate XtQuant payloads."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd


OUTPUT_COLUMNS = (
    "symbol", "timestamp", "open", "high", "low", "close", "volume", "amount"
)
NUMERIC_COLUMNS = ("open", "high", "low", "close", "volume", "amount")


def normalize_symbol_frame(symbol: str, raw: Any) -> pd.DataFrame:
    """Convert one XtQuant symbol payload to the training schema."""

    if raw is None:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    if isinstance(raw, pd.DataFrame):
        frame = raw.copy()
    elif isinstance(raw, Mapping):
        frame = pd.DataFrame(raw)
    elif isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        frame = pd.DataFrame(list(raw))
    else:
        raise TypeError(f"unsupported XtQuant payload for {symbol}: {type(raw).__name__}")
    if frame.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    if "time" not in frame.columns and frame.index.name in {"time", "stime"}:
        frame = frame.reset_index().rename(columns={frame.index.name or "index": "time"})
    required = {"time", *NUMERIC_COLUMNS}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{symbol} is missing columns: {sorted(missing)}")

    result = frame.loc[:, ["time", *NUMERIC_COLUMNS]].copy()
    result["timestamp"] = (
        pd.to_datetime(result.pop("time"), unit="ms", utc=True, errors="coerce")
        .dt.tz_convert("Asia/Shanghai")
        .dt.tz_localize(None)
    )
    result.insert(0, "symbol", symbol.strip().upper())
    for column in NUMERIC_COLUMNS:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    return result.loc[:, OUTPUT_COLUMNS].sort_values("timestamp").reset_index(drop=True)


def validate_bars(frame: pd.DataFrame) -> list[str]:
    """Return human-readable quality errors; an empty list means valid."""

    errors: list[str] = []
    missing = set(OUTPUT_COLUMNS) - set(frame.columns)
    if missing:
        return [f"missing output columns: {sorted(missing)}"]
    if frame.empty:
        return errors
    if frame.loc[:, ["timestamp", *NUMERIC_COLUMNS]].isna().any().any():
        errors.append("contains null or unparseable values")
    numeric = frame.loc[:, NUMERIC_COLUMNS].to_numpy(dtype=float, copy=False)
    if not np.isfinite(numeric).all():
        errors.append("contains non-finite numeric values")
    if (frame["high"] < frame[["open", "low", "close"]].max(axis=1)).any():
        errors.append("contains rows where high is not the maximum price")
    if (frame["low"] > frame[["open", "high", "close"]].min(axis=1)).any():
        errors.append("contains rows where low is not the minimum price")
    if (frame[["volume", "amount"]] < 0).any().any():
        errors.append("contains negative volume or amount")
    if frame.duplicated(["symbol", "timestamp"]).any():
        errors.append("contains duplicate symbol/timestamp rows")
    if not frame["timestamp"].is_monotonic_increasing:
        errors.append("timestamps are not increasing")
    return errors


def validate_partition(frame: pd.DataFrame) -> list[str]:
    """Validate a saved multi-symbol partition."""

    errors: list[str] = []
    missing = set(OUTPUT_COLUMNS) - set(frame.columns)
    if missing:
        return [f"missing output columns: {sorted(missing)}"]
    if frame.empty:
        return ["partition is empty"]
    for symbol, symbol_frame in frame.groupby("symbol", sort=False):
        symbol_errors = validate_bars(symbol_frame.reset_index(drop=True))
        errors.extend(f"{symbol}: {message}" for message in symbol_errors)
    return errors
