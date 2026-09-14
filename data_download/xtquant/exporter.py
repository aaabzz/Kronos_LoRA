"""Atomic, month-partitioned Parquet export."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd


def batch_file_name(batch_id: int) -> str:
    return f"part-batch-{batch_id:05d}.parquet"


def write_month_partitions(
    frame: pd.DataFrame,
    output_dir: Path,
    batch_id: int,
    compression: str = "zstd",
) -> list[dict[str, object]]:
    """Write deterministic monthly parts and return manifest entries."""

    entries: list[dict[str, object]] = []
    if frame.empty:
        return entries
    keyed = frame.assign(
        _year=frame["timestamp"].dt.year,
        _month=frame["timestamp"].dt.month,
    )
    for (year, month), part in keyed.groupby(["_year", "_month"], sort=True):
        directory = output_dir / f"year={int(year):04d}" / f"month={int(month):02d}"
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / batch_file_name(batch_id)
        temporary = target.with_suffix(".parquet.tmp")
        clean = part.drop(columns=["_year", "_month"]).sort_values(
            ["symbol", "timestamp"]
        )
        clean.to_parquet(temporary, index=False, compression=compression)
        os.replace(temporary, target)
        entries.append(
            {
                "path": str(target.relative_to(output_dir)),
                "rows": len(clean),
                "symbols": int(clean["symbol"].nunique()),
            }
        )
    return entries
