"""Validation of already exported XtQuant Parquet partitions."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .manifest import save_json
from .transformer import OUTPUT_COLUMNS, validate_partition


def validate_output(data_dir: Path, write_report: bool = True) -> dict[str, Any]:
    """Validate every partition without concatenating the full market in memory."""

    data_dir = Path(data_dir).resolve()
    files = sorted(data_dir.glob("year=*/month=*/*.parquet"))
    if not files:
        raise FileNotFoundError(f"no partitioned Parquet files found under: {data_dir}")

    issues: dict[str, list[str]] = {}
    rows = 0
    symbols: set[str] = set()
    partition_owners: dict[tuple[int, int, str], str] = {}
    for path in files:
        relative = str(path.relative_to(data_dir))
        try:
            frame = pd.read_parquet(path, columns=list(OUTPUT_COLUMNS))
            frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
            file_issues = validate_partition(frame)
            year = int(path.parent.parent.name.split("=", 1)[1])
            month = int(path.parent.name.split("=", 1)[1])
            if not frame.empty and (
                (frame["timestamp"].dt.year != year)
                | (frame["timestamp"].dt.month != month)
            ).any():
                file_issues.append("contains rows outside its year/month partition")
            rows += len(frame)
            symbols.update(frame["symbol"].astype(str).unique())
            for symbol in frame["symbol"].astype(str).unique():
                key = (year, month, symbol)
                previous = partition_owners.get(key)
                if previous is not None and previous != relative:
                    file_issues.append(
                        f"{symbol} also appears in the same month file {previous}"
                    )
                else:
                    partition_owners[key] = relative
            if file_issues:
                issues[relative] = file_issues
        except Exception as exc:
            issues[relative] = [f"could not read/validate file: {exc}"]

    report: dict[str, Any] = {
        "status": "valid" if not issues else "invalid",
        "files": len(files),
        "rows": rows,
        "symbols": len(symbols),
        "invalid_files": len(issues),
        "issues": issues,
    }
    if write_report:
        save_json(data_dir / "quality_report.json", report)
    return report
