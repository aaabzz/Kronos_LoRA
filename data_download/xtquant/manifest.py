"""Persistent run state used for audit and resume."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def load_manifest(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(temporary, path)


def new_manifest(
    config: dict[str, Any],
    start_date: str,
    end_date: str,
    universe_sha256: str,
    symbol_count: int,
) -> dict[str, Any]:
    return {
        "version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "status": "running",
        "start_date": start_date,
        "end_date": end_date,
        "universe_sha256": universe_sha256,
        "symbol_count": symbol_count,
        "config": config,
        "completed_batches": [],
        "files": [],
        "empty_symbols": [],
        "failed_symbols": {},
        "quality": {"rows": 0, "files": 0},
    }
