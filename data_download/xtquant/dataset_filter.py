"""Materialize a symbol subset from existing month-partitioned Parquet data."""

from __future__ import annotations

import hashlib
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from .downloader import normalize_universe, universe_digest
from .manifest import load_manifest, save_json
from .transformer import OUTPUT_COLUMNS, validate_partition
from .validator import validate_output


def _source_fingerprint(source_dir: Path, files: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in files:
        stat = path.stat()
        value = f"{path.relative_to(source_dir)}\0{stat.st_size}\0{stat.st_mtime_ns}\n"
        digest.update(value.encode("utf-8"))
    return digest.hexdigest()


def _assert_separate_directories(source_dir: Path, output_dir: Path) -> None:
    if (
        source_dir == output_dir
        or source_dir in output_dir.parents
        or output_dir in source_dir.parents
    ):
        raise ValueError("source-dir and output-dir must be separate, non-nested directories")


def filter_existing_dataset(
    source_dir: Path,
    output_dir: Path,
    symbols: Iterable[str],
    *,
    compression: str = "zstd",
    resume: bool = True,
    universe_sources: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Filter one source Parquet at a time and preserve its relative partition path."""

    source_dir = Path(source_dir).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    _assert_separate_directories(source_dir, output_dir)
    source_files = sorted(source_dir.glob("year=*/month=*/*.parquet"))
    if not source_files:
        raise FileNotFoundError(
            f"no partitioned Parquet files found under source-dir: {source_dir}"
        )
    universe = normalize_universe(symbols)
    if not universe:
        raise RuntimeError("the requested A-share universe is empty")

    fingerprint = _source_fingerprint(source_dir, source_files)
    universe_sha256 = universe_digest(universe)
    normalized_sources = sorted(set(universe_sources or []))
    manifest_path = output_dir / "filter_manifest.json"
    existing = load_manifest(manifest_path) if resume else None
    if existing is None:
        if output_dir.exists() and any(output_dir.iterdir()):
            raise FileExistsError(
                f"output directory is not empty and has no resumable filter manifest: "
                f"{output_dir}"
            )
        output_dir.mkdir(parents=True, exist_ok=True)
        manifest: dict[str, Any] = {
            "version": 1,
            "kind": "filtered_xtquant_dataset",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "status": "running",
            "source_dir": str(source_dir),
            "source_fingerprint": fingerprint,
            "universe_sha256": universe_sha256,
            "universe_sources": normalized_sources,
            "requested_symbol_count": len(universe),
            "compression": compression,
            "completed_source_files": [],
            "files": [],
            "found_symbols": [],
            "missing_symbols": [],
        }
    else:
        expected = {
            "source_dir": str(source_dir),
            "source_fingerprint": fingerprint,
            "universe_sha256": universe_sha256,
            "compression": compression,
        }
        for key, value in expected.items():
            if existing.get(key) != value:
                raise ValueError(
                    f"existing filter manifest has incompatible {key}; "
                    "use a new output directory"
                )
        manifest = existing
        manifest["status"] = "running"

    save_json(output_dir / "universe.json", universe)
    save_json(manifest_path, manifest)
    requested = set(universe)
    completed = set(manifest["completed_source_files"])
    found = set(manifest.get("found_symbols", []))

    for source_path in source_files:
        relative = str(source_path.relative_to(source_dir))
        if relative in completed:
            continue
        frame = pd.read_parquet(source_path, columns=list(OUTPUT_COLUMNS))
        normalized_symbols = frame["symbol"].astype(str).str.strip().str.upper()
        selected = frame.loc[normalized_symbols.isin(requested)].copy()
        if not selected.empty:
            selected["symbol"] = normalized_symbols.loc[selected.index]
            selected["timestamp"] = pd.to_datetime(
                selected["timestamp"], errors="coerce"
            )
            selected = selected.sort_values(["symbol", "timestamp"]).reset_index(drop=True)
            errors = validate_partition(selected)
            if errors:
                raise ValueError(f"filtered partition {relative} is invalid: {'; '.join(errors)}")
            target = output_dir / Path(relative)
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_suffix(".parquet.tmp")
            selected.to_parquet(temporary, index=False, compression=compression)
            os.replace(temporary, target)
            values = sorted(selected["symbol"].unique().tolist())
            found.update(values)
            manifest["files"] = [
                entry for entry in manifest["files"] if entry.get("path") != relative
            ]
            manifest["files"].append(
                {
                    "path": relative,
                    "source_path": relative,
                    "rows": len(selected),
                    "symbols": len(values),
                    "symbol_values": values,
                }
            )

        manifest["completed_source_files"].append(relative)
        completed.add(relative)
        manifest["found_symbols"] = sorted(found)
        manifest["updated_at"] = datetime.now(timezone.utc).isoformat()
        save_json(manifest_path, manifest)

    missing = sorted(requested - found)
    manifest["found_symbols"] = sorted(found)
    manifest["missing_symbols"] = missing
    manifest["status"] = "completed" if not missing else "completed_with_missing_symbols"
    manifest["quality"] = {
        "rows": sum(int(entry["rows"]) for entry in manifest["files"]),
        "files": len(manifest["files"]),
        "source_files": len(source_files),
        "requested_symbols": len(universe),
        "found_symbols": len(found),
        "missing_symbols": len(missing),
    }
    manifest["updated_at"] = datetime.now(timezone.utc).isoformat()
    save_json(manifest_path, manifest)
    if not manifest["files"]:
        raise RuntimeError("none of the requested symbols were found in source-dir")
    validation = validate_output(output_dir, write_report=False)
    validation["requested_symbols"] = len(universe)
    validation["missing_symbols"] = missing
    save_json(output_dir / "quality_report.json", validation)
    return manifest
