"""Orchestration for batch XtQuant history download and Parquet export."""

from __future__ import annotations

import hashlib
import json
import logging
from calendar import monthrange
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterable
from zoneinfo import ZoneInfo

import pandas as pd

from .config import XtQuantDownloadConfig
from .exporter import batch_file_name, write_month_partitions
from .manifest import load_manifest, new_manifest, save_json
from .transformer import normalize_symbol_frame, validate_bars


LOGGER = logging.getLogger(__name__)
CN_TZ = ZoneInfo("Asia/Shanghai")
ALLOWED_SYMBOL_SUFFIXES = (".SH", ".SZ", ".BJ")
SECTOR_NAMES = ("沪深A股", "京市A股")


def batched(values: list[str], size: int) -> Iterable[tuple[int, list[str]]]:
    for offset in range(0, len(values), size):
        yield offset // size, values[offset : offset + size]


def universe_digest(symbols: list[str]) -> str:
    payload = "\n".join(symbols).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def normalize_universe(symbols: Iterable[str]) -> list[str]:
    """Return the deterministic A-share universe used by plan and download."""

    return sorted(
        {
            symbol.strip().upper()
            for symbol in symbols
            if isinstance(symbol, str)
            and symbol.strip().upper().endswith(ALLOWED_SYMBOL_SUFFIXES)
        }
    )


class XtQuantDownloader:
    """Download the current A-share universe without holding it all in memory."""

    def __init__(self, config: XtQuantDownloadConfig, xtdata_module: Any) -> None:
        self.config = config
        self.xtdata = xtdata_module
        self.manifest_path = config.output_dir / "download_manifest.json"

    def get_universe(self) -> list[str]:
        updater = getattr(self.xtdata, "download_sector_data", None)
        if callable(updater):
            updater()
        sectors = list(SECTOR_NAMES if self.config.include_beijing else SECTOR_NAMES[:1])
        symbols: set[str] = set()
        for sector in sectors:
            symbols.update(self.xtdata.get_stock_list_in_sector(sector) or [])
        return normalize_universe(symbols)

    def resolve_date_range(self, now: datetime | None = None) -> tuple[str, str]:
        if self.config.start_date and self.config.end_date:
            return self.config.start_date, self.config.end_date

        current = now.astimezone(CN_TZ) if now else datetime.now(CN_TZ)
        candidate = current.date()
        market_close = datetime.strptime("15:05", "%H:%M").time()
        if not self.config.include_today or current.time() < market_close:
            candidate -= timedelta(days=1)

        # Clamp leap day instead of using date.replace(year=...), which fails
        # when candidate is February 29 and the target year is not a leap year.
        target_year = candidate.year - self.config.lookback_years
        target_day = min(candidate.day, monthrange(target_year, candidate.month)[1])
        calendar_start = candidate.replace(year=target_year, day=target_day)
        timestamps = self.xtdata.get_trading_dates(
            "SH",
            start_time=calendar_start.strftime("%Y%m%d"),
            end_time=candidate.strftime("%Y%m%d"),
            count=-1,
        )
        trading_dates = sorted(
            self._timestamp_to_date(value) for value in (timestamps or [])
        )
        if not trading_dates:
            raise RuntimeError("XtQuant returned no trading dates for the requested range")
        return trading_dates[0], trading_dates[-1]

    @staticmethod
    def _timestamp_to_date(value: Any) -> str:
        if isinstance(value, str):
            digits = value.replace("-", "")
            if len(digits) >= 8 and digits[:8].isdigit():
                return digits[:8]
        number = int(value)
        unit = "ms" if number > 10_000_000_000 else "s"
        return (
            pd.to_datetime(number, unit=unit, utc=True)
            .tz_convert(CN_TZ)
            .strftime("%Y%m%d")
        )

    def plan(
        self,
        symbols: list[str] | None = None,
        *,
        universe_source: str | None = None,
    ) -> dict[str, Any]:
        """Describe the exact universe that a matching download would use."""

        start_date, end_date = self.resolve_date_range()
        from_symbols_file = symbols is not None
        universe = normalize_universe(
            symbols if symbols is not None else self.get_universe()
        )
        if not universe:
            raise RuntimeError("the requested A-share universe is empty")
        result = {
            "start_date": start_date,
            "end_date": end_date,
            "symbols": len(universe),
            "batches": (len(universe) + self.config.download_batch_size - 1)
            // self.config.download_batch_size,
            "universe_sha256": universe_digest(universe),
            "universe_source": (
                universe_source
                or ("symbols_file" if from_symbols_file else "market")
            ),
            "output_dir": str(self.config.output_dir),
        }
        if not from_symbols_file:
            result["sectors"] = list(
                SECTOR_NAMES if self.config.include_beijing else SECTOR_NAMES[:1]
            )
        return result

    def run(
        self,
        symbols: list[str] | None = None,
        *,
        universe_source: str | None = None,
    ) -> dict[str, Any]:
        existing = load_manifest(self.manifest_path) if self.config.resume else None
        if (
            existing is not None
            and self.config.start_date is None
            and self.config.end_date is None
        ):
            # A full-market run can span multiple days. Keep its original date
            # range instead of moving the end date every time it is resumed.
            start_date = str(existing["start_date"])
            end_date = str(existing["end_date"])
        else:
            start_date, end_date = self.resolve_date_range()
        source_symbols = symbols
        if source_symbols is None and existing is not None:
            source_symbols = self._load_saved_universe()
        if source_symbols is None:
            source_symbols = self.get_universe()
        universe = normalize_universe(source_symbols)
        if not universe:
            raise RuntimeError("the requested A-share universe is empty")
        digest = universe_digest(universe)

        if existing is not None:
            self._assert_resume_compatible(existing, start_date, end_date, digest)
            manifest = existing
            manifest["status"] = "running"
        else:
            self._assert_safe_new_run()
            manifest = new_manifest(
                self.config.to_dict(), start_date, end_date, digest, len(universe)
            )
            manifest["universe_source"] = universe_source or (
                "symbols" if symbols is not None else "market"
            )
            if len(universe) == 1:
                manifest["symbol"] = universe[0]
        save_json(self.config.output_dir / "universe.json", universe)
        completed = set(manifest["completed_batches"])

        for batch_id, batch_symbols in batched(
            universe, self.config.download_batch_size
        ):
            if batch_id in completed:
                LOGGER.info("Skipping completed batch %05d", batch_id)
                continue
            LOGGER.info(
                "Downloading batch %05d (%d symbols)", batch_id, len(batch_symbols)
            )
            batch_failed = self._download_with_retries(
                batch_symbols, start_date, end_date
            )
            frames: list[pd.DataFrame] = []
            empty_symbols: list[str] = []
            if not batch_failed:
                frames, empty_symbols, read_failed = self._read_batch(
                    batch_symbols, start_date, end_date
                )
                batch_failed.update(read_failed)

            for symbol in batch_symbols:
                if symbol in batch_failed:
                    manifest["failed_symbols"][symbol] = batch_failed[symbol]
                else:
                    manifest["failed_symbols"].pop(symbol, None)
            manifest["empty_symbols"] = sorted(
                (set(manifest["empty_symbols"]) - set(batch_symbols))
                | set(empty_symbols)
            )

            # Only a fully successful batch is resumable. If one symbol fails,
            # rerunning the command retries this whole deterministic batch.
            if not batch_failed:
                self._remove_batch_parts(batch_id)
                self._replace_batch_manifest_files(manifest, batch_id)
                if frames:
                    combined = pd.concat(frames, ignore_index=True)
                    manifest["files"].extend(
                        write_month_partitions(
                            combined,
                            self.config.output_dir,
                            batch_id,
                            self.config.compression,
                        )
                    )
                manifest["completed_batches"].append(batch_id)
                completed.add(batch_id)
            self._refresh_quality(manifest)
            manifest["updated_at"] = datetime.now(CN_TZ).isoformat()
            save_json(self.manifest_path, manifest)

        expected_batches = (
            len(universe) + self.config.download_batch_size - 1
        ) // self.config.download_batch_size
        all_complete = len(set(manifest["completed_batches"])) == expected_batches
        manifest["status"] = (
            "completed"
            if all_complete and not manifest["failed_symbols"]
            else "completed_with_errors"
        )
        self._refresh_quality(manifest)
        manifest["updated_at"] = datetime.now(CN_TZ).isoformat()
        save_json(
            self.config.output_dir / "failed_symbols.json",
            manifest["failed_symbols"],
        )
        save_json(self.config.output_dir / "quality_report.json", manifest["quality"])
        save_json(self.manifest_path, manifest)
        return manifest

    def _load_saved_universe(self) -> list[str]:
        path = self.config.output_dir / "universe.json"
        if not path.exists():
            raise ValueError(
                "resume manifest exists but universe.json is missing; "
                "the original batch mapping cannot be reconstructed"
            )
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
        if not isinstance(value, list) or not all(
            isinstance(symbol, str) for symbol in value
        ):
            raise ValueError("universe.json must contain a JSON array of symbols")
        return value

    def _read_batch(
        self, symbols: list[str], start_date: str, end_date: str
    ) -> tuple[list[pd.DataFrame], list[str], dict[str, str]]:
        frames: list[pd.DataFrame] = []
        empty_symbols: list[str] = []
        failed: dict[str, str] = {}
        for _, read_symbols in batched(symbols, self.config.read_batch_size):
            try:
                payload = self.xtdata.get_market_data_ex(
                    field_list=list(self.config.fields),
                    stock_list=read_symbols,
                    period=self.config.period,
                    start_time=start_date,
                    end_time=end_date,
                    count=-1,
                    dividend_type=self.config.dividend_type,
                    fill_data=self.config.fill_data,
                ) or {}
            except Exception as exc:
                for symbol in read_symbols:
                    failed[symbol] = f"read: {exc}"
                LOGGER.exception("Failed to read %d symbols", len(read_symbols))
                continue
            for symbol in read_symbols:
                try:
                    if symbol not in payload:
                        raise ValueError("XtQuant response does not contain this symbol")
                    frame = normalize_symbol_frame(symbol, payload.get(symbol))
                    errors = validate_bars(frame)
                    if errors:
                        raise ValueError("; ".join(errors))
                    if frame.empty:
                        empty_symbols.append(symbol)
                    else:
                        frames.append(frame)
                except Exception as exc:
                    failed[symbol] = f"read/validate: {exc}"
                    LOGGER.exception("Failed to normalize %s", symbol)
        return frames, empty_symbols, failed

    def _download_with_retries(
        self, symbols: list[str], start_date: str, end_date: str
    ) -> dict[str, str]:
        last_error = "unknown download error"
        for attempt in range(1, self.config.retry_count + 1):
            try:
                self.xtdata.download_history_data2(
                    stock_list=symbols,
                    period=self.config.period,
                    start_time=start_date,
                    end_time=end_date,
                    callback=self._progress_callback,
                )
                return {}
            except Exception as exc:
                last_error = str(exc)
                LOGGER.warning("Download attempt %d failed: %s", attempt, exc)
        return {symbol: f"download: {last_error}" for symbol in symbols}

    @staticmethod
    def _progress_callback(progress: dict[str, Any]) -> None:
        LOGGER.debug(
            "XtQuant download progress: %s/%s %s",
            progress.get("finished"),
            progress.get("total"),
            progress.get("message", ""),
        )

    def _replace_batch_manifest_files(
        self, manifest: dict[str, Any], batch_id: int
    ) -> None:
        name = batch_file_name(batch_id)
        manifest["files"] = [
            entry
            for entry in manifest["files"]
            if Path(str(entry.get("path", ""))).name != name
        ]

    def _remove_batch_parts(self, batch_id: int) -> None:
        """Remove only deterministic files owned by a retried batch."""

        name = batch_file_name(batch_id)
        for path in self.config.output_dir.glob(f"year=*/month=*/{name}"):
            path.unlink()

    @staticmethod
    def _refresh_quality(manifest: dict[str, Any]) -> None:
        manifest["quality"] = {
            "rows": sum(int(entry.get("rows", 0)) for entry in manifest["files"]),
            "files": len(manifest["files"]),
            "completed_batches": len(set(manifest["completed_batches"])),
            "empty_symbols": len(set(manifest["empty_symbols"])),
            "failed_symbols": len(manifest["failed_symbols"]),
        }

    def _assert_safe_new_run(self) -> None:
        existing_parts = list(self.config.output_dir.glob("year=*/month=*/*.parquet"))
        if existing_parts:
            raise ValueError(
                "output directory already contains Parquet parts but no resumable "
                "manifest was selected; use a new output directory"
            )

    def _assert_resume_compatible(
        self,
        manifest: dict[str, Any],
        start_date: str,
        end_date: str,
        digest: str,
    ) -> None:
        if (
            manifest.get("start_date") != start_date
            or manifest.get("end_date") != end_date
        ):
            raise ValueError(
                "existing manifest has a different date range; use a new output directory"
            )
        if manifest.get("universe_sha256") != digest:
            raise ValueError(
                "existing manifest has a different stock universe; use a new output directory"
            )
        old = manifest.get("config", {})
        current = self.config.to_dict()
        for key in (
            "period",
            "dividend_type",
            "fill_data",
            "download_batch_size",
            "fields",
            "compression",
        ):
            if old.get(key) != current.get(key):
                raise ValueError(f"existing manifest has incompatible {key}")
