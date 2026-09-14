"""Offline tests for the XtQuant batch data pipeline."""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from data_download.xtquant.config import XtQuantDownloadConfig
from data_download.xtquant.cli import (
    _load_symbols,
    _requested_symbols,
    parse_args as parse_download_args,
)
from data_download.xtquant.dataset_filter import filter_existing_dataset
from data_download.xtquant.downloader import (
    XtQuantDownloader,
    normalize_universe,
    universe_digest,
)
from data_download.xtquant.exporter import write_month_partitions
from data_download.xtquant.sector_universe import (
    export_sector_universe,
    list_sector_names,
    parse_args as parse_sector_args,
)
from data_download.xtquant.transformer import normalize_symbol_frame, validate_bars
from data_download.xtquant.validator import validate_output


CN_TZ = ZoneInfo("Asia/Shanghai")


def _raw_frame(day: str, close: float = 10.1) -> pd.DataFrame:
    timestamp = pd.Timestamp(f"{day} 09:35:00", tz="Asia/Shanghai").timestamp() * 1000
    return pd.DataFrame(
        {
            "time": [int(timestamp)],
            "open": [10.0],
            "high": [10.2],
            "low": [9.9],
            "close": [close],
            "volume": [1000],
            "amount": [10100],
        }
    )


class FakeXtData:
    def __init__(self) -> None:
        self.download_calls: list[dict[str, object]] = []
        self.read_failures: set[str] = set()
        self.trading_dates = [
            int(pd.Timestamp("2025-08-29", tz="Asia/Shanghai").timestamp() * 1000),
            int(pd.Timestamp("2026-08-28", tz="Asia/Shanghai").timestamp() * 1000),
        ]

    def download_sector_data(self) -> None:
        return None

    def get_stock_list_in_sector(self, sector: str) -> list[str]:
        if sector == "半导体":
            return ["688981.sh", "603501.SH", "688981.SH", "INVALID"]
        if sector == "GN功率半导体":
            return ["688981.SH", "605111.sh", "920001.BJ"]
        if sector == "沪深A股":
            return ["600000.SH", "000001.SZ"]
        if sector == "京市A股":
            return ["920001.BJ"]
        return []

    def get_sector_list(self) -> list[str]:
        return ["半导体", "银行", "半导体", ""]

    def get_trading_dates(self, *_: object, **__: object) -> list[int]:
        return self.trading_dates

    def download_history_data2(self, **kwargs: object) -> None:
        self.download_calls.append(kwargs)

    def get_market_data_ex(
        self, *, stock_list: list[str], **_: object
    ) -> dict[str, pd.DataFrame]:
        return {
            symbol: pd.concat(
                [_raw_frame("2025-08-29"), _raw_frame("2026-08-28", close=10.2)],
                ignore_index=True,
            )
            for symbol in stock_list
            if symbol not in self.read_failures
        }


def test_config_requires_complete_and_valid_explicit_date_range() -> None:
    with pytest.raises(ValueError, match="provided together"):
        XtQuantDownloadConfig(start_date="20250101")
    with pytest.raises(ValueError, match="valid YYYYMMDD"):
        XtQuantDownloadConfig(start_date="20250230", end_date="20250301")
    with pytest.raises(ValueError, match="must not be after"):
        XtQuantDownloadConfig(start_date="20250302", end_date="20250301")


def test_normalize_converts_epoch_to_shanghai_wall_time() -> None:
    result = normalize_symbol_frame("000001.sz", _raw_frame("2026-08-28"))

    assert list(result.columns) == [
        "symbol", "timestamp", "open", "high", "low", "close", "volume", "amount"
    ]
    assert result.loc[0, "symbol"] == "000001.SZ"
    assert result.loc[0, "timestamp"] == pd.Timestamp("2026-08-28 09:35:00")
    assert validate_bars(result) == []


def test_validator_reports_bad_ohlc_and_duplicate_rows() -> None:
    frame = normalize_symbol_frame("000001.SZ", _raw_frame("2026-08-28"))
    frame.loc[0, "high"] = 9.0
    duplicated = pd.concat([frame, frame], ignore_index=True)

    errors = validate_bars(duplicated)

    assert "contains rows where high is not the maximum price" in errors
    assert "contains duplicate symbol/timestamp rows" in errors


def test_downloader_writes_month_partitions_and_resumes(tmp_path: Path) -> None:
    fake = FakeXtData()
    config = XtQuantDownloadConfig(
        output_dir=tmp_path,
        start_date="20250829",
        end_date="20260828",
        download_batch_size=2,
        read_batch_size=1,
    )
    downloader = XtQuantDownloader(config, fake)

    manifest = downloader.run()

    assert manifest["status"] == "completed"
    assert manifest["completed_batches"] == [0, 1]
    assert len(fake.download_calls) == 2
    assert (tmp_path / "year=2025" / "month=08" / "part-batch-00000.parquet").exists()
    assert (tmp_path / "year=2026" / "month=08" / "part-batch-00001.parquet").exists()
    first_part = pd.read_parquet(
        tmp_path / "year=2025" / "month=08" / "part-batch-00000.parquet"
    )
    assert set(first_part["symbol"]) == {"000001.SZ", "600000.SH"}

    resumed = downloader.run()

    assert resumed["status"] == "completed"
    assert len(fake.download_calls) == 2
    report = validate_output(tmp_path)
    assert report["status"] == "valid"
    assert report["symbols"] == 3


def test_failed_read_batch_is_not_marked_complete_and_is_retried(tmp_path: Path) -> None:
    fake = FakeXtData()
    fake.read_failures.add("000001.SZ")
    config = XtQuantDownloadConfig(
        output_dir=tmp_path,
        start_date="20250829",
        end_date="20260828",
        download_batch_size=3,
    )
    downloader = XtQuantDownloader(config, fake)

    failed = downloader.run()

    assert failed["status"] == "completed_with_errors"
    assert failed["completed_batches"] == []
    assert "000001.SZ" in failed["failed_symbols"]
    assert not list(tmp_path.glob("year=*/month=*/*.parquet"))

    fake.read_failures.clear()
    recovered = downloader.run()

    assert recovered["status"] == "completed"
    assert recovered["completed_batches"] == [0]
    assert recovered["failed_symbols"] == {}
    assert len(fake.download_calls) == 2


def test_date_range_excludes_today_by_default(tmp_path: Path) -> None:
    downloader = XtQuantDownloader(XtQuantDownloadConfig(output_dir=tmp_path), FakeXtData())

    start, end = downloader.resolve_date_range(
        datetime(2026, 8, 31, 16, 0, tzinfo=CN_TZ)
    )

    assert (start, end) == ("20250829", "20260828")


def test_resume_keeps_original_dynamic_date_range(tmp_path: Path) -> None:
    fake = FakeXtData()
    fake.read_failures.add("000001.SZ")
    config = XtQuantDownloadConfig(output_dir=tmp_path, download_batch_size=3)
    downloader = XtQuantDownloader(config, fake)

    first = downloader.run()
    fake.trading_dates = [
        int(pd.Timestamp("2025-09-01", tz="Asia/Shanghai").timestamp() * 1000),
        int(pd.Timestamp("2026-08-31", tz="Asia/Shanghai").timestamp() * 1000),
    ]
    fake.read_failures.clear()
    resumed = downloader.run()

    assert resumed["start_date"] == first["start_date"] == "20250829"
    assert resumed["end_date"] == first["end_date"] == "20260828"


def test_plan_uses_the_same_normalized_symbols_as_download(tmp_path: Path) -> None:
    config = XtQuantDownloadConfig(
        output_dir=tmp_path,
        start_date="20250829",
        end_date="20260828",
        download_batch_size=1,
    )
    downloader = XtQuantDownloader(config, FakeXtData())
    requested = ["600000.sh", "600000.SH", "000001.sz", "INVALID"]

    plan = downloader.plan(symbols=requested)
    universe = normalize_universe(requested)

    assert universe == ["000001.SZ", "600000.SH"]
    assert plan["symbols"] == 2
    assert plan["batches"] == 2
    assert plan["universe_source"] == "symbols_file"
    assert plan["universe_sha256"] == universe_digest(universe)
    assert "sectors" not in plan


def test_sector_snapshot_lists_and_exports_fixed_universe(tmp_path: Path) -> None:
    fake = FakeXtData()

    assert list_sector_names(fake) == ["半导体", "银行"]
    result = export_sector_universe(fake, "半导体", tmp_path)

    assert result["symbol_count"] == 2
    assert result["sector_names"] == ["半导体"]
    assert result["overlap_assignments"] == 0
    assert result["membership_semantics"] == "current_constituents_at_generation_time"
    assert (tmp_path / "symbols.txt").read_text(encoding="utf-8").splitlines() == [
        "603501.SH",
        "688981.SH",
    ]
    assert (tmp_path / "sector_metadata.json").exists()
    assert (tmp_path / "sector_membership.json").exists()
    with pytest.raises(FileExistsError, match="already exists"):
        export_sector_universe(fake, "半导体", tmp_path)


def test_sector_snapshot_exports_clean_union_with_membership(tmp_path: Path) -> None:
    result = export_sector_universe(
        FakeXtData(),
        ["半导体", "GN功率半导体", "半导体"],
        tmp_path,
    )

    assert result["sector_name"] is None
    assert result["sector_names"] == ["半导体", "GN功率半导体"]
    assert result["sector_symbol_counts"] == {
        "半导体": 2,
        "GN功率半导体": 3,
    }
    assert result["symbol_count"] == 4
    assert result["source_symbol_assignments"] == 5
    assert result["overlap_assignments"] == 1
    assert (tmp_path / "symbols.txt").read_text(encoding="utf-8").splitlines() == [
        "603501.SH",
        "605111.SH",
        "688981.SH",
        "920001.BJ",
    ]
    membership = json.loads(
        (tmp_path / "sector_membership.json").read_text(encoding="utf-8")
    )
    assert membership["688981.SH"] == ["半导体", "GN功率半导体"]


def test_multiple_symbol_files_are_combined_before_normalization(tmp_path: Path) -> None:
    first = tmp_path / "first.txt"
    second = tmp_path / "second.txt"
    first.write_text("688981.sh\n603501.SH\n", encoding="utf-8")
    second.write_text("# overlap\n688981.SH\n605111.sh\n", encoding="utf-8")

    loaded = _load_symbols([first, second])

    assert loaded is not None
    assert normalize_universe(loaded) == ["603501.SH", "605111.SH", "688981.SH"]


def test_clis_accept_repeated_sector_names_and_symbol_files(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "sector_universe.py",
            "export",
            "--sector-name",
            "SW2半导体",
            "--sector-name",
            "GN功率半导体",
            "--output-dir",
            "snapshot",
        ],
    )
    sector_args = parse_sector_args()
    assert sector_args.sector_names == ["SW2半导体", "GN功率半导体"]

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cli.py",
            "plan",
            "--symbols-file",
            "first.txt",
            "--symbols-file",
            "second.txt",
        ],
    )
    download_args = parse_download_args()
    assert download_args.symbols_files == [Path("first.txt"), Path("second.txt")]


def test_download_cli_accepts_one_direct_symbol(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["cli.py", "plan", "--symbol", "600000.sh"],
    )
    args = parse_download_args()

    assert _requested_symbols(args) == (["600000.SH"], "symbol")


def test_download_cli_rejects_direct_symbol_with_symbol_file(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cli.py",
            "plan",
            "--symbol",
            "600000.SH",
            "--symbols-file",
            "symbols.txt",
        ],
    )
    args = parse_download_args()

    with pytest.raises(ValueError, match="cannot be combined"):
        _requested_symbols(args)


def test_filter_existing_dataset_streams_symbol_subset_and_resumes(
    tmp_path: Path,
) -> None:
    source = tmp_path / "full_market"
    output = tmp_path / "semiconductor"
    frames = [
        normalize_symbol_frame(
            "688981.SH",
            pd.concat(
                [_raw_frame("2025-08-29"), _raw_frame("2026-08-28", close=10.2)],
                ignore_index=True,
            ),
        ),
        normalize_symbol_frame(
            "600000.SH",
            pd.concat(
                [_raw_frame("2025-08-29"), _raw_frame("2026-08-28", close=10.3)],
                ignore_index=True,
            ),
        ),
    ]
    write_month_partitions(pd.concat(frames, ignore_index=True), source, batch_id=0)

    first = filter_existing_dataset(
        source,
        output,
        ["688981.sh", "999999.SH"],
    )
    resumed = filter_existing_dataset(
        source,
        output,
        ["688981.sh", "999999.SH"],
    )

    assert first["status"] == "completed_with_missing_symbols"
    assert first["found_symbols"] == ["688981.SH"]
    assert first["missing_symbols"] == ["999999.SH"]
    assert first["quality"]["rows"] == 2
    assert resumed["completed_source_files"] == first["completed_source_files"]
    saved = pd.concat(
        [pd.read_parquet(path) for path in output.glob("year=*/month=*/*.parquet")],
        ignore_index=True,
    )
    assert set(saved["symbol"]) == {"688981.SH"}
    assert len(saved) == 2
    assert (output / "filter_manifest.json").exists()
    assert (output / "quality_report.json").exists()
