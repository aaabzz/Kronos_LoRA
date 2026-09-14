"""Configuration for downloading A-share 5-minute bars."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FIELDS = ("time", "open", "high", "low", "close", "volume", "amount")


@dataclass(frozen=True, slots=True)
class XtQuantDownloadConfig:
    """Validated settings shared by the CLI and downloader."""

    output_dir: Path = PROJECT_ROOT / "data" / "lora"
    period: str = "5m"
    start_date: str | None = None
    end_date: str | None = None
    lookback_years: int = 1
    include_beijing: bool = True
    include_today: bool = False
    download_batch_size: int = 100
    read_batch_size: int = 100
    dividend_type: str = "front"
    fill_data: bool = False
    retry_count: int = 3
    compression: str = "zstd"
    resume: bool = True
    fields: tuple[str, ...] = field(default=DEFAULT_FIELDS)

    def __post_init__(self) -> None:
        object.__setattr__(self, "output_dir", Path(self.output_dir).resolve())
        for value, name in (
            (self.lookback_years, "lookback_years"),
            (self.download_batch_size, "download_batch_size"),
            (self.read_batch_size, "read_batch_size"),
            (self.retry_count, "retry_count"),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be greater than zero")
        if self.period != "5m":
            raise ValueError("this dataset pipeline currently supports period='5m' only")
        if self.dividend_type not in {"none", "front", "back", "front_ratio", "back_ratio"}:
            raise ValueError(f"unsupported dividend_type: {self.dividend_type}")
        if (self.start_date is None) != (self.end_date is None):
            raise ValueError("start_date and end_date must be provided together")
        for value, name in ((self.start_date, "start_date"), (self.end_date, "end_date")):
            if value is not None:
                try:
                    datetime.strptime(value, "%Y%m%d")
                except ValueError as exc:
                    raise ValueError(f"{name} must be a valid YYYYMMDD date") from exc
        if self.start_date and self.end_date and self.start_date > self.end_date:
            raise ValueError("start_date must not be after end_date")
        missing = set(DEFAULT_FIELDS) - set(self.fields)
        if missing:
            raise ValueError(f"fields are missing required values: {sorted(missing)}")

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["output_dir"] = str(self.output_dir)
        result["fields"] = list(self.fields)
        return result
