"""Read K-line Parquet files and build leakage-aware chronological windows."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .config import DataConfig


SplitName = Literal["train", "validation", "test"]


@dataclass(frozen=True)
class TimeBoundaries:
    train_end: pd.Timestamp
    validation_end: pd.Timestamp

    def validate(self) -> None:
        if self.train_end >= self.validation_end:
            raise ValueError("train_end must be earlier than validation_end.")


@dataclass(frozen=True)
class WindowLocation:
    symbol: str
    start: int


def discover_parquet_files(data_path: Path) -> list[Path]:
    if data_path.is_file():
        if data_path.suffix.lower() != ".parquet":
            raise ValueError(f"Expected a .parquet file, got: {data_path}")
        return [data_path]
    if not data_path.exists():
        raise FileNotFoundError(f"Data path does not exist: {data_path}")

    files = sorted(data_path.rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No Parquet files found under: {data_path}")
    return files


def load_parquet_data(config: DataConfig) -> pd.DataFrame:
    """Load one or more Parquet files and validate the common K-line schema."""
    config.validate()
    files = discover_parquet_files(config.data_path)
    required = [
        config.symbol_column,
        config.timestamp_column,
        *config.feature_columns,
    ]

    frames: list[pd.DataFrame] = []
    for path in files:
        try:
            frame = pd.read_parquet(path, columns=required)
        except ImportError as exc:
            raise RuntimeError(
                "Reading Parquet requires pyarrow. Install finetune_lora/requirements.txt."
            ) from exc
        missing = sorted(set(required) - set(frame.columns))
        if missing:
            raise ValueError(f"{path} is missing columns: {missing}")
        frames.append(frame)

    data = pd.concat(frames, ignore_index=True)
    data[config.timestamp_column] = pd.to_datetime(
        data[config.timestamp_column], errors="raise"
    )
    data[config.symbol_column] = data[config.symbol_column].astype(str)

    if data[required].isnull().any().any():
        bad_columns = data[required].columns[data[required].isnull().any()].tolist()
        raise ValueError(f"Parquet data contains nulls in columns: {bad_columns}")

    feature_values = data[list(config.feature_columns)].to_numpy(dtype=np.float64)
    if not np.isfinite(feature_values).all():
        raise ValueError("Feature columns contain NaN or infinite values.")

    duplicate_mask = data.duplicated(
        [config.symbol_column, config.timestamp_column], keep=False
    )
    if duplicate_mask.any():
        examples = data.loc[
            duplicate_mask, [config.symbol_column, config.timestamp_column]
        ].head(5)
        raise ValueError(f"Duplicate symbol/timestamp rows found:\n{examples}")

    return data.sort_values(
        [config.symbol_column, config.timestamp_column]
    ).reset_index(drop=True)


def resolve_time_boundaries(
    data: pd.DataFrame, config: DataConfig
) -> TimeBoundaries:
    """Resolve explicit boundaries or derive development-only quantile splits."""
    timestamp = config.timestamp_column
    timestamps = pd.Index(data[timestamp].drop_duplicates().sort_values())
    if len(timestamps) < 3:
        raise ValueError("At least three distinct timestamps are required.")

    if config.train_end is not None:
        boundaries = TimeBoundaries(
            train_end=pd.Timestamp(config.train_end),
            validation_end=pd.Timestamp(config.validation_end),
        )
    else:
        train_index = max(0, int(len(timestamps) * config.train_ratio) - 1)
        validation_index = max(
            train_index + 1,
            int(len(timestamps) * (config.train_ratio + config.validation_ratio))
            - 1,
        )
        validation_index = min(validation_index, len(timestamps) - 2)
        boundaries = TimeBoundaries(
            train_end=pd.Timestamp(timestamps[train_index]),
            validation_end=pd.Timestamp(timestamps[validation_index]),
        )

    boundaries.validate()
    minimum = pd.Timestamp(timestamps[0])
    maximum = pd.Timestamp(timestamps[-1])
    if not minimum <= boundaries.train_end < boundaries.validation_end < maximum:
        raise ValueError(
            "Split boundaries must fall inside the dataset timestamp range: "
            f"[{minimum}, {maximum}]."
        )
    return boundaries


def target_mask_for_split(
    target_timestamps: np.ndarray,
    split: SplitName,
    boundaries: TimeBoundaries,
    target_mode: str = "all",
    target_length: int | None = None,
) -> np.ndarray:
    train_end = np.datetime64(boundaries.train_end)
    validation_end = np.datetime64(boundaries.validation_end)
    if split == "train":
        split_mask = target_timestamps <= train_end
    elif split == "validation":
        split_mask = (target_timestamps > train_end) & (
            target_timestamps <= validation_end
        )
    elif split == "test":
        split_mask = target_timestamps > validation_end
    else:
        raise ValueError(f"Unsupported split: {split}")

    if target_mode == "all":
        return split_mask
    if target_mode == "last":
        last_mask = np.zeros_like(split_mask, dtype=bool)
        last_mask[-1] = split_mask[-1]
        return last_mask
    if target_mode == "tail":
        if target_length is None:
            raise ValueError("target_length is required when target_mode=tail.")
        if not 1 <= target_length <= len(split_mask):
            raise ValueError(
                "target_length must be in [1, number of target positions]."
            )
        tail_mask = np.zeros_like(split_mask, dtype=bool)
        tail_mask[-target_length:] = split_mask[-target_length:]
        return tail_mask
    raise ValueError(f"Unsupported target_mode: {target_mode}")


class KlineWindowDataset(Dataset):
    """Sliding windows whose loss only covers targets in the selected split."""

    def __init__(
        self,
        data: pd.DataFrame | dict[str, pd.DataFrame],
        config: DataConfig,
        split: SplitName,
        boundaries: TimeBoundaries,
    ) -> None:
        config.validate()
        boundaries.validate()
        self.config = config
        self.split = split
        self.boundaries = boundaries
        self.window_size = config.context_length + 1
        self.series = (
            data if isinstance(data, dict) else prepare_symbol_series(data, config)
        )
        self.locations: list[WindowLocation] = []

        for symbol, frame in self.series.items():
            if len(frame) < self.window_size:
                continue
            timestamps = frame[config.timestamp_column].to_numpy(
                dtype="datetime64[ns]"
            )

            last_start = len(frame) - self.window_size
            starts = list(range(0, last_start + 1, config.stride))
            if starts[-1] != last_start:
                starts.append(last_start)

            for start in starts:
                window_timestamps = timestamps[start : start + self.window_size]
                # Never let training normalization inspect validation/test rows,
                # and never let validation normalization inspect test rows.
                if split == "train" and window_timestamps[-1] > np.datetime64(
                    boundaries.train_end
                ):
                    continue
                if split == "validation" and window_timestamps[-1] > np.datetime64(
                    boundaries.validation_end
                ):
                    continue
                # Position i predicts i+1, so masks align to rows 1..window_size-1.
                target_timestamps = window_timestamps[1:]
                loss_mask = target_mask_for_split(
                    target_timestamps,
                    split,
                    boundaries,
                    config.target_mode,
                    config.target_length,
                )
                if loss_mask.any():
                    self.locations.append(WindowLocation(str(symbol), start))

        if not self.locations:
            raise ValueError(
                f"No {split} windows were produced. Check date boundaries, "
                "context_length, stride, and per-symbol history length."
            )

    def __len__(self) -> int:
        return len(self.locations)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        location = self.locations[index]
        frame = self.series[location.symbol].iloc[
            location.start : location.start + self.window_size
        ]
        features = frame[list(self.config.feature_columns)].to_numpy(
            dtype=np.float32, copy=True
        )

        # Normalize with the model-input portion only; the final target row is
        # excluded from mean/std estimation to avoid future-value leakage.
        normalization_context = features[: self.config.context_length]
        mean = normalization_context.mean(axis=0)
        std = normalization_context.std(axis=0)
        features = (features - mean) / (std + 1e-5)
        features = np.clip(features, -self.config.clip, self.config.clip)

        timestamps = frame[self.config.timestamp_column].dt
        time_features = np.stack(
            [
                timestamps.minute.to_numpy(),
                timestamps.hour.to_numpy(),
                timestamps.weekday.to_numpy(),
                timestamps.day.to_numpy(),
                timestamps.month.to_numpy(),
            ],
            axis=-1,
        ).astype(np.float32)

        target_timestamps = frame[self.config.timestamp_column].to_numpy(
            dtype="datetime64[ns]"
        )[1:]
        loss_mask = target_mask_for_split(
            target_timestamps,
            self.split,
            self.boundaries,
            self.config.target_mode,
            self.config.target_length,
        )

        return (
            torch.from_numpy(features),
            torch.from_numpy(time_features),
            torch.from_numpy(loss_mask),
        )


def prepare_symbol_series(
    data: pd.DataFrame, config: DataConfig
) -> dict[str, pd.DataFrame]:
    """Sort and materialize each symbol once so all splits can share it."""
    return {
        str(symbol): frame.sort_values(config.timestamp_column).reset_index(drop=True)
        for symbol, frame in data.groupby(config.symbol_column, sort=True)
    }


def build_datasets(
    config: DataConfig,
) -> tuple[dict[SplitName, Dataset], TimeBoundaries]:
    if config.token_cache_path is not None:
        # Local import avoids a module cycle: sharding reuses the online
        # KlineWindowDataset while building the cache.
        from .sharding import build_token_datasets

        return build_token_datasets(config.token_cache_path, config)

    data = load_parquet_data(config)
    boundaries = resolve_time_boundaries(data, config)
    series = prepare_symbol_series(data, config)
    datasets = {
        split: KlineWindowDataset(series, config, split, boundaries)
        for split in ("train", "validation", "test")
    }
    return datasets, boundaries
