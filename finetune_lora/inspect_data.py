"""Validate Parquet schema and report chronological split/window counts."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from .config import DataConfig
from .dataset import KlineWindowDataset, load_parquet_data, resolve_time_boundaries


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect Kronos LoRA Parquet data.")
    parser.add_argument("--data-path", type=Path, default=DataConfig().data_path)
    parser.add_argument("--train-end", default=None)
    parser.add_argument("--validation-end", default=None)
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument("--stride", type=int, default=32)
    parser.add_argument(
        "--target-mode",
        choices=("all", "last", "tail"),
        default=DataConfig().target_mode,
    )
    parser.add_argument("--target-length", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = replace(
        DataConfig(),
        data_path=args.data_path,
        train_end=args.train_end,
        validation_end=args.validation_end,
        context_length=args.context_length,
        stride=args.stride,
        target_mode=args.target_mode,
        target_length=args.target_length,
    )
    data = load_parquet_data(config)
    boundaries = resolve_time_boundaries(data, config)
    print(f"rows: {len(data):,}")
    print(f"symbols: {data[config.symbol_column].nunique():,}")
    print(
        f"timestamp range: {data[config.timestamp_column].min()} -> "
        f"{data[config.timestamp_column].max()}"
    )
    print(f"train_end: {boundaries.train_end}")
    print(f"validation_end: {boundaries.validation_end}")
    print(f"target_mode: {config.target_mode}")
    print(f"target_length: {config.target_length}")
    for split in ("train", "validation", "test"):
        dataset = KlineWindowDataset(data, config, split, boundaries)
        print(f"{split} windows: {len(dataset):,}")


if __name__ == "__main__":
    main()
