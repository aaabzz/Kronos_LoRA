"""CLI for standalone LoRA-compatible Kronos prediction modes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from lora import QVLoRAConfig

from .predictor import FEATURE_COLUMNS, predict_modes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run greedy and sampled Kronos K-line predictions."
    )
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--adapter-path", type=Path, required=True)
    parser.add_argument(
        "--base-model-path",
        type=Path,
        default=QVLoRAConfig().base_model_path,
        help="Local base model on which the adapter was trained.",
    )
    parser.add_argument(
        "--tokenizer-path",
        type=Path,
        default=QVLoRAConfig().tokenizer_path,
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--timestamp-column", default="timestamp")
    parser.add_argument("--symbol-column", default="symbol")
    parser.add_argument(
        "--symbol",
        default=None,
        help="Required when the input Parquet contains more than one symbol.",
    )
    parser.add_argument("--history-length", type=int, default=512)
    parser.add_argument(
        "--future-timestamp",
        action="append",
        required=True,
        help="Future K-line timestamp; repeat this option for multiple bars.",
    )
    parser.add_argument("--sample-count", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--flat-tolerance", type=float, default=0.0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--save-samples",
        action="store_true",
        help="Also save all sampled paths to sampled_paths.npy.",
    )
    return parser.parse_args()


def _prepare_output_dir(path: Path) -> None:
    if path.exists() and not path.is_dir():
        raise FileExistsError(f"Output path is not a directory: {path}")
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty: {path}. Use a new experiment directory."
        )
    path.mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = parse_args()
    if args.history_length <= 0:
        raise ValueError("--history-length must be positive.")
    frame = pd.read_parquet(args.data_path)
    required = {args.timestamp_column, *FEATURE_COLUMNS[:4]}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Input Parquet is missing columns: {missing}")
    frame[args.timestamp_column] = pd.to_datetime(
        frame[args.timestamp_column], errors="raise"
    )
    if args.symbol is not None:
        if args.symbol_column not in frame.columns:
            raise ValueError(
                f"Cannot filter --symbol because {args.symbol_column!r} is missing."
            )
        frame = frame[
            frame[args.symbol_column].astype(str) == str(args.symbol)
        ]
        if frame.empty:
            raise ValueError(f"No rows found for symbol {args.symbol!r}.")
    elif (
        args.symbol_column in frame.columns
        and frame[args.symbol_column].astype(str).nunique() != 1
    ):
        raise ValueError("Input contains multiple symbols; provide --symbol.")
    frame = frame.sort_values(args.timestamp_column).tail(args.history_length)
    x_timestamps = pd.DatetimeIndex(frame[args.timestamp_column])
    y_timestamps = pd.DatetimeIndex(pd.to_datetime(args.future_timestamp))

    result = predict_modes(
        frame,
        x_timestamps,
        y_timestamps,
        adapter_path=args.adapter_path,
        device=args.device,
        sample_count=args.sample_count,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        seed=args.seed,
        flat_tolerance=args.flat_tolerance,
        config=QVLoRAConfig(
            base_model_path=args.base_model_path,
            tokenizer_path=args.tokenizer_path,
        ),
    )

    _prepare_output_dir(args.output_dir)
    result.greedy.to_csv(args.output_dir / "greedy.csv", index_label="timestamp")
    result.sample_mean.to_csv(
        args.output_dir / "sample_mean.csv", index_label="timestamp"
    )
    result.sample_median.to_csv(
        args.output_dir / "sample_median.csv", index_label="timestamp"
    )
    (args.output_dir / "next_bar_probabilities.json").write_text(
        json.dumps(result.probability_summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if args.save_samples:
        np.save(args.output_dir / "sampled_paths.npy", result.sampled_paths)

    print(f"greedy={args.output_dir / 'greedy.csv'}")
    print(f"sample_mean={args.output_dir / 'sample_mean.csv'}")
    print(f"sample_median={args.output_dir / 'sample_median.csv'}")
    print(f"up_probability={result.up_probability:.2%}")
    print(f"down_probability={result.down_probability:.2%}")
    print(f"flat_probability={result.flat_probability:.2%}")


if __name__ == "__main__":
    main()
