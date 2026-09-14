"""CLI for stable symbol buckets and pre-tokenized Kronos training shards."""

from __future__ import annotations

import argparse
import hashlib
from dataclasses import replace
from pathlib import Path

import torch

from lora.config import QVLoRAConfig
from lora.model_factory import load_frozen_tokenizer

from .config import DataConfig
from .sharding import bucketize_parquet, build_token_shards
from .training import select_device


def tokenizer_fingerprint(config: QVLoRAConfig) -> str:
    digest = hashlib.sha256()
    for name in ("config.json", "model.safetensors"):
        path = config.tokenizer_path / name
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build lazy Kronos token shards.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    bucket = subparsers.add_parser("bucketize", help="Hash raw Parquet by symbol.")
    bucket.add_argument("--data-path", type=Path, required=True)
    bucket.add_argument("--bucket-dir", type=Path, required=True)
    bucket.add_argument("--bucket-count", type=int, default=64)
    bucket.add_argument(
        "--symbol",
        help="Optionally bucket only one XtQuant symbol, for example 600000.SH.",
    )

    tokenize = subparsers.add_parser("tokenize", help="Create token shards by bucket.")
    tokenize.add_argument("--bucket-dir", type=Path, required=True)
    tokenize.add_argument("--token-cache-path", type=Path, required=True)
    tokenize.add_argument(
        "--tokenizer-path",
        type=Path,
        default=QVLoRAConfig().tokenizer_path,
        help="Local Kronos tokenizer used to create this cache.",
    )
    tokenize.add_argument("--train-end", required=True)
    tokenize.add_argument("--validation-end", required=True)
    tokenize.add_argument(
        "--symbol",
        help=(
            "Optionally tokenize only one symbol from the bucket data; useful "
            "for reusing existing market or sector buckets."
        ),
    )
    tokenize.add_argument("--context-length", type=int, default=512)
    tokenize.add_argument("--stride", type=int, default=128)
    tokenize.add_argument(
        "--target-mode",
        choices=("all", "last", "tail"),
        default=DataConfig().target_mode,
        help=(
            "all supervises every eligible target; last supervises only the "
            "final target; tail supervises the final --target-length targets."
        ),
    )
    tokenize.add_argument(
        "--target-length",
        type=int,
        default=None,
        help="Number of final target positions supervised by target-mode tail.",
    )
    tokenize.add_argument("--encode-batch-size", type=int, default=64)
    tokenize.add_argument("--samples-per-shard", type=int, default=4096)
    tokenize.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "bucketize":
        metadata = bucketize_parquet(
            replace(DataConfig(), data_path=args.data_path),
            args.bucket_dir,
            bucket_count=args.bucket_count,
            symbol=args.symbol,
        )
        print(f"bucketization complete: {metadata}")
        return

    data_config = replace(
        DataConfig(),
        train_end=args.train_end,
        validation_end=args.validation_end,
        context_length=args.context_length,
        stride=args.stride,
        target_mode=args.target_mode,
        target_length=args.target_length,
    )
    lora_config = replace(QVLoRAConfig(), tokenizer_path=args.tokenizer_path)
    device = select_device(args.device)
    tokenizer = load_frozen_tokenizer(lora_config)
    manifest = build_token_shards(
        data_config,
        args.bucket_dir,
        args.token_cache_path,
        tokenizer,
        device=device,
        encode_batch_size=args.encode_batch_size,
        samples_per_shard=args.samples_per_shard,
        tokenizer_fingerprint=tokenizer_fingerprint(lora_config),
        symbol=args.symbol,
    )
    print(f"token shard build complete: {manifest['counts']}")


if __name__ == "__main__":
    main()
