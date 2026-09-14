"""Streaming Parquet bucketing and lazy pre-tokenized training shards."""

from __future__ import annotations

import hashlib
import json
import math
from bisect import bisect_right
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from data_download.xtquant.downloader import normalize_universe, universe_digest

from .config import DataConfig
from .dataset import (
    KlineWindowDataset,
    SplitName,
    TimeBoundaries,
    discover_parquet_files,
    prepare_symbol_series,
)


CACHE_FORMAT_VERSION = 1
TOKEN_KEYS = ("s1_tokens", "s2_tokens", "stamps", "loss_mask")


def normalize_symbol_selection(symbol: str | None) -> tuple[str, ...] | None:
    """Validate one optional XtQuant symbol used to isolate shard creation."""
    if symbol is None:
        return None
    normalized = normalize_universe([symbol])
    if len(normalized) != 1:
        raise ValueError(
            "symbol must be an A-share XtQuant code ending in .SH, .SZ or .BJ"
        )
    return tuple(normalized)


def stable_bucket_id(symbol: str, bucket_count: int) -> int:
    """Map a symbol to the same bucket across runs and Python processes."""
    if bucket_count <= 0:
        raise ValueError("bucket_count must be positive.")
    digest = hashlib.blake2b(symbol.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, byteorder="little") % bucket_count


def _ensure_empty_output(path: Path, label: str) -> None:
    if path.exists() and not path.is_dir():
        raise FileExistsError(f"{label} is not a directory: {path}")
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(
            f"{label} is not empty: {path}. Use a new directory to avoid "
            "mixing incompatible or partial shards."
        )
    path.mkdir(parents=True, exist_ok=True)


def _required_columns(config: DataConfig) -> list[str]:
    return [
        config.symbol_column,
        config.timestamp_column,
        *config.feature_columns,
    ]


def _validate_frame(frame: pd.DataFrame, config: DataConfig, source: str) -> None:
    required = _required_columns(config)
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"{source} is missing columns: {missing}")
    if frame[required].isnull().any().any():
        raise ValueError(f"{source} contains null values.")
    values = frame[list(config.feature_columns)].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError(f"{source} contains NaN or infinite feature values.")


def bucketize_parquet(
    config: DataConfig,
    bucket_dir: Path,
    *,
    bucket_count: int = 64,
    symbol: str | None = None,
) -> dict[str, object]:
    """Read one source Parquet at a time and write stable symbol fragments."""
    config.validate()
    if bucket_count <= 0:
        raise ValueError("bucket_count must be positive.")
    _ensure_empty_output(bucket_dir, "Bucket output directory")

    files = discover_parquet_files(config.data_path)
    selected_symbols = normalize_symbol_selection(symbol)
    selected = set(selected_symbols or ())
    required = _required_columns(config)
    rows = 0
    fragments = 0
    symbols_seen: set[str] = set()
    for file_index, path in enumerate(files):
        frame = pd.read_parquet(path, columns=required)
        _validate_frame(frame, config, str(path))
        frame[config.symbol_column] = (
            frame[config.symbol_column].astype(str).str.strip().str.upper()
        )
        if selected_symbols is not None:
            frame = frame.loc[frame[config.symbol_column].isin(selected)].copy()
            if frame.empty:
                print(f"bucketized {file_index + 1}/{len(files)}: {path} (no match)")
                continue
        frame[config.timestamp_column] = pd.to_datetime(
            frame[config.timestamp_column], errors="raise"
        )
        symbols_seen.update(frame[config.symbol_column].unique().tolist())
        frame["__bucket_id"] = frame[config.symbol_column].map(
            lambda symbol: stable_bucket_id(symbol, bucket_count)
        )
        rows += len(frame)

        for bucket_id, bucket_frame in frame.groupby("__bucket_id", sort=True):
            destination = bucket_dir / f"bucket={int(bucket_id):03d}"
            destination.mkdir(parents=True, exist_ok=True)
            fragment = destination / f"fragment-{file_index:05d}.parquet"
            bucket_frame.drop(columns="__bucket_id").to_parquet(
                fragment,
                index=False,
                compression="zstd",
            )
            fragments += 1

        del frame
        print(f"bucketized {file_index + 1}/{len(files)}: {path}")

    if selected_symbols is not None and symbols_seen != selected:
        missing = sorted(selected - symbols_seen)
        raise ValueError(f"Requested symbol was not found in source Parquet: {missing}")

    metadata: dict[str, object] = {
        "format_version": CACHE_FORMAT_VERSION,
        "source_data": str(config.data_path.resolve()),
        "source_files": len(files),
        "rows": rows,
        "symbols": sorted(symbols_seen),
        "symbol_count": len(symbols_seen),
        "universe_sha256": universe_digest(sorted(symbols_seen)),
        "requested_symbol": selected_symbols[0] if selected_symbols else None,
        "bucket_count": bucket_count,
        "fragments": fragments,
        "symbol_column": config.symbol_column,
        "timestamp_column": config.timestamp_column,
        "feature_columns": list(config.feature_columns),
    }
    manifest_path = bucket_dir / "bucket_manifest.json"
    temporary = manifest_path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(manifest_path)
    return metadata


class _TokenShardWriter:
    def __init__(self, cache_dir: Path, split: SplitName, samples_per_shard: int):
        self.cache_dir = cache_dir
        self.split = split
        self.samples_per_shard = samples_per_shard
        self.pending: dict[str, list[torch.Tensor]] = {key: [] for key in TOKEN_KEYS}
        self.pending_count = 0
        self.shard_index = 0
        self.entries: list[dict[str, object]] = []

    def add(self, payload: dict[str, torch.Tensor]) -> None:
        batch_size = int(payload["s1_tokens"].shape[0])
        offset = 0
        while offset < batch_size:
            capacity = self.samples_per_shard - self.pending_count
            take = min(capacity, batch_size - offset)
            for key in TOKEN_KEYS:
                self.pending[key].append(payload[key][offset : offset + take].cpu())
            self.pending_count += take
            offset += take
            if self.pending_count == self.samples_per_shard:
                self.flush()

    def flush(self) -> None:
        if self.pending_count == 0:
            return
        payload = {key: torch.cat(self.pending[key], dim=0) for key in TOKEN_KEYS}
        split_dir = self.cache_dir / self.split
        split_dir.mkdir(parents=True, exist_ok=True)
        filename = f"shard-{self.shard_index:05d}.pt"
        path = split_dir / filename
        temporary = path.with_suffix(".pt.tmp")
        torch.save(payload, temporary)
        temporary.replace(path)
        self.entries.append(
            {
                "path": path.relative_to(self.cache_dir).as_posix(),
                "split": self.split,
                "samples": self.pending_count,
            }
        )
        self.pending = {key: [] for key in TOKEN_KEYS}
        self.pending_count = 0
        self.shard_index += 1


def _load_bucket(bucket_path: Path, config: DataConfig) -> pd.DataFrame:
    fragments = sorted(bucket_path.glob("fragment-*.parquet"))
    if not fragments:
        raise FileNotFoundError(f"No fragments found in {bucket_path}.")
    required = _required_columns(config)
    frames = [pd.read_parquet(path, columns=required) for path in fragments]
    frame = pd.concat(frames, ignore_index=True)
    _validate_frame(frame, config, str(bucket_path))
    frame[config.symbol_column] = (
        frame[config.symbol_column].astype(str).str.strip().str.upper()
    )
    frame[config.timestamp_column] = pd.to_datetime(
        frame[config.timestamp_column], errors="raise"
    )
    duplicates = frame.duplicated(
        [config.symbol_column, config.timestamp_column], keep=False
    )
    if duplicates.any():
        examples = frame.loc[
            duplicates, [config.symbol_column, config.timestamp_column]
        ].head(5)
        raise ValueError(f"Duplicate symbol/timestamp rows in {bucket_path}:\n{examples}")
    return frame.sort_values(
        [config.symbol_column, config.timestamp_column]
    ).reset_index(drop=True)


def build_token_shards(
    config: DataConfig,
    bucket_dir: Path,
    cache_dir: Path,
    tokenizer,
    *,
    device: torch.device,
    encode_batch_size: int = 64,
    samples_per_shard: int = 4096,
    tokenizer_fingerprint: str = "unknown",
    symbol: str | None = None,
) -> dict[str, object]:
    """Process one symbol bucket at a time into compact token training shards."""
    config.validate()
    if config.train_end is None or config.validation_end is None:
        raise ValueError("Token shard building requires explicit split boundaries.")
    if encode_batch_size <= 0 or samples_per_shard <= 0:
        raise ValueError("Batch and shard sizes must be positive.")
    if not bucket_dir.is_dir():
        raise FileNotFoundError(f"Bucket directory does not exist: {bucket_dir}")
    _ensure_empty_output(cache_dir, "Token cache directory")

    boundaries = TimeBoundaries(
        pd.Timestamp(config.train_end), pd.Timestamp(config.validation_end)
    )
    boundaries.validate()
    selected_symbols = normalize_symbol_selection(symbol)
    selected = set(selected_symbols or ())
    writers = {
        split: _TokenShardWriter(cache_dir, split, samples_per_shard)
        for split in ("train", "validation", "test")
    }
    bucket_paths = sorted(path for path in bucket_dir.glob("bucket=*") if path.is_dir())
    if not bucket_paths:
        raise FileNotFoundError(f"No bucket directories found under: {bucket_dir}")

    tokenizer.eval()
    tokenizer.requires_grad_(False)
    tokenizer = tokenizer.to(device)
    symbols_seen: set[str] = set()
    for bucket_index, bucket_path in enumerate(bucket_paths, start=1):
        frame = _load_bucket(bucket_path, config)
        series = prepare_symbol_series(frame, config)
        if selected_symbols is not None:
            series = {
                symbol_name: symbol_frame
                for symbol_name, symbol_frame in series.items()
                if symbol_name in selected
            }
            if not series:
                del frame
                print(
                    f"tokenized bucket {bucket_index}/{len(bucket_paths)}: "
                    f"{bucket_path.name} (no match)"
                )
                continue
        symbols_seen.update(series)
        for split, writer in writers.items():
            try:
                dataset = KlineWindowDataset(series, config, split, boundaries)
            except ValueError as exc:
                if str(exc).startswith(f"No {split} windows were produced"):
                    continue
                raise
            loader = DataLoader(dataset, batch_size=encode_batch_size, shuffle=False)
            for features, stamps, loss_mask in loader:
                features = features.to(device)
                with torch.no_grad():
                    s1_tokens, s2_tokens = tokenizer.encode(features, half=True)
                token_min = min(int(s1_tokens.min()), int(s2_tokens.min()))
                token_max = max(int(s1_tokens.max()), int(s2_tokens.max()))
                if token_min < 0 or token_max > 32767:
                    raise ValueError("Tokenizer IDs exceed int16 storage range.")
                writer.add(
                    {
                        # int16 is portable across the supported PyTorch
                        # versions and easily holds Kronos IDs (0..1023).
                        "s1_tokens": s1_tokens.to(torch.int16),
                        "s2_tokens": s2_tokens.to(torch.int16),
                        "stamps": stamps[:, :-1, :].to(torch.uint8),
                        "loss_mask": loss_mask.bool(),
                    }
                )
        del frame, series
        print(f"tokenized bucket {bucket_index}/{len(bucket_paths)}: {bucket_path.name}")

    if selected_symbols is not None and symbols_seen != selected:
        missing = sorted(selected - symbols_seen)
        raise ValueError(f"Requested symbol was not found in bucket data: {missing}")

    entries: list[dict[str, object]] = []
    for writer in writers.values():
        writer.flush()
        entries.extend(writer.entries)
    counts = {
        split: sum(int(entry["samples"]) for entry in entries if entry["split"] == split)
        for split in ("train", "validation", "test")
    }
    empty = [split for split, count in counts.items() if count == 0]
    if empty:
        raise ValueError(f"No tokenized samples were produced for splits: {empty}")

    manifest: dict[str, object] = {
        "format_version": CACHE_FORMAT_VERSION,
        "train_end": str(boundaries.train_end),
        "validation_end": str(boundaries.validation_end),
        "context_length": config.context_length,
        "stride": config.stride,
        "target_mode": config.target_mode,
        "target_length": config.target_length,
        "clip": config.clip,
        "feature_columns": list(config.feature_columns),
        "tokenizer_fingerprint": tokenizer_fingerprint,
        "symbols": sorted(symbols_seen),
        "symbol_count": len(symbols_seen),
        "universe_sha256": universe_digest(sorted(symbols_seen)),
        "requested_symbol": selected_symbols[0] if selected_symbols else None,
        "counts": counts,
        "shards": entries,
    }
    manifest_path = cache_dir / "manifest.json"
    temporary = manifest_path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(manifest_path)
    return manifest


def load_token_manifest(cache_dir: Path) -> dict[str, object]:
    path = cache_dir / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"Token cache manifest does not exist: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("format_version") != CACHE_FORMAT_VERSION:
        raise ValueError(
            f"Unsupported token cache format: {manifest.get('format_version')}"
        )
    return manifest


class TokenShardDataset(Dataset):
    """Map-style dataset that keeps only one pre-tokenized shard in memory."""

    def __init__(
        self,
        cache_dir: Path,
        split: SplitName,
        *,
        target_mode: str | None = None,
        target_length: int | None = None,
    ):
        self.cache_dir = cache_dir
        self.split = split
        self.manifest = load_token_manifest(cache_dir)
        if target_mode is None:
            self.target_mode = str(self.manifest.get("target_mode", "all"))
            manifest_target_length = self.manifest.get("target_length")
            self.target_length = (
                None
                if manifest_target_length is None
                else int(manifest_target_length)
            )
        else:
            self.target_mode = target_mode
            self.target_length = target_length
        self.entries = [
            entry for entry in self.manifest["shards"] if entry["split"] == split
        ]
        if not self.entries:
            raise ValueError(f"Token cache has no {split} shards.")
        self.cumulative_sizes = np.cumsum(
            [int(entry["samples"]) for entry in self.entries]
        ).tolist()
        self._cached_shard_index: int | None = None
        self._cached_payload: dict[str, torch.Tensor] | None = None

    def __len__(self) -> int:
        return int(self.cumulative_sizes[-1])

    @property
    def shard_ranges(self) -> list[tuple[int, int]]:
        starts = [0, *self.cumulative_sizes[:-1]]
        return list(zip(starts, self.cumulative_sizes))

    def _load_shard(self, shard_index: int) -> dict[str, torch.Tensor]:
        if self._cached_shard_index != shard_index:
            path = self.cache_dir / str(self.entries[shard_index]["path"])
            try:
                payload = torch.load(path, map_location="cpu", weights_only=True)
            except TypeError:  # Compatibility with older supported PyTorch.
                payload = torch.load(path, map_location="cpu")
            missing = sorted(set(TOKEN_KEYS) - set(payload))
            if missing:
                raise ValueError(f"Token shard {path} is missing keys: {missing}")
            self._cached_payload = payload
            self._cached_shard_index = shard_index
        assert self._cached_payload is not None
        return self._cached_payload

    def __getitem__(self, index: int):
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        shard_index = bisect_right(self.cumulative_sizes, index)
        previous = 0 if shard_index == 0 else self.cumulative_sizes[shard_index - 1]
        local_index = index - previous
        payload = self._load_shard(shard_index)
        s1 = payload["s1_tokens"][local_index].long()
        s2 = payload["s2_tokens"][local_index].long()
        loss_mask = payload["loss_mask"][local_index].bool()
        if self.target_mode == "last":
            narrowed_mask = torch.zeros_like(loss_mask)
            narrowed_mask[-1] = loss_mask[-1]
            loss_mask = narrowed_mask
        elif self.target_mode == "tail":
            if self.target_length is None:
                raise ValueError("target_length is required when target_mode=tail.")
            narrowed_mask = torch.zeros_like(loss_mask)
            narrowed_mask[-self.target_length :] = loss_mask[-self.target_length :]
            loss_mask = narrowed_mask
        return (
            s1[:-1],
            s2[:-1],
            s1[1:],
            s2[1:],
            payload["stamps"][local_index].float(),
            loss_mask,
        )


class ShardShuffleSampler(Sampler[int]):
    """Shuffle shard order and samples within a shard while preserving locality."""

    def __init__(self, dataset: TokenShardDataset, seed: int = 0):
        self.dataset = dataset
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.dataset)

    def __iter__(self) -> Iterator[int]:
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        ranges = self.dataset.shard_ranges
        shard_order = torch.randperm(len(ranges), generator=generator).tolist()
        for shard_index in shard_order:
            start, end = ranges[shard_index]
            local_order = torch.randperm(end - start, generator=generator).tolist()
            yield from (start + local_index for local_index in local_order)
        self.epoch += 1


class DistributedShardSampler(Sampler[int]):
    """Equal-length rank slices of a shard-local randomized global order."""

    def __init__(
        self,
        dataset: TokenShardDataset,
        *,
        num_replicas: int,
        rank: int,
        seed: int = 0,
    ) -> None:
        if num_replicas <= 0:
            raise ValueError("num_replicas must be positive.")
        if rank < 0 or rank >= num_replicas:
            raise ValueError("rank must be in [0, num_replicas).")
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.seed = seed
        self.epoch = 0
        self.num_samples = math.ceil(len(dataset) / num_replicas)
        self.total_size = self.num_samples * num_replicas

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return self.num_samples

    def _ordered_indices(self, generator: torch.Generator) -> Iterator[int]:
        ranges = self.dataset.shard_ranges
        shard_order = torch.randperm(len(ranges), generator=generator).tolist()
        for shard_index in shard_order:
            start, end = ranges[shard_index]
            local_order = torch.randperm(end - start, generator=generator).tolist()
            yield from (start + local_index for local_index in local_order)

    def __iter__(self) -> Iterator[int]:
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        rank_start = self.rank * self.num_samples
        rank_end = rank_start + self.num_samples
        padding = self.total_size - len(self.dataset)
        prefix: list[int] = []

        for position, index in enumerate(self._ordered_indices(generator)):
            if len(prefix) < padding:
                prefix.append(index)
            if rank_start <= position < min(rank_end, len(self.dataset)):
                yield index

        # Match DistributedSampler semantics: pad by at most world_size - 1
        # samples so every rank performs the same number of backward calls.
        if rank_end > len(self.dataset):
            padding_start = max(0, rank_start - len(self.dataset))
            padding_end = rank_end - len(self.dataset)
            for padding_index in range(padding_start, padding_end):
                yield prefix[padding_index % len(prefix)]


class DistributedSequentialSampler(Sampler[int]):
    """Non-overlapping contiguous rank slices for exact evaluation metrics."""

    def __init__(self, dataset: Dataset, *, num_replicas: int, rank: int) -> None:
        if num_replicas <= 0:
            raise ValueError("num_replicas must be positive.")
        if rank < 0 or rank >= num_replicas:
            raise ValueError("rank must be in [0, num_replicas).")
        total = len(dataset)
        self.start = total * rank // num_replicas
        self.end = total * (rank + 1) // num_replicas

    def __len__(self) -> int:
        return self.end - self.start

    def __iter__(self) -> Iterator[int]:
        return iter(range(self.start, self.end))


def build_token_datasets(
    cache_dir: Path, config: DataConfig
) -> tuple[dict[SplitName, Dataset], TimeBoundaries]:
    manifest = load_token_manifest(cache_dir)
    if int(manifest["context_length"]) != config.context_length:
        raise ValueError(
            "Token cache context_length does not match training config: "
            f"{manifest['context_length']} != {config.context_length}."
        )
    if int(manifest["stride"]) != config.stride:
        raise ValueError(
            "Token cache stride does not match training config: "
            f"{manifest['stride']} != {config.stride}."
        )
    manifest_target_mode = str(manifest.get("target_mode", "all"))
    manifest_target_length = manifest.get("target_length")
    if manifest_target_length is not None:
        manifest_target_length = int(manifest_target_length)
    exact_target_config = (
        manifest_target_mode == config.target_mode
        and manifest_target_length == config.target_length
    )
    safely_narrowing_all_cache = (
        manifest_target_mode == "all"
        and manifest_target_length is None
        and config.target_mode in {"last", "tail"}
    )
    if not exact_target_config and not safely_narrowing_all_cache:
        raise ValueError(
            "Token cache target_mode/target_length cannot satisfy training config: "
            f"mode={manifest_target_mode}, length={manifest_target_length} != "
            f"mode={config.target_mode}, length={config.target_length}. Only an "
            "all cache can be narrowed to last or tail without re-tokenizing."
        )
    boundaries = TimeBoundaries(
        pd.Timestamp(manifest["train_end"]),
        pd.Timestamp(manifest["validation_end"]),
    )
    datasets = {
        split: TokenShardDataset(
            cache_dir,
            split,
            target_mode=config.target_mode,
            target_length=config.target_length,
        )
        for split in ("train", "validation", "test")
    }
    return datasets, boundaries
