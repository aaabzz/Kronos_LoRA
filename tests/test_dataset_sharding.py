from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from finetune_lora.config import DataConfig
from finetune_lora.dataset import KlineWindowDataset, TimeBoundaries
from finetune_lora.sharding import (
    DistributedSequentialSampler,
    DistributedShardSampler,
    ShardShuffleSampler,
    TokenShardDataset,
    bucketize_parquet,
    build_token_datasets,
    build_token_shards,
    stable_bucket_id,
)


class DummyTokenizer(torch.nn.Module):
    def encode(self, features: torch.Tensor, half: bool = False):
        assert half
        length = features.shape[1]
        base = torch.arange(length, device=features.device).unsqueeze(0)
        base = base.expand(features.shape[0], -1)
        return base + 10, base + 100


def make_symbol_frame(symbol: str, start: float) -> pd.DataFrame:
    rows = 30
    timestamp = pd.date_range("2026-01-01 09:35:00", periods=rows, freq="5min")
    value = np.arange(rows, dtype=np.float32) + start
    return pd.DataFrame(
        {
            "symbol": symbol,
            "timestamp": timestamp,
            "open": value,
            "high": value + 0.2,
            "low": value - 0.2,
            "close": value + 0.1,
            "volume": value + 1_000,
            "amount": value + 10_000,
        }
    )


def write_source_fragments(source: Path) -> pd.DataFrame:
    full = pd.concat(
        [make_symbol_frame("000001.SZ", 10), make_symbol_frame("600000.SH", 20)],
        ignore_index=True,
    )
    source.mkdir()
    # Split both symbols across files to verify stable cross-file bucketing.
    first = full.groupby("symbol", group_keys=False).head(15)
    second = full.groupby("symbol", group_keys=False).tail(15)
    first.to_parquet(source / "part-0.parquet", index=False)
    second.to_parquet(source / "part-1.parquet", index=False)
    return full


def test_bucketize_and_lazy_token_shards(tmp_path: Path) -> None:
    source = tmp_path / "source"
    full = write_source_fragments(source)
    bucket_dir = tmp_path / "buckets"
    cache_dir = tmp_path / "cache"
    config = replace(
        DataConfig(),
        data_path=source,
        context_length=4,
        stride=2,
        train_end=str(full["timestamp"].sort_values().drop_duplicates().iloc[14]),
        validation_end=str(
            full["timestamp"].sort_values().drop_duplicates().iloc[22]
        ),
    )

    metadata = bucketize_parquet(config, bucket_dir, bucket_count=4)
    assert metadata["rows"] == len(full)
    for symbol in full["symbol"].unique():
        expected = bucket_dir / f"bucket={stable_bucket_id(symbol, 4):03d}"
        rows = sum(
            int((pd.read_parquet(path)["symbol"] == symbol).sum())
            for path in expected.glob("fragment-*.parquet")
        )
        assert rows == 30

    manifest = build_token_shards(
        config,
        bucket_dir,
        cache_dir,
        DummyTokenizer(),
        device=torch.device("cpu"),
        encode_batch_size=3,
        samples_per_shard=4,
        tokenizer_fingerprint="dummy",
    )
    assert manifest["tokenizer_fingerprint"] == "dummy"
    assert manifest["target_mode"] == "all"
    assert all(manifest["counts"][split] > 0 for split in ("train", "validation", "test"))

    datasets, boundaries = build_token_datasets(cache_dir, config)
    expected_boundaries = TimeBoundaries(
        pd.Timestamp(config.train_end), pd.Timestamp(config.validation_end)
    )
    for split in ("train", "validation", "test"):
        online = KlineWindowDataset(full, config, split, expected_boundaries)
        assert len(datasets[split]) == len(online)
    assert boundaries == expected_boundaries

    with pytest.raises(ValueError, match="stride"):
        build_token_datasets(cache_dir, replace(config, stride=1))

    tail_config = replace(config, target_mode="tail", target_length=3)
    tail_datasets, _ = build_token_datasets(cache_dir, tail_config)
    for split in ("train", "validation", "test"):
        masks = [
            tail_datasets[split][index][-1]
            for index in range(len(tail_datasets[split]))
        ]
        assert all(not mask[:-3].any() for mask in masks)
        assert any(mask.sum().item() == 3 for mask in masks)

    train = datasets["train"]
    assert isinstance(train, TokenShardDataset)
    s1_input, s2_input, s1_target, s2_target, stamps, mask = train[0]
    assert s1_input.dtype == torch.long
    assert s2_input.dtype == torch.long
    assert stamps.dtype == torch.float32
    assert mask.dtype == torch.bool
    assert s1_input.shape == s1_target.shape == (config.context_length,)
    assert s2_input.shape == s2_target.shape == (config.context_length,)
    assert stamps.shape == (config.context_length, 5)
    torch.testing.assert_close(s1_target[:-1], s1_input[1:])
    torch.testing.assert_close(s2_target[:-1], s2_input[1:])

    order = list(ShardShuffleSampler(train, seed=7))
    assert sorted(order) == list(range(len(train)))

    rank_orders = [
        list(
            DistributedShardSampler(
                train,
                num_replicas=2,
                rank=rank,
                seed=7,
            )
        )
        for rank in range(2)
    ]
    assert len(rank_orders[0]) == len(rank_orders[1])
    assert set(range(len(train))).issubset(set(rank_orders[0] + rank_orders[1]))
    assert len(rank_orders[0] + rank_orders[1]) - len(train) <= 1

    eval_orders = [
        list(
            DistributedSequentialSampler(
                train,
                num_replicas=2,
                rank=rank,
            )
        )
        for rank in range(2)
    ]
    assert set(eval_orders[0]).isdisjoint(eval_orders[1])
    assert eval_orders[0] + eval_orders[1] == list(range(len(train)))


def test_single_symbol_bucket_and_token_shard(tmp_path: Path) -> None:
    source = tmp_path / "source"
    full = write_source_fragments(source)
    bucket_dir = tmp_path / "single-buckets"
    cache_dir = tmp_path / "single-cache"
    config = replace(
        DataConfig(),
        data_path=source,
        context_length=4,
        stride=2,
        train_end=str(full["timestamp"].sort_values().drop_duplicates().iloc[14]),
        validation_end=str(
            full["timestamp"].sort_values().drop_duplicates().iloc[22]
        ),
    )

    bucket_manifest = bucketize_parquet(
        config,
        bucket_dir,
        bucket_count=1,
        symbol="600000.sh",
    )
    assert bucket_manifest["symbols"] == ["600000.SH"]
    assert bucket_manifest["symbol_count"] == 1
    assert bucket_manifest["rows"] == 30

    token_manifest = build_token_shards(
        config,
        bucket_dir,
        cache_dir,
        DummyTokenizer(),
        device=torch.device("cpu"),
        encode_batch_size=3,
        samples_per_shard=4,
        tokenizer_fingerprint="dummy",
        symbol="600000.SH",
    )
    assert token_manifest["symbols"] == ["600000.SH"]
    assert token_manifest["symbol_count"] == 1
    assert token_manifest["requested_symbol"] == "600000.SH"
    assert all(
        token_manifest["counts"][split] > 0
        for split in token_manifest["counts"]
    )


def test_last_target_mode_round_trips_through_token_shards(tmp_path: Path) -> None:
    source = tmp_path / "source"
    full = write_source_fragments(source)
    bucket_dir = tmp_path / "buckets"
    cache_dir = tmp_path / "cache"
    config = replace(
        DataConfig(),
        data_path=source,
        context_length=4,
        stride=2,
        target_mode="last",
        train_end=str(full["timestamp"].sort_values().drop_duplicates().iloc[14]),
        validation_end=str(
            full["timestamp"].sort_values().drop_duplicates().iloc[22]
        ),
    )

    bucketize_parquet(config, bucket_dir, bucket_count=2)
    manifest = build_token_shards(
        config,
        bucket_dir,
        cache_dir,
        DummyTokenizer(),
        device=torch.device("cpu"),
        encode_batch_size=3,
        samples_per_shard=4,
        tokenizer_fingerprint="dummy",
    )
    datasets, _ = build_token_datasets(cache_dir, config)

    assert manifest["target_mode"] == "last"
    for split in ("train", "validation", "test"):
        mask = datasets[split][0][-1]
        assert mask.sum().item() == 1
        assert mask[-1].item() is True

    incompatible = replace(config, target_mode="all")
    with pytest.raises(ValueError, match="target_mode"):
        build_token_datasets(cache_dir, incompatible)


def test_tail_target_mode_round_trips_through_token_shards(tmp_path: Path) -> None:
    source = tmp_path / "source"
    full = write_source_fragments(source)
    bucket_dir = tmp_path / "buckets"
    cache_dir = tmp_path / "cache"
    config = replace(
        DataConfig(),
        data_path=source,
        context_length=4,
        stride=2,
        target_mode="tail",
        target_length=3,
        train_end=str(full["timestamp"].sort_values().drop_duplicates().iloc[14]),
        validation_end=str(
            full["timestamp"].sort_values().drop_duplicates().iloc[22]
        ),
    )

    bucketize_parquet(config, bucket_dir, bucket_count=2)
    manifest = build_token_shards(
        config,
        bucket_dir,
        cache_dir,
        DummyTokenizer(),
        device=torch.device("cpu"),
        encode_batch_size=3,
        samples_per_shard=4,
        tokenizer_fingerprint="dummy",
    )
    datasets, _ = build_token_datasets(cache_dir, config)

    assert manifest["target_mode"] == "tail"
    assert manifest["target_length"] == 3
    for split in ("train", "validation", "test"):
        masks = [datasets[split][index][-1] for index in range(len(datasets[split]))]
        assert all(not mask[:-3].any() for mask in masks)
        assert any(mask.sum().item() == 3 for mask in masks)

    incompatible = replace(config, target_length=2)
    with pytest.raises(ValueError, match="target_length"):
        build_token_datasets(cache_dir, incompatible)


def test_stable_bucket_id_validation() -> None:
    assert stable_bucket_id("000001.SZ", 64) == stable_bucket_id("000001.SZ", 64)
    try:
        stable_bucket_id("000001.SZ", 0)
    except ValueError as exc:
        assert "positive" in str(exc)
    else:
        raise AssertionError("bucket_count=0 should fail")


def test_distributed_shard_sampler_pads_tiny_dataset_equally() -> None:
    class TinyDataset:
        shard_ranges = [(0, 1)]

        def __len__(self) -> int:
            return 1

    dataset = TinyDataset()
    orders = [
        list(
            DistributedShardSampler(
                dataset,  # type: ignore[arg-type]
                num_replicas=4,
                rank=rank,
            )
        )
        for rank in range(4)
    ]
    assert orders == [[0], [0], [0], [0]]
