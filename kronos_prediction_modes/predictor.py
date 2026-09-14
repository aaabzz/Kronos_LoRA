"""Prediction modes kept separate from the upstream Kronos implementation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal

import numpy as np
import pandas as pd
import torch

from lora import QVLoRAConfig, load_frozen_tokenizer
from lora.model_factory import load_qv_lora_adapter
from model.kronos import sample_from_logits


FEATURE_COLUMNS = (
    "open",
    "high",
    "low",
    "close",
    "volume",
    "amount",
)
PREDICTION_MODE_KEYS = ("greedy", "median", "mean")


@dataclass(frozen=True)
class PredictionResult:
    """Greedy path, Monte Carlo summaries, and next-bar probabilities."""

    greedy: pd.DataFrame | None
    sample_mean: pd.DataFrame | None
    sample_median: pd.DataFrame | None
    up_probability: float
    down_probability: float
    flat_probability: float
    sampled_paths: np.ndarray

    @property
    def probability_summary(self) -> dict[str, float | int]:
        return {
            "sample_count": int(self.sampled_paths.shape[0]),
            "up_probability": self.up_probability,
            "down_probability": self.down_probability,
            "flat_probability": self.flat_probability,
        }


def _time_features(timestamps) -> np.ndarray:
    index = pd.DatetimeIndex(timestamps)
    if index.hasnans:
        raise ValueError("Timestamps cannot contain NaT.")
    return np.column_stack(
        [index.minute, index.hour, index.weekday, index.day, index.month]
    ).astype(np.float32)


def _prepare_history(
    frame: pd.DataFrame,
    x_timestamps,
    y_timestamps,
    *,
    clip: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    frame = frame.copy()
    x_index = pd.DatetimeIndex(x_timestamps)
    y_index = pd.DatetimeIndex(y_timestamps)
    if not x_index.is_monotonic_increasing:
        raise ValueError("x_timestamps must be sorted in ascending order.")
    if not y_index.is_monotonic_increasing:
        raise ValueError("y_timestamps must be sorted in ascending order.")
    if len(x_index) and len(y_index) and y_index[0] <= x_index[-1]:
        raise ValueError("Every future timestamp must be after the history.")
    if "symbol" in frame.columns and frame["symbol"].astype(str).nunique() != 1:
        raise ValueError("One prediction input cannot contain multiple symbols.")
    missing_prices = [
        column
        for column in ("open", "high", "low", "close")
        if column not in frame.columns
    ]
    if missing_prices:
        raise ValueError(f"Missing price columns: {missing_prices}")
    if frame.empty:
        raise ValueError("Historical frame cannot be empty.")
    if "volume" not in frame.columns:
        frame["volume"] = 0.0
    if "amount" not in frame.columns:
        frame["amount"] = frame["volume"] * frame[
            ["open", "high", "low", "close"]
        ].mean(axis=1)
    if len(frame) != len(x_index):
        raise ValueError("Historical rows and x_timestamps must have equal lengths.")
    if len(y_timestamps) == 0:
        raise ValueError("At least one future timestamp is required.")
    if frame[list(FEATURE_COLUMNS)].isnull().any().any():
        raise ValueError("Historical features contain null values.")

    features = frame[list(FEATURE_COLUMNS)].to_numpy(dtype=np.float32)
    if not np.isfinite(features).all():
        raise ValueError("Historical features contain NaN or infinite values.")
    mean = features.mean(axis=0)
    std = features.std(axis=0)
    normalized = np.clip((features - mean) / (std + 1e-5), -clip, clip)
    return (
        normalized,
        _time_features(x_index),
        _time_features(y_index),
        mean,
        std,
    )


def load_lora_runtime(
    adapter_path: Path,
    *,
    device: str | torch.device = "cuda:0",
    config: QVLoRAConfig | None = None,
):
    """Load a frozen tokenizer and an inference-only LoRA-enabled Kronos model."""
    resolved_config = config or QVLoRAConfig()
    resolved_device = torch.device(device)
    tokenizer = load_frozen_tokenizer(resolved_config).to(resolved_device).eval()
    adapter_model = load_qv_lora_adapter(
        resolved_config,
        adapter_path=adapter_path,
        is_trainable=False,
    ).to(resolved_device).eval()

    # PEFT injects LoRA layers into this underlying Kronos module. Calling its
    # custom decode_s1/decode_s2 methods therefore still uses the active Adapter.
    model = adapter_model.get_base_model()
    model.eval()
    return tokenizer, model


def _next_token(
    logits: torch.Tensor,
    mode: Literal["greedy", "sample"],
    *,
    temperature: float,
    top_k: int,
    top_p: float,
) -> torch.Tensor:
    if mode == "greedy":
        return torch.argmax(logits, dim=-1, keepdim=True)
    return sample_from_logits(
        logits.clone(),
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        sample_logits=True,
    )


@torch.inference_mode()
def _generate_paths(
    tokenizer,
    model,
    normalized_history: np.ndarray,
    x_stamps: np.ndarray,
    y_stamps: np.ndarray,
    *,
    mode: Literal["greedy", "sample"],
    path_count: int,
    device: torch.device,
    max_context: int,
    temperature: float,
    top_k: int,
    top_p: float,
) -> np.ndarray:
    if path_count <= 0:
        raise ValueError("path_count must be positive.")
    pred_len = len(y_stamps)
    if pred_len > max_context:
        raise ValueError("Prediction length cannot exceed max_context.")

    history = torch.from_numpy(normalized_history).unsqueeze(0).to(device)
    stamp_history = torch.from_numpy(x_stamps).unsqueeze(0).to(device)
    future_stamps = torch.from_numpy(y_stamps).unsqueeze(0).to(device)
    s1_history, s2_history = tokenizer.encode(history, half=True)

    s1_history = s1_history.repeat(path_count, 1)
    s2_history = s2_history.repeat(path_count, 1)
    stamp_history = stamp_history.repeat(path_count, 1, 1)
    future_stamps = future_stamps.repeat(path_count, 1, 1)

    for step in range(pred_len):
        context_length = min(s1_history.shape[1], max_context)
        context_s1 = s1_history[:, -context_length:]
        context_s2 = s2_history[:, -context_length:]
        context_stamp = stamp_history[:, -context_length:, :]

        s1_logits, context = model.decode_s1(
            context_s1,
            context_s2,
            context_stamp,
        )
        next_s1 = _next_token(
            s1_logits[:, -1, :],
            mode,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )

        s2_logits = model.decode_s2(context, next_s1)
        next_s2 = _next_token(
            s2_logits[:, -1, :],
            mode,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )

        s1_history = torch.cat([s1_history, next_s1], dim=1)
        s2_history = torch.cat([s2_history, next_s2], dim=1)
        stamp_history = torch.cat(
            [stamp_history, future_stamps[:, step : step + 1, :]],
            dim=1,
        )

    decode_length = min(s1_history.shape[1], max_context)
    decoded = tokenizer.decode(
        [
            s1_history[:, -decode_length:],
            s2_history[:, -decode_length:],
        ],
        half=True,
    )
    return decoded[:, -pred_len:, :].float().cpu().numpy()


def _as_frame(values: np.ndarray, timestamps) -> pd.DataFrame:
    return pd.DataFrame(
        values,
        columns=FEATURE_COLUMNS,
        index=pd.DatetimeIndex(timestamps),
    )


def next_bar_probabilities(
    sampled_paths: np.ndarray,
    reference_close: float,
    *,
    flat_tolerance: float = 0.0,
) -> dict[str, float | int]:
    """Calculate probabilities from only the first bar of every sampled path."""
    if sampled_paths.ndim != 3 or sampled_paths.shape[1] < 1:
        raise ValueError("sampled_paths must have shape [samples, horizon, features].")
    if sampled_paths.shape[2] != len(FEATURE_COLUMNS):
        raise ValueError(f"Expected {len(FEATURE_COLUMNS)} features per K-line.")
    if sampled_paths.shape[0] == 0:
        raise ValueError("At least one sampled path is required.")
    if flat_tolerance < 0:
        raise ValueError("flat_tolerance cannot be negative.")

    sample_count = int(sampled_paths.shape[0])
    close_index = FEATURE_COLUMNS.index("close")
    next_closes = sampled_paths[:, 0, close_index]
    up_count = int(np.count_nonzero(next_closes > reference_close + flat_tolerance))
    down_count = int(
        np.count_nonzero(next_closes < reference_close - flat_tolerance)
    )
    flat_count = sample_count - up_count - down_count
    return {
        "sample_count": sample_count,
        "up_probability": up_count / sample_count,
        "down_probability": down_count / sample_count,
        "flat_probability": flat_count / sample_count,
    }


def predict_modes(
    frame: pd.DataFrame,
    x_timestamps,
    y_timestamps,
    *,
    adapter_path: Path | None = None,
    device: str | torch.device = "cuda:0",
    sample_count: int = 32,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 0.9,
    seed: int = 100,
    max_context: int = 512,
    clip: float = 5.0,
    flat_tolerance: float = 0.0,
    config: QVLoRAConfig | None = None,
    tokenizer=None,
    model=None,
    modes: Iterable[str] | None = None,
) -> PredictionResult:
    """Run only the requested prediction branches; all three are enabled by default.

    Up/down probabilities always describe only the first future bar, comparing
    its sampled close with the final observed close. Later predicted bars do
    not participate in the probability calculation.
    """
    if sample_count <= 0:
        raise ValueError("sample_count must be positive.")
    if temperature <= 0:
        raise ValueError("temperature must be positive.")
    if top_k < 0:
        raise ValueError("top_k cannot be negative.")
    if not 0.0 < top_p <= 1.0:
        raise ValueError("top_p must be in (0, 1].")
    if flat_tolerance < 0:
        raise ValueError("flat_tolerance cannot be negative.")
    selected_modes = (
        PREDICTION_MODE_KEYS
        if modes is None
        else tuple(dict.fromkeys(modes))
    )
    invalid_modes = [mode for mode in selected_modes if mode not in PREDICTION_MODE_KEYS]
    if invalid_modes:
        raise ValueError(f"Unsupported prediction modes: {', '.join(invalid_modes)}")
    if not selected_modes:
        raise ValueError("At least one prediction mode must be selected.")

    resolved_device = torch.device(device)
    torch.manual_seed(seed)
    if resolved_device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    y_index = pd.DatetimeIndex(y_timestamps)
    normalized, x_stamps, y_stamps, mean, std = _prepare_history(
        frame,
        x_timestamps,
        y_index,
        clip=clip,
    )
    if tokenizer is None or model is None:
        if adapter_path is None:
            raise ValueError("adapter_path is required when no runtime is supplied.")
        tokenizer, model = load_lora_runtime(
            adapter_path,
            device=resolved_device,
            config=config,
        )

    greedy_path = None
    if "greedy" in selected_modes:
        # Greedy inference needs exactly one path: repeating it would waste compute.
        greedy_normalized = _generate_paths(
            tokenizer, model, normalized, x_stamps, y_stamps,
            mode="greedy", path_count=1, device=resolved_device,
            max_context=max_context, temperature=1.0, top_k=0, top_p=1.0,
        )
        greedy_path = greedy_normalized[0] * (std + 1e-5) + mean

    needs_samples = bool({"median", "mean"} & set(selected_modes))
    sampled_paths = np.empty((0, len(y_index), len(FEATURE_COLUMNS)), dtype=np.float32)
    sample_mean = None
    sample_median = None
    if needs_samples:
        sampled_normalized = _generate_paths(
            tokenizer, model, normalized, x_stamps, y_stamps,
            mode="sample", path_count=sample_count, device=resolved_device,
            max_context=max_context, temperature=temperature,
            top_k=top_k, top_p=top_p,
        )
        sampled_paths = sampled_normalized * (std + 1e-5) + mean
        if "mean" in selected_modes:
            sample_mean = sampled_paths.mean(axis=0)
        if "median" in selected_modes:
            sample_median = np.median(sampled_paths, axis=0)

    reference_close = float(frame["close"].iloc[-1])
    probability_paths = sampled_paths
    if not len(probability_paths):
        probability_paths = np.expand_dims(greedy_path, axis=0)
    probabilities = next_bar_probabilities(
        probability_paths, reference_close, flat_tolerance=flat_tolerance,
    )

    return PredictionResult(
        greedy=_as_frame(greedy_path, y_index) if greedy_path is not None else None,
        sample_mean=_as_frame(sample_mean, y_index) if sample_mean is not None else None,
        sample_median=_as_frame(sample_median, y_index) if sample_median is not None else None,
        up_probability=float(probabilities["up_probability"]),
        down_probability=float(probabilities["down_probability"]),
        flat_probability=float(probabilities["flat_probability"]),
        sampled_paths=sampled_paths,
    )
