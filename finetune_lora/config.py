"""Configuration for Parquet-based Kronos LoRA fine-tuning."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from lora.config import PROJECT_ROOT, QVLoRAConfig


@dataclass(frozen=True)
class DataConfig:
    """Parquet schema, chronological split, and window settings."""

    data_path: Path = PROJECT_ROOT / "data" / "lora"
    # When set, training reads pre-tokenized shards instead of raw Parquet.
    token_cache_path: Path | None = None
    symbol_column: str = "symbol"
    timestamp_column: str = "timestamp"
    feature_columns: tuple[str, ...] = (
        "open",
        "high",
        "low",
        "close",
        "volume",
        "amount",
    )
    time_columns: tuple[str, ...] = (
        "minute",
        "hour",
        "weekday",
        "day",
        "month",
    )

    # Each sample has context_length + 1 rows. After shifting, the model sees
    # context_length inputs and predicts context_length next-token targets.
    context_length: int = 512
    stride: int = 24
    # all preserves next-token supervision at every split-eligible position;
    # last supervises only the final target; tail supervises the final
    # target_length positions while preserving the full input sequence.
    target_mode: str = "all"
    target_length: int | None = None
    clip: float = 5.0

    # Explicit boundaries are recommended. When both are omitted, global
    # timestamp quantiles are used only as a development fallback.
    train_end: str | None = None
    validation_end: str | None = None
    train_ratio: float = 0.8
    validation_ratio: float = 0.1

    def validate(self) -> None:
        if self.context_length <= 0:
            raise ValueError("context_length must be positive.")
        if self.context_length > 512:
            raise ValueError("Kronos context_length cannot exceed 512.")
        if self.stride <= 0:
            raise ValueError("stride must be positive.")
        if self.target_mode not in {"all", "last", "tail"}:
            raise ValueError("target_mode must be one of: all, last, tail.")
        if self.target_mode == "tail":
            if self.target_length is None:
                raise ValueError("target_length is required when target_mode=tail.")
            if not 1 <= self.target_length <= self.context_length:
                raise ValueError(
                    "target_length must be in [1, context_length] when "
                    "target_mode=tail."
                )
        elif self.target_length is not None:
            raise ValueError("target_length is only valid when target_mode=tail.")
        if self.clip <= 0:
            raise ValueError("clip must be positive.")
        if len(self.feature_columns) != 6:
            raise ValueError("Kronos-Tokenizer-base expects exactly 6 features.")
        if (self.train_end is None) != (self.validation_end is None):
            raise ValueError(
                "train_end and validation_end must be provided together."
            )
        if self.train_end is None:
            if not 0.0 < self.train_ratio < 1.0:
                raise ValueError("train_ratio must be in (0, 1).")
            if not 0.0 < self.validation_ratio < 1.0:
                raise ValueError("validation_ratio must be in (0, 1).")
            if self.train_ratio + self.validation_ratio >= 1.0:
                raise ValueError(
                    "train_ratio + validation_ratio must be less than 1."
                )


@dataclass(frozen=True)
class TrainingConfig:
    """Single-device training defaults for the first real-data baseline."""

    # lora freezes the backbone and trains PEFT adapters; full updates every
    # predictor parameter and saves a complete Kronos model.
    finetune_mode: str = "lora"
    epochs: int = 10
    batch_size: int = 8
    learning_rate: float = 1e-4
    # Number of optimizer updates used for linear LR warmup. Zero preserves
    # the original cosine schedule without warmup.
    warmup_steps: int = 0
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    num_workers: int = 0
    log_interval: int = 50
    # Record pre-clipping gradient norms every N optimizer steps. Zero disables.
    gradient_log_interval: int = 50
    # Replay one accumulation group and compare the separate s1/s2 gradients on
    # shared Transformer LoRA parameters. Zero disables this diagnostic.
    gradient_conflict_interval: int = 0
    # Evaluate the full validation split every N optimizer steps. Zero disables
    # step-level validation; epoch-end validation always remains enabled.
    validation_interval: int = 2000
    checkpoint_interval: int = 2000
    early_stopping_patience: int = 3
    # The two losses are normalized by their weight sum, so 0.5/0.5 and 1/1
    # are equivalent. Defaults preserve the original equal-weight objective.
    s1_loss_weight: float = 0.5
    s2_loss_weight: float = 0.5
    # The s1 ids supplied to the dependency layer during training and the
    # in-training validation used for best-model selection.
    s2_conditioning: str = "teacher"
    # Used only by mixed-greedy/mixed-sample/mixed-topk. The two branch weights are
    # normalized by their sum before they enter the overall s2 objective.
    s2_teacher_loss_weight: float = 0.1
    s2_predicted_loss_weight: float = 0.9
    # topk/mixed-topk marginalize conditional s2 probabilities over these s1
    # candidates. The s1 mixture distribution is detached from the s2 loss.
    s2_top_k: int = 5
    s2_temperature: float = 1.0
    # Only improvements at least this large reset epoch-level early stopping.
    # best_model still tracks every strictly lower validation loss.
    early_stopping_min_delta: float = 0.0
    seed: int = 100
    run_name: str | None = None
    # Complete user-facing label written into each deployable adapter bundle.
    # When omitted, the run name and backbone directory name are combined.
    model_display_name: str | None = None
    # A step directory, a run directory, or its resume/latest.json file.
    resume_from: Path | None = None
    # A PEFT adapter directory or run directory used only to initialize LoRA
    # weights. Optimizer, scheduler, counters, and RNG state start fresh.
    initialize_from_adapter: Path | None = None
    initialize_model_selection: str = "best"
    device: str = "auto"
    # auto: BF16 on supported CUDA devices, otherwise FP32.
    # bf16: require BF16-capable CUDA; fp32: disable mixed precision.
    precision: str = "auto"

    def validate(self) -> None:
        if self.finetune_mode not in {"lora", "full"}:
            raise ValueError("finetune_mode must be one of: lora, full.")
        if self.epochs <= 0 or self.batch_size <= 0:
            raise ValueError("epochs and batch_size must be positive.")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")
        if self.warmup_steps < 0:
            raise ValueError("warmup_steps cannot be negative.")
        if self.gradient_accumulation_steps <= 0:
            raise ValueError("gradient_accumulation_steps must be positive.")
        if self.max_grad_norm <= 0:
            raise ValueError("max_grad_norm must be positive.")
        if self.num_workers < 0:
            raise ValueError("num_workers cannot be negative.")
        if self.gradient_log_interval < 0:
            raise ValueError("gradient_log_interval cannot be negative.")
        if self.gradient_conflict_interval < 0:
            raise ValueError("gradient_conflict_interval cannot be negative.")
        if self.validation_interval < 0:
            raise ValueError("validation_interval cannot be negative.")
        if self.checkpoint_interval <= 0:
            raise ValueError("checkpoint_interval must be positive.")
        if self.early_stopping_patience < 0:
            raise ValueError("early_stopping_patience cannot be negative.")
        if not math.isfinite(self.s1_loss_weight) or not math.isfinite(
            self.s2_loss_weight
        ):
            raise ValueError("s1_loss_weight and s2_loss_weight must be finite.")
        if self.s1_loss_weight < 0 or self.s2_loss_weight < 0:
            raise ValueError("s1_loss_weight and s2_loss_weight cannot be negative.")
        if self.s1_loss_weight + self.s2_loss_weight <= 0:
            raise ValueError("At least one loss weight must be positive.")
        if self.s2_conditioning not in {
            "teacher",
            "greedy",
            "sample",
            "topk",
            "mixed-greedy",
            "mixed-sample",
            "mixed-topk",
        }:
            raise ValueError(
                "s2_conditioning must be one of: teacher, greedy, sample, "
                "topk, mixed-greedy, mixed-sample, mixed-topk."
            )
        if self.s2_top_k <= 0:
            raise ValueError("s2_top_k must be positive.")
        if not math.isfinite(self.s2_temperature) or self.s2_temperature <= 0:
            raise ValueError("s2_temperature must be finite and positive.")
        branch_weights = (
            self.s2_teacher_loss_weight,
            self.s2_predicted_loss_weight,
        )
        if not all(math.isfinite(weight) for weight in branch_weights):
            raise ValueError("Mixed s2 loss weights must be finite.")
        if any(weight < 0 for weight in branch_weights):
            raise ValueError("Mixed s2 loss weights cannot be negative.")
        if self.s2_conditioning.startswith("mixed-") and any(
            weight <= 0 for weight in branch_weights
        ):
            raise ValueError(
                "Mixed s2 loss weights must both be positive; use a pure "
                "conditioning mode to disable one branch."
            )
        if not math.isfinite(self.early_stopping_min_delta):
            raise ValueError("early_stopping_min_delta must be finite.")
        if self.early_stopping_min_delta < 0:
            raise ValueError("early_stopping_min_delta cannot be negative.")
        if self.precision not in {"auto", "bf16", "fp32"}:
            raise ValueError("precision must be one of: auto, bf16, fp32.")
        if self.resume_from is not None and self.initialize_from_adapter is not None:
            raise ValueError(
                "resume_from and initialize_from_adapter are mutually exclusive."
            )
        if self.finetune_mode == "full" and self.initialize_from_adapter is not None:
            raise ValueError(
                "initialize_from_adapter is only supported in finetune_mode=lora."
            )
        if self.initialize_model_selection not in {"best", "last"}:
            raise ValueError(
                "initialize_model_selection must be one of: best, last."
            )
        if self.model_display_name is not None and not self.model_display_name.strip():
            raise ValueError("model_display_name cannot be empty.")


@dataclass(frozen=True)
class ExperimentConfig:
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    lora: QVLoRAConfig = field(default_factory=QVLoRAConfig)

    def validate(self) -> None:
        self.data.validate()
        self.training.validate()
        self.lora.validate_local_models()

    def to_dict(self) -> dict[str, Any]:
        def convert(value: Any) -> Any:
            if isinstance(value, Path):
                return str(value)
            if isinstance(value, tuple):
                return list(value)
            if isinstance(value, dict):
                return {key: convert(item) for key, item in value.items()}
            if isinstance(value, list):
                return [convert(item) for item in value]
            return value

        return convert(asdict(self))
