"""Command-line entry point for Parquet-based QV LoRA training."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from lora.config import (
    LORA_TARGET_PROFILES,
    QVLoRAConfig,
    apply_dependency_override,
    apply_ffn_override,
    apply_fusion_override,
    apply_proj_s1_override,
    apply_target_profile,
)

from .config import DataConfig, ExperimentConfig, TrainingConfig
from .training import train


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune Kronos Q/V with LoRA.")
    parser.add_argument(
        "--base-model-path",
        type=Path,
        default=QVLoRAConfig().base_model_path,
        help="Local Kronos predictor directory (for example Kronos-base or Kronos-small).",
    )
    parser.add_argument(
        "--tokenizer-path",
        type=Path,
        default=QVLoRAConfig().tokenizer_path,
        help="Local Kronos tokenizer directory used for raw Parquet input.",
    )
    parser.add_argument(
        "--adapter-output-dir",
        type=Path,
        default=QVLoRAConfig().adapter_output_dir,
        help="Root directory that receives per-run checkpoints and metrics.",
    )
    parser.add_argument("--data-path", type=Path, default=DataConfig().data_path)
    parser.add_argument(
        "--token-cache-path",
        type=Path,
        default=None,
        help="Read pre-tokenized shards from this directory instead of raw Parquet.",
    )
    parser.add_argument("--train-end", default=None)
    parser.add_argument("--validation-end", default=None)
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument("--stride", type=int, default=32)
    parser.add_argument(
        "--target-mode",
        choices=("all", "last", "tail"),
        default=DataConfig().target_mode,
        help=(
            "all supervises every eligible target; last supervises only the "
            "final target; tail supervises the final --target-length targets."
        ),
    )
    parser.add_argument(
        "--target-length",
        type=int,
        default=None,
        help="Number of final target positions supervised by target-mode tail.",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument(
        "--finetune-mode",
        choices=("lora", "full"),
        default=TrainingConfig().finetune_mode,
        help=(
            "lora trains PEFT adapters; full updates every Kronos predictor "
            "parameter and saves complete model weights."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=TrainingConfig().warmup_steps,
        help=(
            "Linearly warm the learning rate over this many optimizer steps "
            "before cosine decay; use 0 to disable warmup."
        ),
    )
    parser.add_argument(
        "--s1-loss-weight",
        type=float,
        default=TrainingConfig().s1_loss_weight,
        help="Relative weight of the s1 token cross-entropy loss.",
    )
    parser.add_argument(
        "--s2-loss-weight",
        type=float,
        default=TrainingConfig().s2_loss_weight,
        help="Relative weight of the s2 token cross-entropy loss.",
    )
    parser.add_argument(
        "--training-s2-conditioning",
        choices=(
            "teacher",
            "greedy",
            "sample",
            "topk",
            "mixed-greedy",
            "mixed-sample",
            "mixed-topk",
        ),
        default=TrainingConfig().s2_conditioning,
        help=(
            "s1 condition supplied to the dependency layer during training "
            "and in-training validation/best-model selection."
        ),
    )
    parser.add_argument(
        "--s2-teacher-loss-weight",
        type=float,
        default=TrainingConfig().s2_teacher_loss_weight,
        help="Teacher branch weight for mixed-greedy/mixed-sample training.",
    )
    parser.add_argument(
        "--s2-predicted-loss-weight",
        type=float,
        default=TrainingConfig().s2_predicted_loss_weight,
        help="Predicted branch weight for mixed-greedy/mixed-sample training.",
    )
    parser.add_argument(
        "--s2-top-k",
        type=int,
        default=TrainingConfig().s2_top_k,
        help="Number of s1 candidates used by topk/mixed-topk marginalization.",
    )
    parser.add_argument(
        "--s2-temperature",
        type=float,
        default=TrainingConfig().s2_temperature,
        help="Temperature of the s1 distribution used by topk/mixed-topk.",
    )
    parser.add_argument(
        "--lora-target-profile",
        choices=LORA_TARGET_PROFILES,
        default="qv",
        help=(
            "Named reproducible LoRA module set; expanded self-attn profiles "
            "exclude the s2-only dependency cross-attention."
        ),
    )
    parser.add_argument(
        "--lora-r",
        type=int,
        default=QVLoRAConfig().rank,
        help="Global LoRA rank used by target modules without a rank override.",
    )
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=QVLoRAConfig().alpha,
        help="Global LoRA alpha; keeping alpha/r constant preserves scaling.",
    )
    parser.add_argument(
        "--proj-s1-rank",
        type=int,
        default=None,
        help=(
            "Override only head.proj_s1 LoRA rank; requires a profile that "
            "includes proj_s1."
        ),
    )
    parser.add_argument(
        "--proj-s1-alpha",
        type=int,
        default=None,
        help="Override head.proj_s1 LoRA alpha; defaults to 2 * proj-s1-rank.",
    )
    parser.add_argument(
        "--full-finetune-proj-s1",
        action="store_true",
        help=(
            "Train and save the complete head.proj_s1 layer instead of "
            "attaching LoRA to that layer; incompatible with --proj-s1-rank."
        ),
    )
    parser.add_argument(
        "--dependency-lora-r",
        type=int,
        default=None,
        help=(
            "Override only dep_layer.cross_attn Q/V LoRA rank; requires a "
            "target profile that includes dependency Q/V."
        ),
    )
    parser.add_argument(
        "--dependency-lora-alpha",
        type=int,
        default=None,
        help=(
            "Override Dependency Q/V LoRA alpha; defaults to 2 * "
            "dependency-lora-r."
        ),
    )
    parser.add_argument(
        "--fusion-lora-r",
        type=int,
        default=None,
        help="Override only embedding.fusion_proj LoRA rank.",
    )
    parser.add_argument(
        "--fusion-lora-alpha",
        type=int,
        default=None,
        help="Fusion LoRA alpha; defaults to 2 * fusion-lora-r.",
    )
    parser.add_argument(
        "--ffn-lora-r",
        type=int,
        default=None,
        help="Override selected FFN w1/w2/w3 LoRA rank.",
    )
    parser.add_argument(
        "--ffn-lora-alpha",
        type=int,
        default=None,
        help="FFN LoRA alpha; defaults to 2 * ffn-lora-r.",
    )
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--gradient-log-interval",
        type=int,
        default=TrainingConfig().gradient_log_interval,
        help=(
            "Record pre-clipping LoRA gradient norms every N optimizer steps; "
            "use 0 to disable."
        ),
    )
    parser.add_argument(
        "--gradient-conflict-interval",
        type=int,
        default=TrainingConfig().gradient_conflict_interval,
        help=(
            "Measure separate s1/s2 gradients on shared Transformer LoRA "
            "parameters every N optimizer steps; use 0 to disable."
        ),
    )
    parser.add_argument(
        "--validation-interval",
        type=int,
        default=2000,
        help=(
            "Evaluate the full validation split every N optimizer steps; "
            "use 0 to disable step-level validation."
        ),
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=2000,
        help="Save a complete resumable checkpoint every N optimizer steps.",
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=TrainingConfig().early_stopping_patience,
        help="Stop after this many epochs without a significant improvement.",
    )
    parser.add_argument(
        "--early-stopping-min-delta",
        type=float,
        default=TrainingConfig().early_stopping_min_delta,
        help=(
            "Minimum validation-loss decrease that resets early stopping; "
            "best_model still tracks every strictly lower loss."
        ),
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Unique experiment directory name; defaults to timestamp_r{rank}_s{stride}.",
    )
    parser.add_argument(
        "--model-display-name",
        default=None,
        help=(
            "Complete front-end label stored with best_model/last_model, for "
            "example 'SW2半导体 LoRA · Kronos-small'."
        ),
    )
    parser.add_argument(
        "--resume-from",
        type=Path,
        default=None,
        help=(
            "Resume from a resume/step-* directory, a run directory, or "
            "resume/latest.json. The existing run directory is reused."
        ),
    )
    parser.add_argument(
        "--initialize-from-adapter",
        type=Path,
        default=None,
        help=(
            "Initialize LoRA weights from an adapter or run directory while "
            "starting a fresh optimizer, scheduler, RNG stream, and run."
        ),
    )
    parser.add_argument(
        "--initialize-model-selection",
        choices=("best", "last"),
        default=TrainingConfig().initialize_model_selection,
        help=(
            "Adapter subdirectory selected when --initialize-from-adapter "
            "points to a run directory."
        ),
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--precision",
        choices=("auto", "bf16", "fp32"),
        default="auto",
        help="auto uses BF16 on supported CUDA devices and FP32 elsewhere.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = replace(
        DataConfig(),
        data_path=args.data_path,
        token_cache_path=args.token_cache_path,
        train_end=args.train_end,
        validation_end=args.validation_end,
        context_length=args.context_length,
        stride=args.stride,
        target_mode=args.target_mode,
        target_length=args.target_length,
    )
    training = replace(
        TrainingConfig(),
        finetune_mode=args.finetune_mode,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        s1_loss_weight=args.s1_loss_weight,
        s2_loss_weight=args.s2_loss_weight,
        s2_conditioning=args.training_s2_conditioning,
        s2_teacher_loss_weight=args.s2_teacher_loss_weight,
        s2_predicted_loss_weight=args.s2_predicted_loss_weight,
        s2_top_k=args.s2_top_k,
        s2_temperature=args.s2_temperature,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_workers=args.num_workers,
        gradient_log_interval=args.gradient_log_interval,
        gradient_conflict_interval=args.gradient_conflict_interval,
        validation_interval=args.validation_interval,
        checkpoint_interval=args.checkpoint_interval,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_min_delta=args.early_stopping_min_delta,
        run_name=args.run_name,
        model_display_name=args.model_display_name,
        resume_from=args.resume_from,
        initialize_from_adapter=args.initialize_from_adapter,
        initialize_model_selection=args.initialize_model_selection,
        device=args.device,
        precision=args.precision,
    )
    lora = replace(
        QVLoRAConfig(),
        base_model_path=args.base_model_path,
        tokenizer_path=args.tokenizer_path,
        adapter_output_dir=args.adapter_output_dir,
        rank=args.lora_r,
        alpha=args.lora_alpha,
        full_finetune_proj_s1=args.full_finetune_proj_s1,
    )
    lora = apply_target_profile(lora, args.lora_target_profile)
    lora = apply_proj_s1_override(
        lora,
        args.proj_s1_rank,
        args.proj_s1_alpha,
    )
    lora = apply_dependency_override(
        lora,
        args.dependency_lora_r,
        args.dependency_lora_alpha,
    )
    lora = apply_fusion_override(
        lora,
        args.fusion_lora_r,
        args.fusion_lora_alpha,
    )
    lora = apply_ffn_override(
        lora,
        args.ffn_lora_r,
        args.ffn_lora_alpha,
    )
    train(ExperimentConfig(data=data, training=training, lora=lora))


if __name__ == "__main__":
    main()
