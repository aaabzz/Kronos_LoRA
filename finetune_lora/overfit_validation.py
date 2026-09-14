"""Deliberately overfit a small validation subset for training diagnostics."""

from __future__ import annotations

import argparse
import os
from dataclasses import replace
from pathlib import Path

import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Subset

from lora.config import (
    LORA_TARGET_PROFILES,
    QVLoRAConfig,
    apply_dependency_override,
    apply_ffn_override,
    apply_fusion_override,
    apply_proj_s1_override,
    apply_target_profile,
)
from lora.model_factory import build_qv_lora_model, count_parameters

from .config import DataConfig, ExperimentConfig, TrainingConfig
from .dataset import build_datasets
from .training import (
    DistributedContext,
    evaluate,
    forward_batch,
    lora_gradient_norms,
    resolve_precision,
    select_device,
    set_seed,
    task_gradient_conflict_metrics,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train repeatedly on a small validation subset. This is a diagnostic "
            "data-leakage experiment, not a model-selection workflow."
        )
    )
    parser.add_argument(
        "--base-model-path",
        type=Path,
        default=QVLoRAConfig().base_model_path,
    )
    parser.add_argument(
        "--tokenizer-path",
        type=Path,
        default=QVLoRAConfig().tokenizer_path,
    )
    parser.add_argument("--token-cache-path", type=Path, required=True)
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument("--stride", type=int, default=128)
    parser.add_argument(
        "--target-mode",
        choices=("all", "last", "tail"),
        default=DataConfig().target_mode,
        help="Must match the target mode stored in the token cache.",
    )
    parser.add_argument(
        "--target-length",
        type=int,
        default=None,
        help="Must match target_length for a target-mode tail token cache.",
    )
    parser.add_argument("--sample-count", type=int, default=512)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument(
        "--gradient-conflict-interval",
        type=int,
        default=0,
        help=(
            "Measure separate s1/s2 gradients on shared Transformer LoRA "
            "parameters every N optimizer steps; use 0 to disable."
        ),
    )
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--s1-loss-weight", type=float, default=0.5)
    parser.add_argument("--s2-loss-weight", type=float, default=0.5)
    parser.add_argument(
        "--lora-target-profile",
        choices=LORA_TARGET_PROFILES,
        default="qv-s1",
    )
    parser.add_argument("--proj-s1-rank", type=int, default=32)
    parser.add_argument("--proj-s1-alpha", type=int, default=64)
    parser.add_argument(
        "--full-finetune-proj-s1",
        action="store_true",
        help="Train/save the complete head.proj_s1 instead of using LoRA on it.",
    )
    parser.add_argument("--dependency-lora-r", type=int, default=None)
    parser.add_argument("--dependency-lora-alpha", type=int, default=None)
    parser.add_argument("--fusion-lora-r", type=int, default=None)
    parser.add_argument("--fusion-lora-alpha", type=int, default=None)
    parser.add_argument("--ffn-lora-r", type=int, default=None)
    parser.add_argument("--ffn-lora-alpha", type=int, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--precision",
        choices=("auto", "bf16", "fp32"),
        default="bf16",
    )
    parser.add_argument("--seed", type=int, default=100)
    return parser.parse_args()


def validate_diagnostic_args(args: argparse.Namespace) -> None:
    if args.sample_count <= 0:
        raise ValueError("sample_count must be positive.")
    if args.start_index < 0:
        raise ValueError("start_index cannot be negative.")
    if args.epochs <= 0 or args.batch_size <= 0:
        raise ValueError("epochs and batch_size must be positive.")
    if args.learning_rate <= 0:
        raise ValueError("learning_rate must be positive.")
    if args.gradient_accumulation_steps <= 0:
        raise ValueError("gradient_accumulation_steps must be positive.")
    if args.max_grad_norm <= 0:
        raise ValueError("max_grad_norm must be positive.")
    if args.gradient_conflict_interval < 0:
        raise ValueError("gradient_conflict_interval cannot be negative.")
    if args.num_workers < 0:
        raise ValueError("num_workers cannot be negative.")


def main() -> None:
    args = parse_args()
    validate_diagnostic_args(args)
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        raise RuntimeError(
            "overfit_validation is single-process; do not use torchrun."
        )

    print(
        "WARNING: intentionally training on validation data. "
        "Do not use this model or its metrics for model selection."
    )

    device = select_device(args.device)
    set_seed(args.seed)
    torch.set_float32_matmul_precision("high")

    data_config = replace(
        DataConfig(),
        token_cache_path=args.token_cache_path,
        context_length=args.context_length,
        stride=args.stride,
        target_mode=args.target_mode,
        target_length=args.target_length,
    )
    training_config = replace(
        TrainingConfig(),
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_grad_norm=args.max_grad_norm,
        gradient_conflict_interval=args.gradient_conflict_interval,
        num_workers=args.num_workers,
        precision=args.precision,
        device=args.device,
        s1_loss_weight=args.s1_loss_weight,
        s2_loss_weight=args.s2_loss_weight,
    )
    lora_config = apply_target_profile(
        replace(
            QVLoRAConfig(),
            base_model_path=args.base_model_path,
            tokenizer_path=args.tokenizer_path,
            full_finetune_proj_s1=args.full_finetune_proj_s1,
        ),
        args.lora_target_profile,
    )
    if not args.full_finetune_proj_s1:
        lora_config = apply_proj_s1_override(
            lora_config,
            args.proj_s1_rank,
            args.proj_s1_alpha,
        )
    lora_config = apply_dependency_override(
        lora_config,
        args.dependency_lora_r,
        args.dependency_lora_alpha,
    )
    lora_config = apply_fusion_override(
        lora_config,
        args.fusion_lora_r,
        args.fusion_lora_alpha,
    )
    lora_config = apply_ffn_override(
        lora_config,
        args.ffn_lora_r,
        args.ffn_lora_alpha,
    )
    config = ExperimentConfig(
        data=data_config,
        training=training_config,
        lora=lora_config,
    )
    config.validate()
    precision = resolve_precision(args.precision, device)

    datasets, _ = build_datasets(data_config)
    validation_dataset = datasets["validation"]
    if args.start_index >= len(validation_dataset):
        raise ValueError(
            f"start_index={args.start_index} is outside validation dataset "
            f"length {len(validation_dataset)}."
        )
    stop_index = min(
        args.start_index + args.sample_count,
        len(validation_dataset),
    )
    small_dataset = Subset(
        validation_dataset,
        range(args.start_index, stop_index),
    )

    loader_common = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
        "drop_last": False,
    }
    train_loader = DataLoader(
        small_dataset,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
        **loader_common,
    )
    evaluation_loader = DataLoader(
        small_dataset,
        shuffle=False,
        **loader_common,
    )

    model, matched_modules = build_qv_lora_model(lora_config)
    model = model.to(device)
    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=args.learning_rate,
        betas=(training_config.adam_beta1, training_config.adam_beta2),
        weight_decay=training_config.weight_decay,
    )
    trainable, total = count_parameters(model)

    print(f"validation window range : [{args.start_index}, {stop_index})")
    print(f"validation windows used : {len(small_dataset)}")
    print(f"matched LoRA modules    : {len(matched_modules)}")
    print(f"trainable parameters    : {trainable:,}")
    print(f"total parameters        : {total:,}")
    print(f"precision               : {precision}\n")

    initial = evaluate(
        model,
        tokenizer=None,
        loader=evaluation_loader,
        device=device,
        split="small-validation",
        precision=precision,
        s1_loss_weight=args.s1_loss_weight,
        s2_loss_weight=args.s2_loss_weight,
    )
    print(
        f"epoch=00 loss={initial['loss']:.6f} "
        f"s1={initial['s1_loss']:.6f} s2={initial['s2_loss']:.6f} "
        f"s1_top1={initial['s1_top1_accuracy']:.4%} "
        f"s1_top5={initial['s1_top5_accuracy']:.4%} "
        f"s2_top1={initial['s2_top1_accuracy']:.4%} "
        f"s2_top5={initial['s2_top5_accuracy']:.4%} "
        f"s1_entropy={initial['s1_mean_entropy']:.4f} "
        f"s2_entropy={initial['s2_mean_entropy']:.4f}"
    )

    distributed = DistributedContext(False, 0, 0, 1, device)
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total_loss_sum = 0.0
        s1_loss_sum = 0.0
        s2_loss_sum = 0.0
        target_count = 0
        last_gradient_metrics: dict[str, float] | None = None
        conflict_probe_batches = []

        for batch_index, batch in enumerate(train_loader, start=1):
            group_start = (
                (batch_index - 1)
                // args.gradient_accumulation_steps
                * args.gradient_accumulation_steps
                + 1
            )
            group_end = min(
                group_start + args.gradient_accumulation_steps - 1,
                len(train_loader),
            )
            actual_group_size = group_end - group_start + 1
            metrics = forward_batch(
                model,
                tokenizer=None,
                batch=batch,
                device=device,
                precision=precision,
                s1_loss_weight=args.s1_loss_weight,
                s2_loss_weight=args.s2_loss_weight,
            )
            (metrics.loss / actual_group_size).backward()

            count = metrics.target_count
            total_loss_sum += metrics.loss.item() * count
            s1_loss_sum += metrics.s1_loss.item() * count
            s2_loss_sum += metrics.s2_loss.item() * count
            target_count += count

            should_update = (
                batch_index % args.gradient_accumulation_steps == 0
                or batch_index == len(train_loader)
            )
            should_probe_group = (
                args.gradient_conflict_interval > 0
                and (global_step + 1) % args.gradient_conflict_interval == 0
            )
            if should_probe_group:
                conflict_probe_batches.append(batch)
            if should_update:
                last_gradient_metrics = lora_gradient_norms(model)
                clip_grad_norm_(trainable_parameters, args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                if should_probe_group:
                    conflict_metrics = task_gradient_conflict_metrics(
                        model,
                        tokenizer=None,
                        batches=conflict_probe_batches,
                        device=device,
                        precision=precision,
                        distributed=distributed,
                        s1_weight=args.s1_loss_weight,
                        s2_weight=args.s2_loss_weight,
                    )
                    conflict_probe_batches.clear()
                    print(
                        f"epoch={epoch:02d} step={global_step} "
                        f"shared_g1="
                        f"{conflict_metrics['transformer_s1_grad_norm']:.3e} "
                        f"shared_g2="
                        f"{conflict_metrics['transformer_s2_grad_norm']:.3e} "
                        f"ratio="
                        f"{conflict_metrics['transformer_s2_to_s1_ratio']:.3f} "
                        f"cos="
                        f"{conflict_metrics['transformer_grad_cosine']:.4f}"
                    )

        if last_gradient_metrics is None or target_count == 0:
            raise RuntimeError("The diagnostic loader produced no trainable batch.")

        same_set = evaluate(
            model,
            tokenizer=None,
            loader=evaluation_loader,
            device=device,
            split="small-validation",
            epoch=epoch,
            precision=precision,
            s1_loss_weight=args.s1_loss_weight,
            s2_loss_weight=args.s2_loss_weight,
        )
        print(
            f"epoch={epoch:02d} "
            f"train_loss={total_loss_sum / target_count:.6f} "
            f"train_s1={s1_loss_sum / target_count:.6f} "
            f"train_s2={s2_loss_sum / target_count:.6f} | "
            f"same_set_loss={same_set['loss']:.6f} "
            f"same_set_s1={same_set['s1_loss']:.6f} "
            f"same_set_s2={same_set['s2_loss']:.6f} | "
            f"s1_top1={same_set['s1_top1_accuracy']:.4%} "
            f"s1_top5={same_set['s1_top5_accuracy']:.4%} "
            f"s2_top1={same_set['s2_top1_accuracy']:.4%} "
            f"s2_top5={same_set['s2_top5_accuracy']:.4%} "
            f"s1_entropy={same_set['s1_mean_entropy']:.4f} "
            f"s2_entropy={same_set['s2_mean_entropy']:.4f} | "
            f"grad_total={last_gradient_metrics['grad_norm_total']:.3e} "
            f"grad_s1={last_gradient_metrics['grad_norm_proj_s1']:.3e} "
            f"grad_transformer="
            f"{last_gradient_metrics['grad_norm_transformer']:.3e}"
        )

    print("\nDiagnostic experiment finished.")
    print("No adapter was saved. Do not use these metrics for model selection.")


if __name__ == "__main__":
    main()
