"""Evaluate the frozen Kronos base model on a token-cache data split."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import replace
from pathlib import Path

import torch

from lora.config import QVLoRAConfig
from model import Kronos

from .config import DataConfig, ExperimentConfig, TrainingConfig
from .evaluate_adapter import (
    conditioning_display_name,
    evaluate_s2_conditioning,
    teacher_forcing_gap,
)
from .training import (
    DistributedContext,
    create_loaders,
    evaluate,
    resolve_precision,
    select_device,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a Kronos base model without a LoRA adapter on the cached "
            "validation or test split."
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
        help="Tokenizer metadata path used to validate the experiment configuration.",
    )
    parser.add_argument(
        "--token-cache-path",
        type=Path,
        required=True,
        help="Pre-tokenized cache used by the corresponding training run.",
    )
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument(
        "--s2-conditioning",
        choices=(
            "teacher",
            "greedy",
            "sample",
            "topk",
            "shuffle",
            "random",
            "constant",
            "both",
            "all",
        ),
        default="teacher",
        help=(
            "Choose the s1 condition used by s2. 'both' compares teacher and "
            "greedy; 'all' runs every conditioning ablation."
        ),
    )
    parser.add_argument(
        "--constant-s1-id",
        type=int,
        default=0,
        help="Token id used by the constant s1 conditioning ablation.",
    )
    parser.add_argument("--s2-top-k", type=int, default=5)
    parser.add_argument("--s2-temperature", type=float, default=1.0)
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
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--s1-loss-weight", type=float, default=0.5)
    parser.add_argument("--s2-loss-weight", type=float, default=0.5)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--precision",
        choices=("auto", "bf16", "fp32"),
        default="bf16",
    )
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path for a machine-readable copy of the evaluation result.",
    )
    return parser.parse_args()


def print_result(result: dict[str, object]) -> None:
    """Print base-model metrics using the same layout as adapter evaluation."""
    split = str(result["split"])
    metrics = result["metrics"]
    assert isinstance(metrics, dict)
    print(f"\nKronos base model {split} result")
    print(f"base model          : {result['base_model_path']}")
    print(f"s2 conditioning     : {result['s2_conditioning']}")
    print(f"target mode         : {result['target_mode']}")
    if result["target_mode"] == "tail":
        print(f"target length       : {result['target_length']}")
    if result["s2_conditioning"] in {"topk", "all"}:
        print(f"marginalization k   : {int(result.get('s2_top_k', 5))}")
        print(
            "s1 temperature      : "
            f"{float(result.get('s2_temperature', 1.0)):g}"
        )
    print(f"{split} windows     : {result['window_count']}")
    print(f"{split} loss        : {metrics['loss']:.6f}")
    print(f"s1 loss             : {metrics['s1_loss']:.6f}")
    print(f"s2 loss             : {metrics['s2_loss']:.6f}")
    print(f"s1 top-1 accuracy   : {metrics['s1_top1_accuracy']:.4%}")
    print(f"s1 top-5 accuracy   : {metrics['s1_top5_accuracy']:.4%}")
    print(f"s2 top-1 accuracy   : {metrics['s2_top1_accuracy']:.4%}")
    print(f"s2 top-5 accuracy   : {metrics['s2_top5_accuracy']:.4%}")
    print(f"s1 mean entropy     : {metrics['s1_mean_entropy']:.6f}")
    print(f"s2 mean entropy     : {metrics['s2_mean_entropy']:.6f}")
    print(f"target count        : {int(metrics['targets'])}")
    metrics_by_conditioning = result.get("metrics_by_conditioning")
    if isinstance(metrics_by_conditioning, dict):
        print("\nConditioning ablation results")
        for mode, mode_metrics in metrics_by_conditioning.items():
            assert isinstance(mode_metrics, dict)
            display_name = conditioning_display_name(mode, result)
            print(
                f"{display_name:>20} | "
                f"s2_loss={mode_metrics['s2_loss']:.6f} "
                f"s2_top1_acc={mode_metrics['s2_top1_accuracy']:.4%} "
                f"s2_top5_acc={mode_metrics['s2_top5_accuracy']:.4%} "
                f"entropy={mode_metrics['s2_mean_entropy']:.6f}"
            )
    gap = result.get("greedy_minus_teacher")
    if isinstance(gap, dict):
        print("\nGreedy minus teacher gap")
        print(f"s2 loss delta       : {gap['s2_loss_delta']:+.6f}")
        print(f"s2 top-1 delta      : {gap['s2_top1_accuracy_delta']:+.4%}")
        print(f"s2 top-5 delta      : {gap['s2_top5_accuracy_delta']:+.4%}")
        print(f"s2 entropy delta    : {gap['s2_mean_entropy_delta']:+.6f}")


def main() -> None:
    args = parse_args()
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        raise RuntimeError("evaluate_base is single-process; do not use torchrun.")

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
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
        precision=args.precision,
        s1_loss_weight=args.s1_loss_weight,
        s2_loss_weight=args.s2_loss_weight,
    )
    config = ExperimentConfig(
        data=data_config,
        training=training_config,
        lora=replace(
            QVLoRAConfig(),
            base_model_path=args.base_model_path,
            tokenizer_path=args.tokenizer_path,
        ),
    )
    config.validate()

    distributed = DistributedContext(
        enabled=False,
        rank=0,
        local_rank=0,
        world_size=1,
        device=device,
    )
    loaders, datasets, _ = create_loaders(config, distributed)
    precision = resolve_precision(args.precision, device)

    model = Kronos.from_pretrained(config.lora.base_model_path).to(device).eval()
    metrics_by_conditioning = None
    gap = None
    if args.s2_conditioning == "teacher":
        metrics = evaluate(
            model,
            tokenizer=None,
            loader=loaders[args.split],
            device=device,
            split=args.split,
            precision=precision,
            s1_loss_weight=args.s1_loss_weight,
            s2_loss_weight=args.s2_loss_weight,
            distributed=distributed,
        )
    else:
        if args.s2_conditioning == "both":
            modes = ("teacher", "greedy")
        elif args.s2_conditioning == "all":
            modes = (
                "teacher",
                "greedy",
                "sample",
                "topk",
                "shuffle",
                "random",
                "constant",
            )
        else:
            modes = (args.s2_conditioning,)
        metrics_by_conditioning = evaluate_s2_conditioning(
            model,
            loaders[args.split],
            device,
            precision=precision,
            modes=modes,
            s1_loss_weight=args.s1_loss_weight,
            s2_loss_weight=args.s2_loss_weight,
            constant_s1_id=args.constant_s1_id,
            s2_top_k=args.s2_top_k,
            s2_temperature=args.s2_temperature,
        )
        metrics = metrics_by_conditioning[
            "teacher"
            if args.s2_conditioning in {"both", "all"}
            else args.s2_conditioning
        ]
        if args.s2_conditioning in {"both", "all"}:
            gap = teacher_forcing_gap(metrics_by_conditioning)

    result: dict[str, object] = {
        "base_model_path": str(config.lora.base_model_path),
        "tokenizer_path": str(config.lora.tokenizer_path),
        "token_cache_path": str(args.token_cache_path.resolve()),
        "split": args.split,
        "window_count": len(datasets[args.split]),
        "resolved_precision": precision,
        "s2_conditioning": args.s2_conditioning,
        "target_mode": args.target_mode,
        "target_length": args.target_length,
        "s2_top_k": args.s2_top_k,
        "s2_temperature": args.s2_temperature,
        "metrics": metrics,
    }
    if metrics_by_conditioning is not None:
        result["metrics_by_conditioning"] = metrics_by_conditioning
    if gap is not None:
        result["greedy_minus_teacher"] = gap
    print_result(result)

    if args.output_json is not None:
        output_path = args.output_json.expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = output_path.with_name(f".{output_path.name}.tmp")
        temporary.write_text(
            json.dumps(result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temporary.replace(output_path)
        print(f"saved result        : {output_path}")


if __name__ == "__main__":
    main()
