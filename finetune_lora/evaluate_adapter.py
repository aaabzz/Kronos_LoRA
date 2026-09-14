"""Evaluate a locally saved Kronos LoRA adapter from the command line."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import replace
from pathlib import Path

import torch

from lora.config import QVLoRAConfig
from lora.model_factory import load_qv_lora_adapter

from .config import DataConfig, ExperimentConfig, TrainingConfig
from .training import (
    DistributedContext,
    LossMetrics,
    autocast_context,
    create_loaders,
    evaluate,
    hierarchical_token_loss,
    prepare_batch,
    resolve_precision,
    select_device,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a saved Kronos LoRA adapter with loss, token top-k "
            "accuracy, and token entropy."
        )
    )
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
        help="Tokenizer metadata path used to validate the experiment configuration.",
    )
    parser.add_argument(
        "--adapter-path",
        type=Path,
        required=True,
        help=(
            "Direct PEFT adapter directory or a training run directory "
            "containing best_model/ and last_model/."
        ),
    )
    parser.add_argument(
        "--model-selection",
        choices=("best", "last"),
        default="best",
        help="Adapter subdirectory to use when --adapter-path is a run directory.",
    )
    parser.add_argument("--token-cache-path", type=Path, required=True)
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
    parser.add_argument(
        "--s2-top-k",
        type=int,
        default=5,
        help="Number of s1 candidates marginalized by topk evaluation.",
    )
    parser.add_argument(
        "--s2-temperature",
        type=float,
        default=1.0,
        help="Temperature applied to the s1 distribution before top-k selection.",
    )
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


def resolve_adapter_path(path: Path, model_selection: str) -> Path:
    """Resolve a direct PEFT directory or best/last model below a run directory."""
    requested = path.expanduser().resolve()
    if (requested / "adapter_config.json").is_file():
        return requested
    subdirectory = "best_model" if model_selection == "best" else "last_model"
    candidate = requested / subdirectory
    if (candidate / "adapter_config.json").is_file():
        return candidate
    raise FileNotFoundError(
        f"No PEFT adapter_config.json found in {requested} or {candidate}."
    )


def _new_metric_totals() -> dict[str, float]:
    return {
        "weighted_loss": 0.0,
        "weighted_s1_loss": 0.0,
        "weighted_s2_loss": 0.0,
        "s1_top1_correct": 0.0,
        "s1_top5_correct": 0.0,
        "s1_entropy_sum": 0.0,
        "s2_top1_correct": 0.0,
        "s2_top5_correct": 0.0,
        "s2_entropy_sum": 0.0,
        "targets": 0.0,
    }


def _accumulate_metrics(totals: dict[str, float], metrics: LossMetrics) -> None:
    if metrics.s1_prediction is None or metrics.s2_prediction is None:
        raise RuntimeError("Conditioning evaluation metrics were not computed.")
    count = metrics.target_count
    totals["weighted_loss"] += metrics.loss.item() * count
    totals["weighted_s1_loss"] += metrics.s1_loss.item() * count
    totals["weighted_s2_loss"] += metrics.s2_loss.item() * count
    totals["s1_top1_correct"] += metrics.s1_prediction.top1_correct
    totals["s1_top5_correct"] += metrics.s1_prediction.top5_correct
    totals["s1_entropy_sum"] += metrics.s1_prediction.entropy_sum
    totals["s2_top1_correct"] += metrics.s2_prediction.top1_correct
    totals["s2_top5_correct"] += metrics.s2_prediction.top5_correct
    totals["s2_entropy_sum"] += metrics.s2_prediction.entropy_sum
    totals["targets"] += count


def _finalize_metrics(totals: dict[str, float]) -> dict[str, float]:
    targets = totals["targets"]
    if targets <= 0:
        raise RuntimeError("Conditioning evaluation produced no targets.")
    return {
        "loss": totals["weighted_loss"] / targets,
        "s1_loss": totals["weighted_s1_loss"] / targets,
        "s2_loss": totals["weighted_s2_loss"] / targets,
        "s1_top1_accuracy": totals["s1_top1_correct"] / targets,
        "s1_top5_accuracy": totals["s1_top5_correct"] / targets,
        "s2_top1_accuracy": totals["s2_top1_correct"] / targets,
        "s2_top5_accuracy": totals["s2_top5_correct"] / targets,
        "s1_mean_entropy": totals["s1_entropy_sum"] / targets,
        "s2_mean_entropy": totals["s2_entropy_sum"] / targets,
        "targets": targets,
    }


def build_s1_conditioning_ids(
    mode: str,
    s1_logits: torch.Tensor,
    s1_targets: torch.Tensor,
    *,
    vocab_size: int,
    constant_s1_id: int = 0,
) -> torch.Tensor:
    """Construct the s1 ids used to condition s2 for one ablation mode."""
    if vocab_size <= 0:
        raise ValueError("vocab_size must be positive.")
    if mode == "teacher":
        return s1_targets
    if mode == "greedy":
        return s1_logits.detach().argmax(dim=-1)
    if mode == "sample":
        probabilities = s1_logits.detach().float().softmax(dim=-1)
        return torch.multinomial(
            probabilities.reshape(-1, vocab_size),
            num_samples=1,
        ).reshape_as(s1_targets)
    if mode == "shuffle":
        flattened = s1_targets.reshape(-1)
        permutation = torch.randperm(flattened.numel(), device=flattened.device)
        return flattened[permutation].reshape_as(s1_targets)
    if mode == "random":
        return torch.randint(
            0,
            vocab_size,
            s1_targets.shape,
            device=s1_targets.device,
            dtype=s1_targets.dtype,
        )
    if mode == "constant":
        if not 0 <= constant_s1_id < vocab_size:
            raise ValueError(
                f"constant_s1_id must be in [0, {vocab_size}), "
                f"got {constant_s1_id}."
            )
        return torch.full_like(s1_targets, constant_s1_id)
    raise ValueError(f"Unsupported s2 conditioning mode: {mode!r}.")


def evaluate_s2_conditioning(
    model,
    loader,
    device: torch.device,
    *,
    precision: str,
    modes: tuple[str, ...],
    s1_loss_weight: float,
    s2_loss_weight: float,
    constant_s1_id: int = 0,
    s2_top_k: int = 5,
    s2_temperature: float = 1.0,
) -> dict[str, dict[str, float]]:
    """Evaluate s2 conditioning ablations from one shared context pass."""
    supported_modes = {
        "teacher",
        "greedy",
        "sample",
        "topk",
        "shuffle",
        "random",
        "constant",
    }
    invalid_modes = set(modes) - supported_modes
    if invalid_modes or not modes:
        raise ValueError(f"Unsupported s2 conditioning modes: {modes!r}.")
    backbone = model.get_base_model() if hasattr(model, "get_base_model") else model
    totals_by_mode = {mode: _new_metric_totals() for mode in modes}
    model.eval()

    with torch.no_grad():
        for batch_index, batch in enumerate(loader, start=1):
            (
                s1_input,
                s2_input,
                s1_targets,
                s2_targets,
                stamps,
                loss_mask,
            ) = prepare_batch(None, batch, device)
            with autocast_context(device, precision):
                s1_logits, context = backbone.decode_s1(
                    s1_input,
                    s2_input,
                    stamps,
                )
                vocab_size = backbone.embedding.emb_s1.num_embeddings
                for mode in modes:
                    if mode == "topk":
                        s2_logits = backbone.topk_marginalized_s2_log_probs(
                            context,
                            s1_logits,
                            top_k=s2_top_k,
                            temperature=s2_temperature,
                            is_causal=True,
                        )
                    else:
                        conditioning_ids = build_s1_conditioning_ids(
                            mode,
                            s1_logits,
                            s1_targets,
                            vocab_size=vocab_size,
                            constant_s1_id=constant_s1_id,
                        )
                        sibling_embed = backbone.embedding.emb_s1(
                            conditioning_ids
                        )
                        dependency_context = backbone.dep_layer(
                            context,
                            sibling_embed,
                            is_causal=True,
                        )
                        s2_logits = backbone.head.cond_forward(
                            dependency_context
                        )
                    metrics = hierarchical_token_loss(
                        s1_logits,
                        s2_logits,
                        s1_targets,
                        s2_targets,
                        loss_mask,
                        s1_weight=s1_loss_weight,
                        s2_weight=s2_loss_weight,
                        compute_prediction_metrics=True,
                    )
                    if not torch.isfinite(metrics.loss):
                        raise FloatingPointError(
                            f"Non-finite {mode} conditioning loss at "
                            f"batch={batch_index}."
                        )
                    _accumulate_metrics(totals_by_mode[mode], metrics)

    return {
        mode: _finalize_metrics(totals)
        for mode, totals in totals_by_mode.items()
    }


def teacher_forcing_gap(
    metrics_by_conditioning: dict[str, dict[str, float]],
) -> dict[str, float]:
    """Return greedy-minus-teacher deltas; positive loss means degradation."""
    teacher = metrics_by_conditioning["teacher"]
    greedy = metrics_by_conditioning["greedy"]
    return {
        "loss_delta": greedy["loss"] - teacher["loss"],
        "s2_loss_delta": greedy["s2_loss"] - teacher["s2_loss"],
        "s2_top1_accuracy_delta": (
            greedy["s2_top1_accuracy"] - teacher["s2_top1_accuracy"]
        ),
        "s2_top5_accuracy_delta": (
            greedy["s2_top5_accuracy"] - teacher["s2_top5_accuracy"]
        ),
        "s2_mean_entropy_delta": (
            greedy["s2_mean_entropy"] - teacher["s2_mean_entropy"]
        ),
    }


def conditioning_display_name(mode: str, result: dict[str, object]) -> str:
    """Disambiguate marginalization k from the fixed top-5 accuracy metric."""
    if mode != "topk":
        return mode
    top_k = int(result.get("s2_top_k", 5))
    temperature = float(result.get("s2_temperature", 1.0))
    return f"topk(k={top_k},T={temperature:g})"


def print_result(result: dict[str, object]) -> None:
    split = str(result["split"])
    metrics = result["metrics"]
    assert isinstance(metrics, dict)
    print(f"\nKronos LoRA {split} result")
    print(f"adapter             : {result['adapter_path']}")
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
        raise RuntimeError("evaluate_adapter is single-process; do not use torchrun.")

    adapter_path = resolve_adapter_path(args.adapter_path, args.model_selection)
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
    lora_config = replace(
        QVLoRAConfig(),
        base_model_path=args.base_model_path,
        tokenizer_path=args.tokenizer_path,
    )
    config = ExperimentConfig(data_config, training_config, lora_config)
    config.validate()

    distributed = DistributedContext(False, 0, 0, 1, device)
    loaders, datasets, _ = create_loaders(config, distributed)
    precision = resolve_precision(args.precision, device)
    model = load_qv_lora_adapter(lora_config, adapter_path).to(device).eval()
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
        "adapter_path": str(adapter_path),
        "base_model_path": str(lora_config.base_model_path),
        "tokenizer_path": str(lora_config.tokenizer_path),
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
