"""Model-only QV LoRA smoke test using synthetic token IDs."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import torch

from .config import (
    LORA_TARGET_PROFILES,
    QVLoRAConfig,
    apply_dependency_override,
    apply_ffn_override,
    apply_fusion_override,
    apply_proj_s1_override,
    apply_target_profile,
)
from .model_factory import (
    build_qv_lora_model,
    count_parameters,
    load_frozen_tokenizer,
)


def select_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build QV LoRA and run a forward pass without real data."
    )
    parser.add_argument(
        "--base-model-path", type=Path, default=QVLoRAConfig().base_model_path
    )
    parser.add_argument(
        "--tokenizer-path", type=Path, default=QVLoRAConfig().tokenizer_path
    )
    parser.add_argument(
        "--adapter-output-dir",
        type=Path,
        default=QVLoRAConfig().adapter_output_dir,
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--sequence-length", type=int, default=8)
    parser.add_argument(
        "--lora-target-profile",
        choices=LORA_TARGET_PROFILES,
        default="qv",
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
    parser.add_argument("--proj-s1-rank", type=int, default=None)
    parser.add_argument("--proj-s1-alpha", type=int, default=None)
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
    parser.add_argument(
        "--save-adapter",
        type=Path,
        default=None,
        help="Optionally save the untrained adapter for serialization testing.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = replace(
        QVLoRAConfig(),
        base_model_path=args.base_model_path,
        tokenizer_path=args.tokenizer_path,
        adapter_output_dir=args.adapter_output_dir,
        rank=args.lora_r,
        alpha=args.lora_alpha,
        full_finetune_proj_s1=args.full_finetune_proj_s1,
    )
    config = apply_target_profile(config, args.lora_target_profile)
    config = apply_proj_s1_override(
        config,
        args.proj_s1_rank,
        args.proj_s1_alpha,
    )
    config = apply_dependency_override(
        config,
        args.dependency_lora_r,
        args.dependency_lora_alpha,
    )
    config = apply_fusion_override(
        config,
        args.fusion_lora_r,
        args.fusion_lora_alpha,
    )
    config = apply_ffn_override(
        config,
        args.ffn_lora_r,
        args.ffn_lora_alpha,
    )
    device = select_device(args.device)

    tokenizer = load_frozen_tokenizer(config).to(device)
    model, matched_modules = build_qv_lora_model(config)
    model = model.to(device).eval()

    trainable, total = count_parameters(model)
    print(f"Matched trainable modules ({args.lora_target_profile}):")
    for name in matched_modules:
        print(f"  - {name}")
    print(f"Matched module count: {len(matched_modules)}")
    print(f"Trainable parameters: {trainable:,}")
    print(f"Total parameters with adapters: {total:,}")
    print(f"Trainable ratio: {100.0 * trainable / total:.4f}%")
    print(f"Tokenizer trainable parameters: {sum(p.numel() for p in tokenizer.parameters() if p.requires_grad)}")

    s1_vocab_size = 2 ** model.base_model.model.s1_bits
    s2_vocab_size = 2 ** model.base_model.model.s2_bits
    shape = (args.batch_size, args.sequence_length)
    s1_ids = torch.randint(s1_vocab_size, shape, device=device)
    s2_ids = torch.randint(s2_vocab_size, shape, device=device)
    s1_targets = torch.randint(s1_vocab_size, shape, device=device)

    with torch.no_grad():
        s1_logits, s2_logits = model(
            s1_ids,
            s2_ids,
            use_teacher_forcing=True,
            s1_targets=s1_targets,
        )

    expected_s1_shape = (*shape, s1_vocab_size)
    expected_s2_shape = (*shape, s2_vocab_size)
    if tuple(s1_logits.shape) != expected_s1_shape:
        raise RuntimeError(
            f"Unexpected s1 shape: {tuple(s1_logits.shape)} != {expected_s1_shape}"
        )
    if tuple(s2_logits.shape) != expected_s2_shape:
        raise RuntimeError(
            f"Unexpected s2 shape: {tuple(s2_logits.shape)} != {expected_s2_shape}"
        )

    print(f"s1 logits shape: {tuple(s1_logits.shape)}")
    print(f"s2 logits shape: {tuple(s2_logits.shape)}")
    print("LoRA smoke test passed.")

    if args.save_adapter is not None:
        model.save_pretrained(args.save_adapter)
        print(f"Untrained adapter saved to: {args.save_adapter.resolve()}")


if __name__ == "__main__":
    main()
