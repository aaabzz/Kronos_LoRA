"""Build and restore a QV LoRA model on top of local Kronos weights."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import replace
from pathlib import Path

import torch
from torch import nn

from model import Kronos, KronosTokenizer

from .config import QVLoRAConfig, TOP4_FFN_TARGETS


def _import_peft():
    try:
        from peft import LoraConfig, PeftModel, get_peft_model
    except ImportError as exc:
        raise RuntimeError(
            "PEFT is not installed. Install it with: pip install peft"
        ) from exc
    return LoraConfig, PeftModel, get_peft_model


def load_frozen_tokenizer(config: QVLoRAConfig) -> KronosTokenizer:
    """Load the local tokenizer and freeze all of its parameters."""
    config.validate_local_models()
    tokenizer = KronosTokenizer.from_pretrained(config.tokenizer_path)
    tokenizer.eval()
    tokenizer.requires_grad_(False)
    return tokenizer


def build_full_finetune_model(config: QVLoRAConfig) -> Kronos:
    """Load a Kronos predictor with every parameter enabled for training."""
    config.validate_local_models()
    model = Kronos.from_pretrained(config.base_model_path)
    model.requires_grad_(True)
    return model


def load_full_finetune_model(model_path: str | Path) -> Kronos:
    """Load a complete Kronos checkpoint saved by full-parameter training."""
    return Kronos.from_pretrained(Path(model_path))


def find_target_linear_modules(
    model: nn.Module,
    target_suffixes: Iterable[str],
    exclude_suffixes: Iterable[str] = (),
) -> list[str]:
    """Return Linear names selected by PEFT-style target/exclude matching."""
    suffixes = tuple(target_suffixes)
    exclusions = tuple(exclude_suffixes)
    return [
        name
        for name, module in model.named_modules()
        if (
            isinstance(module, nn.Linear)
            and name.endswith(suffixes)
            and not (exclusions and name.endswith(exclusions))
        )
    ]


def assert_only_lora_parameters_are_trainable(
    model: nn.Module,
    modules_to_save: Iterable[str] = (),
) -> None:
    """Fail early if a parameter outside LoRA/saved full modules is trainable."""
    saved_modules = tuple(modules_to_save)
    unexpected = [
        name
        for name, parameter in model.named_parameters()
        if (
            parameter.requires_grad
            and "lora_" not in name
            and not (
                ".modules_to_save." in name
                and any(module_name in name for module_name in saved_modules)
            )
        )
    ]
    if unexpected:
        formatted = "\n".join(f"- {name}" for name in unexpected)
        raise RuntimeError(f"Unexpected trainable base parameters:\n{formatted}")


def count_parameters(model: nn.Module) -> tuple[int, int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def resolve_target_modules(model: nn.Module, config: QVLoRAConfig) -> tuple[str, ...]:
    """Resolve architecture-relative profile entries against the loaded model."""
    configured = tuple(config.target_modules)
    if not set(TOP4_FFN_TARGETS).issubset(configured):
        return configured

    transformer = getattr(model, "transformer", None)
    if transformer is None:
        raise RuntimeError("The selected top-4 FFN profile requires model.transformer.")
    layer_count = len(transformer)
    if layer_count < 4:
        raise RuntimeError(
            f"The selected top-4 FFN profile requires at least 4 layers; got {layer_count}."
        )
    fixed_targets = tuple(name for name in configured if name not in TOP4_FFN_TARGETS)
    dynamic_targets = tuple(
        f"transformer.{layer}.ffn.{projection}"
        for layer in range(layer_count - 4, layer_count)
        for projection in ("w1", "w2", "w3")
    )
    return (*fixed_targets, *dynamic_targets)


def build_rank_alpha_patterns(
    config: QVLoRAConfig,
    model: nn.Module | None = None,
) -> tuple[dict[str, int], dict[str, int]]:
    """Build PEFT per-family rank/alpha overrides."""
    rank_pattern: dict[str, int] = {}
    alpha_pattern: dict[str, int] = {}
    if config.proj_s1_rank is not None and not config.full_finetune_proj_s1:
        rank_pattern["head.proj_s1"] = config.proj_s1_rank
        alpha_pattern["head.proj_s1"] = config.proj_s1_alpha
    if config.dependency_rank is not None:
        for name in (
            "dep_layer.cross_attn.q_proj",
            "dep_layer.cross_attn.v_proj",
        ):
            rank_pattern[name] = config.dependency_rank
            alpha_pattern[name] = config.dependency_alpha
    if config.fusion_rank is not None:
        rank_pattern["embedding.fusion_proj"] = config.fusion_rank
        alpha_pattern["embedding.fusion_proj"] = config.fusion_alpha
    if config.ffn_rank is not None:
        if model is not None:
            ffn_modules = tuple(
                name
                for name, module in model.named_modules()
                if isinstance(module, nn.Linear)
                and ".ffn." in name
                and name.endswith(config.target_modules)
                and not (
                    config.exclude_modules
                    and name.endswith(config.exclude_modules)
                )
            )
        elif {"w1", "w2", "w3"}.intersection(config.target_modules):
            # Backward-compatible representation for callers that only inspect
            # the configuration. Model construction passes the actual backbone
            # above, so non-base layer counts are discovered dynamically.
            ffn_modules = tuple(
                f"transformer.{layer}.ffn.{projection}"
                for layer in range(12)
                for projection in ("w1", "w2", "w3")
            )
        else:
            ffn_modules = tuple(
                name for name in config.target_modules if ".ffn.w" in name
            )
        for name in ffn_modules:
            rank_pattern[name] = config.ffn_rank
            alpha_pattern[name] = config.ffn_alpha
    return rank_pattern, alpha_pattern


def build_qv_lora_model(config: QVLoRAConfig):
    """Load a local Kronos backbone, inject adapters, and validate the result."""
    config.validate_local_models()
    LoraConfig, _, get_peft_model = _import_peft()

    base_model = Kronos.from_pretrained(config.base_model_path)
    resolved_targets = resolve_target_modules(base_model, config)
    lora_target_modules = tuple(
        name
        for name in resolved_targets
        if not (config.full_finetune_proj_s1 and name.endswith("proj_s1"))
    )
    matched_modules = find_target_linear_modules(
        base_model,
        lora_target_modules,
        config.exclude_modules,
    )
    if not matched_modules:
        raise RuntimeError(
            "No Linear modules matched target/exclude selection: "
            f"targets={lora_target_modules}, excludes={config.exclude_modules}."
        )

    modules_to_save = (
        ("head.proj_s1",)
        if getattr(config, "full_finetune_proj_s1", False)
        else ()
    )
    if modules_to_save:
        proj_s1 = dict(base_model.named_modules()).get("head.proj_s1")
        if not isinstance(proj_s1, nn.Linear):
            raise RuntimeError("head.proj_s1 is not a Linear module.")
        matched_modules.append("head.proj_s1")

    pattern_config = replace(config, target_modules=resolved_targets)
    rank_pattern, alpha_pattern = build_rank_alpha_patterns(pattern_config, base_model)
    peft_config = LoraConfig(
        r=config.rank,
        lora_alpha=config.alpha,
        rank_pattern=rank_pattern,
        alpha_pattern=alpha_pattern,
        lora_dropout=config.dropout,
        target_modules=list(lora_target_modules),
        exclude_modules=list(config.exclude_modules) or None,
        modules_to_save=list(modules_to_save) or None,
        bias=config.bias,
        inference_mode=False,
    )
    model = get_peft_model(base_model, peft_config)
    # Kronos is a custom nn.Module, so get_peft_model cannot infer this field
    # and overwrites it with None. Set it on the installed config afterwards;
    # deployment.json remains the portable front-end source of truth.
    for installed_config in model.peft_config.values():
        installed_config.base_model_name_or_path = str(config.base_model_path)
    assert_only_lora_parameters_are_trainable(model, modules_to_save)

    return model, matched_modules


def load_qv_lora_adapter(
    config: QVLoRAConfig,
    adapter_path=None,
    *,
    is_trainable: bool = False,
):
    """Restore a previously saved adapter on the frozen local base model."""
    config.validate_local_models()
    _, PeftModel, _ = _import_peft()
    base_model = Kronos.from_pretrained(config.base_model_path)
    path = adapter_path or config.adapter_output_dir
    return PeftModel.from_pretrained(
        base_model,
        path,
        is_trainable=is_trainable,
    )


def resolve_adapter_directory(
    path: str | Path,
    model_selection: str = "best",
) -> Path:
    """Resolve a direct PEFT adapter or best/last adapter below a run."""
    requested = Path(path).expanduser().resolve()
    if (requested / "adapter_config.json").is_file():
        return requested
    if model_selection not in {"best", "last"}:
        raise ValueError("model_selection must be one of: best, last.")
    subdirectory = "best_model" if model_selection == "best" else "last_model"
    candidate = requested / subdirectory
    if (candidate / "adapter_config.json").is_file():
        return candidate
    raise FileNotFoundError(
        f"No adapter_config.json found in {requested} or {candidate}."
    )


def initialize_qv_lora_model(
    config: QVLoRAConfig,
    adapter_path: str | Path,
    *,
    model_selection: str = "best",
):
    """Build a fresh target adapter and copy compatible stage-one weights.

    Missing destination tensors are allowed so a stage-two target profile can
    be a superset of stage one. Shape mismatches and source LoRA tensors that
    do not exist in the destination remain hard errors.
    """
    source_directory = resolve_adapter_directory(
        adapter_path,
        model_selection,
    )
    model, matched_modules = build_qv_lora_model(config)
    try:
        from peft.utils.save_and_load import (
            load_peft_weights,
            set_peft_model_state_dict,
        )
    except ImportError as exc:
        raise RuntimeError(
            "This PEFT version cannot initialize from adapter weights. "
            "Install the version listed in requirements-train-server.txt."
        ) from exc

    source_state = load_peft_weights(str(source_directory), device="cpu")
    if not source_state:
        raise RuntimeError(f"Adapter contains no weights: {source_directory}")
    load_result = set_peft_model_state_dict(
        model,
        source_state,
        adapter_name="default",
    )
    unexpected_lora = [
        key for key in load_result.unexpected_keys if "lora_" in key
    ]
    if unexpected_lora:
        formatted = "\n".join(f"- {key}" for key in unexpected_lora)
        raise ValueError(
            "The initialization adapter contains LoRA modules absent from "
            f"the requested target profile:\n{formatted}"
        )
    modules_to_save = (
        ("head.proj_s1",)
        if getattr(config, "full_finetune_proj_s1", False)
        else ()
    )
    assert_only_lora_parameters_are_trainable(model, modules_to_save)
    return model, matched_modules, source_directory, len(source_state)
