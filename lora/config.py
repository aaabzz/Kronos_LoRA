"""Configuration for QV LoRA adaptation of Kronos."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

BASELINE_QV_PROFILE = "qv"
QV_S1_PROFILE = "qv-s1"
SELF_ATTN_QV_S1_PROFILE = "self-attn-qv-s1"
SELF_ATTN_QKVO_FUSION_S1_PROFILE = "self-attn-qkvo-fusion-s1"
SELF_ATTN_QKVO_FUSION_TOP4_FFN_S1_PROFILE = (
    "self-attn-qkvo-fusion-top4-ffn-s1"
)
SELF_ATTN_QKVO_FUSION_FFN_S1_PROFILE = "self-attn-qkvo-fusion-ffn-s1"
LORA_TARGET_PROFILES: tuple[str, ...] = (
    BASELINE_QV_PROFILE,
    QV_S1_PROFILE,
    SELF_ATTN_QV_S1_PROFILE,
    SELF_ATTN_QKVO_FUSION_S1_PROFILE,
    SELF_ATTN_QKVO_FUSION_TOP4_FFN_S1_PROFILE,
    SELF_ATTN_QKVO_FUSION_FFN_S1_PROFILE,
)

SELF_ATTN_DEPENDENCY_EXCLUSIONS: tuple[str, ...] = (
    "dep_layer.cross_attn.q_proj",
    "dep_layer.cross_attn.k_proj",
    "dep_layer.cross_attn.v_proj",
    "dep_layer.cross_attn.out_proj",
)
TOP4_FFN_TARGETS: tuple[str, ...] = tuple(
    f"transformer.{layer}.ffn.{projection}"
    for layer in range(8, 12)
    for projection in ("w1", "w2", "w3")
)


@dataclass(frozen=True)
class QVLoRAConfig:
    """Model-only configuration; no dataset settings are included yet."""

    base_model_path: Path = PROJECT_ROOT / "pretrained" / "Kronos-base"
    tokenizer_path: Path = PROJECT_ROOT / "pretrained" / "Kronos-Tokenizer-base"
    adapter_output_dir: Path = PROJECT_ROOT / "checkpoints" / "Kronos-base-QV-LoRA"

    rank: int = 8
    alpha: int = 16
    # Optional PEFT rank/alpha override for the s1 output projection only.
    # Q/V modules continue to use the global rank and alpha above.
    proj_s1_rank: int | None = None
    proj_s1_alpha: int | None = None
    # Train and save the complete head.proj_s1 Linear layer instead of
    # attaching a LoRA adapter to that layer.
    full_finetune_proj_s1: bool = False
    # Optional PEFT rank/alpha override for dependency cross-attention Q/V.
    # Other Q/V modules continue to use the global rank and alpha above.
    dependency_rank: int | None = None
    dependency_alpha: int | None = None
    # Optional independent overrides for the input token fusion projection
    # and any FFN modules selected by the active target profile.
    fusion_rank: int | None = None
    fusion_alpha: int | None = None
    ffn_rank: int | None = None
    ffn_alpha: int | None = None
    dropout: float = 0.05
    target_modules: tuple[str, ...] = ("q_proj", "v_proj")
    exclude_modules: tuple[str, ...] = ()
    bias: str = "none"

    def validate_local_models(self) -> None:
        required_files = (
            self.base_model_path / "config.json",
            self.base_model_path / "model.safetensors",
            self.tokenizer_path / "config.json",
            self.tokenizer_path / "model.safetensors",
        )
        missing = [path for path in required_files if not path.is_file()]
        if missing:
            formatted = "\n".join(f"- {path}" for path in missing)
            raise FileNotFoundError(f"Missing local pretrained files:\n{formatted}")

        if self.rank <= 0:
            raise ValueError("LoRA rank must be positive.")
        if self.alpha <= 0:
            raise ValueError("LoRA alpha must be positive.")
        if (self.proj_s1_rank is None) != (self.proj_s1_alpha is None):
            raise ValueError(
                "proj_s1_rank and proj_s1_alpha must be configured together."
            )
        if self.proj_s1_rank is not None:
            if self.proj_s1_rank <= 0 or self.proj_s1_alpha <= 0:
                raise ValueError("proj_s1 rank and alpha must be positive.")
            if "proj_s1" not in self.target_modules:
                raise ValueError(
                    "A proj_s1 rank override requires a LoRA target profile "
                    "that includes proj_s1."
                )
        if self.full_finetune_proj_s1 and self.proj_s1_rank is not None:
            raise ValueError(
                "full_finetune_proj_s1 cannot be combined with a proj_s1 "
                "LoRA rank override."
            )
        if (self.dependency_rank is None) != (self.dependency_alpha is None):
            raise ValueError(
                "dependency_rank and dependency_alpha must be configured together."
            )
        if self.dependency_rank is not None:
            if self.dependency_rank <= 0 or self.dependency_alpha <= 0:
                raise ValueError("Dependency rank and alpha must be positive.")
            dependency_modules = {
                "dep_layer.cross_attn.q_proj",
                "dep_layer.cross_attn.v_proj",
            }
            if dependency_modules.intersection(self.exclude_modules):
                raise ValueError(
                    "A Dependency rank override requires a target profile "
                    "that includes dependency cross-attention Q/V."
                )
            if not {"q_proj", "v_proj"}.issubset(self.target_modules):
                raise ValueError(
                    "A Dependency rank override requires q_proj and v_proj "
                    "LoRA targets."
                )
        if (self.fusion_rank is None) != (self.fusion_alpha is None):
            raise ValueError("fusion_rank and fusion_alpha must be configured together.")
        if self.fusion_rank is not None:
            if self.fusion_rank <= 0 or self.fusion_alpha <= 0:
                raise ValueError("Fusion rank and alpha must be positive.")
            if not any(name.endswith("fusion_proj") for name in self.target_modules):
                raise ValueError(
                    "A Fusion rank override requires a target profile that "
                    "includes embedding.fusion_proj."
                )
        if (self.ffn_rank is None) != (self.ffn_alpha is None):
            raise ValueError("ffn_rank and ffn_alpha must be configured together.")
        if self.ffn_rank is not None:
            if self.ffn_rank <= 0 or self.ffn_alpha <= 0:
                raise ValueError("FFN rank and alpha must be positive.")
            if not any(
                name in {"w1", "w2", "w3"} or ".ffn.w" in name
                for name in self.target_modules
            ):
                raise ValueError(
                    "An FFN rank override requires a target profile that "
                    "includes FFN w1/w2/w3 modules."
                )
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("LoRA dropout must be in [0, 1).")
        if not self.target_modules:
            raise ValueError("At least one LoRA target module is required.")


def apply_target_profile(config: QVLoRAConfig, profile: str) -> QVLoRAConfig:
    """Return a config for a named, reproducible LoRA target selection."""

    if profile == BASELINE_QV_PROFILE:
        return replace(
            config,
            target_modules=("q_proj", "v_proj"),
            exclude_modules=(),
        )
    if profile == QV_S1_PROFILE:
        return replace(
            config,
            target_modules=("q_proj", "v_proj", "proj_s1"),
            exclude_modules=(),
        )
    if profile == SELF_ATTN_QV_S1_PROFILE:
        return replace(
            config,
            target_modules=("q_proj", "v_proj", "proj_s1"),
            exclude_modules=(
                "dep_layer.cross_attn.q_proj",
                "dep_layer.cross_attn.v_proj",
            ),
        )
    expanded_attention_targets = (
        "q_proj",
        "k_proj",
        "v_proj",
        "out_proj",
        "fusion_proj",
        "proj_s1",
    )
    if profile == SELF_ATTN_QKVO_FUSION_S1_PROFILE:
        return replace(
            config,
            target_modules=expanded_attention_targets,
            exclude_modules=SELF_ATTN_DEPENDENCY_EXCLUSIONS,
        )
    if profile == SELF_ATTN_QKVO_FUSION_TOP4_FFN_S1_PROFILE:
        return replace(
            config,
            target_modules=(*expanded_attention_targets, *TOP4_FFN_TARGETS),
            exclude_modules=SELF_ATTN_DEPENDENCY_EXCLUSIONS,
        )
    if profile == SELF_ATTN_QKVO_FUSION_FFN_S1_PROFILE:
        return replace(
            config,
            target_modules=(*expanded_attention_targets, "w1", "w2", "w3"),
            exclude_modules=SELF_ATTN_DEPENDENCY_EXCLUSIONS,
        )
    choices = ", ".join(LORA_TARGET_PROFILES)
    raise ValueError(
        f"Unknown LoRA target profile {profile!r}; choose from: {choices}."
    )


def apply_proj_s1_override(
    config: QVLoRAConfig,
    rank: int | None,
    alpha: int | None = None,
) -> QVLoRAConfig:
    """Configure a higher-capacity s1 head while leaving Q/V defaults intact."""
    if rank is None:
        if alpha is not None:
            raise ValueError("--proj-s1-alpha requires --proj-s1-rank.")
        return config
    resolved_alpha = 2 * rank if alpha is None else alpha
    return replace(
        config,
        proj_s1_rank=rank,
        proj_s1_alpha=resolved_alpha,
    )


def apply_dependency_override(
    config: QVLoRAConfig,
    rank: int | None,
    alpha: int | None = None,
) -> QVLoRAConfig:
    """Override Dependency cross-attention Q/V while preserving global rank."""
    if rank is None:
        if alpha is not None:
            raise ValueError("--dependency-lora-alpha requires --dependency-lora-r.")
        return config
    dependency_modules = {
        "dep_layer.cross_attn.q_proj",
        "dep_layer.cross_attn.v_proj",
    }
    if dependency_modules.intersection(config.exclude_modules):
        raise ValueError(
            "--dependency-lora-r requires a target profile that includes "
            "dependency cross-attention Q/V."
        )
    if not {"q_proj", "v_proj"}.issubset(config.target_modules):
        raise ValueError(
            "--dependency-lora-r requires q_proj and v_proj LoRA targets."
        )
    resolved_alpha = 2 * rank if alpha is None else alpha
    return replace(
        config,
        dependency_rank=rank,
        dependency_alpha=resolved_alpha,
    )


def apply_fusion_override(
    config: QVLoRAConfig,
    rank: int | None,
    alpha: int | None = None,
) -> QVLoRAConfig:
    """Override embedding.fusion_proj without changing other LoRA ranks."""
    if rank is None:
        if alpha is not None:
            raise ValueError("--fusion-lora-alpha requires --fusion-lora-r.")
        return config
    if not any(name.endswith("fusion_proj") for name in config.target_modules):
        raise ValueError(
            "--fusion-lora-r requires a target profile that includes "
            "embedding.fusion_proj."
        )
    return replace(
        config,
        fusion_rank=rank,
        fusion_alpha=2 * rank if alpha is None else alpha,
    )


def apply_ffn_override(
    config: QVLoRAConfig,
    rank: int | None,
    alpha: int | None = None,
) -> QVLoRAConfig:
    """Override selected FFN w1/w2/w3 modules independently."""
    if rank is None:
        if alpha is not None:
            raise ValueError("--ffn-lora-alpha requires --ffn-lora-r.")
        return config
    if not any(
        name in {"w1", "w2", "w3"} or ".ffn.w" in name
        for name in config.target_modules
    ):
        raise ValueError(
            "--ffn-lora-r requires a target profile that includes FFN "
            "w1/w2/w3 modules."
        )
    return replace(
        config,
        ffn_rank=rank,
        ffn_alpha=2 * rank if alpha is None else alpha,
    )
