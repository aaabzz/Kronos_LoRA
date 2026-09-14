"""Validation and path resolution for deployable LoRA inference bundles."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


@dataclass(frozen=True, slots=True)
class LoRADeployment:
    """Files required to serve one trained PEFT adapter for inference."""

    adapter_path: Path
    base_model_path: Path
    tokenizer_path: Path
    metadata: dict[str, Any] | None = None
    metadata_error: str | None = None

    @classmethod
    def from_values(
        cls, adapter_path: str, base_model_path: str, tokenizer_path: str
    ) -> "LoRADeployment":
        return cls(
            adapter_path=project_path(adapter_path),
            base_model_path=project_path(base_model_path),
            tokenizer_path=project_path(tokenizer_path),
        )

    @classmethod
    def from_adapter_directory(
        cls,
        adapter_path: str | Path,
        fallback_base_model_path: str | Path,
        fallback_tokenizer_path: str | Path,
    ) -> "LoRADeployment":
        """Resolve per-adapter metadata, with legacy global-path fallback."""
        resolved_adapter = project_path(adapter_path)
        metadata_path = resolved_adapter / "deployment.json"
        if not metadata_path.is_file():
            return cls.from_values(
                str(resolved_adapter),
                str(fallback_base_model_path),
                str(fallback_tokenizer_path),
            )

        def invalid(reason: str) -> "LoRADeployment":
            fallback = cls.from_values(
                str(resolved_adapter),
                str(fallback_base_model_path),
                str(fallback_tokenizer_path),
            )
            return cls(
                adapter_path=fallback.adapter_path,
                base_model_path=fallback.base_model_path,
                tokenizer_path=fallback.tokenizer_path,
                metadata_error=f"{metadata_path}: {reason}",
            )

        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return invalid("invalid JSON")
        if not isinstance(metadata, dict) or metadata.get("format_version") != 1:
            return invalid("unsupported format_version")
        base = metadata.get("base_model")
        tokenizer = metadata.get("tokenizer")
        if not isinstance(base, dict) or not isinstance(base.get("path"), str):
            return invalid("missing base_model.path")
        if not isinstance(tokenizer, dict) or not isinstance(
            tokenizer.get("path"), str
        ):
            return invalid("missing tokenizer.path")
        return cls(
            adapter_path=resolved_adapter,
            base_model_path=project_path(base["path"]),
            tokenizer_path=project_path(tokenizer["path"]),
            metadata=metadata,
        )

    @property
    def weight_path(self) -> Path | None:
        weights = self.weight_paths
        return weights[0] if len(weights) == 1 else None

    @property
    def weight_paths(self) -> tuple[Path, ...]:
        return tuple(
            candidate
            for candidate in (
                self.adapter_path / "adapter_model.safetensors",
                self.adapter_path / "adapter_model.bin",
            )
            if candidate.is_file()
        )

    @property
    def missing_files(self) -> tuple[str, ...]:
        required = (
            self.adapter_path / "adapter_config.json",
            self.base_model_path / "config.json",
            self.base_model_path / "model.safetensors",
            self.tokenizer_path / "config.json",
            self.tokenizer_path / "model.safetensors",
        )
        missing = [str(path) for path in required if not path.is_file()]
        if self.metadata_error is not None:
            missing.append(self.metadata_error)
        if not self.weight_paths:
            missing.append(
                str(self.adapter_path / "adapter_model.safetensors")
                + " (or adapter_model.bin)"
            )
        elif len(self.weight_paths) > 1:
            missing.append(
                f"{self.adapter_path}: keep exactly one adapter weight file"
            )
        return tuple(missing)

    @property
    def available(self) -> bool:
        return not self.missing_files


def discover_lora_deployments(
    adapters_root: str | Path,
    base_model_path: str | Path,
    tokenizer_path: str | Path,
) -> tuple[tuple[str, LoRADeployment], ...]:
    """Discover complete adapter bundles one directory below ``adapters_root``."""

    root = project_path(adapters_root)
    if not root.is_dir():
        return ()
    deployments: list[tuple[str, LoRADeployment]] = []
    for adapter_path in sorted(root.iterdir(), key=lambda item: item.name.lower()):
        if not adapter_path.is_dir():
            continue
        deployment = LoRADeployment.from_adapter_directory(
            adapter_path, base_model_path, tokenizer_path
        )
        if deployment.available:
            deployments.append((f"lora:{adapter_path.name}", deployment))
    return tuple(deployments)
