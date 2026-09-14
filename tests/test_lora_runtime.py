import json
from pathlib import Path

from api.lora_runtime import LoRADeployment, discover_lora_deployments


def _write(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"test")


def test_lora_deployment_requires_complete_inference_bundle(tmp_path: Path) -> None:
    deployment = LoRADeployment(
        adapter_path=tmp_path / "adapters" / "model",
        base_model_path=tmp_path / "pretrained" / "base",
        tokenizer_path=tmp_path / "pretrained" / "tokenizer",
    )
    assert deployment.available is False
    assert deployment.missing_files

    for path in (
        deployment.adapter_path / "adapter_config.json",
        deployment.adapter_path / "adapter_model.safetensors",
        deployment.base_model_path / "config.json",
        deployment.base_model_path / "model.safetensors",
        deployment.tokenizer_path / "config.json",
        deployment.tokenizer_path / "model.safetensors",
    ):
        _write(path)

    assert deployment.available is True
    assert deployment.missing_files == ()
    assert deployment.weight_path == (
        deployment.adapter_path / "adapter_model.safetensors"
    )


def test_discovers_multiple_independent_adapter_directories(tmp_path: Path) -> None:
    base = tmp_path / "base"
    tokenizer = tmp_path / "tokenizer"
    adapters = tmp_path / "adapters"
    for path in (base, tokenizer):
        _write(path / "config.json")
        _write(path / "model.safetensors")
    for name in ("model-a", "model-b"):
        _write(adapters / name / "adapter_config.json")
        _write(adapters / name / "adapter_model.safetensors")

    discovered = discover_lora_deployments(adapters, base, tokenizer)

    assert [key for key, _ in discovered] == ["lora:model-a", "lora:model-b"]


def test_rejects_directory_with_two_adapter_weight_files(tmp_path: Path) -> None:
    deployment = LoRADeployment(
        tmp_path / "adapter", tmp_path / "base", tmp_path / "tokenizer"
    )
    for path in (
        deployment.adapter_path / "adapter_config.json",
        deployment.adapter_path / "adapter_model.safetensors",
        deployment.adapter_path / "adapter_model.bin",
        deployment.base_model_path / "config.json",
        deployment.base_model_path / "model.safetensors",
        deployment.tokenizer_path / "config.json",
        deployment.tokenizer_path / "model.safetensors",
    ):
        _write(path)

    assert deployment.available is False
    assert any("exactly one" in item for item in deployment.missing_files)


def test_deployment_metadata_overrides_legacy_global_base(tmp_path: Path) -> None:
    fallback_base = tmp_path / "pretrained" / "base"
    small_base = tmp_path / "pretrained" / "small"
    fallback_tokenizer = tmp_path / "pretrained" / "legacy-tokenizer"
    tokenizer = tmp_path / "pretrained" / "tokenizer"
    adapter = tmp_path / "adapters" / "small-adapter"
    for path in (fallback_base, small_base, fallback_tokenizer, tokenizer):
        _write(path / "config.json")
        _write(path / "model.safetensors")
    _write(adapter / "adapter_config.json")
    _write(adapter / "adapter_model.safetensors")
    (adapter / "deployment.json").write_text(
        json.dumps(
            {
                "format_version": 1,
                "base_model": {"path": str(small_base)},
                "tokenizer": {"path": str(tokenizer)},
            }
        ),
        encoding="utf-8",
    )

    discovered = discover_lora_deployments(
        tmp_path / "adapters", fallback_base, fallback_tokenizer
    )

    assert len(discovered) == 1
    key, deployment = discovered[0]
    assert key == "lora:small-adapter"
    assert deployment.base_model_path == small_base
    assert deployment.tokenizer_path == tokenizer
    assert deployment.metadata is not None


def test_malformed_deployment_metadata_does_not_use_fallback_base(
    tmp_path: Path,
) -> None:
    base = tmp_path / "base"
    tokenizer = tmp_path / "tokenizer"
    adapter = tmp_path / "adapters" / "broken"
    for path in (base, tokenizer):
        _write(path / "config.json")
        _write(path / "model.safetensors")
    _write(adapter / "adapter_config.json")
    _write(adapter / "adapter_model.safetensors")
    (adapter / "deployment.json").write_text("{}", encoding="utf-8")

    assert discover_lora_deployments(tmp_path / "adapters", base, tokenizer) == ()
