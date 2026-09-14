from dataclasses import replace
import json

import numpy as np
import pandas as pd
import pytest
import torch

import finetune_lora.training as training_module
from finetune_lora.config import DataConfig, ExperimentConfig, TrainingConfig
from finetune_lora.train import parse_args
from lora.config import (
    QVLoRAConfig,
    apply_dependency_override,
    apply_ffn_override,
    apply_fusion_override,
    apply_proj_s1_override,
    apply_target_profile,
)
from lora.model_factory import (
    build_full_finetune_model,
    build_rank_alpha_patterns,
    build_qv_lora_model,
    count_parameters,
    find_target_linear_modules,
    initialize_qv_lora_model,
    load_full_finetune_model,
    load_qv_lora_adapter,
)
from model import Kronos
from finetune_lora.dataset import (
    KlineWindowDataset,
    TimeBoundaries,
    resolve_time_boundaries,
)
from finetune_lora.training import (
    build_learning_rate_scheduler,
    create_run_directory,
    DistributedContext,
    LossMetrics,
    evaluate,
    forward_batch,
    hierarchical_token_loss,
    is_significant_validation_improvement,
    lora_gradient_norms,
    shared_transformer_parameters,
    summarize_gradient_conflicts,
    task_gradient_conflict_metrics,
    load_resume_state,
    resolve_precision,
    resolve_resume_checkpoint,
    save_resume_checkpoint,
    should_run_step_validation,
    write_training_plots,
    write_loss_svg,
)


def make_frame(rows: int = 900) -> pd.DataFrame:
    timestamp = pd.date_range("2026-01-01 09:35:00", periods=rows, freq="5min")
    base = np.linspace(10.0, 20.0, rows, dtype=np.float32)
    return pd.DataFrame(
        {
            "symbol": "000001.SZ",
            "timestamp": timestamp,
            "open": base,
            "high": base + 0.2,
            "low": base - 0.2,
            "close": base + 0.1,
            "volume": np.linspace(1_000, 2_000, rows, dtype=np.float32),
            "amount": np.linspace(10_000, 40_000, rows, dtype=np.float32),
        }
    )


def test_explicit_boundaries_and_split_masks() -> None:
    frame = make_frame()
    config = replace(
        DataConfig(),
        context_length=32,
        stride=8,
        train_end=str(frame.loc[499, "timestamp"]),
        validation_end=str(frame.loc[699, "timestamp"]),
    )
    boundaries = resolve_time_boundaries(frame, config)

    train = KlineWindowDataset(frame, config, "train", boundaries)
    validation = KlineWindowDataset(frame, config, "validation", boundaries)
    test = KlineWindowDataset(frame, config, "test", boundaries)

    assert len(train) > 0
    assert len(validation) > 0
    assert len(test) > 0
    assert all(validation[index][2].any() for index in range(len(validation)))
    assert all(test[index][2].any() for index in range(len(test)))

    # Training windows end before validation begins; validation windows end
    # before the test boundary. Context may come from the preceding split.
    for location in train.locations:
        window = train.series[location.symbol].iloc[
            location.start : location.start + train.window_size
        ]
        assert window[config.timestamp_column].max() <= boundaries.train_end
    for location in validation.locations:
        window = validation.series[location.symbol].iloc[
            location.start : location.start + validation.window_size
        ]
        assert window[config.timestamp_column].max() <= boundaries.validation_end


def test_window_shapes_and_normalization_excludes_final_target() -> None:
    frame = make_frame(100)
    config = replace(DataConfig(), context_length=16, stride=16)
    boundaries = TimeBoundaries(frame.loc[60, "timestamp"], frame.loc[80, "timestamp"])
    dataset = KlineWindowDataset(frame, config, "train", boundaries)

    features, stamps, mask = dataset[0]
    assert features.shape == (17, 6)
    assert stamps.shape == (17, 5)
    assert mask.shape == (16,)
    assert features.dtype == torch.float32
    assert mask.dtype == torch.bool
    torch.testing.assert_close(features[:16].mean(dim=0), torch.zeros(6), atol=2e-4, rtol=0)


def test_last_target_mode_supervises_only_the_full_context_prediction() -> None:
    frame = make_frame(100)
    config = replace(
        DataConfig(),
        context_length=16,
        stride=16,
        target_mode="last",
    )
    boundaries = TimeBoundaries(frame.loc[60, "timestamp"], frame.loc[80, "timestamp"])
    dataset = KlineWindowDataset(frame, config, "train", boundaries)

    _, _, mask = dataset[0]

    assert mask.shape == (config.context_length,)
    assert mask.sum().item() == 1
    assert mask[-1].item() is True
    assert not mask[:-1].any()


def test_tail_target_mode_supervises_configured_final_positions() -> None:
    frame = make_frame(100)
    config = replace(
        DataConfig(),
        context_length=16,
        stride=16,
        target_mode="tail",
        target_length=5,
    )
    boundaries = TimeBoundaries(frame.loc[60, "timestamp"], frame.loc[80, "timestamp"])
    dataset = KlineWindowDataset(frame, config, "train", boundaries)

    _, _, mask = dataset[0]

    assert mask.shape == (config.context_length,)
    assert mask.sum().item() == 5
    assert not mask[:-5].any()
    assert mask[-5:].all()


def test_tail_target_mode_validates_target_length() -> None:
    with pytest.raises(ValueError, match="target_length is required"):
        replace(DataConfig(), target_mode="tail").validate()
    with pytest.raises(ValueError, match=r"\[1, context_length\]"):
        replace(
            DataConfig(),
            context_length=16,
            target_mode="tail",
            target_length=17,
        ).validate()
    with pytest.raises(ValueError, match="only valid"):
        replace(DataConfig(), target_mode="all", target_length=1).validate()


def test_hierarchical_loss_respects_mask() -> None:
    torch.manual_seed(1)
    s1_logits = torch.randn(1, 3, 4, requires_grad=True)
    s2_logits = torch.randn(1, 3, 4, requires_grad=True)
    s1_targets = torch.tensor([[0, 1, 2]])
    s2_targets = torch.tensor([[1, 2, 3]])
    mask = torch.tensor([[False, True, True]])

    metrics = hierarchical_token_loss(
        s1_logits, s2_logits, s1_targets, s2_targets, mask
    )
    expected_s1 = torch.nn.functional.cross_entropy(
        s1_logits[mask], s1_targets[mask]
    )
    expected_s2 = torch.nn.functional.cross_entropy(
        s2_logits[mask], s2_targets[mask]
    )

    torch.testing.assert_close(metrics.s1_loss, expected_s1)
    torch.testing.assert_close(metrics.s2_loss, expected_s2)
    torch.testing.assert_close(metrics.loss, (expected_s1 + expected_s2) / 2)
    assert metrics.target_count == 2
    metrics.loss.backward()
    assert s1_logits.grad is not None
    assert s2_logits.grad is not None


def test_hierarchical_loss_supports_normalized_weights() -> None:
    s1_logits = torch.tensor([[[2.0, 0.0]]])
    s2_logits = torch.tensor([[[0.0, 2.0]]])
    targets = torch.tensor([[0]])
    mask = torch.tensor([[True]])

    metrics = hierarchical_token_loss(
        s1_logits,
        s2_logits,
        targets,
        targets,
        mask,
        s1_weight=0.7,
        s2_weight=0.3,
    )

    expected = 0.7 * metrics.s1_loss + 0.3 * metrics.s2_loss
    torch.testing.assert_close(metrics.loss, expected)


def test_hierarchical_loss_computes_masked_topk_and_entropy() -> None:
    logits = torch.tensor(
        [
            [
                [9.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 5.0, 4.0, 3.0, 2.0, 1.0],
                [6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
            ]
        ]
    )
    targets = torch.tensor([[5, 1, 5]])
    mask = torch.tensor([[False, True, True]])

    metrics = hierarchical_token_loss(
        logits,
        logits,
        targets,
        targets,
        mask,
        compute_prediction_metrics=True,
    )

    assert metrics.s1_prediction is not None
    assert metrics.s2_prediction is not None
    assert metrics.s1_prediction.top1_correct == 1
    assert metrics.s1_prediction.top5_correct == 1
    expected_log_probs = torch.log_softmax(logits[mask].float(), dim=-1)
    expected_entropy = -(expected_log_probs.exp() * expected_log_probs).sum()
    assert metrics.s1_prediction.entropy_sum == pytest.approx(
        expected_entropy.item()
    )


def test_evaluate_returns_global_token_metrics() -> None:
    class PerfectModel(torch.nn.Module):
        def forward(
            self,
            _s1_ids,
            _s2_ids,
            _stamps,
            *,
            use_teacher_forcing,
            s1_targets,
        ):
            assert use_teacher_forcing
            logits = 10.0 * torch.nn.functional.one_hot(
                s1_targets, num_classes=6
            ).float()
            return logits, logits

    targets = torch.tensor([[1, 2, 3]])
    batch = (
        torch.tensor([[0, 1, 2]]),
        torch.tensor([[0, 1, 2]]),
        targets,
        targets,
        torch.zeros(1, 3, 5),
        torch.tensor([[False, True, True]]),
    )

    result = evaluate(
        PerfectModel(),
        tokenizer=None,
        loader=[batch],
        device=torch.device("cpu"),
        split="validation",
    )

    assert result["targets"] == 2
    assert result["s1_top1_accuracy"] == 1.0
    assert result["s1_top5_accuracy"] == 1.0
    assert result["s2_top1_accuracy"] == 1.0
    assert result["s2_top5_accuracy"] == 1.0
    assert 0.0 < result["s1_mean_entropy"] < 0.01
    assert result["s2_mean_entropy"] == pytest.approx(
        result["s1_mean_entropy"]
    )


def test_precision_config_and_cpu_auto_fallback() -> None:
    assert resolve_precision("auto", torch.device("cpu")) == "fp32"
    assert resolve_precision("fp32", torch.device("cpu")) == "fp32"
    with pytest.raises(RuntimeError, match="BF16 training requires a CUDA device"):
        resolve_precision("bf16", torch.device("cpu"))

    with pytest.raises(ValueError, match="precision must be one of"):
        replace(TrainingConfig(), precision="fp16").validate()
    with pytest.raises(ValueError, match="warmup_steps cannot be negative"):
        replace(TrainingConfig(), warmup_steps=-1).validate()
    with pytest.raises(ValueError, match="checkpoint_interval must be positive"):
        replace(TrainingConfig(), checkpoint_interval=0).validate()
    replace(TrainingConfig(), validation_interval=0).validate()
    with pytest.raises(ValueError, match="validation_interval cannot be negative"):
        replace(TrainingConfig(), validation_interval=-1).validate()
    with pytest.raises(ValueError, match="gradient_log_interval cannot be negative"):
        replace(TrainingConfig(), gradient_log_interval=-1).validate()
    with pytest.raises(
        ValueError, match="gradient_conflict_interval cannot be negative"
    ):
        replace(TrainingConfig(), gradient_conflict_interval=-1).validate()
    with pytest.raises(ValueError, match="At least one loss weight"):
        replace(TrainingConfig(), s1_loss_weight=0.0, s2_loss_weight=0.0).validate()
    with pytest.raises(ValueError, match="s2_conditioning must be one of"):
        replace(TrainingConfig(), s2_conditioning="random").validate()
    with pytest.raises(ValueError, match="both be positive"):
        replace(
            TrainingConfig(),
            s2_conditioning="mixed-sample",
            s2_teacher_loss_weight=0.0,
            s2_predicted_loss_weight=0.0,
        ).validate()
    with pytest.raises(ValueError, match="s2_top_k must be positive"):
        replace(TrainingConfig(), s2_top_k=0).validate()
    with pytest.raises(ValueError, match="s2_temperature"):
        replace(TrainingConfig(), s2_temperature=float("nan")).validate()
    with pytest.raises(ValueError, match="mutually exclusive"):
        replace(
            TrainingConfig(),
            resume_from="resume",
            initialize_from_adapter="adapter",
        ).validate()
    with pytest.raises(ValueError, match="early_stopping_min_delta cannot be negative"):
        replace(TrainingConfig(), early_stopping_min_delta=-0.1).validate()


def test_step_validation_interval_uses_optimizer_boundaries() -> None:
    assert not should_run_step_validation(1999, 2000, is_epoch_end=False)
    assert should_run_step_validation(2000, 2000, is_epoch_end=False)
    assert not should_run_step_validation(2000, 2000, is_epoch_end=True)
    assert not should_run_step_validation(2000, 0, is_epoch_end=False)


def test_early_stopping_min_delta_requires_significant_decrease() -> None:
    assert is_significant_validation_improvement(0.998, 1.0, 0.001)
    assert is_significant_validation_improvement(0.999, 1.0, 0.001)
    assert not is_significant_validation_improvement(0.9995, 1.0, 0.001)
    assert not is_significant_validation_improvement(1.0, 1.0, 0.0)
    assert is_significant_validation_improvement(1.0, float("inf"), 0.001)


def test_cli_accepts_custom_step_validation_interval(monkeypatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "finetune_lora.train",
            "--validation-interval",
            "750",
            "--warmup-steps",
            "500",
            "--s1-loss-weight",
            "0.7",
            "--s2-loss-weight",
            "0.3",
            "--training-s2-conditioning",
            "mixed-sample",
            "--s2-teacher-loss-weight",
            "0.1",
            "--s2-predicted-loss-weight",
            "0.9",
            "--s2-top-k",
            "7",
            "--s2-temperature",
            "0.8",
            "--early-stopping-min-delta",
            "0.001",
            "--lora-r",
            "16",
            "--lora-alpha",
            "32",
            "--target-mode",
            "tail",
            "--target-length",
            "278",
            "--proj-s1-rank",
            "32",
            "--proj-s1-alpha",
            "64",
            "--dependency-lora-r",
            "8",
            "--dependency-lora-alpha",
            "16",
            "--fusion-lora-r",
            "8",
            "--fusion-lora-alpha",
            "16",
            "--ffn-lora-r",
            "4",
            "--ffn-lora-alpha",
            "8",
            "--gradient-log-interval",
            "25",
            "--gradient-conflict-interval",
            "200",
            "--initialize-from-adapter",
            "stage-one",
            "--initialize-model-selection",
            "last",
        ],
    )

    args = parse_args()
    assert args.validation_interval == 750
    assert args.warmup_steps == 500
    assert args.s1_loss_weight == 0.7
    assert args.s2_loss_weight == 0.3
    assert args.training_s2_conditioning == "mixed-sample"
    assert args.s2_teacher_loss_weight == 0.1
    assert args.s2_predicted_loss_weight == 0.9
    assert args.s2_top_k == 7
    assert args.s2_temperature == 0.8
    assert args.early_stopping_min_delta == 0.001
    assert args.lora_r == 16
    assert args.lora_alpha == 32
    assert args.target_mode == "tail"
    assert args.target_length == 278
    assert args.proj_s1_rank == 32
    assert args.proj_s1_alpha == 64
    assert args.full_finetune_proj_s1 is False
    assert args.dependency_lora_r == 8
    assert args.dependency_lora_alpha == 16
    assert args.fusion_lora_r == 8
    assert args.fusion_lora_alpha == 16
    assert args.ffn_lora_r == 4
    assert args.ffn_lora_alpha == 8
    assert args.gradient_log_interval == 25
    assert args.gradient_conflict_interval == 200
    assert str(args.initialize_from_adapter) == "stage-one"
    assert args.initialize_model_selection == "last"


def test_cli_accepts_full_parameter_mode(monkeypatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        ["finetune_lora.train", "--finetune-mode", "full"],
    )

    args = parse_args()

    assert args.finetune_mode == "full"


def test_full_parameter_mode_rejects_adapter_initialization() -> None:
    with pytest.raises(ValueError, match="only supported"):
        replace(
            TrainingConfig(),
            finetune_mode="full",
            initialize_from_adapter="adapter",
        ).validate()


def test_full_finetune_model_enables_every_parameter(monkeypatch) -> None:
    model = Kronos(
        s1_bits=3,
        s2_bits=3,
        n_layers=1,
        d_model=32,
        n_heads=4,
        ff_dim=64,
        ffn_dropout_p=0.0,
        attn_dropout_p=0.0,
        resid_dropout_p=0.0,
        token_dropout_p=0.0,
        learn_te=False,
    )
    model.requires_grad_(False)
    monkeypatch.setattr(QVLoRAConfig, "validate_local_models", lambda self: None)
    monkeypatch.setattr(
        "lora.model_factory.Kronos.from_pretrained",
        lambda *_args, **_kwargs: model,
    )

    restored = build_full_finetune_model(QVLoRAConfig())

    assert restored is model
    assert all(parameter.requires_grad for parameter in restored.parameters())


def test_full_finetune_checkpoint_round_trip(tmp_path) -> None:
    source = Kronos(
        s1_bits=3,
        s2_bits=3,
        n_layers=1,
        d_model=32,
        n_heads=4,
        ff_dim=64,
        ffn_dropout_p=0.0,
        attn_dropout_p=0.0,
        resid_dropout_p=0.0,
        token_dropout_p=0.0,
        learn_te=False,
    )
    with torch.no_grad():
        source.head.proj_s1.weight.fill_(0.125)
    checkpoint = tmp_path / "full-model"
    source.save_pretrained(checkpoint)

    restored = load_full_finetune_model(checkpoint)

    torch.testing.assert_close(
        restored.head.proj_s1.weight,
        torch.full_like(restored.head.proj_s1.weight, 0.125),
    )


def test_full_finetune_deployment_metadata_marks_complete_model(tmp_path) -> None:
    config = ExperimentConfig(
        training=replace(TrainingConfig(), finetune_mode="full")
    )

    metadata = training_module.deployment_metadata(
        config,
        tmp_path / "full-run",
        {"tokenizer_fingerprint": "tokenizer-hash"},
    )

    assert metadata["model_key"] == "full:full-run"
    assert metadata["adapter"] == {
        "type": "full",
        "target_profile": None,
        "rank": None,
        "alpha": None,
        "full_finetune_proj_s1": True,
    }


def test_learning_rate_scheduler_warms_up_then_decays() -> None:
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.AdamW([parameter], lr=1e-3)
    scheduler = build_learning_rate_scheduler(
        optimizer,
        total_steps=6,
        warmup_steps=2,
    )

    used_learning_rates = []
    for _ in range(6):
        used_learning_rates.append(optimizer.param_groups[0]["lr"])
        optimizer.step()
        scheduler.step()

    assert used_learning_rates[0] == pytest.approx(1e-6)
    assert used_learning_rates[1] > used_learning_rates[0]
    assert used_learning_rates[2] == pytest.approx(1e-3)
    assert used_learning_rates[3:] == sorted(
        used_learning_rates[3:], reverse=True
    )


def test_training_cli_accepts_full_proj_s1(monkeypatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        ["finetune_lora.train", "--full-finetune-proj-s1"],
    )

    args = parse_args()

    assert args.full_finetune_proj_s1 is True
    assert args.proj_s1_rank is None


def test_learning_rate_scheduler_rejects_warmup_covering_training() -> None:
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.AdamW([parameter], lr=1e-3)
    with pytest.raises(ValueError, match="smaller than the total optimizer steps"):
        build_learning_rate_scheduler(
            optimizer,
            total_steps=10,
            warmup_steps=10,
        )


def test_self_attn_qv_s1_profile_excludes_dependency_qv() -> None:
    config = apply_target_profile(QVLoRAConfig(), "self-attn-qv-s1")
    model = Kronos(
        s1_bits=3,
        s2_bits=3,
        n_layers=2,
        d_model=32,
        n_heads=4,
        ff_dim=64,
        ffn_dropout_p=0.0,
        attn_dropout_p=0.0,
        resid_dropout_p=0.0,
        token_dropout_p=0.0,
        learn_te=False,
    )

    matched = find_target_linear_modules(
        model,
        config.target_modules,
        config.exclude_modules,
    )

    assert matched == [
        "transformer.0.self_attn.q_proj",
        "transformer.0.self_attn.v_proj",
        "transformer.1.self_attn.q_proj",
        "transformer.1.self_attn.v_proj",
        "head.proj_s1",
    ]
    assert not any(name.startswith("dep_layer.") for name in matched)


def test_qv_s1_profile_keeps_dependency_qv_and_adds_s1_head() -> None:
    config = apply_target_profile(QVLoRAConfig(), "qv-s1")
    model = Kronos(
        s1_bits=3,
        s2_bits=3,
        n_layers=2,
        d_model=32,
        n_heads=4,
        ff_dim=64,
        ffn_dropout_p=0.0,
        attn_dropout_p=0.0,
        resid_dropout_p=0.0,
        token_dropout_p=0.0,
        learn_te=False,
    )

    matched = find_target_linear_modules(
        model,
        config.target_modules,
        config.exclude_modules,
    )

    assert matched == [
        "transformer.0.self_attn.q_proj",
        "transformer.0.self_attn.v_proj",
        "transformer.1.self_attn.q_proj",
        "transformer.1.self_attn.v_proj",
        "dep_layer.cross_attn.q_proj",
        "dep_layer.cross_attn.v_proj",
        "head.proj_s1",
    ]
    assert config.exclude_modules == ()


def test_expanded_s1_profile_targets_self_attention_and_fusion_only() -> None:
    config = apply_target_profile(
        QVLoRAConfig(),
        "self-attn-qkvo-fusion-s1",
    )
    model = Kronos(
        s1_bits=3,
        s2_bits=3,
        n_layers=2,
        d_model=32,
        n_heads=4,
        ff_dim=64,
        ffn_dropout_p=0.0,
        attn_dropout_p=0.0,
        resid_dropout_p=0.0,
        token_dropout_p=0.0,
        learn_te=False,
    )

    matched = find_target_linear_modules(
        model,
        config.target_modules,
        config.exclude_modules,
    )

    assert "embedding.fusion_proj" in matched
    assert "head.proj_s1" in matched
    for layer in range(2):
        for projection in ("q_proj", "k_proj", "v_proj", "out_proj"):
            assert f"transformer.{layer}.self_attn.{projection}" in matched
    assert not any(name.startswith("dep_layer.") for name in matched)
    assert not any(".ffn." in name for name in matched)


def test_top4_ffn_profile_does_not_touch_lower_ffn_layers() -> None:
    config = apply_target_profile(
        QVLoRAConfig(),
        "self-attn-qkvo-fusion-top4-ffn-s1",
    )
    model = Kronos(
        s1_bits=3,
        s2_bits=3,
        n_layers=12,
        d_model=32,
        n_heads=4,
        ff_dim=64,
        ffn_dropout_p=0.0,
        attn_dropout_p=0.0,
        resid_dropout_p=0.0,
        token_dropout_p=0.0,
        learn_te=False,
    )

    matched = find_target_linear_modules(
        model,
        config.target_modules,
        config.exclude_modules,
    )
    ffn_modules = [name for name in matched if ".ffn." in name]

    assert len(ffn_modules) == 12
    assert all(int(name.split(".")[1]) >= 8 for name in ffn_modules)


def test_fusion_and_ffn_rank_overrides_are_independent() -> None:
    config = apply_target_profile(
        QVLoRAConfig(),
        "self-attn-qkvo-fusion-top4-ffn-s1",
    )
    config = apply_fusion_override(config, rank=8)
    config = apply_ffn_override(config, rank=4)

    rank_pattern, alpha_pattern = build_rank_alpha_patterns(config)

    assert rank_pattern["embedding.fusion_proj"] == 8
    assert alpha_pattern["embedding.fusion_proj"] == 16
    ffn_patterns = {name: rank for name, rank in rank_pattern.items() if ".ffn." in name}
    assert len(ffn_patterns) == 12
    assert set(ffn_patterns.values()) == {4}
    assert all(alpha_pattern[name] == 8 for name in ffn_patterns)


def test_fusion_and_ffn_overrides_require_matching_profiles() -> None:
    config = apply_target_profile(QVLoRAConfig(), "self-attn-qv-s1")
    with pytest.raises(ValueError, match="includes embedding.fusion_proj"):
        apply_fusion_override(config, rank=8)
    with pytest.raises(ValueError, match="includes FFN"):
        apply_ffn_override(config, rank=4)


def test_proj_s1_rank_override_leaves_global_qv_rank_unchanged() -> None:
    config = apply_target_profile(QVLoRAConfig(), "qv-s1")
    config = apply_proj_s1_override(config, rank=32)

    rank_pattern, alpha_pattern = build_rank_alpha_patterns(config)

    assert config.rank == 8
    assert config.alpha == 16
    assert config.proj_s1_rank == 32
    assert config.proj_s1_alpha == 64
    assert rank_pattern == {"head.proj_s1": 32}
    assert alpha_pattern == {"head.proj_s1": 64}


def test_proj_s1_alpha_requires_rank() -> None:
    with pytest.raises(ValueError, match="requires --proj-s1-rank"):
        apply_proj_s1_override(QVLoRAConfig(), rank=None, alpha=64)


def test_full_proj_s1_rejects_lora_rank_override() -> None:
    config = apply_target_profile(QVLoRAConfig(), "self-attn-qv-s1")
    config = replace(
        config,
        full_finetune_proj_s1=True,
        proj_s1_rank=8,
        proj_s1_alpha=16,
    )
    with pytest.raises(ValueError, match="cannot be combined"):
        config.validate_local_models()


def test_full_proj_s1_is_trainable_and_saved_with_adapter(
    monkeypatch, tmp_path
) -> None:
    def tiny_model() -> Kronos:
        return Kronos(
            s1_bits=3,
            s2_bits=3,
            n_layers=1,
            d_model=32,
            n_heads=4,
            ff_dim=64,
            ffn_dropout_p=0.0,
            attn_dropout_p=0.0,
            resid_dropout_p=0.0,
            token_dropout_p=0.0,
            learn_te=False,
        )

    source_base = tiny_model()
    restored_base = tiny_model()
    models = iter((source_base, restored_base))
    monkeypatch.setattr(QVLoRAConfig, "validate_local_models", lambda self: None)
    monkeypatch.setattr(
        "lora.model_factory.Kronos.from_pretrained",
        lambda *_args, **_kwargs: next(models),
    )
    config = replace(
        apply_target_profile(QVLoRAConfig(), "self-attn-qv-s1"),
        full_finetune_proj_s1=True,
    )

    model, matched = build_qv_lora_model(config)
    head = model.base_model.model.head.proj_s1
    assert "head.proj_s1" in matched
    assert not hasattr(head, "lora_A")
    full_weight = head.modules_to_save.default.weight
    assert full_weight.requires_grad
    with torch.no_grad():
        full_weight.fill_(0.125)
    trainable, _ = count_parameters(model)
    assert trainable > full_weight.numel()

    adapter_dir = tmp_path / "adapter"
    model.save_pretrained(adapter_dir)
    adapter_config = json.loads(
        (adapter_dir / "adapter_config.json").read_text(encoding="utf-8")
    )
    assert adapter_config["base_model_name_or_path"] == str(config.base_model_path)
    restored = load_qv_lora_adapter(config, adapter_dir)
    restored_weight = (
        restored.base_model.model.head.proj_s1.modules_to_save.default.weight
    )
    torch.testing.assert_close(
        restored_weight,
        torch.full_like(restored_weight, 0.125),
    )


def test_dependency_rank_override_is_independent_from_global_and_s1() -> None:
    config = replace(QVLoRAConfig(), rank=16, alpha=32)
    config = apply_target_profile(config, "qv-s1")
    config = apply_proj_s1_override(config, rank=32, alpha=64)
    config = apply_dependency_override(config, rank=8, alpha=16)

    rank_pattern, alpha_pattern = build_rank_alpha_patterns(config)

    assert config.rank == 16
    assert config.alpha == 32
    assert config.dependency_rank == 8
    assert config.dependency_alpha == 16
    assert rank_pattern == {
        "head.proj_s1": 32,
        "dep_layer.cross_attn.q_proj": 8,
        "dep_layer.cross_attn.v_proj": 8,
    }
    assert alpha_pattern == {
        "head.proj_s1": 64,
        "dep_layer.cross_attn.q_proj": 16,
        "dep_layer.cross_attn.v_proj": 16,
    }


def test_dependency_override_rejects_excluded_profile_and_alpha_only() -> None:
    excluded = apply_target_profile(QVLoRAConfig(), "self-attn-qv-s1")
    with pytest.raises(ValueError, match="includes dependency"):
        apply_dependency_override(excluded, rank=8)
    with pytest.raises(ValueError, match="requires --dependency-lora-r"):
        apply_dependency_override(QVLoRAConfig(), rank=None, alpha=16)


def test_peft_applies_independent_dependency_rank_pattern() -> None:
    from peft import LoraConfig, get_peft_model

    model = Kronos(
        s1_bits=3,
        s2_bits=3,
        n_layers=1,
        d_model=32,
        n_heads=4,
        ff_dim=64,
        ffn_dropout_p=0.0,
        attn_dropout_p=0.0,
        resid_dropout_p=0.0,
        token_dropout_p=0.0,
        learn_te=False,
    )
    config = replace(QVLoRAConfig(), rank=16, alpha=32)
    config = apply_target_profile(config, "qv-s1")
    config = apply_proj_s1_override(config, rank=32, alpha=64)
    config = apply_dependency_override(config, rank=8, alpha=16)
    rank_pattern, alpha_pattern = build_rank_alpha_patterns(config)
    adapted = get_peft_model(
        model,
        LoraConfig(
            r=config.rank,
            lora_alpha=config.alpha,
            rank_pattern=rank_pattern,
            alpha_pattern=alpha_pattern,
            target_modules=list(config.target_modules),
        ),
    )

    backbone = adapted.base_model.model
    assert (
        backbone.transformer[0].self_attn.q_proj.lora_A.default.out_features
        == 16
    )
    assert backbone.dep_layer.cross_attn.q_proj.lora_A.default.out_features == 8
    assert backbone.dep_layer.cross_attn.v_proj.lora_A.default.out_features == 8
    assert backbone.head.proj_s1.lora_A.default.out_features == 32


def test_gradient_norms_are_grouped_by_model_path() -> None:
    class GradientModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.transformer = torch.nn.Linear(2, 2, bias=False)
            self.dep_layer = torch.nn.Linear(2, 2, bias=False)
            self.head = torch.nn.Module()
            self.head.proj_s1 = torch.nn.Linear(2, 2, bias=False)

    model = GradientModel()
    loss = sum(parameter.sum() for parameter in model.parameters())
    loss.backward()

    norms = lora_gradient_norms(model)

    assert norms["grad_norm_total"] > 0
    assert norms["grad_norm_proj_s1"] > 0
    assert norms["grad_norm_transformer"] > 0
    assert norms["grad_norm_dependency"] > 0
    assert norms["grad_norm_other"] == 0


def test_task_gradient_conflict_metrics_use_shared_transformer_only() -> None:
    class SameTaskModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.transformer = torch.nn.Linear(2, 2, bias=False)
            self.dep_layer = torch.nn.Linear(2, 2, bias=False)
            self.head = torch.nn.Module()
            self.head.proj_s1 = torch.nn.Linear(2, 2, bias=False)

        def forward(
            self,
            s1_ids,
            _s2_ids,
            _stamps,
            *,
            use_teacher_forcing,
            s1_targets,
        ):
            assert use_teacher_forcing and s1_targets is not None
            features = torch.nn.functional.one_hot(s1_ids, 2).float()
            logits = self.transformer(features)
            return logits, logits

    model = SameTaskModel()
    batch = (
        torch.tensor([[0, 1]]),
        torch.tensor([[1, 0]]),
        torch.tensor([[1, 0]]),
        torch.tensor([[1, 0]]),
        torch.zeros(1, 2, 5),
        torch.tensor([[True, True]]),
    )
    context = DistributedContext(False, 0, 0, 1, torch.device("cpu"))

    selected = shared_transformer_parameters(model)
    result = task_gradient_conflict_metrics(
        model,
        tokenizer=None,
        batches=[batch, batch],
        device=torch.device("cpu"),
        precision="fp32",
        distributed=context,
        s1_weight=0.5,
        s2_weight=0.5,
    )

    assert [name for name, _ in selected] == ["transformer.weight"]
    assert result["transformer_s1_grad_norm"] > 0
    assert result["transformer_s2_grad_norm"] == pytest.approx(
        result["transformer_s1_grad_norm"]
    )
    assert result["transformer_s2_to_s1_ratio"] == pytest.approx(1.0)
    assert result["effective_s2_to_s1_ratio"] == pytest.approx(1.0)
    assert result["transformer_grad_cosine"] == pytest.approx(1.0)
    assert all(parameter.grad is None for parameter in model.parameters())


def test_gradient_conflict_summary_reports_direction_frequencies() -> None:
    records = [
        {
            "split": "gradient_conflict",
            "transformer_grad_cosine": value,
            "transformer_s2_to_s1_ratio": ratio,
        }
        for value, ratio in [(-0.2, 2.0), (-0.05, 1.0), (0.3, 0.5)]
    ]

    summary = summarize_gradient_conflicts(records)

    assert summary is not None
    assert summary["sample_count"] == 3
    assert summary["median_cosine"] == pytest.approx(-0.05)
    assert summary["negative_cosine_ratio"] == pytest.approx(2 / 3)
    assert summary["strong_conflict_ratio"] == pytest.approx(1 / 3)
    assert summary["median_s2_to_s1_ratio"] == pytest.approx(1.0)


def test_create_run_directory_refuses_explicit_overwrite(tmp_path) -> None:
    context = DistributedContext(
        enabled=False,
        rank=0,
        local_rank=0,
        world_size=1,
        device=torch.device("cpu"),
    )
    run_dir = create_run_directory(
        tmp_path,
        context,
        requested_name="baseline_r8_s128",
        lora_rank=8,
        stride=128,
    )

    assert run_dir == tmp_path / "runs" / "baseline_r8_s128"
    assert (run_dir / "plots").is_dir()
    with pytest.raises(RuntimeError, match="will not be overwritten"):
        create_run_directory(
            tmp_path,
            context,
            requested_name="baseline_r8_s128",
            lora_rank=8,
            stride=128,
        )


def test_create_run_directory_auto_name_never_overwrites(
    tmp_path, monkeypatch
) -> None:
    context = DistributedContext(
        enabled=False,
        rank=0,
        local_rank=0,
        world_size=1,
        device=torch.device("cpu"),
    )

    class FixedDateTime:
        @staticmethod
        def now():
            return pd.Timestamp("2026-09-07 14:30:00")

    monkeypatch.setattr(training_module, "datetime", FixedDateTime)
    first = create_run_directory(
        tmp_path,
        context,
        requested_name=None,
        lora_rank=8,
        stride=128,
    )
    second = create_run_directory(
        tmp_path,
        context,
        requested_name=None,
        lora_rank=8,
        stride=128,
    )

    assert first.name == "20260907_143000_r8_s128"
    assert second.name == "20260907_143000_r8_s128_01"


def test_write_loss_svg_contains_train_and_validation_series(tmp_path) -> None:
    output = tmp_path / "plots" / "loss.svg"
    write_loss_svg(
        output,
        [
            {"split": "train", "global_step": 50, "loss": 1.2},
            {"split": "train", "global_step": 100, "loss": 0.9},
            {"split": "validation", "global_step": 100, "loss": 1.0},
        ],
    )

    svg = output.read_text(encoding="utf-8")
    assert "<svg" in svg
    assert "train" in svg
    assert "validation" in svg


def test_write_training_plots_creates_loss_and_learning_rate_png(tmp_path) -> None:
    write_training_plots(
        tmp_path,
        [
            {
                "split": "train",
                "global_step": 50,
                "loss": 1.2,
                "s1_loss": 1.4,
                "s2_loss": 1.0,
                "learning_rate": 1e-4,
            },
            {
                "split": "validation",
                "global_step": 50,
                "loss": 1.0,
                "s1_loss": 1.2,
                "s2_loss": 0.8,
                "learning_rate": 1e-4,
                "s1_top1_accuracy": 0.1,
                "s1_top5_accuracy": 0.4,
                "s2_top1_accuracy": 0.3,
                "s2_top5_accuracy": 0.8,
                "s1_mean_entropy": 2.5,
                "s2_mean_entropy": 1.2,
                "predicted_s2_loss": 0.8,
                "teacher_s2_loss": 0.6,
                "mixed_s2_loss": 0.78,
                "teacher_s2_top1_accuracy": 0.5,
                "teacher_s2_top5_accuracy": 0.9,
                "teacher_s2_mean_entropy": 1.0,
            },
            {
                "split": "gradient",
                "global_step": 50,
                "grad_norm_total": 1.0,
                "grad_norm_proj_s1": 0.2,
                "grad_norm_transformer": 0.8,
                "grad_norm_dependency": 0.4,
            },
            {
                "split": "gradient_conflict",
                "global_step": 50,
                "transformer_s1_grad_norm": 0.3,
                "transformer_s2_grad_norm": 0.7,
                "transformer_grad_cosine": -0.2,
            },
        ],
    )

    assert (tmp_path / "loss.png").read_bytes().startswith(b"\x89PNG")
    assert (tmp_path / "learning_rate.png").read_bytes().startswith(b"\x89PNG")
    assert (tmp_path / "component_loss.png").read_bytes().startswith(b"\x89PNG")
    assert (tmp_path / "gradient_norms.png").read_bytes().startswith(b"\x89PNG")
    assert (tmp_path / "gradient_conflict.png").read_bytes().startswith(b"\x89PNG")
    assert (tmp_path / "token_accuracy.png").read_bytes().startswith(b"\x89PNG")
    assert (tmp_path / "token_entropy.png").read_bytes().startswith(b"\x89PNG")
    assert (tmp_path / "conditioning_loss.png").read_bytes().startswith(b"\x89PNG")


def test_complete_resume_checkpoint_round_trip(tmp_path) -> None:
    class DummyAdapter(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(1.0))

        def save_pretrained(self, path) -> None:
            path.mkdir(parents=True, exist_ok=True)
            (path / "adapter_model.safetensors").write_bytes(b"adapter")
            (path / "adapter_config.json").write_text("{}", encoding="utf-8")

    context = DistributedContext(
        enabled=False,
        rank=0,
        local_rank=0,
        world_size=1,
        device=torch.device("cpu"),
    )
    run_dir = tmp_path / "runs" / "experiment"
    (run_dir / "resume").mkdir(parents=True)
    model = DummyAdapter()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    signature = {"dataset": "fixed"}

    checkpoint = save_resume_checkpoint(
        model,
        optimizer,
        scheduler,
        run_dir,
        context,
        epoch=2,
        next_batch_index=17,
        global_step=2000,
        best_validation_loss=0.8,
        early_stopping_reference_loss=0.85,
        stale_epochs=1,
        validation_improved_this_epoch=False,
        history_count=1,
        metric_count=5,
        running_loss=2.5,
        running_s1_loss=3.0,
        running_s2_loss=2.0,
        running_batches=2,
        signature=signature,
    )

    assert checkpoint.name == "step-00002000"
    assert (checkpoint / "optimizer.pt").is_file()
    assert (checkpoint / "scheduler.pt").is_file()
    assert (checkpoint / "rng_state_rank-0000.pt").is_file()
    resolved_checkpoint, resolved_run = resolve_resume_checkpoint(
        run_dir,
        tmp_path,
        context,
        requested_run_name="experiment",
    )
    assert resolved_checkpoint == checkpoint.resolve()
    assert resolved_run == run_dir.resolve()

    state, running_loss, running_s1_loss, running_s2_loss, running_batches = (
        load_resume_state(
            checkpoint,
            optimizer,
            scheduler,
            context,
            expected_signature=signature,
        )
    )
    assert state["global_step"] == 2000
    assert state["next_batch_index"] == 17
    assert state["early_stopping_reference_loss"] == 0.85
    assert running_loss == 2.5
    assert running_s1_loss == 3.0
    assert running_s2_loss == 2.0
    assert running_batches == 2
    rank_state_path = checkpoint / "rng_state_rank-0000.pt"
    legacy_rank_state = torch.load(
        rank_state_path, map_location="cpu", weights_only=False
    )
    legacy_rank_state.pop("running_s1_loss")
    legacy_rank_state.pop("running_s2_loss")
    torch.save(legacy_rank_state, rank_state_path)
    _, legacy_loss, legacy_s1, legacy_s2, legacy_batches = load_resume_state(
        checkpoint,
        optimizer,
        scheduler,
        context,
        expected_signature=signature,
    )
    assert (legacy_loss, legacy_s1, legacy_s2, legacy_batches) == (0.0, 0.0, 0.0, 0)
    legacy_state, _, _, _, _ = load_resume_state(
        checkpoint,
        optimizer,
        scheduler,
        context,
        expected_signature={"dataset": "fixed", "validation_interval": 0},
    )
    assert legacy_state["global_step"] == 2000
    warmup_legacy_state, _, _, _, _ = load_resume_state(
        checkpoint,
        optimizer,
        scheduler,
        context,
        expected_signature={"dataset": "fixed", "warmup_steps": 0},
    )
    assert warmup_legacy_state["global_step"] == 2000
    full_head_legacy_state, _, _, _, _ = load_resume_state(
        checkpoint,
        optimizer,
        scheduler,
        context,
        expected_signature={"dataset": "fixed", "full_finetune_proj_s1": False},
    )
    assert full_head_legacy_state["global_step"] == 2000
    with pytest.raises(ValueError, match="does not match the checkpoint"):
        load_resume_state(
            checkpoint,
            optimizer,
            scheduler,
            context,
            expected_signature={"dataset": "fixed", "warmup_steps": 500},
        )
    with pytest.raises(ValueError, match="does not match the checkpoint"):
        load_resume_state(
            checkpoint,
            optimizer,
            scheduler,
            context,
            expected_signature={"dataset": "fixed", "validation_interval": 2000},
        )
    with pytest.raises(ValueError, match="does not match the checkpoint"):
        load_resume_state(
            checkpoint,
            optimizer,
            scheduler,
            context,
            expected_signature={"dataset": "changed"},
        )


def test_forward_batch_accepts_pre_tokenized_shards_without_tokenizer() -> None:
    class DummyModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.scale = torch.nn.Parameter(torch.tensor(1.0))

        def forward(self, s1, s2, stamps, **_kwargs):
            batch, length = s1.shape
            logits = self.scale * torch.ones(batch, length, 8)
            return logits, logits.clone()

    batch = (
        torch.tensor([[0, 1, 2]]),
        torch.tensor([[1, 2, 3]]),
        torch.tensor([[1, 2, 3]]),
        torch.tensor([[2, 3, 4]]),
        torch.zeros(1, 3, 5),
        torch.tensor([[True, True, True]]),
    )
    model = DummyModel()
    metrics = forward_batch(
        model,
        tokenizer=None,
        batch=batch,
        device=torch.device("cpu"),
    )
    assert torch.isfinite(metrics.loss)
    assert metrics.target_count == 3
    metrics.loss.backward()
    assert model.scale.grad is not None


def test_forward_batch_passes_explicit_s2_conditioning() -> None:
    class CapturingModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.scale = torch.nn.Parameter(torch.tensor(1.0))
            self.conditioning = None

        def forward(self, s1, _s2, _stamps, **kwargs):
            self.conditioning = kwargs["s2_conditioning"]
            batch, length = s1.shape
            logits = self.scale * torch.ones(batch, length, 8)
            return logits, logits.clone()

    batch = (
        torch.tensor([[0, 1]]),
        torch.tensor([[1, 2]]),
        torch.tensor([[1, 2]]),
        torch.tensor([[2, 3]]),
        torch.zeros(1, 2, 5),
        torch.tensor([[True, True]]),
    )
    model = CapturingModel()

    forward_batch(
        model,
        tokenizer=None,
        batch=batch,
        device=torch.device("cpu"),
        s2_conditioning="greedy",
    )

    assert model.conditioning == "greedy"


def test_mixed_s2_loss_uses_two_branches_and_predicted_validation() -> None:
    class MixedModel(torch.nn.Module):
        def forward(self, _s1, _s2, _stamps, **kwargs):
            assert kwargs["s2_conditioning"] == "mixed-greedy"
            targets = kwargs["s1_targets"]
            s1_logits = 10.0 * torch.nn.functional.one_hot(
                targets, num_classes=4
            ).float()
            predicted_s2_logits = torch.zeros_like(s1_logits)
            teacher_s2_logits = s1_logits.clone()
            return s1_logits, predicted_s2_logits, teacher_s2_logits

    targets = torch.tensor([[1, 2]])
    batch = (
        torch.tensor([[0, 1]]),
        torch.tensor([[1, 2]]),
        targets,
        targets,
        torch.zeros(1, 2, 5),
        torch.tensor([[True, True]]),
    )
    metrics = forward_batch(
        MixedModel(),
        tokenizer=None,
        batch=batch,
        device=torch.device("cpu"),
        s2_conditioning="mixed-greedy",
        s2_teacher_loss_weight=0.1,
        s2_predicted_loss_weight=0.9,
        compute_prediction_metrics=True,
    )

    assert metrics.teacher_s2_loss is not None
    assert metrics.predicted_s2_loss is not None
    expected_mixed = (
        0.1 * metrics.teacher_s2_loss + 0.9 * metrics.predicted_s2_loss
    )
    torch.testing.assert_close(metrics.s2_loss, expected_mixed)
    assert metrics.teacher_s2_prediction is not None
    assert metrics.teacher_s2_prediction.top1_correct == 2

    evaluated = evaluate(
        MixedModel(),
        tokenizer=None,
        loader=[batch],
        device=torch.device("cpu"),
        split="validation",
        s2_conditioning="mixed-greedy",
        s2_teacher_loss_weight=0.1,
        s2_predicted_loss_weight=0.9,
    )
    assert evaluated["s2_loss"] == pytest.approx(
        metrics.predicted_s2_loss.item()
    )
    assert evaluated["mixed_s2_loss"] == pytest.approx(metrics.s2_loss.item())
    assert evaluated["teacher_s2_top1_accuracy"] == 1.0
    assert evaluated["predicted_minus_teacher_s2_loss_delta"] > 0


def test_kronos_forward_supports_explicit_predicted_s1_conditioning() -> None:
    model = Kronos(
        s1_bits=3,
        s2_bits=3,
        n_layers=1,
        d_model=32,
        n_heads=4,
        ff_dim=64,
        ffn_dropout_p=0.0,
        attn_dropout_p=0.0,
        resid_dropout_p=0.0,
        token_dropout_p=0.0,
        learn_te=False,
    ).eval()
    s1 = torch.zeros(1, 4, dtype=torch.long)
    s2 = torch.zeros_like(s1)

    predicted_outputs = {}
    for conditioning in ("greedy", "sample", "topk"):
        s1_logits, s2_logits = model(
            s1,
            s2,
            s2_conditioning=conditioning,
            s2_top_k=3,
        )
        predicted_outputs[conditioning] = s2_logits
        assert s1_logits.shape == (1, 4, 8)
        assert s2_logits.shape == (1, 4, 8)

    _, greedy_logits = model(s1, s2, s2_conditioning="greedy")
    _, top1_log_probs = model(
        s1,
        s2,
        s2_conditioning="topk",
        s2_top_k=1,
    )
    torch.testing.assert_close(
        top1_log_probs.exp(),
        greedy_logits.float().softmax(dim=-1),
    )
    torch.testing.assert_close(
        torch.logsumexp(predicted_outputs["topk"], dim=-1),
        torch.zeros_like(predicted_outputs["topk"][..., 0]),
        atol=1e-6,
        rtol=1e-6,
    )

    for conditioning in ("mixed-greedy", "mixed-sample", "mixed-topk"):
        outputs = model(
            s1,
            s2,
            s1_targets=s1,
            s2_conditioning=conditioning,
            s2_top_k=3,
        )
        assert len(outputs) == 3
        assert all(output.shape == (1, 4, 8) for output in outputs)

    with pytest.raises(ValueError, match="requires s1_targets"):
        model(s1, s2, s2_conditioning="teacher")
    with pytest.raises(ValueError, match="top_k"):
        model(s1, s2, s2_conditioning="topk", s2_top_k=0)


def test_initialize_adapter_copies_subset_into_fresh_superset(
    tmp_path, monkeypatch
) -> None:
    from peft import LoraConfig, get_peft_model

    class TinyModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.q_proj = torch.nn.Linear(4, 4, bias=False)
            self.v_proj = torch.nn.Linear(4, 4, bias=False)

        def forward(self, value):
            return self.q_proj(value) + self.v_proj(value)

    source = get_peft_model(
        TinyModel(),
        LoraConfig(r=2, lora_alpha=4, target_modules=["q_proj"]),
    )
    source.base_model.model.q_proj.lora_A.default.weight.data.fill_(0.25)
    source.base_model.model.q_proj.lora_B.default.weight.data.fill_(0.5)
    source.save_pretrained(tmp_path)

    destination = get_peft_model(
        TinyModel(),
        LoraConfig(
            r=2,
            lora_alpha=4,
            target_modules=["q_proj", "v_proj"],
        ),
    )
    monkeypatch.setattr(
        "lora.model_factory.build_qv_lora_model",
        lambda _config: (destination, ["q_proj", "v_proj"]),
    )

    initialized, matched, source_directory, tensor_count = (
        initialize_qv_lora_model(object(), tmp_path)
    )

    assert initialized is destination
    assert matched == ["q_proj", "v_proj"]
    assert source_directory == tmp_path.resolve()
    assert tensor_count == 2
    torch.testing.assert_close(
        destination.base_model.model.q_proj.lora_A.default.weight,
        torch.full((2, 4), 0.25),
    )
    torch.testing.assert_close(
        destination.base_model.model.q_proj.lora_B.default.weight,
        torch.full((4, 2), 0.5),
    )
    torch.testing.assert_close(
        destination.base_model.model.v_proj.lora_B.default.weight,
        torch.zeros((4, 2)),
    )


def test_evaluate_reports_split_epoch_and_batch_for_non_finite_loss(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class DummyModel:
        def eval(self):
            return self

    def non_finite_forward(*_args, **_kwargs) -> LossMetrics:
        value = torch.tensor(float("nan"))
        return LossMetrics(value, value, value, target_count=1)

    monkeypatch.setattr(training_module, "forward_batch", non_finite_forward)

    with pytest.raises(
        FloatingPointError,
        match=r"Non-finite validation loss at batch=1, epoch=4\.",
    ):
        evaluate(
            DummyModel(),
            tokenizer=None,
            loader=[object()],
            device=torch.device("cpu"),
            split="validation",
            epoch=4,
        )
