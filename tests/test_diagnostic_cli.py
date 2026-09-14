from argparse import Namespace

import pytest
import torch

from finetune_lora.evaluate_base import parse_args as parse_base_args
from finetune_lora.evaluate_adapter import (
    build_s1_conditioning_ids,
    conditioning_display_name,
    evaluate_s2_conditioning,
    parse_args as parse_adapter_args,
    resolve_adapter_path,
    teacher_forcing_gap,
)
from finetune_lora.overfit_validation import (
    parse_args as parse_overfit_args,
    validate_diagnostic_args,
)


def test_base_evaluation_cli_defaults(monkeypatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        ["finetune_lora.evaluate_base", "--token-cache-path", "cache"],
    )

    args = parse_base_args()

    assert args.split == "validation"
    assert args.s2_conditioning == "teacher"
    assert args.constant_s1_id == 0
    assert args.s2_top_k == 5
    assert args.s2_temperature == 1.0
    assert args.context_length == 512
    assert args.target_mode == "all"
    assert args.target_length is None
    assert args.s1_loss_weight == 0.5
    assert args.s2_loss_weight == 0.5


def test_adapter_evaluation_cli_defaults(monkeypatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "finetune_lora.evaluate_adapter",
            "--adapter-path",
            "run",
            "--token-cache-path",
            "cache",
        ],
    )

    args = parse_adapter_args()

    assert args.model_selection == "best"
    assert args.split == "validation"
    assert args.s2_conditioning == "teacher"
    assert args.constant_s1_id == 0
    assert args.s2_top_k == 5
    assert args.s2_temperature == 1.0
    assert args.context_length == 512
    assert args.target_mode == "all"
    assert args.target_length is None
    assert args.s1_loss_weight == 0.5
    assert args.s2_loss_weight == 0.5


def test_resolve_adapter_path_accepts_run_or_direct_directory(tmp_path) -> None:
    best = tmp_path / "best_model"
    best.mkdir()
    (best / "adapter_config.json").write_text("{}", encoding="utf-8")

    assert resolve_adapter_path(tmp_path, "best") == best.resolve()
    assert resolve_adapter_path(best, "last") == best.resolve()


def test_resolve_adapter_path_rejects_missing_adapter(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="adapter_config.json"):
        resolve_adapter_path(tmp_path, "best")


def test_topk_display_name_separates_marginalization_from_top5_accuracy() -> None:
    assert conditioning_display_name(
        "topk",
        {"s2_top_k": 10, "s2_temperature": 1.2},
    ) == "topk(k=10,T=1.2)"
    assert conditioning_display_name("greedy", {}) == "greedy"


def test_conditioning_comparison_reuses_context_and_reports_gap() -> None:
    class Dependency(torch.nn.Module):
        def forward(self, context, sibling, *, is_causal):
            assert is_causal
            return context + sibling

    class Head(torch.nn.Module):
        def cond_forward(self, value):
            return value

    class ConditioningModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embedding = torch.nn.Module()
            self.embedding.emb_s1 = torch.nn.Embedding.from_pretrained(
                torch.tensor([[10.0, 0.0], [0.0, 10.0]])
            )
            self.dep_layer = Dependency()
            self.head = Head()
            self.decode_calls = 0

        def decode_s1(self, s1_ids, _s2_ids, _stamps):
            self.decode_calls += 1
            logits = torch.zeros((*s1_ids.shape, 2))
            logits[..., 0] = 10.0
            return logits, torch.zeros_like(logits)

        def topk_marginalized_s2_log_probs(
            self,
            context,
            _s1_logits,
            *,
            top_k,
            temperature,
            is_causal,
        ):
            assert top_k == 2
            assert temperature == 1.0
            assert is_causal
            return torch.log_softmax(context, dim=-1)

    model = ConditioningModel()
    targets = torch.ones((1, 2), dtype=torch.long)
    batch = (
        torch.zeros((1, 2), dtype=torch.long),
        torch.zeros((1, 2), dtype=torch.long),
        targets,
        targets,
        torch.zeros(1, 2, 5),
        torch.tensor([[True, True]]),
    )

    compared = evaluate_s2_conditioning(
        model,
        loader=[batch],
        device=torch.device("cpu"),
        precision="fp32",
        modes=(
            "teacher",
            "greedy",
            "sample",
            "topk",
            "shuffle",
            "random",
            "constant",
        ),
        s1_loss_weight=0.5,
        s2_loss_weight=0.5,
        s2_top_k=2,
    )
    gap = teacher_forcing_gap(compared)

    assert model.decode_calls == 1
    assert compared["teacher"]["s2_top1_accuracy"] == 1.0
    assert compared["greedy"]["s2_top1_accuracy"] == 0.0
    assert "topk" in compared
    assert gap["s2_loss_delta"] > 0
    assert gap["s2_top1_accuracy_delta"] == -1.0


def test_build_s1_conditioning_ids_supports_ablation_modes() -> None:
    logits = torch.tensor(
        [[[10.0, 0.0, 0.0], [0.0, 10.0, 0.0]]],
    )
    targets = torch.tensor([[2, 1]])

    assert torch.equal(
        build_s1_conditioning_ids(
            "teacher", logits, targets, vocab_size=3
        ),
        targets,
    )
    assert torch.equal(
        build_s1_conditioning_ids(
            "greedy", logits, targets, vocab_size=3
        ),
        torch.tensor([[0, 1]]),
    )
    assert torch.equal(
        build_s1_conditioning_ids(
            "constant", logits, targets, vocab_size=3, constant_s1_id=2
        ),
        torch.full_like(targets, 2),
    )
    for mode in ("sample", "shuffle", "random"):
        ids = build_s1_conditioning_ids(mode, logits, targets, vocab_size=3)
        assert ids.shape == targets.shape
        assert ids.dtype == targets.dtype
        assert ids.min().item() >= 0
        assert ids.max().item() < 3


def test_constant_s1_conditioning_rejects_out_of_range_id() -> None:
    with pytest.raises(ValueError, match="constant_s1_id"):
        build_s1_conditioning_ids(
            "constant",
            torch.zeros(1, 1, 3),
            torch.zeros(1, 1, dtype=torch.long),
            vocab_size=3,
            constant_s1_id=3,
        )


def test_validation_overfit_cli_defaults(monkeypatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        ["finetune_lora.overfit_validation", "--token-cache-path", "cache"],
    )

    args = parse_overfit_args()

    assert args.sample_count == 512
    assert args.epochs == 20
    assert args.lora_target_profile == "qv-s1"
    assert args.proj_s1_rank == 32
    assert args.proj_s1_alpha == 64
    assert args.gradient_conflict_interval == 0


def test_validation_overfit_rejects_empty_subset() -> None:
    args = Namespace(
        sample_count=0,
        start_index=0,
        epochs=1,
        batch_size=1,
        learning_rate=1e-4,
        gradient_accumulation_steps=1,
        max_grad_norm=1.0,
        gradient_conflict_interval=0,
        num_workers=0,
    )

    with pytest.raises(ValueError, match="sample_count must be positive"):
        validate_diagnostic_args(args)
