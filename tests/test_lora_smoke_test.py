from lora.smoke_test import parse_args


def test_smoke_cli_accepts_global_lora_rank_and_alpha(monkeypatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "lora.smoke_test",
            "--lora-target-profile",
            "self-attn-qkvo-fusion-top4-ffn-s1",
            "--lora-r",
            "4",
            "--lora-alpha",
            "8",
            "--proj-s1-rank",
            "4",
            "--proj-s1-alpha",
            "8",
            "--fusion-lora-r",
            "8",
            "--fusion-lora-alpha",
            "16",
            "--ffn-lora-r",
            "4",
            "--ffn-lora-alpha",
            "8",
        ],
    )

    args = parse_args()

    assert args.lora_target_profile == "self-attn-qkvo-fusion-top4-ffn-s1"
    assert args.lora_r == 4
    assert args.lora_alpha == 8
    assert args.proj_s1_rank == 4
    assert args.proj_s1_alpha == 8
    assert args.full_finetune_proj_s1 is False
    assert args.fusion_lora_r == 8
    assert args.fusion_lora_alpha == 16
    assert args.ffn_lora_r == 4
    assert args.ffn_lora_alpha == 8


def test_smoke_cli_accepts_full_proj_s1(monkeypatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        ["lora.smoke_test", "--full-finetune-proj-s1"],
    )

    args = parse_args()

    assert args.full_finetune_proj_s1 is True
