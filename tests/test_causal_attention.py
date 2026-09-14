"""Regression tests for causal attention in every model mode."""

from __future__ import annotations

import pytest
import torch

from model.module import MultiHeadCrossAttentionWithRoPE


@pytest.mark.parametrize("training", [True, False])
def test_cross_attention_future_values_do_not_change_prefix(
    training: bool,
) -> None:
    """Changing future K/V must not affect outputs at earlier positions."""

    torch.manual_seed(7)
    layer = MultiHeadCrossAttentionWithRoPE(
        d_model=16,
        n_heads=4,
        attn_dropout_p=0.0,
        resid_dropout=0.0,
    )
    layer.train(training)

    query = torch.randn(2, 6, 16)
    key = torch.randn(2, 6, 16)
    value = torch.randn(2, 6, 16)
    changed_key = key.clone()
    changed_value = value.clone()
    changed_key[:, 3:] += 20.0
    changed_value[:, 3:] -= 20.0

    original = layer(query, key, value)
    changed = layer(query, changed_key, changed_value)

    torch.testing.assert_close(
        original[:, :3],
        changed[:, :3],
        rtol=1e-5,
        atol=1e-6,
    )
    assert not torch.allclose(original[:, 3:], changed[:, 3:])


def test_incremental_query_can_read_the_entire_available_history() -> None:
    """A one-token incremental query must not be upper-left masked."""

    torch.manual_seed(11)
    layer = MultiHeadCrossAttentionWithRoPE(
        d_model=16,
        n_heads=4,
        attn_dropout_p=0.0,
        resid_dropout=0.0,
    ).eval()
    query = torch.randn(1, 1, 16)
    key = torch.randn(1, 6, 16)
    value = torch.randn(1, 6, 16)
    changed_value = value.clone()
    changed_value[:, -1] += 20.0

    original = layer(query, key, value, is_causal=False)
    changed = layer(query, key, changed_value, is_causal=False)

    assert not torch.allclose(original, changed)
