import numpy as np
import pandas as pd
import pytest
import torch

import kronos_prediction_modes.predictor as predictor_module
from kronos_prediction_modes.predictor import (
    _next_token,
    next_bar_probabilities,
    predict_modes,
)


def test_greedy_always_selects_one_highest_probability_path() -> None:
    logits = torch.tensor(
        [
            [0.1, 2.0, 0.3],
            [4.0, 1.0, 3.0],
        ]
    )

    selected = _next_token(
        logits,
        "greedy",
        temperature=1.0,
        top_k=0,
        top_p=1.0,
    )

    assert selected.shape == (2, 1)
    assert selected.tolist() == [[1], [0]]


def test_probabilities_use_only_first_future_bar() -> None:
    paths = np.zeros((4, 2, 6), dtype=np.float32)
    # First future close: one up, one down, and two within the flat tolerance.
    paths[:, 0, 3] = [10.02, 9.98, 10.00, 10.004]
    # These later closes must not affect the next-bar probabilities.
    paths[:, 1, 3] = [1.0, 100.0, 1.0, 100.0]

    probabilities = next_bar_probabilities(
        paths,
        reference_close=10.0,
        flat_tolerance=0.005,
    )

    assert probabilities == {
        "sample_count": 4,
        "up_probability": 0.25,
        "down_probability": 0.25,
        "flat_probability": 0.5,
    }


def test_predict_modes_skips_unselected_generation_branches(monkeypatch) -> None:
    calls: list[str] = []

    def fake_generate_paths(*args, mode, path_count, **kwargs):
        calls.append(mode)
        return np.zeros((path_count, 1, 6), dtype=np.float32)

    monkeypatch.setattr(predictor_module, "_generate_paths", fake_generate_paths)
    frame = pd.DataFrame({
        "open": [10.0, 10.0], "high": [10.1, 10.1],
        "low": [9.9, 9.9], "close": [10.0, 10.0],
        "volume": [100.0, 100.0], "amount": [1000.0, 1000.0],
    })
    x_timestamps = pd.date_range("2026-08-03 09:30", periods=2, freq="5min")
    y_timestamps = pd.date_range("2026-08-03 09:40", periods=1, freq="5min")

    greedy = predict_modes(
        frame, x_timestamps, y_timestamps, tokenizer=object(), model=object(),
        device="cpu", modes=("greedy",),
    )
    assert calls == ["greedy"]
    assert greedy.greedy is not None
    assert greedy.sample_median is None
    assert greedy.sample_mean is None
    assert greedy.sampled_paths.shape[0] == 0

    calls.clear()
    median = predict_modes(
        frame, x_timestamps, y_timestamps, tokenizer=object(), model=object(),
        device="cpu", modes=("median",), sample_count=3,
    )
    assert calls == ["sample"]
    assert median.greedy is None
    assert median.sample_median is not None
    assert median.sample_mean is None
    assert median.sampled_paths.shape[0] == 3


def test_predict_modes_defaults_to_all_three_modes(monkeypatch) -> None:
    calls: list[tuple[str, int]] = []

    def fake_generate_paths(*args, mode, path_count, **kwargs):
        calls.append((mode, path_count))
        return np.zeros((path_count, 1, 6), dtype=np.float32)

    monkeypatch.setattr(predictor_module, "_generate_paths", fake_generate_paths)
    frame = pd.DataFrame({
        "open": [10.0], "high": [10.1], "low": [9.9], "close": [10.0],
        "volume": [100.0], "amount": [1000.0],
    })
    x_timestamps = pd.date_range("2026-08-03 09:30", periods=1, freq="5min")
    y_timestamps = pd.date_range("2026-08-03 09:35", periods=1, freq="5min")
    result = predict_modes(
        frame, x_timestamps, y_timestamps, tokenizer=object(), model=object(),
        device="cpu", sample_count=4,
    )

    assert calls == [("greedy", 1), ("sample", 4)]
    assert result.greedy is not None
    assert result.sample_median is not None
    assert result.sample_mean is not None


def test_predict_modes_rejects_an_explicit_empty_mode_selection() -> None:
    frame = pd.DataFrame({
        "open": [10.0], "high": [10.1], "low": [9.9], "close": [10.0],
        "volume": [100.0], "amount": [1000.0],
    })
    timestamps = pd.date_range("2026-08-03 09:30", periods=1, freq="5min")

    with pytest.raises(ValueError, match="At least one prediction mode"):
        predict_modes(
            frame, timestamps, timestamps, tokenizer=object(), model=object(),
            device="cpu", modes=(),
        )
