"""Unit tests for Module 27 (Visualizing attention)."""

from __future__ import annotations

import pytest
import torch

from app.inference.attention_viz import extract_attention_weights, plot_attention_heatmap
from app.transformer.gpt_model import GPTModel

VOCAB_SIZE = 20
D_MODEL = 16
N_HEADS = 4
N_LAYERS = 3
MAX_SEQ_LEN = 16


def _build_model() -> GPTModel:
    return GPTModel(VOCAB_SIZE, D_MODEL, N_HEADS, N_LAYERS, MAX_SEQ_LEN)


def test_extract_returns_one_entry_per_layer() -> None:
    """The result must have exactly n_layers entries, one per decoder block."""
    model = _build_model()
    token_ids = torch.randint(0, VOCAB_SIZE, (1, 6))
    weights = extract_attention_weights(model, token_ids)
    assert len(weights) == N_LAYERS


def test_extracted_weights_have_the_expected_shape() -> None:
    """Each layer's weights must have shape (n_heads, seq_len, seq_len)."""
    model = _build_model()
    token_ids = torch.randint(0, VOCAB_SIZE, (1, 6))
    weights = extract_attention_weights(model, token_ids)
    for layer_weights in weights:
        assert layer_weights.shape == (N_HEADS, 6, 6)


def test_extracted_weights_sum_to_one_per_row() -> None:
    """Every row of every head's weights must still be a valid probability
    distribution, confirming extraction doesn't corrupt the values."""
    model = _build_model()
    token_ids = torch.randint(0, VOCAB_SIZE, (1, 6))
    weights = extract_attention_weights(model, token_ids)
    for layer_weights in weights:
        row_sums = layer_weights.sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)


def test_extracted_weights_respect_the_causal_mask() -> None:
    """Extracted weights must still show zero weight on future positions,
    at the full-model level, not just in isolated attention tests."""
    model = _build_model()
    token_ids = torch.randint(0, VOCAB_SIZE, (1, 6))
    weights = extract_attention_weights(model, token_ids)
    for layer_weights in weights:
        for i in range(6):
            for j in range(i + 1, 6):
                assert torch.all(layer_weights[:, i, j] == 0.0)


def test_extract_rejects_batch_size_other_than_one() -> None:
    """Extraction is defined for a single sequence at a time."""
    model = _build_model()
    token_ids = torch.randint(0, VOCAB_SIZE, (2, 6))
    with pytest.raises(ValueError, match=r"\(1, seq_len\)"):
        extract_attention_weights(model, token_ids)


def test_extract_rejects_wrong_rank_input() -> None:
    """A 1-D tensor (missing the batch dimension entirely) must be rejected
    with a clear error, not a confusing failure deeper inside the model."""
    model = _build_model()
    token_ids = torch.randint(0, VOCAB_SIZE, (6,))
    with pytest.raises(ValueError, match=r"\(1, seq_len\)"):
        extract_attention_weights(model, token_ids)


def test_extract_restores_the_models_prior_training_mode() -> None:
    """Regression test: extraction must not permanently leave the model in
    eval mode if it was training beforehand."""
    model = _build_model()
    model.train()
    token_ids = torch.randint(0, VOCAB_SIZE, (1, 6))
    extract_attention_weights(model, token_ids)
    assert model.training is True


def test_plot_attention_heatmap_creates_a_valid_png(tmp_path) -> None:
    """The plotting function must write an actual PNG file to disk."""
    weights = torch.softmax(torch.randn(4, 4), dim=-1)
    output_path = tmp_path / "heatmap.png"
    tokens = ["a", "b", "c", "d"]

    plot_attention_heatmap(weights, tokens=tokens, title="test", output_path=output_path)

    assert output_path.exists()
    assert output_path.stat().st_size > 0
    with open(output_path, "rb") as f:
        magic_bytes = f.read(8)
    assert magic_bytes == b"\x89PNG\r\n\x1a\n"


def test_plot_attention_heatmap_creates_missing_parent_directories(tmp_path) -> None:
    """A nested output path must not require the directory to pre-exist."""
    weights = torch.softmax(torch.randn(3, 3), dim=-1)
    output_path = tmp_path / "nested" / "dir" / "heatmap.png"

    plot_attention_heatmap(weights, tokens=["a", "b", "c"], title="test", output_path=output_path)
    assert output_path.exists()
