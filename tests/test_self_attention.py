"""Unit tests for Module 9 (Self attention)."""

from __future__ import annotations

import math

import pytest
import torch

from app.attention.self_attention import SelfAttention

D_MODEL = 16
D_HEAD = 8
MAX_SEQ_LEN = 32


def test_output_shape() -> None:
    """Output must have shape (batch_size, seq_len, d_head)."""
    attention = SelfAttention(D_MODEL, D_HEAD, MAX_SEQ_LEN)
    x = torch.randn(2, 5, D_MODEL)
    output = attention(x)
    assert output.shape == (2, 5, D_HEAD)


def test_attention_weights_shape_and_sum_to_one() -> None:
    """Each query position's attention weights must form a probability distribution."""
    attention = SelfAttention(D_MODEL, D_HEAD, MAX_SEQ_LEN)
    x = torch.randn(2, 5, D_MODEL)
    _, weights = attention(x, return_weights=True)

    assert weights.shape == (2, 5, 5)
    row_sums = weights.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)


def test_causal_mask_zeroes_out_future_attention_weights() -> None:
    """Weight from position i to a future position j > i must be exactly 0."""
    attention = SelfAttention(D_MODEL, D_HEAD, MAX_SEQ_LEN)
    x = torch.randn(1, 6, D_MODEL)
    _, weights = attention(x, return_weights=True)

    for i in range(6):
        for j in range(i + 1, 6):
            assert weights[0, i, j].item() == 0.0


def test_causal_mask_blocks_future_positions_from_affecting_output() -> None:
    """Perturbing only the LAST token must not change earlier positions' output --
    this is the concrete, end-to-end test of the causal masking property."""
    attention = SelfAttention(D_MODEL, D_HEAD, MAX_SEQ_LEN)
    torch.manual_seed(0)
    x = torch.randn(1, 6, D_MODEL)
    x_perturbed = x.clone()
    x_perturbed[0, -1] = torch.randn(D_MODEL) * 100  # drastic change, last position only

    output = attention(x)
    output_perturbed = attention(x_perturbed)

    for i in range(5):  # every position except the last must be unaffected
        assert torch.allclose(output[0, i], output_perturbed[0, i], atol=1e-5)
    assert not torch.allclose(output[0, 5], output_perturbed[0, 5], atol=1e-5)


def test_single_token_sequence_attends_fully_to_itself() -> None:
    """With seq_len=1, the only allowed position is itself, so weight must be exactly 1.0."""
    attention = SelfAttention(D_MODEL, D_HEAD, MAX_SEQ_LEN)
    x = torch.randn(1, 1, D_MODEL)
    _, weights = attention(x, return_weights=True)
    assert weights[0, 0, 0].item() == pytest.approx(1.0)


def test_returned_weights_sum_to_one_even_with_dropout_during_training() -> None:
    """Regression test: returned/cached weights must be captured BEFORE
    dropout, so they stay a valid probability distribution even when
    dropout > 0 and the module is in train mode (inverted dropout rescales
    survivors, breaking the sum-to-1 property of a post-dropout tensor)."""
    attention = SelfAttention(D_MODEL, D_HEAD, MAX_SEQ_LEN, dropout=0.5)
    attention.train()
    x = torch.randn(2, 5, D_MODEL)
    _, weights = attention(x, return_weights=True)
    row_sums = weights.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)


def test_matches_manual_reference_computation() -> None:
    """The module's output must match an independent, manual implementation of
    scaled dot-product attention with causal masking, using the same projections."""
    attention = SelfAttention(D_MODEL, D_HEAD, MAX_SEQ_LEN)
    x = torch.randn(2, 5, D_MODEL)

    output = attention(x)

    q = attention.query_proj(x)
    k = attention.key_proj(x)
    v = attention.value_proj(x)
    scores = (q @ k.transpose(-2, -1)) / math.sqrt(D_HEAD)
    causal_mask = torch.tril(torch.ones(5, 5, dtype=torch.bool))
    scores = scores.masked_fill(~causal_mask, float("-inf"))
    expected = torch.softmax(scores, dim=-1) @ v

    assert torch.allclose(output, expected, atol=1e-6)


def test_seq_len_exceeding_max_raises() -> None:
    """Exceeding the precomputed mask size must fail loudly, not silently misbehave."""
    attention = SelfAttention(D_MODEL, D_HEAD, max_seq_len=4)
    x = torch.randn(1, 5, D_MODEL)
    with pytest.raises(ValueError, match="exceeds max_seq_len"):
        attention(x)


def test_gradients_flow_to_all_projections() -> None:
    """Backprop must produce gradients for the query, key, and value projections."""
    attention = SelfAttention(D_MODEL, D_HEAD, MAX_SEQ_LEN)
    x = torch.randn(2, 5, D_MODEL, requires_grad=True)
    output = attention(x)
    output.sum().backward()

    assert attention.query_proj.weight.grad is not None
    assert attention.key_proj.weight.grad is not None
    assert attention.value_proj.weight.grad is not None
    assert x.grad is not None
