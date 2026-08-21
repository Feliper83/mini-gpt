"""Unit tests for Module 10 (Multi-head attention)."""

from __future__ import annotations

import math

import pytest
import torch

from app.attention.multi_head_attention import MultiHeadAttention

D_MODEL = 12
N_HEADS = 3
MAX_SEQ_LEN = 32


def test_output_shape_matches_input() -> None:
    """Output must have shape (batch_size, seq_len, d_model), same as the input."""
    mha = MultiHeadAttention(D_MODEL, N_HEADS, MAX_SEQ_LEN)
    x = torch.randn(2, 5, D_MODEL)
    output = mha(x)
    assert output.shape == (2, 5, D_MODEL)


def test_d_model_not_divisible_by_n_heads_raises() -> None:
    """d_model must split evenly across heads."""
    with pytest.raises(ValueError, match="must be divisible by"):
        MultiHeadAttention(d_model=10, n_heads=3, max_seq_len=MAX_SEQ_LEN)


def test_weights_shape_and_sum_to_one_per_head() -> None:
    """Every head's attention weights must form a probability distribution."""
    mha = MultiHeadAttention(D_MODEL, N_HEADS, MAX_SEQ_LEN)
    x = torch.randn(2, 5, D_MODEL)
    _, weights = mha(x, return_weights=True)

    assert weights.shape == (2, N_HEADS, 5, 5)
    row_sums = weights.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)


def test_returned_and_cached_weights_sum_to_one_even_with_dropout_during_training() -> None:
    """Regression test: returned/cached weights must be captured BEFORE
    dropout, so they stay a valid probability distribution even when
    dropout > 0 and the module is in train mode."""
    mha = MultiHeadAttention(D_MODEL, N_HEADS, MAX_SEQ_LEN, dropout=0.5)
    mha.train()
    x = torch.randn(2, 5, D_MODEL)
    output, weights = mha(x, return_weights=True)
    row_sums = weights.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)
    assert torch.equal(mha.last_attention_weights, weights)
    assert output.shape == (2, 5, D_MODEL)


def test_causal_mask_blocks_future_positions_from_affecting_output() -> None:
    """Perturbing only the LAST token must not change any earlier position's output."""
    mha = MultiHeadAttention(D_MODEL, N_HEADS, MAX_SEQ_LEN)
    torch.manual_seed(0)
    x = torch.randn(1, 6, D_MODEL)
    x_perturbed = x.clone()
    x_perturbed[0, -1] = torch.randn(D_MODEL) * 100

    output = mha(x)
    output_perturbed = mha(x_perturbed)

    for i in range(5):
        assert torch.allclose(output[0, i], output_perturbed[0, i], atol=1e-5)
    assert not torch.allclose(output[0, 5], output_perturbed[0, 5], atol=1e-5)


def test_split_and_merge_heads_are_inverse_operations() -> None:
    """_merge_heads(_split_heads(x)) must reconstruct x exactly."""
    mha = MultiHeadAttention(D_MODEL, N_HEADS, MAX_SEQ_LEN)
    x = torch.randn(2, 5, D_MODEL)
    split = mha._split_heads(x, batch_size=2, seq_len=5)
    assert split.shape == (2, N_HEADS, 5, D_MODEL // N_HEADS)

    merged = mha._merge_heads(split, batch_size=2, seq_len=5)
    assert torch.equal(merged, x)


def test_matches_naive_per_head_loop_implementation() -> None:
    """The vectorized reshape/transpose implementation must exactly match a
    naive Python loop over heads using explicit weight slices. This is the
    classic source of MHA bugs: mixing up which axis is heads vs. seq_len
    inside the reshape/transpose calls."""
    d_model, n_heads, seq_len = 12, 3, 5
    d_head = d_model // n_heads
    mha = MultiHeadAttention(d_model, n_heads, max_seq_len=16)
    x = torch.randn(1, seq_len, d_model)

    output = mha(x)

    q_full = mha.query_proj(x)
    k_full = mha.key_proj(x)
    v_full = mha.value_proj(x)
    causal_mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool))

    head_outputs = []
    for h in range(n_heads):
        q_h = q_full[..., h * d_head : (h + 1) * d_head]
        k_h = k_full[..., h * d_head : (h + 1) * d_head]
        v_h = v_full[..., h * d_head : (h + 1) * d_head]
        scores = (q_h @ k_h.transpose(-2, -1)) / math.sqrt(d_head)
        scores = scores.masked_fill(~causal_mask, float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        head_outputs.append(weights @ v_h)

    expected_merged = torch.cat(head_outputs, dim=-1)
    expected = mha.output_proj(expected_merged)

    assert torch.allclose(output, expected, atol=1e-6)


def test_seq_len_exceeding_max_raises() -> None:
    """Exceeding the precomputed mask size must fail loudly."""
    mha = MultiHeadAttention(D_MODEL, N_HEADS, max_seq_len=4)
    x = torch.randn(1, 5, D_MODEL)
    with pytest.raises(ValueError, match="exceeds max_seq_len"):
        mha(x)


def test_last_attention_weights_are_cached_after_forward() -> None:
    """MultiHeadAttention must cache its most recent attention weights so
    they can be recovered later without changing the call site -- needed by
    Module 27 (attention visualization) to extract weights from a full
    stacked model, where DecoderBlock always calls with return_weights=False."""
    mha = MultiHeadAttention(D_MODEL, N_HEADS, MAX_SEQ_LEN, dropout=0.0)
    x = torch.randn(1, 5, D_MODEL)
    _, weights = mha(x, return_weights=True)
    assert torch.equal(mha.last_attention_weights, weights)


def test_gradients_flow_to_all_projections() -> None:
    """Backprop must produce gradients for Q, K, V, and the output projection."""
    mha = MultiHeadAttention(D_MODEL, N_HEADS, MAX_SEQ_LEN)
    x = torch.randn(2, 5, D_MODEL, requires_grad=True)
    output = mha(x)
    output.sum().backward()

    assert mha.query_proj.weight.grad is not None
    assert mha.key_proj.weight.grad is not None
    assert mha.value_proj.weight.grad is not None
    assert mha.output_proj.weight.grad is not None
    assert x.grad is not None
