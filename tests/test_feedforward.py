"""Unit tests for Module 8 (Feed forward network)."""

from __future__ import annotations

import torch
from torch.nn.functional import gelu as torch_gelu

from app.transformer.feedforward import GELU, FeedForward

D_MODEL = 16


# --- GELU ---


def test_gelu_exact_matches_pytorch_reference() -> None:
    """The exact (erf-based) GELU must match torch's own 'none' approximation."""
    gelu = GELU(approximate=False)
    x = torch.linspace(-5, 5, steps=50)
    expected = torch_gelu(x, approximate="none")
    assert torch.allclose(gelu(x), expected, atol=1e-6)


def test_gelu_tanh_approximation_matches_pytorch_reference() -> None:
    """The tanh approximation must match torch's 'tanh' approximation."""
    gelu = GELU(approximate=True)
    x = torch.linspace(-5, 5, steps=50)
    expected = torch_gelu(x, approximate="tanh")
    assert torch.allclose(gelu(x), expected, atol=1e-6)


def test_gelu_at_zero_is_zero() -> None:
    """GELU(0) = 0 * Phi(0) = 0 * 0.5 = 0."""
    gelu = GELU()
    output = gelu(torch.tensor([0.0]))
    assert torch.allclose(output, torch.tensor([0.0]), atol=1e-6)


def test_gelu_large_positive_input_approaches_identity() -> None:
    """As x -> +inf, Phi(x) -> 1, so GELU(x) -> x."""
    gelu = GELU()
    output = gelu(torch.tensor([10.0]))
    assert torch.allclose(output, torch.tensor([10.0]), atol=1e-3)


def test_gelu_large_negative_input_approaches_zero() -> None:
    """As x -> -inf, Phi(x) -> 0, so GELU(x) -> 0."""
    gelu = GELU()
    output = gelu(torch.tensor([-10.0]))
    assert torch.allclose(output, torch.tensor([0.0]), atol=1e-3)


def test_gelu_small_negative_input_is_small_but_nonzero() -> None:
    """Unlike ReLU, GELU does not hard-zero small negative inputs."""
    gelu = GELU()
    output = gelu(torch.tensor([-0.5])).item()
    assert output != 0.0
    assert output < 0.0  # GELU dips slightly negative around here


# --- FeedForward ---


def test_feedforward_preserves_input_output_shape() -> None:
    """Output must have the same shape as the input, for residual compatibility."""
    ffn = FeedForward(D_MODEL)
    x = torch.randn(2, 5, D_MODEL)
    output = ffn(x)
    assert output.shape == x.shape


def test_feedforward_default_hidden_dim_is_four_times_d_model() -> None:
    """Absent an explicit d_ff, the GPT-2 convention 4*d_model must be used."""
    ffn = FeedForward(D_MODEL)
    assert ffn.linear_1.out_features == 4 * D_MODEL
    assert ffn.linear_2.in_features == 4 * D_MODEL


def test_feedforward_respects_custom_hidden_dim() -> None:
    """An explicit d_ff must override the default 4x expansion."""
    ffn = FeedForward(D_MODEL, d_ff=64)
    assert ffn.linear_1.out_features == 64
    assert ffn.linear_2.in_features == 64


def test_feedforward_is_position_independent() -> None:
    """Changing one position's input must not affect any other position's output,
    since the FFN is applied identically and independently per position."""
    ffn = FeedForward(D_MODEL)
    torch.manual_seed(0)
    x_a = torch.randn(1, 3, D_MODEL)
    x_b = x_a.clone()
    x_b[0, 1] = torch.randn(D_MODEL) * 100  # drastically perturb only position 1

    out_a = ffn(x_a)
    out_b = ffn(x_b)

    assert torch.allclose(out_a[0, 0], out_b[0, 0], atol=1e-6)
    assert torch.allclose(out_a[0, 2], out_b[0, 2], atol=1e-6)
    assert not torch.allclose(out_a[0, 1], out_b[0, 1], atol=1e-6)


def test_feedforward_gradients_flow_to_both_linear_layers() -> None:
    """Backprop must produce gradients for both projection matrices."""
    ffn = FeedForward(D_MODEL)
    x = torch.randn(2, 3, D_MODEL, requires_grad=True)
    output = ffn(x)
    output.sum().backward()

    assert ffn.linear_1.weight.grad is not None
    assert ffn.linear_2.weight.grad is not None
    assert not torch.all(ffn.linear_1.weight.grad == 0)


def test_feedforward_zero_dropout_is_deterministic() -> None:
    """With dropout=0.0, repeated forward passes on the same input must match exactly."""
    ffn = FeedForward(D_MODEL, dropout=0.0)
    x = torch.randn(1, 4, D_MODEL)
    assert torch.equal(ffn(x), ffn(x))
