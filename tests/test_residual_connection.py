"""Unit tests for Module 11 (Residual connections)."""

from __future__ import annotations

import torch

from app.attention.multi_head_attention import MultiHeadAttention
from app.transformer.feedforward import FeedForward
from app.transformer.residual import ResidualConnection

D_MODEL = 16


def _identity(x: torch.Tensor) -> torch.Tensor:
    return x


def _zeros(x: torch.Tensor) -> torch.Tensor:
    return torch.zeros_like(x)


def test_output_shape_matches_input() -> None:
    """The residual wrapper must preserve the input's shape."""
    residual = ResidualConnection(D_MODEL, dropout=0.0)
    x = torch.randn(2, 5, D_MODEL)
    output = residual(x, _identity)
    assert output.shape == x.shape


def test_zero_output_sublayer_reduces_to_exact_identity() -> None:
    """A sublayer that contributes nothing must leave the residual stream unchanged:
    output = x + 0 = x exactly."""
    residual = ResidualConnection(D_MODEL, dropout=0.0)
    x = torch.randn(2, 5, D_MODEL)
    output = residual(x, _zeros)
    assert torch.equal(output, x)


def test_identity_sublayer_equals_x_plus_layernorm() -> None:
    """With sublayer = identity, output must equal x + LayerNorm(x) exactly."""
    residual = ResidualConnection(D_MODEL, dropout=0.0)
    x = torch.randn(2, 5, D_MODEL)
    output = residual(x, _identity)
    expected = x + residual.norm(x)
    assert torch.allclose(output, expected, atol=1e-6)


def test_sublayer_is_called_with_normalized_input_not_raw_x() -> None:
    """The sublayer must receive LayerNorm(x) (Pre-LN), never raw x."""
    residual = ResidualConnection(D_MODEL, dropout=0.0)
    x = torch.randn(2, 5, D_MODEL)
    received: dict[str, torch.Tensor] = {}

    def capturing_sublayer(t: torch.Tensor) -> torch.Tensor:
        received["input"] = t
        return t

    residual(x, capturing_sublayer)
    expected_norm = residual.norm(x)
    assert torch.equal(received["input"], expected_norm)


def test_gradient_flows_through_identity_path_even_if_sublayer_gradient_is_zero() -> None:
    """Even if the sublayer contributes exactly zero gradient, the residual
    identity path must still deliver full gradient to the input -- this is
    the core mathematical claim of the module."""
    residual = ResidualConnection(D_MODEL, dropout=0.0)
    x = torch.randn(2, 3, D_MODEL, requires_grad=True)

    def zero_gradient_sublayer(t: torch.Tensor) -> torch.Tensor:
        return t.detach() * 0.0  # constant zero: contributes NO gradient to x

    output = residual(x, zero_gradient_sublayer)
    output.sum().backward()

    assert x.grad is not None
    assert torch.allclose(x.grad, torch.ones_like(x))


def test_works_with_real_multi_head_attention_sublayer() -> None:
    """Integration check: wrapping real MultiHeadAttention must preserve its
    causal-masking property through the residual wrapper."""
    n_heads = 4
    mha = MultiHeadAttention(D_MODEL, n_heads, max_seq_len=16)
    residual = ResidualConnection(D_MODEL, dropout=0.0)

    torch.manual_seed(0)
    x = torch.randn(1, 6, D_MODEL)
    x_perturbed = x.clone()
    x_perturbed[0, -1] = torch.randn(D_MODEL) * 100

    output = residual(x, mha)
    output_perturbed = residual(x_perturbed, mha)

    for i in range(5):
        assert torch.allclose(output[0, i], output_perturbed[0, i], atol=1e-5)
    assert not torch.allclose(output[0, 5], output_perturbed[0, 5], atol=1e-5)


def test_works_with_real_feedforward_sublayer() -> None:
    """Integration check: wrapping real FeedForward must preserve its
    position-independence property through the residual wrapper."""
    ffn = FeedForward(D_MODEL)
    residual = ResidualConnection(D_MODEL, dropout=0.0)

    torch.manual_seed(0)
    x = torch.randn(1, 3, D_MODEL)
    x_perturbed = x.clone()
    x_perturbed[0, 1] = torch.randn(D_MODEL) * 100

    output = residual(x, ffn)
    output_perturbed = residual(x_perturbed, ffn)

    assert torch.allclose(output[0, 0], output_perturbed[0, 0], atol=1e-5)
    assert torch.allclose(output[0, 2], output_perturbed[0, 2], atol=1e-5)
    assert not torch.allclose(output[0, 1], output_perturbed[0, 1], atol=1e-5)


def test_zero_dropout_is_deterministic() -> None:
    """With dropout=0.0, repeated calls on the same input must match exactly."""
    residual = ResidualConnection(D_MODEL, dropout=0.0)
    x = torch.randn(1, 4, D_MODEL)
    assert torch.equal(residual(x, _identity), residual(x, _identity))
