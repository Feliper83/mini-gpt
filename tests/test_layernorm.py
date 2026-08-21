"""Unit tests for Module 7 (LayerNorm)."""

from __future__ import annotations

import torch
from torch import nn

from app.transformer.layernorm import LayerNorm

D_MODEL = 16


def test_output_shape_is_preserved() -> None:
    """LayerNorm must not change the shape of its input."""
    layer_norm = LayerNorm(D_MODEL)
    x = torch.randn(4, 10, D_MODEL)
    output = layer_norm(x)
    assert output.shape == x.shape


def test_gamma_and_beta_initialized_to_identity_affine() -> None:
    """At init, gamma=1 and beta=0, so the affine part starts as a no-op."""
    layer_norm = LayerNorm(D_MODEL)
    assert torch.equal(layer_norm.gamma, torch.ones(D_MODEL))
    assert torch.equal(layer_norm.beta, torch.zeros(D_MODEL))


def test_output_has_zero_mean_unit_variance_at_init() -> None:
    """With gamma=1, beta=0, the output must be normalized: mean 0, var 1 per token."""
    layer_norm = LayerNorm(D_MODEL)
    x = torch.randn(4, 10, D_MODEL) * 5 + 3  # arbitrary scale/shift
    output = layer_norm(x)

    mean = output.mean(dim=-1)
    variance = output.var(dim=-1, unbiased=False)
    assert torch.allclose(mean, torch.zeros_like(mean), atol=1e-5)
    assert torch.allclose(variance, torch.ones_like(variance), atol=1e-3)


def test_matches_pytorch_reference_implementation() -> None:
    """Our from-scratch LayerNorm must numerically match torch.nn.LayerNorm
    when given identical gamma/beta, validating correctness against a known-good implementation."""
    torch.manual_seed(0)
    x = torch.randn(4, 10, D_MODEL)

    ours = LayerNorm(D_MODEL, eps=1e-5)
    ours.gamma.data = torch.randn(D_MODEL)
    ours.beta.data = torch.randn(D_MODEL)

    reference = nn.LayerNorm(D_MODEL, eps=1e-5)
    reference.weight.data = ours.gamma.data.clone()
    reference.bias.data = ours.beta.data.clone()

    assert torch.allclose(ours(x), reference(x), atol=1e-5)


def test_custom_gamma_beta_apply_affine_transform_exactly() -> None:
    """With gamma != 1 and beta != 0, the output must equal gamma*normalized + beta."""
    layer_norm = LayerNorm(D_MODEL)
    layer_norm.gamma.data = torch.full((D_MODEL,), 2.0)
    layer_norm.beta.data = torch.full((D_MODEL,), 10.0)

    x = torch.randn(2, 5, D_MODEL)
    output = layer_norm(x)

    mean = x.mean(dim=-1, keepdim=True)
    variance = x.var(dim=-1, keepdim=True, unbiased=False)
    expected = 2.0 * (x - mean) / torch.sqrt(variance + layer_norm.eps) + 10.0
    assert torch.allclose(output, expected, atol=1e-5)


def test_invariant_to_uniform_positive_scaling_of_input() -> None:
    """Scaling a token's activations by a positive constant should not change
    the normalized output (up to eps), since LayerNorm removes overall scale."""
    layer_norm = LayerNorm(D_MODEL, eps=1e-8)
    x = torch.randn(1, 1, D_MODEL) + 2.0  # avoid a near-zero-variance edge case
    scaled_x = x * 100.0

    output = layer_norm(x)
    scaled_output = layer_norm(scaled_x)
    assert torch.allclose(output, scaled_output, atol=1e-3)


def test_zero_variance_input_does_not_produce_nan_or_inf() -> None:
    """A token where every feature is identical (variance=0) must not divide by zero."""
    layer_norm = LayerNorm(D_MODEL)
    x = torch.full((1, 1, D_MODEL), 7.0)
    output = layer_norm(x)
    assert torch.isfinite(output).all()


def test_gradients_flow_to_input_and_parameters() -> None:
    """Backprop must produce gradients for gamma, beta, and the input."""
    layer_norm = LayerNorm(D_MODEL)
    x = torch.randn(2, 3, D_MODEL, requires_grad=True)

    output = layer_norm(x)
    output.sum().backward()

    assert x.grad is not None
    assert layer_norm.gamma.grad is not None
    assert layer_norm.beta.grad is not None
    assert not torch.all(x.grad == 0)
