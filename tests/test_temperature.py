"""Unit tests for Module 21 (Temperature)."""

from __future__ import annotations

import pytest
import torch

from app.inference.temperature import apply_temperature


def test_temperature_one_leaves_logits_unchanged() -> None:
    """T=1 must be a no-op."""
    logits = torch.tensor([1.0, 2.0, 3.0])
    scaled = apply_temperature(logits, temperature=1.0)
    assert torch.equal(scaled, logits)


def test_low_temperature_sharpens_the_distribution() -> None:
    """T<1 must make the resulting softmax distribution MORE peaked (lower entropy)."""
    logits = torch.tensor([1.0, 2.0, 3.0])
    base_probs = torch.softmax(logits, dim=-1)
    sharp_probs = torch.softmax(apply_temperature(logits, temperature=0.3), dim=-1)

    base_entropy = -(base_probs * base_probs.log()).sum()
    sharp_entropy = -(sharp_probs * sharp_probs.log()).sum()
    assert sharp_entropy < base_entropy


def test_high_temperature_flattens_the_distribution() -> None:
    """T>1 must make the resulting softmax distribution LESS peaked (higher entropy)."""
    logits = torch.tensor([1.0, 2.0, 3.0])
    base_probs = torch.softmax(logits, dim=-1)
    flat_probs = torch.softmax(apply_temperature(logits, temperature=3.0), dim=-1)

    base_entropy = -(base_probs * base_probs.log()).sum()
    flat_entropy = -(flat_probs * flat_probs.log()).sum()
    assert flat_entropy > base_entropy


def test_temperature_preserves_token_ranking() -> None:
    """Scaling must never change which token is more likely than another."""
    logits = torch.tensor([0.5, 3.0, -1.0, 2.0])
    for temperature in [0.1, 0.5, 1.0, 2.0, 10.0]:
        scaled = apply_temperature(logits, temperature)
        assert torch.equal(scaled.argsort(), logits.argsort())


def test_very_low_temperature_approaches_greedy() -> None:
    """As T -> 0, softmax(logits/T) should collapse almost entirely onto the argmax."""
    logits = torch.tensor([1.0, 2.0, 5.0, 3.0])
    probs = torch.softmax(apply_temperature(logits, temperature=0.01), dim=-1)
    assert probs.argmax().item() == logits.argmax().item()
    assert probs.max().item() > 0.999


def test_high_temperature_approaches_uniform() -> None:
    """As T -> inf, softmax(logits/T) should approach a uniform distribution."""
    logits = torch.tensor([1.0, 2.0, 5.0, 3.0])
    probs = torch.softmax(apply_temperature(logits, temperature=10_000.0), dim=-1)
    assert torch.allclose(probs, torch.full((4,), 0.25), atol=1e-3)


def test_zero_or_negative_temperature_raises() -> None:
    """Temperature must be strictly positive -- division by zero is undefined."""
    logits = torch.tensor([1.0, 2.0])
    with pytest.raises(ValueError, match="must be > 0"):
        apply_temperature(logits, temperature=0.0)
    with pytest.raises(ValueError, match="must be > 0"):
        apply_temperature(logits, temperature=-1.0)


def test_output_shape_is_unchanged() -> None:
    """Temperature scaling must preserve the logits' shape."""
    logits = torch.randn(50)
    scaled = apply_temperature(logits, temperature=0.7)
    assert scaled.shape == logits.shape
