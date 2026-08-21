"""Unit tests for Module 23 (Top-P / nucleus sampling)."""

from __future__ import annotations

import pytest
import torch

from app.inference.sampling import sample_from_logits
from app.inference.top_p import apply_top_p


def test_output_shape_is_unchanged() -> None:
    """Top-p filtering must preserve the logits' shape."""
    logits = torch.randn(30)
    filtered = apply_top_p(logits, p=0.9)
    assert filtered.shape == logits.shape


def test_top_token_is_always_kept_even_with_tiny_p() -> None:
    """The single highest-probability token must never be masked, no matter how small p is."""
    logits = torch.tensor([5.0, 1.0, 0.5, 0.0, -1.0])
    filtered = apply_top_p(logits, p=0.01)
    assert filtered[0].item() != float("-inf")


def test_kept_probability_mass_reaches_at_least_p() -> None:
    """The defining property: the surviving tokens' probability must sum to >= p."""
    logits = torch.tensor([2.0, 1.5, 1.0, 0.5, 0.0, -0.5, -1.0])
    probs = torch.softmax(logits, dim=-1)
    for p in [0.5, 0.7, 0.9, 0.99]:
        filtered = apply_top_p(logits, p)
        kept_mass = probs[filtered != float("-inf")].sum().item()
        assert kept_mass >= p - 1e-6


def test_kept_set_is_minimal() -> None:
    """Removing the lowest-probability kept token must drop mass below p --
    the nucleus should not keep unnecessarily many tokens."""
    logits = torch.tensor([2.0, 1.5, 1.0, 0.5, 0.0, -0.5, -1.0])
    probs = torch.softmax(logits, dim=-1)
    p = 0.8
    filtered = apply_top_p(logits, p)
    kept_indices = (filtered != float("-inf")).nonzero(as_tuple=True)[0]

    kept_probs_sorted = probs[kept_indices].sort(descending=True).values
    mass_without_last = kept_probs_sorted[:-1].sum().item()
    assert mass_without_last < p


def test_p_equal_to_one_keeps_everything() -> None:
    """With p=1.0, essentially the whole distribution should survive."""
    logits = torch.tensor([2.0, 1.0, 0.0, -1.0, -2.0])
    filtered = apply_top_p(logits, p=1.0)
    assert torch.equal(filtered, logits)


def test_invalid_p_raises() -> None:
    """p must be strictly in (0, 1]."""
    logits = torch.tensor([1.0, 2.0])
    with pytest.raises(ValueError, match="must be in"):
        apply_top_p(logits, p=0.0)
    with pytest.raises(ValueError, match="must be in"):
        apply_top_p(logits, p=1.5)


def test_confident_distribution_keeps_fewer_tokens_than_uncertain_one() -> None:
    """The adaptive property that distinguishes top-p from top-k: a peaked
    distribution should need fewer tokens to reach the same p than a flat one."""
    confident_logits = torch.tensor([8.0, 1.0, 0.5, 0.0, -1.0, -2.0])
    uncertain_logits = torch.tensor([2.0, 1.9, 1.8, 1.7, 1.6, 1.5])
    p = 0.9

    confident_kept = (apply_top_p(confident_logits, p) != float("-inf")).sum().item()
    uncertain_kept = (apply_top_p(uncertain_logits, p) != float("-inf")).sum().item()

    assert confident_kept < uncertain_kept


def test_sampling_after_top_p_stays_within_the_nucleus() -> None:
    """Composed with Module 20's sampler, draws must always land inside the nucleus."""
    torch.manual_seed(0)
    logits = torch.randn(20)
    p = 0.8
    filtered = apply_top_p(logits, p)
    nucleus_indices = {i for i in range(20) if filtered[i].item() != float("-inf")}

    draws = [sample_from_logits(filtered) for _ in range(300)]
    assert all(draw in nucleus_indices for draw in draws)
