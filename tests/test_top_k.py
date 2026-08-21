"""Unit tests for Module 22 (Top-K sampling)."""

from __future__ import annotations

import pytest
import torch

from app.inference.sampling import sample_from_logits
from app.inference.top_k import apply_top_k


def test_only_k_tokens_have_nonzero_probability_after_softmax() -> None:
    """Exactly k tokens must survive softmax with nonzero probability."""
    logits = torch.tensor([5.0, 4.0, 3.0, 2.0, 1.0, 0.0])
    masked = apply_top_k(logits, k=3)
    probs = torch.softmax(masked, dim=-1)
    assert (probs > 0).sum().item() == 3


def test_top_k_keeps_the_actual_highest_logits() -> None:
    """The surviving tokens must be exactly the k highest-logit indices."""
    logits = torch.tensor([1.0, 5.0, 3.0, 0.0, 4.0])
    masked = apply_top_k(logits, k=2)
    kept_indices = {i for i in range(5) if masked[i].item() != float("-inf")}
    assert kept_indices == {1, 4}  # logits 5.0 and 4.0


def test_kept_logit_values_are_unchanged() -> None:
    """Surviving positions must retain their exact original logit value."""
    logits = torch.tensor([1.0, 5.0, 3.0, 0.0, 4.0])
    masked = apply_top_k(logits, k=2)
    assert masked[1].item() == 5.0
    assert masked[4].item() == 4.0


def test_k_equal_to_vocab_size_is_a_no_op() -> None:
    """With k == vocab_size, nothing should be masked."""
    logits = torch.tensor([1.0, 5.0, 3.0, 0.0, 4.0])
    masked = apply_top_k(logits, k=5)
    assert torch.equal(masked, logits)


def test_k_greater_than_vocab_size_is_clamped() -> None:
    """k larger than the vocabulary must behave like no filtering, not error."""
    logits = torch.tensor([1.0, 5.0, 3.0])
    masked = apply_top_k(logits, k=1000)
    assert torch.equal(masked, logits)


def test_invalid_k_raises() -> None:
    """k must be strictly positive."""
    logits = torch.tensor([1.0, 2.0])
    with pytest.raises(ValueError, match="must be a positive integer"):
        apply_top_k(logits, k=0)
    with pytest.raises(ValueError, match="must be a positive integer"):
        apply_top_k(logits, k=-1)


def test_ties_at_the_boundary_do_not_let_extra_tokens_through() -> None:
    """Even with duplicate values at the k-th position, exactly k must survive."""
    logits = torch.tensor([5.0, 3.0, 3.0, 3.0, 1.0])  # three-way tie at rank 2-4
    masked = apply_top_k(logits, k=2)
    kept = (masked != float("-inf")).sum().item()
    assert kept == 2


def test_output_shape_is_unchanged() -> None:
    """Top-k filtering must preserve the logits' shape."""
    logits = torch.randn(30)
    masked = apply_top_k(logits, k=5)
    assert masked.shape == logits.shape


def test_sampling_after_top_k_never_leaves_the_kept_set() -> None:
    """Composed with Module 20's sampler, draws must always land inside top-k,
    regardless of how many times we sample."""
    torch.manual_seed(0)
    logits = torch.randn(20)
    k = 4
    top_k_indices = set(torch.topk(logits, k).indices.tolist())

    masked = apply_top_k(logits, k)
    draws = [sample_from_logits(masked) for _ in range(300)]

    assert all(draw in top_k_indices for draw in draws)
