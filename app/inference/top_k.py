"""Top-K filtering: restrict sampling to the K highest-probability tokens."""

from __future__ import annotations

import torch


def apply_top_k(logits: torch.Tensor, k: int) -> torch.Tensor:
    """Masks every logit except the K largest to -inf.

    After softmax, masked positions get exactly 0 probability (exp(-inf) = 0)
    -- not just reduced, eliminated. This caps how much of the vocabulary can
    ever be sampled from, regardless of temperature (Module 21): no matter
    how flat the distribution gets, only these K tokens are reachable.

    Uses the indices returned by `torch.topk` directly (rather than a value
    threshold) so that ties at the boundary can never let more than K tokens
    survive.

    Args:
        logits: Unnormalized scores over the vocabulary.
        k: Number of top tokens to keep. Values >= vocab_size are clamped
            (equivalent to no filtering).

    Shape:
        logits: (vocab_size,)
        output: (vocab_size,), unchanged shape.

    Returns:
        Logits with all but the top-k positions set to -inf.

    Raises:
        ValueError: If `k` is not a positive integer.
    """
    if k <= 0:
        raise ValueError(f"k must be a positive integer, got {k}")

    vocab_size = logits.shape[-1]
    k = min(k, vocab_size)

    top_values, top_indices = torch.topk(logits, k, dim=-1)
    masked = torch.full_like(logits, float("-inf"))
    masked.scatter_(-1, top_indices, top_values)
    return masked
