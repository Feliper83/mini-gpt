"""Top-P (nucleus) filtering: restrict sampling to the smallest set of tokens
whose cumulative probability reaches a target threshold."""

from __future__ import annotations

import torch


def apply_top_p(logits: torch.Tensor, p: float) -> torch.Tensor:
    """Masks tokens outside the smallest top-probability set reaching mass p.

    Unlike top-k (Module 22), which always keeps a fixed COUNT of tokens,
    this keeps a fixed cumulative PROBABILITY MASS -- the number of tokens
    kept shrinks automatically when the model is confident (probability
    concentrated on few tokens) and grows automatically when it's uncertain
    (probability spread thin), without needing to hand-tune a count.

    Args:
        logits: Unnormalized scores over the vocabulary.
        p: Target cumulative probability mass to keep. Must be in (0, 1].

    Shape:
        logits: (vocab_size,)
        output: (vocab_size,), unchanged shape.

    Returns:
        Logits with every token outside the nucleus set to -inf. The single
        highest-probability token is always kept, regardless of `p`.

    Raises:
        ValueError: If `p` is not in (0, 1].
    """
    if not (0.0 < p <= 1.0):
        raise ValueError(f"p must be in (0, 1], got {p}")

    probs = torch.softmax(logits, dim=-1)
    sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    # Remove a (sorted-order) token if the cumulative mass BEFORE it already
    # reached p -- earlier, higher-ranked tokens already cover what's needed.
    sorted_indices_to_remove = (cumulative_probs - sorted_probs) >= p

    indices_to_remove = torch.zeros_like(sorted_indices_to_remove)
    indices_to_remove.scatter_(-1, sorted_indices, sorted_indices_to_remove)

    return logits.masked_fill(indices_to_remove, float("-inf"))
