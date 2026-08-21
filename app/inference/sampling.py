"""Stochastic sampling: draw a token id from the model's predicted distribution."""

from __future__ import annotations

import torch


def sample_from_logits(logits: torch.Tensor, generator: torch.Generator | None = None) -> int:
    """Samples a token id proportionally to softmax(logits).

    Unlike greedy (argmax) selection -- which always picks the single most
    likely token and is fully deterministic -- this treats the predicted
    distribution as an actual probability distribution and draws a random
    sample from it. A token the model assigns 30% probability to gets
    chosen roughly 30% of the time, not systematically ignored just because
    it isn't the single most likely option.

    Args:
        logits: Unnormalized scores over the vocabulary, for ONE position.
        generator: Optional `torch.Generator` for reproducible sampling
            (e.g. in tests or demos with a fixed seed).

    Shape:
        logits: (vocab_size,)

    Returns:
        A sampled token id.
    """
    probabilities = torch.softmax(logits, dim=-1)
    sampled_index = torch.multinomial(probabilities, num_samples=1, generator=generator)
    return int(sampled_index.item())
