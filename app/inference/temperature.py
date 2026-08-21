"""Temperature scaling: reshape a logit distribution's sharpness before sampling."""

from __future__ import annotations

import torch


def apply_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Rescales logits by 1/temperature, controlling distribution sharpness.

    softmax(logits / T):
      T < 1 sharpens the distribution (more confident, closer to greedy)
      T = 1 leaves it unchanged
      T > 1 flattens it (more random, closer to uniform)

    This does not change the RANKING of tokens by probability -- only how
    pronounced the differences between them are. It's meant to be applied
    before `sample_from_logits` (Module 20), and composes with top-k/top-p
    (Modules 22-23), which are applied afterward.

    Args:
        logits: Unnormalized scores over the vocabulary.
        temperature: Must be strictly positive. Values < 1 sharpen the
            distribution; values > 1 flatten it.

    Shape:
        logits: (vocab_size,)
        output: (vocab_size,), unchanged shape.

    Returns:
        The rescaled logits (not yet softmax-normalized).

    Raises:
        ValueError: If `temperature` is not strictly positive.
    """
    if temperature <= 0.0:
        raise ValueError(f"temperature must be > 0, got {temperature}")
    return logits / temperature
