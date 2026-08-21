"""Autoregressive text generation loop."""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import nn


class Generator:
    """Generates text autoregressively, one token at a time.

    At each step: run the full model on the current (possibly truncated)
    sequence, take the logits at the LAST position (the prediction for the
    next token), pick a token id from those logits, append it to the
    sequence, and repeat.

    By default, token selection is greedy (argmax) -- deterministic, always
    picks the single most likely token. Modules 20-23 introduce real
    sampling strategies (temperature, top-k, top-p); they plug in here via
    `sample_fn`, without changing this loop.

    This recomputes the full forward pass over the whole growing sequence at
    every step: O(max_new_tokens * seq_len^2) total, since each step re-does
    an O(seq_len^2) attention computation over an ever-longer sequence.
    Production systems avoid this with a "KV cache" (caching the K/V
    projections already computed for earlier positions instead of
    recomputing them every step) -- an optimization this project doesn't
    implement, but worth knowing exists once you feel this loop's cost
    firsthand.
    """

    def __init__(
        self,
        model: nn.Module,
        max_seq_len: int,
        sample_fn: Callable[[torch.Tensor], int] | None = None,
    ) -> None:
        """Builds the generator around an already-constructed model.

        Args:
            model: A `GPTModel` (trained or untrained).
            max_seq_len: The model's maximum context length. Generation
                truncates the context to this many of the most recent
                tokens before each forward pass.
            sample_fn: A callable mapping the next-token logits (shape
                `(vocab_size,)`) to a chosen token id. Defaults to greedy
                argmax selection.
        """
        self.model = model
        self.max_seq_len = max_seq_len
        self.sample_fn = sample_fn if sample_fn is not None else self._greedy_sample

    @staticmethod
    def _greedy_sample(logits: torch.Tensor) -> int:
        """Picks the highest-probability token id. Deterministic, no randomness."""
        return int(logits.argmax().item())

    @torch.no_grad()
    def generate(self, prompt_ids: list[int], max_new_tokens: int) -> list[int]:
        """Generates `max_new_tokens` additional token ids after `prompt_ids`.

        Args:
            prompt_ids: The initial sequence of token ids to continue from.
            max_new_tokens: Number of new tokens to generate.

        Returns:
            `prompt_ids` extended with `max_new_tokens` generated ids.

        Raises:
            ValueError: If `prompt_ids` is empty.
        """
        if len(prompt_ids) == 0:
            raise ValueError("prompt_ids must contain at least one token")

        was_training = self.model.training
        device = next(self.model.parameters()).device
        self.model.eval()
        try:
            ids = list(prompt_ids)
            for _ in range(max_new_tokens):
                context = ids[-self.max_seq_len :]
                input_tensor = torch.tensor([context], device=device)
                logits = self.model(input_tensor)
                next_token_logits = logits[0, -1]
                next_id = self.sample_fn(next_token_logits)
                ids.append(next_id)
            return ids
        finally:
            self.model.train(was_training)
