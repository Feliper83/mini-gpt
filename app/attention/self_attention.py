"""Single-head causal self-attention: the core mechanism of a decoder-only transformer."""

from __future__ import annotations

import math

import torch
from torch import nn

from app.utils.validation import check_seq_len


class SelfAttention(nn.Module):
    """Single-head scaled dot-product self-attention with a causal mask.

    Attention(Q, K, V) = softmax(QK^T / sqrt(d_head) + M) V

    Every position can attend to itself and every position before it, and
    never to positions after it (the causal mask M) -- this is what makes
    the model usable for left-to-right autoregressive generation (Module 19):
    at inference time, future tokens simply don't exist yet.
    """

    def __init__(self, d_model: int, d_head: int, max_seq_len: int, dropout: float = 0.0) -> None:
        """Builds the attention layer and precomputes its causal mask.

        Args:
            d_model: Dimensionality of the input token representations.
            d_head: Dimensionality of the query/key/value projections.
            max_seq_len: Longest sequence length the causal mask is
                precomputed for.
            dropout: Dropout probability applied to the attention weights.
        """
        super().__init__()
        self.d_head = d_head

        self.query_proj = nn.Linear(d_model, d_head)
        self.key_proj = nn.Linear(d_model, d_head)
        self.value_proj = nn.Linear(d_model, d_head)
        self.dropout = nn.Dropout(dropout)

        # causal_mask[i, j] is True where position i is ALLOWED to attend to
        # position j (j <= i): a lower-triangular matrix, including the diagonal.
        causal_mask = torch.tril(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool))
        self.register_buffer("causal_mask", causal_mask, persistent=False)

    def forward(
        self, x: torch.Tensor, return_weights: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Applies causal self-attention.

        Args:
            x: Input token representations.
            return_weights: If True, also return the post-softmax attention
                weights (needed later for Module 27, attention visualization).

        Shape:
            x: (batch_size, seq_len, d_model)
            output: (batch_size, seq_len, d_head)
            weights (if returned): (batch_size, seq_len, seq_len)

        Raises:
            ValueError: If `seq_len` exceeds the precomputed `max_seq_len`.
        """
        _, seq_len, _ = x.shape
        check_seq_len(seq_len, self.causal_mask.shape[0], "this causal mask")

        q = self.query_proj(x)  # (batch, seq_len, d_head)
        k = self.key_proj(x)  # (batch, seq_len, d_head)
        v = self.value_proj(x)  # (batch, seq_len, d_head)

        scores = q @ k.transpose(-2, -1)  # (batch, seq_len, seq_len)
        scores = scores / math.sqrt(self.d_head)

        mask = self.causal_mask[:seq_len, :seq_len]  # (seq_len, seq_len)
        scores = scores.masked_fill(~mask, float("-inf"))

        weights = torch.softmax(scores, dim=-1)  # (batch, seq_len, seq_len)

        # Returned to the caller (if requested) BEFORE dropout: dropout
        # rescales surviving entries by 1/(1-p), so a post-dropout row no
        # longer sums to 1 whenever dropout > 0 and the module is training --
        # these weights are meant to represent the attention distribution
        # itself, not the (dropout-regularized) tensor used below.
        weights_for_output = self.dropout(weights)
        output = weights_for_output @ v  # (batch, seq_len, d_head)

        if return_weights:
            return output, weights
        return output
