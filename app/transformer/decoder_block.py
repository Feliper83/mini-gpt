"""A single GPT-style decoder block: attention and feed-forward, each wrapped
in its own Pre-LN residual connection."""

from __future__ import annotations

import torch
from torch import nn

from app.attention.multi_head_attention import MultiHeadAttention
from app.transformer.feedforward import FeedForward
from app.transformer.residual import ResidualConnection


class DecoderBlock(nn.Module):
    """One GPT-style transformer decoder block.

    x = x + MultiHeadAttention(LayerNorm(x))
    x = x + FeedForward(LayerNorm(x))

    Two independent residual connections, each with its own LayerNorm: one
    around the attention sublayer (which mixes information across
    positions), one around the feed-forward sublayer (which processes each
    position independently). GPT stacks many of these blocks (Module 13) to
    build the full model.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        max_seq_len: int,
        d_ff: int | None = None,
        dropout: float = 0.0,
    ) -> None:
        """Builds the block's sublayers and their residual wrappers.

        Args:
            d_model: Dimensionality of the token representations.
            n_heads: Number of attention heads.
            max_seq_len: Longest sequence length the causal mask is
                precomputed for.
            d_ff: Hidden dimensionality of the feed-forward expansion.
                Defaults to `4 * d_model` (see `FeedForward`).
            dropout: Dropout probability used throughout the block
                (attention weights and both sublayer outputs).
        """
        super().__init__()
        self.attention = MultiHeadAttention(d_model, n_heads, max_seq_len, dropout=dropout)
        self.feed_forward = FeedForward(d_model, d_ff=d_ff, dropout=dropout)
        self.attention_residual = ResidualConnection(d_model, dropout=dropout)
        self.feed_forward_residual = ResidualConnection(d_model, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies one decoder block.

        Args:
            x: Input token representations.

        Shape:
            x: (batch_size, seq_len, d_model)
            output: (batch_size, seq_len, d_model), unchanged shape.
        """
        x = self.attention_residual(x, self.attention)
        x = self.feed_forward_residual(x, self.feed_forward)
        return x
