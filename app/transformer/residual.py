"""Residual (skip) connections wrapping a sublayer, GPT-2 style (Pre-LN)."""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import nn

from app.transformer.layernorm import LayerNorm


class ResidualConnection(nn.Module):
    """Wraps a sublayer with a Pre-LN residual connection.

    output = x + dropout(sublayer(LayerNorm(x)))

    LayerNorm is applied to the input BEFORE it reaches the sublayer
    (Pre-LN), not after the residual addition (Post-LN, the original
    Transformer paper's placement). Pre-LN keeps the residual stream itself
    unnormalized end to end, which trains more stably at the depths
    GPT-style models use.

    The identity term (`x +`) gives gradients a direct path back through
    every layer regardless of what the sublayer does:
    d(output)/dx = I + d(sublayer)/dx, so even a sublayer with a near-zero
    or unstable gradient cannot fully block gradient flow.
    """

    def __init__(self, d_model: int, dropout: float = 0.0) -> None:
        """Builds the wrapper.

        Args:
            d_model: Dimensionality of the residual stream.
            dropout: Dropout probability applied to the sublayer's output
                before it is added back to the residual stream.
        """
        super().__init__()
        self.norm = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, sublayer: Callable[[torch.Tensor], torch.Tensor]
    ) -> torch.Tensor:
        """Applies `sublayer` inside a Pre-LN residual connection.

        Args:
            x: Input to the residual block.
            sublayer: A callable (typically an `nn.Module`'s forward, or a
                closure wrapping one) that must return a tensor of the same
                shape as its input.

        Shape:
            x: (batch_size, seq_len, d_model)
            output: (batch_size, seq_len, d_model), unchanged shape.
        """
        return x + self.dropout(sublayer(self.norm(x)))
