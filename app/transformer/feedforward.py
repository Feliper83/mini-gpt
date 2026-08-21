"""Position-wise feed-forward network: GELU activation and the FFN block."""

from __future__ import annotations

import math

import torch
from torch import nn


class GELU(nn.Module):
    """Gaussian Error Linear Unit activation, implemented from scratch.

    GELU(x) = x * Phi(x)

    where Phi is the standard normal CDF. Unlike ReLU's hard cutoff at 0,
    GELU weights each input by the probability that a standard normal
    variable is less than it -- a smooth, differentiable gate instead of an
    all-or-nothing one. This smoothness is why GELU largely replaced ReLU in
    transformer feed-forward blocks (BERT, GPT-2/3).
    """

    def __init__(self, approximate: bool = False) -> None:
        """Builds the activation.

        Args:
            approximate: If True, use the tanh-based approximation that the
                original GPT-2 codebase uses (cheaper, avoids `erf`). If
                False (default), compute the exact form via `torch.erf`.
        """
        super().__init__()
        self.approximate = approximate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies GELU elementwise.

        Shape:
            x: (...)
            output: (...), unchanged.
        """
        if self.approximate:
            inner = math.sqrt(2.0 / math.pi) * (x + 0.044715 * x**3)
            return 0.5 * x * (1.0 + torch.tanh(inner))
        return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class FeedForward(nn.Module):
    """Position-wise feed-forward network: Linear -> GELU -> Linear.

    Applied identically and independently to every position in the
    sequence -- unlike attention (Module 9/10), it never mixes information
    across positions. This is the per-token nonlinear processing step that
    follows attention's context-mixing step in a transformer block.
    """

    def __init__(self, d_model: int, d_ff: int | None = None, dropout: float = 0.0) -> None:
        """Builds the feed-forward block.

        Args:
            d_model: Input/output dimensionality.
            d_ff: Hidden dimensionality of the expansion. Defaults to
                `4 * d_model`, the convention used by GPT-2.
            dropout: Dropout probability applied after the second projection.
        """
        super().__init__()
        d_ff = d_ff if d_ff is not None else 4 * d_model
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.activation = GELU()
        self.linear_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies the position-wise transformation.

        Args:
            x: Input activations.

        Shape:
            x: (batch_size, seq_len, d_model)
            output: (batch_size, seq_len, d_model), unchanged shape.
        """
        x = self.linear_1(x)
        x = self.activation(x)
        x = self.linear_2(x)
        return self.dropout(x)
