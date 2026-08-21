"""Layer normalization, implemented from scratch (no `torch.nn.LayerNorm`)."""

from __future__ import annotations

import torch
from torch import nn


class LayerNorm(nn.Module):
    """Normalizes activations across the feature dimension, per token.

    y = gamma * (x - mean) / sqrt(var + eps) + beta

    Statistics (mean, variance) are computed independently for each token
    position over its `d_model` features -- unlike BatchNorm, there is no
    dependency on batch size or on other positions in the sequence, so this
    layer behaves identically during training and inference.
    """

    def __init__(self, d_model: int, eps: float = 1e-5) -> None:
        """Builds the layer with learnable per-feature scale and shift.

        Args:
            d_model: Size of the feature dimension being normalized.
            eps: Small constant added to the variance for numerical
                stability, avoiding division by zero when variance is near 0.
        """
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalizes `x` over its last dimension.

        Args:
            x: Input activations.

        Shape:
            x: (..., d_model)
            output: (..., d_model), unchanged.
        """
        mean = x.mean(dim=-1, keepdim=True)
        variance = x.var(dim=-1, keepdim=True, unbiased=False)
        x_normalized = (x - mean) / torch.sqrt(variance + self.eps)
        return self.gamma * x_normalized + self.beta
