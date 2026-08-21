"""Runnable example for Module 7 (LayerNorm).

Run with:

    uv run python scripts/layernorm_demo.py
"""

from __future__ import annotations

import torch
from torch import nn

from app.transformer.layernorm import LayerNorm

D_MODEL = 8


def main() -> None:
    """Normalize activations with wildly different per-token scales and inspect the result."""
    torch.manual_seed(0)
    layer_norm = LayerNorm(D_MODEL)

    # Three "tokens" with very different scales/offsets, to make normalization visible.
    x = torch.stack(
        [
            torch.randn(D_MODEL) * 1.0,
            torch.randn(D_MODEL) * 50.0 + 100.0,
            torch.randn(D_MODEL) * 0.01 - 5.0,
        ]
    ).unsqueeze(0)  # (1, 3, D_MODEL)

    print("Before LayerNorm (per-token mean, std):")
    for i in range(x.shape[1]):
        print(f"  token {i}: mean={x[0, i].mean().item():8.3f}  std={x[0, i].std().item():8.3f}")

    output = layer_norm(x)

    print("\nAfter LayerNorm (per-token mean, std) -- should all be ~0, ~1:")
    for i in range(output.shape[1]):
        mean = output[0, i].mean().item()
        std = output[0, i].std().item()
        print(f"  token {i}: mean={mean:8.3f}  std={std:8.3f}")

    print("\nCross-checking against torch.nn.LayerNorm with the same gamma/beta:")
    reference = nn.LayerNorm(D_MODEL)
    reference.weight.data = layer_norm.gamma.data.clone()
    reference.bias.data = layer_norm.beta.data.clone()
    reference_output = reference(x)
    max_diff = (output - reference_output).abs().max().item()
    print(f"  Max absolute difference vs torch.nn.LayerNorm: {max_diff:.2e}")
    print(
        "  (Token 2 has std~0.01, so var~eps: floating-point round-off from "
        "computing mean/var in two passes vs. PyTorch's fused kernel gets "
        "amplified there -- this is why production LayerNorm implementations "
        "use fused kernels rather than naive multi-pass reductions.)"
    )


if __name__ == "__main__":
    main()
