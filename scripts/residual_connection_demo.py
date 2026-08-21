"""Runnable example for Module 11 (Residual connections).

Run with:

    uv run python scripts/residual_connection_demo.py
"""

from __future__ import annotations

import torch
from torch import nn

from app.transformer.residual import ResidualConnection

D_MODEL = 16
N_LAYERS = 30


def zero_sublayer(x: torch.Tensor) -> torch.Tensor:
    """A sublayer that always outputs zero, to demonstrate the no-op property."""
    return torch.zeros_like(x)


def build_weak_sublayers(n_layers: int) -> list[nn.Module]:
    """Linear sublayers with deliberately tiny weights, simulating a
    transformation whose own gradient is individually very weak."""
    torch.manual_seed(0)
    layers = []
    for _ in range(n_layers):
        linear = nn.Linear(D_MODEL, D_MODEL)
        nn.init.normal_(linear.weight, mean=0.0, std=0.01)
        nn.init.zeros_(linear.bias)
        layers.append(linear)
    return layers


def main() -> None:
    """Show the zero-sublayer no-op property, then compare gradient magnitude
    at the input of a deep stack, with vs. without residual connections."""
    print("--- Property: a zero-output sublayer makes the residual block a no-op ---")
    residual = ResidualConnection(D_MODEL, dropout=0.0)
    x = torch.randn(1, 3, D_MODEL)
    output = residual(x, zero_sublayer)
    print(f"  output == x exactly: {torch.equal(output, x)}\n")

    print(f"--- Gradient flow through {N_LAYERS} stacked weak (std=0.01) linear sublayers ---")

    # (a) Plain stack: the same weak layers, chained directly, no residuals.
    plain_layers = build_weak_sublayers(N_LAYERS)
    x_plain = torch.randn(1, 3, D_MODEL, requires_grad=True)
    h = x_plain
    for layer in plain_layers:
        h = layer(h)
    h.sum().backward()
    plain_grad_norm = x_plain.grad.norm().item()

    # (b) Residual stack: the SAME weak layers, each wrapped in a residual connection.
    residual_layers = build_weak_sublayers(N_LAYERS)
    residual_blocks = [ResidualConnection(D_MODEL, dropout=0.0) for _ in range(N_LAYERS)]
    x_residual = torch.randn(1, 3, D_MODEL, requires_grad=True)
    h = x_residual
    for block, layer in zip(residual_blocks, residual_layers, strict=True):
        h = block(h, layer)
    h.sum().backward()
    residual_grad_norm = x_residual.grad.norm().item()

    print(f"  Gradient norm at input, WITHOUT residual connections: {plain_grad_norm:.2e}")
    print(f"  Gradient norm at input, WITH residual connections:    {residual_grad_norm:.2e}")
    ratio = residual_grad_norm / max(plain_grad_norm, 1e-30)
    print(f"  Ratio (with / without): {ratio:.1f}x")
    print("\nWithout residuals, the gradient shrinks by roughly (0.01 * sqrt(d_model))^30,")
    print("compounding multiplicatively across 30 layers -- vanishing gradient in practice.")
    print("With residuals, each layer's identity term keeps the gradient from collapsing.")


if __name__ == "__main__":
    main()
