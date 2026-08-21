"""Runnable example for Module 13 (Transformer stack).

Run with:

    uv run python scripts/transformer_stack_demo.py
"""

from __future__ import annotations

import torch

from app.transformer.transformer_stack import TransformerStack

D_MODEL = 32
N_HEADS = 4
MAX_SEQ_LEN = 16


def main() -> None:
    """Show parameter scaling, residual stream growth, and causal masking at depth."""
    torch.manual_seed(0)

    print("--- Parameter count scaling with n_layers ---")
    for n_layers in [2, 4, 8, 12]:
        stack = TransformerStack(D_MODEL, N_HEADS, n_layers, MAX_SEQ_LEN)
        total_params = sum(p.numel() for p in stack.parameters())
        print(f"  n_layers={n_layers:>2}: {total_params:>7} parameters")

    print("\n--- Residual stream magnitude grows across blocks (Pre-LN never normalizes it) ---")
    stack = TransformerStack(D_MODEL, N_HEADS, n_layers=8, max_seq_len=MAX_SEQ_LEN)
    x = torch.randn(1, 6, D_MODEL)
    h = x
    print(f"  input norm:            {h.norm().item():8.3f}")
    for i, block in enumerate(stack.blocks):
        h = block(h)
        print(f"  after block {i}: norm={h.norm().item():8.3f}")
    final = stack.final_norm(h)
    print(f"  after FINAL LayerNorm: norm={final.norm().item():8.3f}")

    print("\n--- Causal masking preserved through the entire stack ---")
    x = torch.randn(1, 6, D_MODEL)
    x_perturbed = x.clone()
    x_perturbed[0, -1] = torch.randn(D_MODEL) * 100

    output = stack(x)
    output_perturbed = stack(x_perturbed)
    for i in range(5):
        unchanged = torch.allclose(output[0, i], output_perturbed[0, i], atol=1e-4)
        print(f"  Position {i} unchanged: {unchanged}")
    changed = not torch.allclose(output[0, 5], output_perturbed[0, 5], atol=1e-4)
    print(f"  Position 5 (perturbed) changed: {changed}")


if __name__ == "__main__":
    main()
