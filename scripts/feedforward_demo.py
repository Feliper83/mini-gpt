"""Runnable example for Module 8 (Feed forward network).

Run with:

    uv run python scripts/feedforward_demo.py
"""

from __future__ import annotations

import torch

from app.transformer.feedforward import GELU, FeedForward

D_MODEL = 8


def main() -> None:
    """Compare GELU to ReLU, then demonstrate the FFN's position independence."""
    print("--- GELU vs ReLU at a few points ---")
    gelu = GELU()
    xs = torch.tensor([-3.0, -1.0, -0.5, 0.0, 0.5, 1.0, 3.0])
    relu_vals = torch.relu(xs)
    gelu_vals = gelu(xs)
    print(f"{'x':>6} {'ReLU(x)':>10} {'GELU(x)':>10}")
    for x, r, g in zip(xs.tolist(), relu_vals.tolist(), gelu_vals.tolist(), strict=True):
        print(f"{x:6.2f} {r:10.4f} {g:10.4f}")
    print("Note GELU(-0.5) is small-but-negative, unlike ReLU's hard zero.\n")

    ffn = FeedForward(d_model=D_MODEL)
    d_ff = ffn.linear_1.out_features
    print(f"FeedForward: d_model={D_MODEL} -> d_ff={d_ff} -> d_model={D_MODEL}")

    ffn_params = sum(p.numel() for p in ffn.parameters())
    attention_like_params = 4 * D_MODEL * D_MODEL  # Q,K,V,O projections, for comparison
    print(f"FeedForward parameter count:               {ffn_params}")
    print(f"(For reference) 4 attention projections:   {attention_like_params}")
    print("FFN has roughly 2x the parameters of the attention projections at this d_model.\n")

    print("--- Position independence: changing position 1 must not affect position 0 ---")
    torch.manual_seed(0)
    x_a = torch.randn(1, 3, D_MODEL)
    x_b = x_a.clone()
    x_b[0, 1] = torch.randn(D_MODEL) * 100  # drastically change only position 1

    out_a = ffn(x_a)
    out_b = ffn(x_b)

    pos0_unchanged = torch.allclose(out_a[0, 0], out_b[0, 0], atol=1e-6)
    pos1_changed = not torch.allclose(out_a[0, 1], out_b[0, 1], atol=1e-6)
    pos2_unchanged = torch.allclose(out_a[0, 2], out_b[0, 2], atol=1e-6)

    print(f"  Position 0 unchanged: {pos0_unchanged}")
    print(f"  Position 1 changed:   {pos1_changed}")
    print(f"  Position 2 unchanged: {pos2_unchanged}")
    print("\n(Attention, in Module 9, will behave the opposite way: changing one")
    print("position's input WILL affect other positions' outputs.)")


if __name__ == "__main__":
    main()
