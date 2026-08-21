"""Runnable example for Module 10 (Multi-head attention).

Run with:

    uv run python scripts/multi_head_attention_demo.py
"""

from __future__ import annotations

import torch

from app.attention.multi_head_attention import MultiHeadAttention

D_MODEL = 12
N_HEADS = 3
MAX_SEQ_LEN = 16


def print_attention_matrix(weights: torch.Tensor, tokens: list[str]) -> None:
    """Prints an attention weight matrix as an ASCII table."""
    header = "        " + " ".join(f"{t:>6}" for t in tokens)
    print(header)
    for i, row in enumerate(weights):
        row_str = " ".join(f"{v:6.2f}" for v in row.tolist())
        print(f"{tokens[i]:>7} {row_str}")


def main() -> None:
    """Show each head's attention pattern and confirm they genuinely differ."""
    torch.manual_seed(0)
    mha = MultiHeadAttention(D_MODEL, N_HEADS, MAX_SEQ_LEN)

    tokens = ["I", "like", "cats", "a", "lot"]
    seq_len = len(tokens)
    x = torch.randn(1, seq_len, D_MODEL)

    output, weights = mha(x, return_weights=True)
    print(f"Input shape:   {tuple(x.shape)}")
    print(f"Output shape:  {tuple(output.shape)}  (back to d_model, same as input)")
    print(f"Weights shape: {tuple(weights.shape)}  (batch, n_heads, seq_len, seq_len)")
    print(f"\nd_model={D_MODEL}, n_heads={N_HEADS} -> d_head={mha.d_head} per head")

    print("\nEach head learns a DIFFERENT attention pattern from the same input:")
    for h in range(N_HEADS):
        print(f"\nHead {h}:")
        print_attention_matrix(weights[0, h], tokens)

    head0_vs_head1 = torch.allclose(weights[0, 0], weights[0, 1], atol=1e-3)
    print(f"\nHead 0 and Head 1 patterns are (nearly) identical: {head0_vs_head1}")

    total_params = sum(p.numel() for p in mha.parameters())
    print(f"\nTotal MultiHeadAttention parameters: {total_params}")
    print("(4 projections of size d_model x d_model: Q, K, V, and the output projection O)")


if __name__ == "__main__":
    main()
