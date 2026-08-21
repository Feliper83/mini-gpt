"""Runnable example for Module 9 (Self attention).

Run with:

    uv run python scripts/self_attention_demo.py
"""

from __future__ import annotations

import torch

from app.attention.self_attention import SelfAttention

D_MODEL = 8
D_HEAD = 8
MAX_SEQ_LEN = 16


def print_attention_matrix(weights: torch.Tensor, tokens: list[str]) -> None:
    """Prints an attention weight matrix as an ASCII table."""
    header = "        " + " ".join(f"{t:>6}" for t in tokens)
    print(header)
    for i, row in enumerate(weights):
        row_str = " ".join(f"{v:6.2f}" for v in row.tolist())
        print(f"{tokens[i]:>7} {row_str}")


def main() -> None:
    """Visualize the causal attention matrix and verify the masking property live."""
    torch.manual_seed(0)
    attention = SelfAttention(D_MODEL, D_HEAD, MAX_SEQ_LEN)

    tokens = ["I", "like", "cats", "a", "lot"]
    seq_len = len(tokens)
    x = torch.randn(1, seq_len, D_MODEL)

    output, weights = attention(x, return_weights=True)
    print(f"Input shape:  {tuple(x.shape)}")
    print(f"Output shape: {tuple(output.shape)}\n")

    print("Attention weight matrix (rows = query position, cols = key position):")
    print_attention_matrix(weights[0], tokens)
    print("\nThe upper triangle is all 0.00: no token attends to a future token.")
    print("Each row sums to 1.00 -- a probability distribution over allowed positions.")

    row_sums = weights[0].sum(dim=-1)
    print(f"\nRow sums: {[round(s, 4) for s in row_sums.tolist()]}")

    print("\n--- Causal masking check: perturbing only the LAST token ---")
    x_perturbed = x.clone()
    x_perturbed[0, -1] = torch.randn(D_MODEL) * 100
    output_perturbed = attention(x_perturbed)

    for i in range(seq_len - 1):
        unchanged = torch.allclose(output[0, i], output_perturbed[0, i], atol=1e-5)
        print(f"  Position {i} ({tokens[i]:>5}) unchanged: {unchanged}")
    changed = not torch.allclose(output[0, -1], output_perturbed[0, -1], atol=1e-5)
    print(f"  Position {seq_len - 1} ({tokens[-1]:>5}) changed:   {changed}")


if __name__ == "__main__":
    main()
