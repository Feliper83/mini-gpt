"""Runnable example for Module 22 (Top-K sampling).

Run with:

    uv run python scripts/top_k_demo.py
"""

from __future__ import annotations

import torch

from app.inference.sampling import sample_from_logits
from app.inference.temperature import apply_temperature
from app.inference.top_k import apply_top_k

VOCAB = ["cat", "dog", "fish", "bird", "ant", "moth", "worm", "slug"]
LOGITS = torch.tensor([3.0, 2.0, 0.5, -1.0, -3.0, -4.0, -5.0, -6.0])


def ascii_bar(value: float, width: int = 40) -> str:
    """Renders a probability as an ASCII bar."""
    return "#" * round(value * width)


def print_distribution(label: str, probs: torch.Tensor) -> None:
    nonzero = (probs > 1e-6).sum().item()
    print(f"{label} ({nonzero} tokens with nonzero probability):")
    for token, prob in zip(VOCAB, probs.tolist(), strict=True):
        print(f"  {token:>5} {prob:6.3f} {ascii_bar(prob)}")
    print()


def main() -> None:
    """Show the long-tail problem from high temperature, then top-k fixing it."""
    base_probs = torch.softmax(LOGITS, dim=-1)
    print_distribution("No filtering (T=1)", base_probs)

    print("--- The problem (Module 21's exercise): high temperature revives the tail ---")
    hot_probs = torch.softmax(apply_temperature(LOGITS, temperature=2.5), dim=-1)
    print_distribution("T=2.5, no top-k", hot_probs)

    print("--- The fix: top-k caps how many tokens can EVER be chosen ---")
    for k in [1, 3, 5]:
        filtered_logits = apply_top_k(apply_temperature(LOGITS, temperature=2.5), k=k)
        filtered_probs = torch.softmax(filtered_logits, dim=-1)
        print_distribution(f"T=2.5, top_k={k}", filtered_probs)

    print("--- Sampling always stays within the top-k set, no matter how many draws ---")
    torch.manual_seed(0)
    k = 3
    top_k_indices = set(torch.topk(LOGITS, k).indices.tolist())
    draws = [sample_from_logits(apply_top_k(LOGITS, k)) for _ in range(500)]
    outside_top_k = [d for d in draws if d not in top_k_indices]
    print(f"  k={k}, drew 500 samples, {len(outside_top_k)} landed OUTSIDE the top-{k} set.")


if __name__ == "__main__":
    main()
