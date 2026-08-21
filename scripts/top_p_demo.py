"""Runnable example for Module 23 (Top-P / nucleus sampling).

Run with:

    uv run python scripts/top_p_demo.py
"""

from __future__ import annotations

import torch

from app.inference.top_k import apply_top_k
from app.inference.top_p import apply_top_p

CONFIDENT_LOGITS = torch.tensor([8.0, 1.0, 0.5, 0.0, -1.0, -2.0])  # one token dominates
UNCERTAIN_LOGITS = torch.tensor([2.0, 1.9, 1.8, 1.7, 1.6, 1.5])  # nearly tied


def describe_top_p(label: str, logits: torch.Tensor, p: float) -> None:
    probs = torch.softmax(logits, dim=-1)
    filtered = apply_top_p(logits, p)
    kept = (filtered != float("-inf")).sum().item()
    rounded = [round(v, 3) for v in probs.tolist()]
    print(f"{label}: probs = {rounded}")
    print(f"  top_p={p} keeps {kept} token(s)\n")


def main() -> None:
    """Show top-p's adaptive nucleus size, contrasted against fixed top-k."""
    p = 0.9
    print("--- Top-P adapts the nucleus size to the model's actual certainty ---\n")
    describe_top_p("Confident distribution", CONFIDENT_LOGITS, p)
    describe_top_p("Uncertain distribution", UNCERTAIN_LOGITS, p)

    print("--- Contrast: fixed top-k keeps the SAME count regardless of certainty ---\n")
    k = 4
    for label, logits in [("Confident", CONFIDENT_LOGITS), ("Uncertain", UNCERTAIN_LOGITS)]:
        filtered = apply_top_k(logits, k)
        kept = (filtered != float("-inf")).sum().item()
        print(f"  {label} distribution, top_k={k}: keeps {kept} token(s) always")

    print("\n--- Verifying the defining property: kept probability mass >= p ---")
    for label, logits in [("Confident", CONFIDENT_LOGITS), ("Uncertain", UNCERTAIN_LOGITS)]:
        probs = torch.softmax(logits, dim=-1)
        filtered = apply_top_p(logits, p)
        kept_mass = probs[filtered != float("-inf")].sum().item()
        print(f"  {label}: kept probability mass = {kept_mass:.4f} (target >= {p})")


if __name__ == "__main__":
    main()
