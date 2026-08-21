"""Runnable example for Module 21 (Temperature).

Run with:

    uv run python scripts/temperature_demo.py
"""

from __future__ import annotations

import torch

from app.inference.temperature import apply_temperature

VOCAB = ["cat", "dog", "fish", "bird", "ant"]
LOGITS = torch.tensor([3.0, 2.0, 0.5, -1.0, -3.0])


def ascii_bar(value: float, width: int = 40) -> str:
    """Renders a probability as an ASCII bar."""
    return "#" * round(value * width)


def print_distribution(label: str, probs: torch.Tensor) -> None:
    entropy = -(probs * probs.clamp_min(1e-12).log()).sum().item()
    print(f"{label} (entropy={entropy:.3f}):")
    for token, prob in zip(VOCAB, probs.tolist(), strict=True):
        print(f"  {token:>5} {prob:6.3f} {ascii_bar(prob)}")
    print()


def main() -> None:
    """Show how temperature reshapes the same logits into very different distributions."""
    print(f"Base logits: {dict(zip(VOCAB, LOGITS.tolist(), strict=True))}\n")

    for temperature in [0.2, 1.0, 3.0]:
        scaled = apply_temperature(LOGITS, temperature)
        probs = torch.softmax(scaled, dim=-1)
        print_distribution(f"T={temperature}", probs)

    print("Note: the RANKING (cat > dog > fish > bird > ant) never changes --")
    print("only how sharply the probability mass concentrates on the top choice.")


if __name__ == "__main__":
    main()
