"""Runnable example for Module 18 (LR scheduler).

Run with:

    uv run python scripts/scheduler_demo.py
"""

from __future__ import annotations

import torch
from torch import nn

from app.training.optimizer import AdamW
from app.training.scheduler import WarmupCosineScheduler

MAX_LR = 3e-4
MIN_LR = 3e-5
WARMUP_STEPS = 20
MAX_STEPS = 100


def ascii_bar(value: float, max_value: float, width: int = 40) -> str:
    """Renders a value as a simple ASCII bar for visualizing the LR curve."""
    filled = round(width * value / max_value)
    return "#" * filled


def main() -> None:
    """Visualize the warmup + cosine decay curve, then apply it to a real optimizer."""
    scheduler = WarmupCosineScheduler(MAX_LR, WARMUP_STEPS, MAX_STEPS, MIN_LR)

    print(f"max_lr={MAX_LR}, min_lr={MIN_LR}, warmup_steps={WARMUP_STEPS}, max_steps={MAX_STEPS}\n")
    print("Learning rate schedule:")
    for step in range(0, MAX_STEPS + 20, 5):
        lr = scheduler.get_lr(step)
        if step < WARMUP_STEPS:
            phase = "warmup"
        elif step <= MAX_STEPS:
            phase = "decay"
        else:
            phase = "floor"
        bar = ascii_bar(lr, MAX_LR)
        print(f"  step {step:4d} [{phase:6s}]: lr={lr:.6f} {bar}")

    print("\n--- Applying the scheduler to a real AdamW optimizer ---")
    param = nn.Parameter(torch.zeros(4))
    optimizer = AdamW([param], lr=MAX_LR)
    for step in [0, 5, 10, 20, 60, 100, 150]:
        scheduler.step_optimizer(optimizer, step)
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"  step {step:4d}: optimizer.param_groups[0]['lr'] = {current_lr:.6f}")


if __name__ == "__main__":
    main()
