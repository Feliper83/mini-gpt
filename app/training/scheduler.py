"""Learning rate scheduler: linear warmup followed by cosine decay."""

from __future__ import annotations

import math

import torch


class WarmupCosineScheduler:
    """Linear warmup, then cosine decay down to a minimum learning rate.

    lr(t) = max_lr * (t / warmup_steps)                        for t < warmup_steps
    lr(t) = min_lr + 0.5*(max_lr-min_lr)*(1+cos(pi*progress))   for warmup_steps <= t <= max_steps
    lr(t) = min_lr                                             for t > max_steps

    where progress = (t - warmup_steps) / (max_steps - warmup_steps).

    Warmup avoids taking large, potentially destabilizing steps while the
    optimizer's moment estimates (Module 17) are still unreliable right
    after initialization. Cosine decay then reduces the step size smoothly
    as training progresses, staying near the peak learning rate longer
    before decaying more steeply near the end -- the convention used by
    GPT-2/GPT-3 pretraining.
    """

    def __init__(
        self,
        max_lr: float,
        warmup_steps: int,
        max_steps: int,
        min_lr: float = 0.0,
    ) -> None:
        """Builds the schedule.

        Args:
            max_lr: Peak learning rate, reached at the end of warmup.
            warmup_steps: Number of steps to linearly ramp from 0 to max_lr.
            max_steps: Total number of training steps; the cosine decay
                reaches `min_lr` exactly at this step.
            min_lr: Learning rate floor, held constant after `max_steps`.

        Raises:
            ValueError: If `warmup_steps` is negative, `max_steps` is not
                greater than `warmup_steps`, or `max_lr`/`min_lr` are invalid.
        """
        if max_lr <= 0.0:
            raise ValueError(f"max_lr must be positive, got {max_lr}")
        if min_lr < 0.0 or min_lr > max_lr:
            raise ValueError(f"min_lr must be in [0, max_lr], got {min_lr}")
        if warmup_steps < 0:
            raise ValueError(f"warmup_steps must be >= 0, got {warmup_steps}")
        if max_steps <= warmup_steps:
            raise ValueError(
                f"max_steps ({max_steps}) must be greater than warmup_steps ({warmup_steps})"
            )

        self.max_lr = max_lr
        self.min_lr = min_lr
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps

    def get_lr(self, step: int) -> float:
        """Computes the learning rate for a given step.

        Args:
            step: Current training step (0-indexed).

        Returns:
            The learning rate to use at `step`.
        """
        if step < self.warmup_steps:
            return self.max_lr * (step / self.warmup_steps)

        if step > self.max_steps:
            return self.min_lr

        progress = (step - self.warmup_steps) / (self.max_steps - self.warmup_steps)
        cosine_factor = 0.5 * (1 + math.cos(math.pi * progress))
        return self.min_lr + (self.max_lr - self.min_lr) * cosine_factor

    def step_optimizer(self, optimizer: torch.optim.Optimizer, step: int) -> float:
        """Computes lr(step) and writes it into every param group of `optimizer`.

        Args:
            optimizer: The optimizer whose param groups' `lr` will be updated.
            step: Current training step.

        Returns:
            The learning rate that was set.
        """
        lr = self.get_lr(step)
        for group in optimizer.param_groups:
            group["lr"] = lr
        return lr
