"""Training loop orchestration for a GPTModel."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.data import DataLoader

from app.training.loss import CrossEntropyLoss
from app.training.scheduler import WarmupCosineScheduler


@dataclass
class TrainingMetrics:
    """Metrics recorded for a single training step."""

    step: int
    loss: float
    lr: float


class Trainer:
    """Orchestrates the training loop: forward, loss, backward, optimizer step.

    This is intentionally a thin orchestrator: the model (Module 14) and the
    optimizer are constructed elsewhere and handed in. Module 17 goes deeper
    into how the optimizer itself should be built (parameter groups, weight
    decay exclusions); this module uses a plain, directly-constructed
    `torch.optim.Optimizer` to get the loop itself working end to end first.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device | str = "cpu",
    ) -> None:
        """Builds the trainer around an already-constructed model and optimizer.

        Args:
            model: The model being trained (e.g. a `GPTModel`).
            optimizer: A constructed PyTorch optimizer over `model`'s parameters.
            device: Device to run training on.
        """
        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = device
        self.loss_fn = CrossEntropyLoss()

    def train_step(self, inputs: torch.Tensor, targets: torch.Tensor) -> float:
        """Runs one optimization step: forward, loss, backward, parameter update.

        Args:
            inputs: Input token ids.
            targets: Target token ids (inputs shifted by one, from Module 4's
                `TokenDataset`).

        Shape:
            inputs: (batch_size, seq_len)
            targets: (batch_size, seq_len)

        Returns:
            The scalar loss value for this step, as a Python float.
        """
        self.model.train()
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)

        logits = self.model(inputs)
        loss = self.loss_fn(logits, targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def fit(
        self,
        dataloader: DataLoader,
        max_steps: int,
        log_every: int = 10,
        scheduler: WarmupCosineScheduler | None = None,
        start_step: int = 0,
    ) -> list[TrainingMetrics]:
        """Runs training until the step counter reaches `max_steps`.

        Args:
            dataloader: Yields `(inputs, targets)` batches, e.g. from
                `create_dataloader` (Module 4).
            max_steps: The ABSOLUTE step to stop at, not a step count. With
                the default `start_step=0`, this is the same as running
                `max_steps` steps from scratch. The dataloader is cycled
                (re-iterated from the start) if it's exhausted before
                reaching `max_steps`.
            log_every: Print a progress line every this many steps.
            scheduler: Optional `WarmupCosineScheduler` (Module 18). If
                given, the learning rate is updated on every optimizer
                param group before each step. If `None`, whatever `lr` the
                optimizer was constructed with stays fixed throughout.
            start_step: The step number to begin counting from -- e.g.
                `loaded.step` from `load_checkpoint` (Module 25), so a
                resumed run continues the LR schedule from the correct
                point instead of restarting warmup from step 0.

        Returns:
            A list of `TrainingMetrics`, one entry per step, in order.
        """
        history: list[TrainingMetrics] = []
        step = start_step
        while step < max_steps:
            for inputs, targets in dataloader:
                if step >= max_steps:
                    break
                if scheduler is not None:
                    scheduler.step_optimizer(self.optimizer, step)
                current_lr = self.optimizer.param_groups[0]["lr"]

                loss_value = self.train_step(inputs, targets)
                history.append(TrainingMetrics(step=step, loss=loss_value, lr=current_lr))
                if step % log_every == 0:
                    print(f"step {step:5d} | loss {loss_value:.4f} | lr {current_lr:.6f}")
                step += 1
        return history
