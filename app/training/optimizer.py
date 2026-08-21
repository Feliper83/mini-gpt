"""AdamW optimizer, implemented from scratch, plus GPT-2 style parameter grouping."""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import nn


class AdamW(torch.optim.Optimizer):
    """AdamW: Adam with decoupled weight decay, implemented from scratch.

    m_t = beta1 * m_{t-1} + (1 - beta1) * g_t          (first moment: EMA of gradient)
    v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2         (second moment: EMA of squared gradient)
    m_hat_t = m_t / (1 - beta1^t)                        (bias correction)
    v_hat_t = v_t / (1 - beta2^t)
    theta_t = theta_{t-1} * (1 - lr * weight_decay) - lr * m_hat_t / (sqrt(v_hat_t) + eps)

    Unlike plain Adam with L2 regularization (which folds weight decay into
    the gradient itself, so it gets distorted by the adaptive 1/sqrt(v_hat)
    scaling), AdamW applies weight decay as a separate, DECOUPLED step
    directly on the parameters -- the fix proposed by Loshchilov & Hutter
    (2019), and what the "W" stands for.
    """

    def __init__(
        self,
        params: Iterable[nn.Parameter] | Iterable[dict],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ) -> None:
        """Builds the optimizer.

        Args:
            params: Parameters to optimize, or parameter group dicts (see
                `build_param_groups` for the GPT-2 convention of excluding
                1D parameters from weight decay).
            lr: Learning rate.
            betas: (beta1, beta2), the exponential decay rates for the
                first and second moment estimates.
            eps: Small constant added for numerical stability in the denominator.
            weight_decay: Decoupled weight decay coefficient.

        Raises:
            ValueError: If any hyperparameter is out of its valid range.
        """
        if lr <= 0.0:
            raise ValueError(f"lr must be positive, got {lr}")
        if not (0.0 <= betas[0] < 1.0 and 0.0 <= betas[1] < 1.0):
            raise ValueError(f"betas must be in [0, 1), got {betas}")
        if eps <= 0.0:
            raise ValueError(f"eps must be positive, got {eps}")

        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay}
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self) -> None:  # type: ignore[override]
        """Performs a single optimization step over all parameter groups."""
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for param in group["params"]:
                if param.grad is None:
                    continue

                grad = param.grad
                state = self.state[param]

                if len(state) == 0:
                    state["step"] = 0
                    state["m"] = torch.zeros_like(param)
                    state["v"] = torch.zeros_like(param)

                state["step"] += 1
                step = state["step"]
                m, v = state["m"], state["v"]

                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                bias_correction1 = 1 - beta1**step
                bias_correction2 = 1 - beta2**step
                m_hat = m / bias_correction1
                v_hat = v / bias_correction2

                if weight_decay != 0.0:
                    param.mul_(1 - lr * weight_decay)  # decoupled weight decay

                param.addcdiv_(m_hat, v_hat.sqrt().add_(eps), value=-lr)


def build_param_groups(model: nn.Module, weight_decay: float) -> list[dict]:
    """Splits a model's parameters into GPT-2 style weight-decay groups.

    Only 2D+ parameters (matmul weight matrices: attention/FFN/embedding
    weights) get weight decay. 1D parameters (biases, LayerNorm gamma/beta)
    are excluded: decaying LayerNorm's gamma toward 0 would actively fight
    its purpose (gamma starts at 1, meaning "no rescaling" -- see Module 7),
    and there is little regularization benefit to decaying biases.

    Args:
        model: The model whose parameters will be grouped.
        weight_decay: Weight decay coefficient applied to the decayed group.

    Returns:
        Two parameter-group dicts suitable for passing to `AdamW`.
    """
    decay_params = [p for p in model.parameters() if p.requires_grad and p.dim() >= 2]
    no_decay_params = [p for p in model.parameters() if p.requires_grad and p.dim() < 2]

    return [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
