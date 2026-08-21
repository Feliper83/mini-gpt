"""Unit tests for Module 17 (Optimizer)."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from app.training.optimizer import AdamW, build_param_groups
from app.transformer.gpt_model import GPTModel


def test_invalid_lr_raises() -> None:
    """A non-positive learning rate must fail loudly."""
    with pytest.raises(ValueError, match="lr must be positive"):
        AdamW([nn.Parameter(torch.zeros(1))], lr=0.0)


def test_invalid_betas_raise() -> None:
    """Betas outside [0, 1) must fail loudly."""
    with pytest.raises(ValueError, match="betas must be"):
        AdamW([nn.Parameter(torch.zeros(1))], betas=(1.0, 0.999))


def test_invalid_eps_raises() -> None:
    """A non-positive eps must fail loudly."""
    with pytest.raises(ValueError, match="eps must be positive"):
        AdamW([nn.Parameter(torch.zeros(1))], eps=0.0)


def test_step_skips_parameters_with_no_gradient() -> None:
    """A parameter with grad=None must be left untouched, not raise."""
    param = nn.Parameter(torch.tensor([1.0, 2.0]))
    optimizer = AdamW([param], lr=0.1)
    optimizer.step()  # no .grad was ever set
    assert torch.equal(param, torch.tensor([1.0, 2.0]))


def test_matches_pytorch_adamw_reference_over_many_steps() -> None:
    """Our from-scratch AdamW must numerically match torch.optim.AdamW,
    given identical initial parameters and identical gradients each step."""
    torch.manual_seed(0)
    param_ours = nn.Parameter(torch.randn(10))
    param_ref = nn.Parameter(param_ours.data.clone())

    ours = AdamW([param_ours], lr=1e-2, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)
    reference = torch.optim.AdamW(
        [param_ref], lr=1e-2, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01
    )

    for _ in range(30):
        grad = torch.randn(10)
        param_ours.grad = grad.clone()
        param_ref.grad = grad.clone()
        ours.step()
        reference.step()

    assert torch.allclose(param_ours, param_ref, atol=1e-5)


def test_converges_on_a_toy_quadratic() -> None:
    """Minimizing f(x) = (x - target)^2 must converge close to `target`."""
    target = 3.0
    x = nn.Parameter(torch.tensor([0.0]))
    optimizer = AdamW([x], lr=0.1)

    for _ in range(200):
        loss = (x - target) ** 2
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    assert x.item() == pytest.approx(target, abs=0.05)


def test_zero_weight_decay_matches_reference_without_decay() -> None:
    """With weight_decay=0, our AdamW must still match the PyTorch reference."""
    torch.manual_seed(1)
    param_ours = nn.Parameter(torch.randn(5))
    param_ref = nn.Parameter(param_ours.data.clone())

    ours = AdamW([param_ours], lr=1e-2, weight_decay=0.0)
    reference = torch.optim.AdamW([param_ref], lr=1e-2, weight_decay=0.0)

    for _ in range(10):
        grad = torch.randn(5)
        param_ours.grad = grad.clone()
        param_ref.grad = grad.clone()
        ours.step()
        reference.step()

    assert torch.allclose(param_ours, param_ref, atol=1e-5)


# --- build_param_groups ---


def test_param_groups_split_by_dimensionality() -> None:
    """2D+ params go in the decay group; 1D params go in the no-decay group."""

    class TinyModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = nn.Linear(4, 4)  # weight is 2D, bias is 1D

    model = TinyModel()
    decay_group, no_decay_group = build_param_groups(model, weight_decay=0.1)

    assert decay_group["weight_decay"] == 0.1
    assert no_decay_group["weight_decay"] == 0.0
    assert any(p is model.linear.weight for p in decay_group["params"])
    assert any(p is model.linear.bias for p in no_decay_group["params"])
    assert not any(p is model.linear.bias for p in decay_group["params"])


def test_param_groups_cover_every_parameter_exactly_once() -> None:
    """Every parameter in the model must appear in exactly one of the two groups,
    with tied (shared) parameters counted only once -- consistent with Module 14."""
    model = GPTModel(vocab_size=30, d_model=16, n_heads=4, n_layers=2, max_seq_len=16)
    decay_group, no_decay_group = build_param_groups(model, weight_decay=0.1)

    grouped_ids = {id(p) for p in decay_group["params"]} | {
        id(p) for p in no_decay_group["params"]
    }
    model_param_ids = {id(p) for p in model.parameters()}

    assert grouped_ids == model_param_ids


def test_layernorm_gamma_beta_land_in_no_decay_group() -> None:
    """LayerNorm's gamma and beta (both 1D) must never receive weight decay."""
    model = GPTModel(vocab_size=30, d_model=16, n_heads=4, n_layers=2, max_seq_len=16)
    _, no_decay_group = build_param_groups(model, weight_decay=0.1)

    first_block = model.transformer.blocks[0]
    gamma = first_block.attention_residual.norm.gamma
    beta = first_block.attention_residual.norm.beta

    assert any(p is gamma for p in no_decay_group["params"])
    assert any(p is beta for p in no_decay_group["params"])
