"""Unit tests for Module 18 (LR scheduler)."""

from __future__ import annotations

import math

import pytest
import torch
from torch import nn

from app.training.optimizer import AdamW
from app.training.scheduler import WarmupCosineScheduler

MAX_LR = 1e-3
MIN_LR = 1e-4
WARMUP_STEPS = 10
MAX_STEPS = 100


def _build_scheduler() -> WarmupCosineScheduler:
    return WarmupCosineScheduler(MAX_LR, WARMUP_STEPS, MAX_STEPS, MIN_LR)


def test_lr_starts_near_zero() -> None:
    """At step 0, warmup has barely begun: lr should be 0."""
    scheduler = _build_scheduler()
    assert scheduler.get_lr(0) == 0.0


def test_lr_ramps_linearly_during_warmup() -> None:
    """Halfway through warmup, lr should be exactly half of max_lr."""
    scheduler = _build_scheduler()
    half_warmup = WARMUP_STEPS // 2
    assert scheduler.get_lr(half_warmup) == pytest.approx(MAX_LR * half_warmup / WARMUP_STEPS)


def test_lr_at_warmup_boundary_equals_max_lr() -> None:
    """The transition from warmup to cosine decay must be continuous: no jump."""
    scheduler = _build_scheduler()
    assert scheduler.get_lr(WARMUP_STEPS) == pytest.approx(MAX_LR, abs=1e-9)


def test_lr_at_max_steps_equals_min_lr() -> None:
    """The cosine decay must reach exactly min_lr at max_steps."""
    scheduler = _build_scheduler()
    assert scheduler.get_lr(MAX_STEPS) == pytest.approx(MIN_LR, abs=1e-9)


def test_lr_beyond_max_steps_stays_at_min_lr() -> None:
    """Past max_steps, lr must hold at the floor, not go negative or oscillate."""
    scheduler = _build_scheduler()
    assert scheduler.get_lr(MAX_STEPS + 500) == MIN_LR


def test_lr_is_non_increasing_throughout_the_decay_phase() -> None:
    """Cosine decay must never increase the learning rate once warmup ends."""
    scheduler = _build_scheduler()
    lrs = [scheduler.get_lr(step) for step in range(WARMUP_STEPS, MAX_STEPS + 1)]
    assert all(lrs[i] >= lrs[i + 1] - 1e-12 for i in range(len(lrs) - 1))


def test_matches_closed_form_cosine_formula_at_midpoint() -> None:
    """At the midpoint of decay (progress=0.5), lr must match the closed-form value."""
    scheduler = _build_scheduler()
    midpoint = WARMUP_STEPS + (MAX_STEPS - WARMUP_STEPS) // 2
    progress = (midpoint - WARMUP_STEPS) / (MAX_STEPS - WARMUP_STEPS)
    expected = MIN_LR + 0.5 * (MAX_LR - MIN_LR) * (1 + math.cos(math.pi * progress))
    assert scheduler.get_lr(midpoint) == pytest.approx(expected)


def test_zero_warmup_steps_starts_immediately_at_max_lr() -> None:
    """With warmup_steps=0, there is no ramp: step 0 should already be at max_lr."""
    scheduler = WarmupCosineScheduler(MAX_LR, warmup_steps=0, max_steps=MAX_STEPS, min_lr=MIN_LR)
    assert scheduler.get_lr(0) == pytest.approx(MAX_LR)


def test_invalid_max_lr_raises() -> None:
    with pytest.raises(ValueError, match="max_lr must be positive"):
        WarmupCosineScheduler(max_lr=0.0, warmup_steps=10, max_steps=100)


def test_invalid_min_lr_raises() -> None:
    with pytest.raises(ValueError, match="min_lr must be in"):
        WarmupCosineScheduler(max_lr=1e-3, warmup_steps=10, max_steps=100, min_lr=2e-3)


def test_negative_warmup_steps_raises() -> None:
    with pytest.raises(ValueError, match="warmup_steps must be >= 0"):
        WarmupCosineScheduler(max_lr=1e-3, warmup_steps=-1, max_steps=100)


def test_max_steps_not_greater_than_warmup_raises() -> None:
    with pytest.raises(ValueError, match="must be greater than warmup_steps"):
        WarmupCosineScheduler(max_lr=1e-3, warmup_steps=100, max_steps=100)


def test_step_optimizer_updates_all_param_groups() -> None:
    """step_optimizer must write get_lr(step) into every param group."""
    scheduler = _build_scheduler()
    param = nn.Parameter(torch.zeros(3))
    optimizer = AdamW([param], lr=999.0)  # sentinel value, should get overwritten

    lr = scheduler.step_optimizer(optimizer, step=WARMUP_STEPS)
    assert lr == pytest.approx(MAX_LR, abs=1e-9)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(MAX_LR, abs=1e-9)
