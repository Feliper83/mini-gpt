"""Unit tests for Module 15 (Loss function)."""

from __future__ import annotations

import math

import pytest
import torch
from torch.nn.functional import cross_entropy as torch_cross_entropy

from app.training.loss import CrossEntropyLoss

VOCAB_SIZE = 20


def test_loss_is_a_scalar() -> None:
    """The loss must reduce to a single scalar value."""
    loss_fn = CrossEntropyLoss()
    logits = torch.randn(4, 6, VOCAB_SIZE)
    targets = torch.randint(0, VOCAB_SIZE, (4, 6))
    loss = loss_fn(logits, targets)
    assert loss.shape == ()


def test_matches_pytorch_cross_entropy_reference() -> None:
    """Our from-scratch loss must numerically match torch's cross_entropy."""
    loss_fn = CrossEntropyLoss()
    torch.manual_seed(0)
    logits = torch.randn(4, 6, VOCAB_SIZE)
    targets = torch.randint(0, VOCAB_SIZE, (4, 6))

    ours = loss_fn(logits, targets)
    reference = torch_cross_entropy(logits.reshape(-1, VOCAB_SIZE), targets.reshape(-1))

    assert torch.allclose(ours, reference, atol=1e-5)


def test_uniform_logits_give_loss_equal_to_log_vocab_size() -> None:
    """With all logits equal, softmax is uniform (1/V each), so
    loss = -log(1/V) = log(V) exactly -- a clean closed-form special case."""
    loss_fn = CrossEntropyLoss()
    logits = torch.zeros(2, 3, VOCAB_SIZE)  # every logit identical -> uniform softmax
    targets = torch.randint(0, VOCAB_SIZE, (2, 3))

    loss = loss_fn(logits, targets)
    assert loss.item() == pytest.approx(math.log(VOCAB_SIZE), abs=1e-5)


def test_loss_decreases_as_confidence_in_correct_class_increases() -> None:
    """As the logit for the TRUE class grows relative to the others, loss must
    monotonically decrease -- the model is "more confidently correct"."""
    loss_fn = CrossEntropyLoss()
    correct_class = 3
    confidences = [0.0, 2.0, 5.0, 10.0]
    losses = []

    for confidence in confidences:
        logits = torch.zeros(1, 1, VOCAB_SIZE)
        logits[0, 0, correct_class] = confidence
        targets = torch.tensor([[correct_class]])
        losses.append(loss_fn(logits, targets).item())

    assert losses == sorted(losses, reverse=True)  # strictly non-increasing


def test_confident_correct_prediction_gives_near_zero_loss() -> None:
    """A near-one-hot correct prediction must have loss close to 0."""
    loss_fn = CrossEntropyLoss()
    correct_class = 5
    logits = torch.full((1, 1, VOCAB_SIZE), -20.0)
    logits[0, 0, correct_class] = 20.0
    targets = torch.tensor([[correct_class]])

    loss = loss_fn(logits, targets)
    assert loss.item() < 1e-6


def test_confident_wrong_prediction_gives_large_loss() -> None:
    """A near-one-hot WRONG prediction must have a large loss."""
    loss_fn = CrossEntropyLoss()
    logits = torch.full((1, 1, VOCAB_SIZE), -20.0)
    logits[0, 0, 0] = 20.0  # model is very confident in class 0
    targets = torch.tensor([[1]])  # but the true class is 1

    loss = loss_fn(logits, targets)
    assert loss.item() > 10.0


def test_gradient_flows_to_logits() -> None:
    """Backprop must produce a gradient on the logits."""
    loss_fn = CrossEntropyLoss()
    logits = torch.randn(2, 3, VOCAB_SIZE, requires_grad=True)
    targets = torch.randint(0, VOCAB_SIZE, (2, 3))

    loss = loss_fn(logits, targets)
    loss.backward()

    assert logits.grad is not None
    assert not torch.all(logits.grad == 0)
