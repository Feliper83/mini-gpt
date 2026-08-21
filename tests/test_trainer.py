"""Unit tests for Module 16 (Training loop)."""

from __future__ import annotations

import copy

import torch

from app.dataset.dataloader import create_dataloader
from app.dataset.token_dataset import TokenDataset
from app.training.scheduler import WarmupCosineScheduler
from app.training.trainer import Trainer
from app.transformer.gpt_model import GPTModel

VOCAB_SIZE = 20
D_MODEL = 16
N_HEADS = 4
N_LAYERS = 2
MAX_SEQ_LEN = 16


def _build_trainer(lr: float = 1e-2) -> Trainer:
    model = GPTModel(VOCAB_SIZE, D_MODEL, N_HEADS, N_LAYERS, MAX_SEQ_LEN)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    return Trainer(model, optimizer)


def test_train_step_returns_a_finite_loss() -> None:
    """A single training step must return a finite scalar loss."""
    trainer = _build_trainer()
    inputs = torch.randint(0, VOCAB_SIZE, (2, 8))
    targets = torch.randint(0, VOCAB_SIZE, (2, 8))

    loss = trainer.train_step(inputs, targets)
    assert isinstance(loss, float)
    assert loss == loss  # not NaN
    assert loss != float("inf")


def test_train_step_updates_model_parameters() -> None:
    """After a training step, at least one parameter must have changed value."""
    trainer = _build_trainer()
    inputs = torch.randint(0, VOCAB_SIZE, (2, 8))
    targets = torch.randint(0, VOCAB_SIZE, (2, 8))

    before = copy.deepcopy(list(trainer.model.parameters()))
    trainer.train_step(inputs, targets)
    after = list(trainer.model.parameters())

    changed = any(not torch.equal(b, a) for b, a in zip(before, after, strict=True))
    assert changed


def test_train_step_overfits_a_tiny_batch() -> None:
    """The single most important integration test in this project: train
    repeatedly on ONE tiny batch and confirm the loss collapses toward 0.
    If the model wiring, loss, or optimizer step were broken, this MUST fail."""
    torch.manual_seed(0)
    trainer = _build_trainer(lr=1e-2)

    inputs = torch.randint(0, VOCAB_SIZE, (4, 8))
    targets = torch.randint(0, VOCAB_SIZE, (4, 8))

    first_loss = trainer.train_step(inputs, targets)
    final_loss = first_loss
    for _ in range(300):
        final_loss = trainer.train_step(inputs, targets)

    assert final_loss < first_loss * 0.1
    assert final_loss < 0.5


def test_fit_runs_for_exactly_max_steps() -> None:
    """fit() must produce exactly max_steps history entries, correctly numbered."""
    trainer = _build_trainer()
    token_ids = (list(range(VOCAB_SIZE)) * 8)[:50]
    dataset = TokenDataset(token_ids, block_size=8)
    dataloader = create_dataloader(dataset, batch_size=4, shuffle=True)

    history = trainer.fit(dataloader, max_steps=7, log_every=100)

    assert len(history) == 7
    assert [entry.step for entry in history] == list(range(7))


def test_fit_cycles_the_dataloader_when_exhausted() -> None:
    """If max_steps exceeds one epoch's worth of batches, fit() must keep
    going by cycling back through the dataloader instead of stopping early."""
    trainer = _build_trainer()
    token_ids = list(range(20))  # small corpus -> few batches per epoch
    dataset = TokenDataset(token_ids, block_size=4)
    dataloader = create_dataloader(dataset, batch_size=2, shuffle=True)
    batches_per_epoch = len(dataloader)

    history = trainer.fit(dataloader, max_steps=batches_per_epoch * 3, log_every=100)
    assert len(history) == batches_per_epoch * 3


def test_fit_applies_the_scheduler_lr_when_provided() -> None:
    """Passing a scheduler must make the optimizer's lr follow get_lr(step)
    exactly, instead of staying fixed at whatever the optimizer started with."""
    trainer = _build_trainer(lr=999.0)  # sentinel: should never actually be used
    token_ids = (list(range(VOCAB_SIZE)) * 8)[:50]
    dataset = TokenDataset(token_ids, block_size=8)
    dataloader = create_dataloader(dataset, batch_size=4, shuffle=False)

    scheduler = WarmupCosineScheduler(max_lr=1e-3, warmup_steps=3, max_steps=20, min_lr=1e-4)
    history = trainer.fit(dataloader, max_steps=5, log_every=100, scheduler=scheduler)

    for entry in history:
        assert entry.lr == scheduler.get_lr(entry.step)


def test_fit_without_a_scheduler_keeps_lr_fixed() -> None:
    """Without a scheduler, the optimizer's lr must stay exactly as constructed."""
    trainer = _build_trainer(lr=1e-2)
    token_ids = (list(range(VOCAB_SIZE)) * 8)[:50]
    dataset = TokenDataset(token_ids, block_size=8)
    dataloader = create_dataloader(dataset, batch_size=4, shuffle=False)

    history = trainer.fit(dataloader, max_steps=5, log_every=100)
    assert all(entry.lr == 1e-2 for entry in history)


def test_fit_with_start_step_continues_numbering_instead_of_restarting() -> None:
    """start_step must offset both the step numbering and (critically) what
    the scheduler sees -- a resumed run must NOT re-enter warmup."""
    trainer = _build_trainer()
    token_ids = (list(range(VOCAB_SIZE)) * 8)[:50]
    dataset = TokenDataset(token_ids, block_size=8)
    dataloader = create_dataloader(dataset, batch_size=4, shuffle=False)

    scheduler = WarmupCosineScheduler(max_lr=1e-3, warmup_steps=100, max_steps=1000, min_lr=1e-4)
    history = trainer.fit(
        dataloader, max_steps=155, log_every=100, scheduler=scheduler, start_step=150
    )

    assert [entry.step for entry in history] == list(range(150, 155))
    for entry in history:
        assert entry.lr == scheduler.get_lr(entry.step)
        # Past warmup_steps=100, so lr must NOT be a small warmup-region value.
        assert entry.lr > scheduler.get_lr(50)


def test_fit_default_start_step_is_zero_and_backward_compatible() -> None:
    """Omitting start_step must behave exactly as before: numbering from 0."""
    trainer = _build_trainer()
    token_ids = (list(range(VOCAB_SIZE)) * 8)[:50]
    dataset = TokenDataset(token_ids, block_size=8)
    dataloader = create_dataloader(dataset, batch_size=4, shuffle=False)

    history = trainer.fit(dataloader, max_steps=5, log_every=100)
    assert [entry.step for entry in history] == list(range(5))
