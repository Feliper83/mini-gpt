"""Unit tests for Module 26 (Fine-tuning)."""

from __future__ import annotations

import pytest
import torch

from app.training.finetune import (
    count_trainable_parameters,
    freeze_all_parameters,
    unfreeze_last_n_blocks,
)
from app.training.optimizer import AdamW, build_param_groups
from app.transformer.gpt_model import GPTModel

VOCAB_SIZE = 30
D_MODEL = 16
N_HEADS = 4
N_LAYERS = 4
MAX_SEQ_LEN = 16


def test_freeze_all_sets_requires_grad_false_everywhere() -> None:
    """Every parameter must be frozen after freeze_all_parameters."""
    model = GPTModel(VOCAB_SIZE, D_MODEL, N_HEADS, N_LAYERS, MAX_SEQ_LEN, tie_weights=False)
    freeze_all_parameters(model)
    assert all(not p.requires_grad for p in model.parameters())


def test_unfreeze_last_n_blocks_unfreezes_only_the_expected_parts() -> None:
    """With tie_weights=False, unfreezing must leave early blocks and the
    embedding frozen, and only touch the last N blocks + head + final norm."""
    model = GPTModel(VOCAB_SIZE, D_MODEL, N_HEADS, N_LAYERS, MAX_SEQ_LEN, tie_weights=False)
    freeze_all_parameters(model)
    unfreeze_last_n_blocks(model, n_trainable_blocks=1)

    assert not model.token_embedding.embedding.weight.requires_grad
    assert not any(p.requires_grad for p in model.transformer.blocks[0].parameters())
    assert not any(p.requires_grad for p in model.transformer.blocks[1].parameters())
    assert not any(p.requires_grad for p in model.transformer.blocks[2].parameters())
    assert all(p.requires_grad for p in model.transformer.blocks[3].parameters())
    assert all(p.requires_grad for p in model.output_head.parameters())
    assert all(p.requires_grad for p in model.transformer.final_norm.parameters())


def test_unfreeze_zero_blocks_only_unfreezes_head_and_final_norm() -> None:
    """n_trainable_blocks=0 must still unfreeze the head and final norm, no blocks."""
    model = GPTModel(VOCAB_SIZE, D_MODEL, N_HEADS, N_LAYERS, MAX_SEQ_LEN, tie_weights=False)
    freeze_all_parameters(model)
    unfreeze_last_n_blocks(model, n_trainable_blocks=0)

    assert not any(
        p.requires_grad for block in model.transformer.blocks for p in block.parameters()
    )
    assert all(p.requires_grad for p in model.output_head.parameters())


def test_invalid_n_trainable_blocks_raises() -> None:
    """n_trainable_blocks outside [0, total_blocks] must fail loudly."""
    model = GPTModel(VOCAB_SIZE, D_MODEL, N_HEADS, N_LAYERS, MAX_SEQ_LEN)
    with pytest.raises(ValueError, match="must be in"):
        unfreeze_last_n_blocks(model, n_trainable_blocks=-1)
    with pytest.raises(ValueError, match="must be in"):
        unfreeze_last_n_blocks(model, n_trainable_blocks=N_LAYERS + 1)


def test_weight_tying_couples_output_head_and_embedding_freezing() -> None:
    """With tie_weights=True (the default), unfreezing the output head must
    ALSO unfreeze the token embedding -- they are the same Parameter object.
    This is a documented gotcha, not a bug."""
    model = GPTModel(VOCAB_SIZE, D_MODEL, N_HEADS, N_LAYERS, MAX_SEQ_LEN, tie_weights=True)
    freeze_all_parameters(model)
    unfreeze_last_n_blocks(model, n_trainable_blocks=0)

    assert model.token_embedding.embedding.weight.requires_grad


def test_count_trainable_parameters_matches_manual_count() -> None:
    """The helper's counts must match a direct manual sum."""
    model = GPTModel(VOCAB_SIZE, D_MODEL, N_HEADS, N_LAYERS, MAX_SEQ_LEN, tie_weights=False)
    freeze_all_parameters(model)
    unfreeze_last_n_blocks(model, n_trainable_blocks=1)

    trainable, total = count_trainable_parameters(model)
    expected_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    expected_total = sum(p.numel() for p in model.parameters())
    assert trainable == expected_trainable
    assert total == expected_total
    assert 0 < trainable < total


def test_frozen_parameters_receive_no_gradient() -> None:
    """After backward(), frozen params must have grad=None; unfrozen ones must not."""
    model = GPTModel(VOCAB_SIZE, D_MODEL, N_HEADS, N_LAYERS, MAX_SEQ_LEN, tie_weights=False)
    freeze_all_parameters(model)
    unfreeze_last_n_blocks(model, n_trainable_blocks=1)

    inputs = torch.randint(0, VOCAB_SIZE, (2, 8))
    logits = model(inputs)
    logits.sum().backward()

    assert model.transformer.blocks[0].attention.query_proj.weight.grad is None
    assert model.transformer.blocks[-1].attention.query_proj.weight.grad is not None


def test_build_param_groups_excludes_frozen_parameters() -> None:
    """Module 17's build_param_groups must only include trainable parameters."""
    model = GPTModel(VOCAB_SIZE, D_MODEL, N_HEADS, N_LAYERS, MAX_SEQ_LEN, tie_weights=False)
    freeze_all_parameters(model)
    unfreeze_last_n_blocks(model, n_trainable_blocks=1)

    trainable, _ = count_trainable_parameters(model)
    decay_group, no_decay_group = build_param_groups(model, weight_decay=0.1)
    grouped_count = sum(p.numel() for p in decay_group["params"]) + sum(
        p.numel() for p in no_decay_group["params"]
    )
    assert grouped_count == trainable


def test_finetuning_only_updates_unfrozen_weights() -> None:
    """End-to-end: training with frozen layers must leave the frozen block's
    weights byte-identical, while the unfrozen block's weights change."""
    torch.manual_seed(0)
    model = GPTModel(VOCAB_SIZE, D_MODEL, N_HEADS, N_LAYERS, MAX_SEQ_LEN, tie_weights=False)
    freeze_all_parameters(model)
    unfreeze_last_n_blocks(model, n_trainable_blocks=1)

    optimizer = AdamW(build_param_groups(model, weight_decay=0.0), lr=1e-2)

    frozen_before = model.transformer.blocks[0].attention.query_proj.weight.clone()
    unfrozen_before = model.transformer.blocks[-1].attention.query_proj.weight.clone()

    inputs = torch.randint(0, VOCAB_SIZE, (2, 8))
    targets = torch.randint(0, VOCAB_SIZE, (2, 8))
    for _ in range(10):
        logits = model(inputs)
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, VOCAB_SIZE), targets.reshape(-1)
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    frozen_after = model.transformer.blocks[0].attention.query_proj.weight
    unfrozen_after = model.transformer.blocks[-1].attention.query_proj.weight

    assert torch.equal(frozen_before, frozen_after)
    assert not torch.equal(unfrozen_before, unfrozen_after)
