"""Unit tests for Module 24 (Saving checkpoints) and Module 25 (Loading checkpoints)."""

from __future__ import annotations

import pytest
import torch

from app.training.checkpoint import ModelConfig, load_checkpoint, save_checkpoint
from app.training.optimizer import AdamW
from app.transformer.gpt_model import GPTModel

VOCAB_SIZE = 20
D_MODEL = 16
N_HEADS = 4
N_LAYERS = 2
MAX_SEQ_LEN = 16


def _build_model_and_optimizer() -> tuple[GPTModel, AdamW]:
    model = GPTModel(VOCAB_SIZE, D_MODEL, N_HEADS, N_LAYERS, MAX_SEQ_LEN)
    optimizer = AdamW(model.parameters(), lr=1e-2)
    return model, optimizer


def test_save_creates_a_file(tmp_path) -> None:
    """save_checkpoint must actually write a file to disk."""
    model, optimizer = _build_model_and_optimizer()
    config = ModelConfig(VOCAB_SIZE, D_MODEL, N_HEADS, N_LAYERS, MAX_SEQ_LEN)
    path = tmp_path / "ckpt.pt"

    save_checkpoint(path, model, optimizer, step=1, config=config)
    assert path.exists()


def test_save_creates_missing_parent_directories(tmp_path) -> None:
    """A nested destination path must not require the directory to pre-exist."""
    model, optimizer = _build_model_and_optimizer()
    config = ModelConfig(VOCAB_SIZE, D_MODEL, N_HEADS, N_LAYERS, MAX_SEQ_LEN)
    path = tmp_path / "nested" / "dir" / "ckpt.pt"

    save_checkpoint(path, model, optimizer, step=1, config=config)
    assert path.exists()


def test_checkpoint_contains_all_expected_keys(tmp_path) -> None:
    """The saved dict must contain every field needed to resume training."""
    model, optimizer = _build_model_and_optimizer()
    config = ModelConfig(VOCAB_SIZE, D_MODEL, N_HEADS, N_LAYERS, MAX_SEQ_LEN)
    path = tmp_path / "ckpt.pt"

    save_checkpoint(path, model, optimizer, step=42, config=config)
    raw = torch.load(path, weights_only=False)

    assert set(raw.keys()) == {
        "model_state_dict",
        "optimizer_state_dict",
        "step",
        "config",
        "extra",
    }


def test_saved_model_state_dict_matches_model_exactly(tmp_path) -> None:
    """Every saved tensor must exactly equal the model's live parameters at save time."""
    model, optimizer = _build_model_and_optimizer()
    config = ModelConfig(VOCAB_SIZE, D_MODEL, N_HEADS, N_LAYERS, MAX_SEQ_LEN)
    path = tmp_path / "ckpt.pt"

    save_checkpoint(path, model, optimizer, step=1, config=config)
    raw = torch.load(path, weights_only=False)

    live_state = model.state_dict()
    for name, tensor in raw["model_state_dict"].items():
        assert torch.equal(tensor, live_state[name])


def test_saved_optimizer_state_preserves_moment_buffers(tmp_path) -> None:
    """After real optimizer steps, the saved state must include AdamW's m/v
    buffers, not just an empty/default state."""
    model, optimizer = _build_model_and_optimizer()
    inputs = torch.randint(0, VOCAB_SIZE, (2, 8))
    targets = torch.randint(0, VOCAB_SIZE, (2, 8))

    for _ in range(3):
        logits = model(inputs)
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, VOCAB_SIZE), targets.reshape(-1)
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    config = ModelConfig(VOCAB_SIZE, D_MODEL, N_HEADS, N_LAYERS, MAX_SEQ_LEN)
    path = tmp_path / "ckpt.pt"
    save_checkpoint(path, model, optimizer, step=3, config=config)

    raw = torch.load(path, weights_only=False)
    optimizer_state = raw["optimizer_state_dict"]["state"]
    assert len(optimizer_state) > 0
    first_param_state = next(iter(optimizer_state.values()))
    assert first_param_state["step"] == 3
    assert torch.any(first_param_state["m"] != 0)


def test_config_is_saved_as_a_plain_dict(tmp_path) -> None:
    """The config must round-trip via dataclasses.asdict for JSON-friendly storage."""
    model, optimizer = _build_model_and_optimizer()
    config = ModelConfig(VOCAB_SIZE, D_MODEL, N_HEADS, N_LAYERS, MAX_SEQ_LEN, dropout=0.1)
    path = tmp_path / "ckpt.pt"

    save_checkpoint(path, model, optimizer, step=1, config=config)
    raw = torch.load(path, weights_only=False)

    assert raw["config"]["vocab_size"] == VOCAB_SIZE
    assert raw["config"]["dropout"] == 0.1


def test_extra_metadata_is_saved(tmp_path) -> None:
    """Arbitrary extra metadata passed by the caller must be preserved."""
    model, optimizer = _build_model_and_optimizer()
    config = ModelConfig(VOCAB_SIZE, D_MODEL, N_HEADS, N_LAYERS, MAX_SEQ_LEN)
    path = tmp_path / "ckpt.pt"

    save_checkpoint(path, model, optimizer, step=1, config=config, extra={"loss": 1.23})
    raw = torch.load(path, weights_only=False)

    assert raw["extra"] == {"loss": 1.23}


def test_extra_defaults_to_empty_dict_not_none(tmp_path) -> None:
    """Omitting `extra` must save an empty dict, so Module 25's loader never
    has to guard against `None`."""
    model, optimizer = _build_model_and_optimizer()
    config = ModelConfig(VOCAB_SIZE, D_MODEL, N_HEADS, N_LAYERS, MAX_SEQ_LEN)
    path = tmp_path / "ckpt.pt"

    save_checkpoint(path, model, optimizer, step=1, config=config)
    raw = torch.load(path, weights_only=False)

    assert raw["extra"] == {}


# --- load_checkpoint (Module 25) ---


def test_load_reconstructs_the_correct_architecture(tmp_path) -> None:
    """The reconstructed model must have the exact shapes described by the saved config."""
    model, optimizer = _build_model_and_optimizer()
    config = ModelConfig(VOCAB_SIZE, D_MODEL, N_HEADS, N_LAYERS, MAX_SEQ_LEN)
    path = tmp_path / "ckpt.pt"
    save_checkpoint(path, model, optimizer, step=1, config=config)

    loaded = load_checkpoint(path)

    assert loaded.model.token_embedding.embedding.weight.shape == (VOCAB_SIZE, D_MODEL)
    assert len(loaded.model.transformer.blocks) == N_LAYERS


def test_load_restores_exact_weight_values(tmp_path) -> None:
    """Every loaded parameter value must exactly match the original model's."""
    model, optimizer = _build_model_and_optimizer()
    config = ModelConfig(VOCAB_SIZE, D_MODEL, N_HEADS, N_LAYERS, MAX_SEQ_LEN)
    path = tmp_path / "ckpt.pt"
    save_checkpoint(path, model, optimizer, step=1, config=config)

    loaded = load_checkpoint(path)

    original_state = model.state_dict()
    for name, tensor in loaded.model.state_dict().items():
        assert torch.equal(tensor, original_state[name])


def test_loaded_model_produces_identical_output_to_the_original(tmp_path) -> None:
    """The strongest correctness test: same input through the original and the
    reloaded model must produce byte-identical output."""
    model, optimizer = _build_model_and_optimizer()
    config = ModelConfig(VOCAB_SIZE, D_MODEL, N_HEADS, N_LAYERS, MAX_SEQ_LEN)
    path = tmp_path / "ckpt.pt"
    save_checkpoint(path, model, optimizer, step=1, config=config)

    loaded = load_checkpoint(path)

    model.eval()
    loaded.model.eval()
    inputs = torch.randint(0, VOCAB_SIZE, (2, 8))
    with torch.no_grad():
        original_output = model(inputs)
        loaded_output = loaded.model(inputs)

    assert torch.equal(original_output, loaded_output)


def test_load_without_an_optimizer_only_reconstructs_the_model(tmp_path) -> None:
    """Passing no optimizer must be a valid, inference-only load path."""
    model, optimizer = _build_model_and_optimizer()
    config = ModelConfig(VOCAB_SIZE, D_MODEL, N_HEADS, N_LAYERS, MAX_SEQ_LEN)
    path = tmp_path / "ckpt.pt"
    save_checkpoint(path, model, optimizer, step=7, config=config, extra={"note": "hi"})

    loaded = load_checkpoint(path, optimizer=None)

    assert loaded.step == 7
    assert loaded.extra == {"note": "hi"}
    assert loaded.config.vocab_size == VOCAB_SIZE


def test_load_restores_optimizer_state_onto_a_fresh_optimizer(tmp_path) -> None:
    """A fresh optimizer, built over a fresh model's parameters, must receive
    the exact m/v buffers from the original training run after loading."""
    model, optimizer = _build_model_and_optimizer()
    inputs = torch.randint(0, VOCAB_SIZE, (2, 8))
    targets = torch.randint(0, VOCAB_SIZE, (2, 8))
    for _ in range(4):
        logits = model(inputs)
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, VOCAB_SIZE), targets.reshape(-1)
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    config = ModelConfig(VOCAB_SIZE, D_MODEL, N_HEADS, N_LAYERS, MAX_SEQ_LEN)
    path = tmp_path / "ckpt.pt"
    save_checkpoint(path, model, optimizer, step=4, config=config)

    fresh_model = GPTModel(VOCAB_SIZE, D_MODEL, N_HEADS, N_LAYERS, MAX_SEQ_LEN)
    fresh_optimizer = AdamW(fresh_model.parameters(), lr=1e-2)
    loaded = load_checkpoint(path, model=fresh_model, optimizer=fresh_optimizer)

    original_state = list(optimizer.state.values())
    fresh_state = list(fresh_optimizer.state.values())
    assert len(original_state) == len(fresh_state)
    for orig, fresh in zip(original_state, fresh_state, strict=True):
        assert orig["step"] == fresh["step"]
        assert torch.equal(orig["m"], fresh["m"])
        assert torch.equal(orig["v"], fresh["v"])
    assert loaded.step == 4


def test_full_save_load_and_resume_training_cycle(tmp_path) -> None:
    """End-to-end: train, save, reconstruct fresh model+optimizer, load, and
    keep training -- the loss must keep DECREASING after resume, not just
    stay finite (a broken optimizer sync would still produce finite losses,
    just ones that never improve -- see the regression test below)."""
    torch.manual_seed(0)
    model, optimizer = _build_model_and_optimizer()
    inputs = torch.randint(0, VOCAB_SIZE, (2, 8))
    targets = torch.randint(0, VOCAB_SIZE, (2, 8))

    def train_step(m: GPTModel, opt: AdamW) -> float:
        logits = m(inputs)
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, VOCAB_SIZE), targets.reshape(-1)
        )
        opt.zero_grad()
        loss.backward()
        opt.step()
        return loss.item()

    for _ in range(5):
        train_step(model, optimizer)

    config = ModelConfig(VOCAB_SIZE, D_MODEL, N_HEADS, N_LAYERS, MAX_SEQ_LEN)
    path = tmp_path / "ckpt.pt"
    save_checkpoint(path, model, optimizer, step=5, config=config)

    fresh_model = GPTModel(VOCAB_SIZE, D_MODEL, N_HEADS, N_LAYERS, MAX_SEQ_LEN)
    fresh_optimizer = AdamW(fresh_model.parameters(), lr=1e-2)
    loaded = load_checkpoint(path, model=fresh_model, optimizer=fresh_optimizer)

    losses = [train_step(loaded.model, fresh_optimizer) for _ in range(30)]
    assert all(loss == loss and loss != float("inf") for loss in losses)  # all finite
    assert losses[-1] < losses[0] * 0.5  # real learning happened, not a no-op optimizer


def test_load_checkpoint_with_optimizer_but_no_model_raises(tmp_path) -> None:
    """Regression test: passing optimizer without model must fail loudly --
    silently constructing a fresh model internally in that case is exactly
    the bug where the optimizer ends up referencing a different model's
    parameters than the one actually used for training, making resumed
    training a silent no-op."""
    model, optimizer = _build_model_and_optimizer()
    config = ModelConfig(VOCAB_SIZE, D_MODEL, N_HEADS, N_LAYERS, MAX_SEQ_LEN)
    path = tmp_path / "ckpt.pt"
    save_checkpoint(path, model, optimizer, step=1, config=config)

    fresh_model = GPTModel(VOCAB_SIZE, D_MODEL, N_HEADS, N_LAYERS, MAX_SEQ_LEN)
    fresh_optimizer = AdamW(fresh_model.parameters(), lr=1e-2)
    with pytest.raises(ValueError, match="also requires model="):
        load_checkpoint(path, optimizer=fresh_optimizer)
