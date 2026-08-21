"""Unit tests for Module 19 (Text generation)."""

from __future__ import annotations

import pytest

from app.inference.generator import Generator
from app.transformer.gpt_model import GPTModel

VOCAB_SIZE = 20
D_MODEL = 16
N_HEADS = 4
N_LAYERS = 2
MAX_SEQ_LEN = 16


def _build_generator() -> Generator:
    model = GPTModel(VOCAB_SIZE, D_MODEL, N_HEADS, N_LAYERS, MAX_SEQ_LEN)
    return Generator(model, max_seq_len=MAX_SEQ_LEN)


def test_generate_returns_correct_total_length() -> None:
    """Output length must equal prompt length plus max_new_tokens."""
    generator = _build_generator()
    prompt = [1, 2, 3]
    result = generator.generate(prompt, max_new_tokens=10)
    assert len(result) == len(prompt) + 10


def test_generate_preserves_the_prompt_prefix() -> None:
    """Generation must never alter the original prompt tokens."""
    generator = _build_generator()
    prompt = [4, 5, 6, 7]
    result = generator.generate(prompt, max_new_tokens=5)
    assert result[: len(prompt)] == prompt


def test_generate_empty_prompt_raises() -> None:
    """An empty prompt has no context to generate from -- must fail loudly."""
    generator = _build_generator()
    with pytest.raises(ValueError, match="at least one token"):
        generator.generate([], max_new_tokens=5)


def test_greedy_generation_is_deterministic() -> None:
    """Greedy (argmax) selection has no randomness: repeated calls must match."""
    generator = _build_generator()
    prompt = [1, 2, 3]
    result_a = generator.generate(prompt, max_new_tokens=10)
    result_b = generator.generate(prompt, max_new_tokens=10)
    assert result_a == result_b


def test_generate_truncates_context_beyond_max_seq_len() -> None:
    """Generating more tokens than max_seq_len must not crash: the context
    fed to the model must always be truncated to the most recent max_seq_len tokens."""
    generator = _build_generator()
    prompt = [0, 1, 2]
    result = generator.generate(prompt, max_new_tokens=MAX_SEQ_LEN * 2)
    assert len(result) == len(prompt) + MAX_SEQ_LEN * 2


def test_custom_sample_fn_is_used() -> None:
    """A custom sample_fn must fully control token selection."""
    model = GPTModel(VOCAB_SIZE, D_MODEL, N_HEADS, N_LAYERS, MAX_SEQ_LEN)
    fixed_token = 7
    generator = Generator(model, max_seq_len=MAX_SEQ_LEN, sample_fn=lambda _logits: fixed_token)

    result = generator.generate([1, 2], max_new_tokens=5)
    assert result[2:] == [fixed_token] * 5


def test_generate_uses_eval_mode_internally() -> None:
    """During generation, the model must be in eval mode (e.g. so dropout
    is disabled), even if it was in train() mode beforehand."""
    model = GPTModel(VOCAB_SIZE, D_MODEL, N_HEADS, N_LAYERS, MAX_SEQ_LEN)
    model.train()
    observed_training_states = []

    def spy_sample_fn(logits):
        observed_training_states.append(model.training)
        return int(logits.argmax().item())

    generator = Generator(model, max_seq_len=MAX_SEQ_LEN, sample_fn=spy_sample_fn)
    generator.generate([1, 2, 3], max_new_tokens=3)

    assert observed_training_states == [False] * 3


def test_generate_restores_the_models_prior_training_mode() -> None:
    """Regression test: generate() must restore whatever mode the model was
    in before the call, not permanently leave it in eval mode -- a caller
    that had called model.train() once and interleaves manual training
    steps with generate() previews must not have dropout silently disabled
    for the rest of training."""
    model = GPTModel(VOCAB_SIZE, D_MODEL, N_HEADS, N_LAYERS, MAX_SEQ_LEN)
    model.train()
    generator = Generator(model, max_seq_len=MAX_SEQ_LEN)

    generator.generate([1, 2, 3], max_new_tokens=3)
    assert model.training is True


def test_generate_leaves_an_already_eval_model_in_eval_mode() -> None:
    """If the model was already in eval mode, it must stay that way after."""
    model = GPTModel(VOCAB_SIZE, D_MODEL, N_HEADS, N_LAYERS, MAX_SEQ_LEN)
    model.eval()
    generator = Generator(model, max_seq_len=MAX_SEQ_LEN)

    generator.generate([1, 2, 3], max_new_tokens=3)
    assert model.training is False


def test_generate_does_not_accumulate_gradients() -> None:
    """Generation must run under no_grad: parameters must receive no gradient."""
    model = GPTModel(VOCAB_SIZE, D_MODEL, N_HEADS, N_LAYERS, MAX_SEQ_LEN)
    generator = Generator(model, max_seq_len=MAX_SEQ_LEN)

    generator.generate([1, 2, 3], max_new_tokens=5)

    assert all(p.grad is None for p in model.parameters())
