"""Unit tests for Module 20 (Sampling)."""

from __future__ import annotations

import torch

from app.inference.generator import Generator
from app.inference.sampling import sample_from_logits
from app.transformer.gpt_model import GPTModel

VOCAB_SIZE = 20
D_MODEL = 16
N_HEADS = 4
N_LAYERS = 2
MAX_SEQ_LEN = 16


def test_returns_a_valid_token_id() -> None:
    """The sampled id must always fall within [0, vocab_size)."""
    logits = torch.randn(VOCAB_SIZE)
    for _ in range(20):
        token_id = sample_from_logits(logits)
        assert 0 <= token_id < VOCAB_SIZE


def test_sampling_frequency_matches_softmax_probability() -> None:
    """Over many draws, empirical sampling frequency must track the true
    softmax probability of each class -- the core statistical property."""
    torch.manual_seed(0)
    logits = torch.tensor([2.0, 1.0, 0.5, -1.0])
    true_probs = torch.softmax(logits, dim=-1)

    n_draws = 5000
    counts = torch.zeros(4)
    for _ in range(n_draws):
        counts[sample_from_logits(logits)] += 1
    empirical_probs = counts / n_draws

    assert torch.allclose(empirical_probs, true_probs, atol=0.03)


def test_uniform_logits_give_roughly_uniform_sampling() -> None:
    """With all logits equal, every class should be drawn roughly equally often."""
    torch.manual_seed(0)
    logits = torch.zeros(5)
    n_draws = 5000
    counts = torch.zeros(5)
    for _ in range(n_draws):
        counts[sample_from_logits(logits)] += 1
    empirical_probs = counts / n_draws

    assert torch.allclose(empirical_probs, torch.full((5,), 0.2), atol=0.03)


def test_extremely_peaked_distribution_almost_always_picks_the_dominant_class() -> None:
    """A distribution with one hugely dominant logit should behave almost
    like greedy selection -- a sanity bridge to Module 19's argmax default."""
    torch.manual_seed(0)
    logits = torch.tensor([-20.0, -20.0, 20.0, -20.0])  # class 2 dominates completely
    results = [sample_from_logits(logits) for _ in range(200)]
    assert results.count(2) == 200


def test_reproducible_with_a_seeded_generator() -> None:
    """Passing a seeded torch.Generator must make sampling deterministic."""
    logits = torch.randn(VOCAB_SIZE)
    gen_a = torch.Generator().manual_seed(42)
    gen_b = torch.Generator().manual_seed(42)

    result_a = [sample_from_logits(logits, generator=gen_a) for _ in range(10)]
    result_b = [sample_from_logits(logits, generator=gen_b) for _ in range(10)]

    assert result_a == result_b


def test_plugs_into_generator_and_produces_varying_output() -> None:
    """Unlike Module 19's deterministic greedy default, stochastic sampling
    plugged into Generator must be able to produce DIFFERENT completions
    for the same prompt across calls."""
    torch.manual_seed(0)
    model = GPTModel(VOCAB_SIZE, D_MODEL, N_HEADS, N_LAYERS, MAX_SEQ_LEN)
    generator = Generator(model, max_seq_len=MAX_SEQ_LEN, sample_fn=sample_from_logits)

    prompt = [1, 2, 3]
    results = {tuple(generator.generate(prompt, max_new_tokens=10)) for _ in range(10)}

    assert len(results) > 1  # at least some variation across the 10 draws
