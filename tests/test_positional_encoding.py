"""Unit tests for Module 6 (Positional encoding)."""

from __future__ import annotations

import math

import pytest
import torch

from app.embeddings.positional_encoding import (
    LearnedPositionalEncoding,
    SinusoidalPositionalEncoding,
)
from app.embeddings.token_embedding import TokenEmbedding

D_MODEL = 8
MAX_SEQ_LEN = 32


# --- SinusoidalPositionalEncoding ---


def test_sinusoidal_output_shape() -> None:
    """Output must have shape (seq_len, d_model)."""
    pe = SinusoidalPositionalEncoding(D_MODEL, MAX_SEQ_LEN)
    output = pe(10)
    assert output.shape == (10, D_MODEL)


def test_sinusoidal_rejects_odd_d_model() -> None:
    """d_model must be even, since dimensions come in sin/cos pairs."""
    with pytest.raises(ValueError, match="must be even"):
        SinusoidalPositionalEncoding(d_model=7, max_seq_len=MAX_SEQ_LEN)


def test_sinusoidal_values_are_bounded() -> None:
    """Every entry must lie in [-1, 1], since it's built from sin/cos."""
    pe = SinusoidalPositionalEncoding(D_MODEL, MAX_SEQ_LEN)
    output = pe(MAX_SEQ_LEN)
    assert torch.all(output >= -1.0)
    assert torch.all(output <= 1.0)


def test_sinusoidal_position_zero_matches_closed_form() -> None:
    """At pos=0, every angle is 0, so even dims are sin(0)=0 and odd dims are cos(0)=1."""
    pe = SinusoidalPositionalEncoding(D_MODEL, MAX_SEQ_LEN)
    pos_zero = pe(1)[0]
    expected = torch.tensor([0.0, 1.0] * (D_MODEL // 2))
    assert torch.allclose(pos_zero, expected, atol=1e-6)


def test_sinusoidal_matches_reference_formula() -> None:
    """The precomputed table must match a direct evaluation of the PE formula."""
    pe = SinusoidalPositionalEncoding(D_MODEL, MAX_SEQ_LEN)
    output = pe(MAX_SEQ_LEN)

    for pos in [0, 1, 5, 17]:
        for i in range(D_MODEL // 2):
            angle = pos / (10000 ** (2 * i / D_MODEL))
            assert output[pos, 2 * i].item() == pytest.approx(math.sin(angle), abs=1e-5)
            assert output[pos, 2 * i + 1].item() == pytest.approx(math.cos(angle), abs=1e-5)


def test_sinusoidal_has_no_trainable_parameters() -> None:
    """The sinusoidal encoding is a fixed buffer, not a learned parameter."""
    pe = SinusoidalPositionalEncoding(D_MODEL, MAX_SEQ_LEN)
    assert list(pe.parameters()) == []


def test_sinusoidal_seq_len_exceeding_max_raises() -> None:
    """Requesting more positions than precomputed must fail loudly, not silently truncate."""
    pe = SinusoidalPositionalEncoding(D_MODEL, MAX_SEQ_LEN)
    with pytest.raises(ValueError, match="exceeds max_seq_len"):
        pe(MAX_SEQ_LEN + 1)


# --- LearnedPositionalEncoding ---


def test_learned_output_shape() -> None:
    """Output must have shape (seq_len, d_model)."""
    pe = LearnedPositionalEncoding(D_MODEL, MAX_SEQ_LEN)
    output = pe(10)
    assert output.shape == (10, D_MODEL)


def test_learned_has_trainable_parameters() -> None:
    """Unlike the sinusoidal version, this encoding has a learned weight table."""
    pe = LearnedPositionalEncoding(D_MODEL, MAX_SEQ_LEN)
    params = list(pe.parameters())
    assert len(params) == 1
    assert params[0].shape == (MAX_SEQ_LEN, D_MODEL)


def test_learned_different_positions_give_different_vectors() -> None:
    """Distinct positions must (almost surely) map to distinct vectors under random init."""
    pe = LearnedPositionalEncoding(D_MODEL, MAX_SEQ_LEN)
    output = pe(2)
    assert not torch.equal(output[0], output[1])


def test_learned_seq_len_exceeding_max_raises() -> None:
    """Requesting more positions than the table holds must fail loudly."""
    pe = LearnedPositionalEncoding(D_MODEL, MAX_SEQ_LEN)
    with pytest.raises(ValueError, match="exceeds max_seq_len"):
        pe(MAX_SEQ_LEN + 1)


# --- Integration: closes the Module 5 exercise ---


def test_adding_positional_encoding_breaks_position_blindness() -> None:
    """Module 5 showed the same token id gives an identical vector at any position.
    Adding a positional encoding must break that: the combined vectors at two
    positions sharing the same token id must now differ.
    """
    vocab_size = 20
    token_embedding = TokenEmbedding(vocab_size, D_MODEL)
    sinusoidal_pe = SinusoidalPositionalEncoding(D_MODEL, MAX_SEQ_LEN)
    learned_pe = LearnedPositionalEncoding(D_MODEL, MAX_SEQ_LEN)

    token_ids = torch.tensor([[3, 7, 3]])  # id 3 repeats at position 0 and 2
    token_vectors = token_embedding(token_ids)
    seq_len = token_ids.shape[1]

    # Baseline: raw token embeddings alone ARE identical at positions 0 and 2.
    assert torch.equal(token_vectors[0, 0], token_vectors[0, 2])

    with_sinusoidal = token_vectors + sinusoidal_pe(seq_len)
    with_learned = token_vectors + learned_pe(seq_len)

    assert not torch.equal(with_sinusoidal[0, 0], with_sinusoidal[0, 2])
    assert not torch.equal(with_learned[0, 0], with_learned[0, 2])
