"""Unit tests for Module 5 (Embedding layer)."""

from __future__ import annotations

import pytest
import torch

from app.embeddings.token_embedding import TokenEmbedding

VOCAB_SIZE = 50
D_MODEL = 16


def test_output_shape() -> None:
    """Output must have shape (batch_size, seq_len, d_model)."""
    embedding = TokenEmbedding(VOCAB_SIZE, D_MODEL)
    token_ids = torch.randint(0, VOCAB_SIZE, (4, 10))
    output = embedding(token_ids)
    assert output.shape == (4, 10, D_MODEL)


def test_output_dtype_is_float() -> None:
    """Embedding vectors must be floating point even though ids are integers."""
    embedding = TokenEmbedding(VOCAB_SIZE, D_MODEL)
    token_ids = torch.randint(0, VOCAB_SIZE, (2, 5))
    output = embedding(token_ids)
    assert output.dtype == torch.float32


def test_same_token_id_gives_identical_vector_regardless_of_position() -> None:
    """The embedding table has no positional awareness: same id -> same vector."""
    embedding = TokenEmbedding(VOCAB_SIZE, D_MODEL)
    token_ids = torch.tensor([[3, 7, 3, 3, 12]])  # id 3 repeats at positions 0, 2, 3
    output = embedding(token_ids)
    assert torch.equal(output[0, 0], output[0, 2])
    assert torch.equal(output[0, 2], output[0, 3])


def test_different_token_ids_give_different_vectors() -> None:
    """Distinct ids must (almost surely) map to distinct vectors under random init."""
    embedding = TokenEmbedding(VOCAB_SIZE, D_MODEL)
    token_ids = torch.tensor([[0, 1]])
    output = embedding(token_ids)
    assert not torch.equal(output[0, 0], output[0, 1])


def test_init_std_controls_weight_scale() -> None:
    """The empirical std of the weight matrix must track the requested init_std."""
    torch.manual_seed(0)
    embedding = TokenEmbedding(vocab_size=5000, d_model=32, init_std=0.02)
    empirical_std = embedding.embedding.weight.std().item()
    assert empirical_std == pytest.approx(0.02, abs=0.002)


def test_gradient_is_sparse_only_looked_up_rows_get_gradient() -> None:
    """Only the rows of W_E used in the forward pass should receive gradient."""
    embedding = TokenEmbedding(VOCAB_SIZE, D_MODEL)
    token_ids = torch.tensor([[1, 2, 1]])  # ids 1 and 2 are used; the rest are not

    output = embedding(token_ids)
    output.sum().backward()

    grad = embedding.embedding.weight.grad
    assert grad is not None
    used_ids = {1, 2}
    for row in range(VOCAB_SIZE):
        if row in used_ids:
            assert not torch.all(grad[row] == 0)
        else:
            assert torch.all(grad[row] == 0)


def test_out_of_range_token_id_raises() -> None:
    """An id outside [0, vocab_size) must fail, mirroring the tokenizers' behavior."""
    embedding = TokenEmbedding(VOCAB_SIZE, D_MODEL)
    token_ids = torch.tensor([[VOCAB_SIZE]])  # first invalid id
    with pytest.raises(IndexError):
        embedding(token_ids)
