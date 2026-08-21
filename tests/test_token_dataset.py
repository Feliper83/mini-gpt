"""Unit tests for Module 4 (Dataset loader)."""

from __future__ import annotations

import pytest
import torch

from app.dataset.dataloader import create_dataloader
from app.dataset.token_dataset import TokenDataset

TOKEN_IDS = list(range(100))  # deterministic, easy-to-reason-about token stream
BLOCK_SIZE = 8


def test_len_matches_expected_window_count() -> None:
    """len(dataset) must equal N - block_size, per the formula in the write-up."""
    dataset = TokenDataset(TOKEN_IDS, block_size=BLOCK_SIZE)
    assert len(dataset) == len(TOKEN_IDS) - BLOCK_SIZE


def test_getitem_shapes() -> None:
    """Both x and y must have shape (block_size,)."""
    dataset = TokenDataset(TOKEN_IDS, block_size=BLOCK_SIZE)
    x, y = dataset[0]
    assert x.shape == (BLOCK_SIZE,)
    assert y.shape == (BLOCK_SIZE,)


def test_getitem_dtype_is_long() -> None:
    """Token ids must be torch.long, matching what nn.Embedding expects."""
    dataset = TokenDataset(TOKEN_IDS, block_size=BLOCK_SIZE)
    x, y = dataset[0]
    assert x.dtype == torch.long
    assert y.dtype == torch.long


def test_target_is_input_shifted_by_one_against_raw_tokens() -> None:
    """x and y must exactly match slices of the raw token stream."""
    dataset = TokenDataset(TOKEN_IDS, block_size=BLOCK_SIZE)
    for index in [0, 1, 17, len(dataset) - 1]:
        x, y = dataset[index]
        assert x.tolist() == TOKEN_IDS[index : index + BLOCK_SIZE]
        assert y.tolist() == TOKEN_IDS[index + 1 : index + BLOCK_SIZE + 1]


def test_target_overlaps_input_shifted_by_one() -> None:
    """y[:-1] must equal x[1:], since y is x shifted one position ahead."""
    dataset = TokenDataset(TOKEN_IDS, block_size=BLOCK_SIZE)
    x, y = dataset[5]
    assert torch.equal(y[:-1], x[1:])


def test_block_size_zero_or_negative_raises() -> None:
    """block_size=0 (or negative) must be rejected, not silently accepted
    as a degenerate empty-shape window."""
    with pytest.raises(ValueError, match="block_size must be positive"):
        TokenDataset(TOKEN_IDS, block_size=0)
    with pytest.raises(ValueError, match="block_size must be positive"):
        TokenDataset(TOKEN_IDS, block_size=-1)


def test_block_size_too_large_raises() -> None:
    """block_size must be strictly smaller than the corpus length."""
    with pytest.raises(ValueError, match="must be smaller than"):
        TokenDataset(TOKEN_IDS, block_size=len(TOKEN_IDS))


def test_dataloader_batches_have_expected_shape() -> None:
    """A DataLoader batch must have shape (batch_size, block_size)."""
    dataset = TokenDataset(TOKEN_IDS, block_size=BLOCK_SIZE)
    dataloader = create_dataloader(dataset, batch_size=4, shuffle=False)
    batch_x, batch_y = next(iter(dataloader))
    assert batch_x.shape == (4, BLOCK_SIZE)
    assert batch_y.shape == (4, BLOCK_SIZE)


def test_dataloader_without_shuffle_is_deterministic_and_sequential() -> None:
    """With shuffle=False, the first batch must match dataset[0..batch_size-1] in order."""
    dataset = TokenDataset(TOKEN_IDS, block_size=BLOCK_SIZE)
    dataloader = create_dataloader(dataset, batch_size=4, shuffle=False)
    batch_x, batch_y = next(iter(dataloader))

    for row in range(4):
        expected_x, expected_y = dataset[row]
        assert torch.equal(batch_x[row], expected_x)
        assert torch.equal(batch_y[row], expected_y)
