"""Token-sequence dataset for causal language model training."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch.utils.data import Dataset


class TokenDataset(Dataset):
    """Yields fixed-length (input, target) windows for next-token prediction.

    Given a flat sequence of token ids, each item is a contiguous window of
    `block_size` ids as the input, and the same window shifted one position
    to the right as the target -- the standard causal language modeling setup
    (teacher forcing).
    """

    def __init__(self, token_ids: Sequence[int], block_size: int) -> None:
        """Builds the dataset from a flat token stream.

        Args:
            token_ids: The full tokenized corpus, as a flat sequence of ids.
            block_size: Context length. Must be positive and smaller than
                `len(token_ids)`.

        Raises:
            ValueError: If `block_size` is not positive, or not smaller
                than the corpus length, since one (input, target) pair
                requires `block_size + 1` tokens.
        """
        if block_size <= 0:
            raise ValueError(f"block_size must be positive, got {block_size}")
        if block_size >= len(token_ids):
            raise ValueError(
                f"block_size ({block_size}) must be smaller than the corpus "
                f"length ({len(token_ids)}) so at least one training example exists."
            )
        self.data = torch.tensor(token_ids, dtype=torch.long)
        self.block_size = block_size

    def __len__(self) -> int:
        """Number of valid starting positions for a `block_size` window."""
        return len(self.data) - self.block_size

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns one (input, target) pair starting at `index`.

        Args:
            index: Starting position in `[0, len(self))`.

        Returns:
            A tuple `(x, y)` where `y` equals `x` shifted one position ahead
            in the underlying token stream.

        Shape:
            x: (block_size,)
            y: (block_size,)
        """
        x = self.data[index : index + self.block_size]
        y = self.data[index + 1 : index + self.block_size + 1]
        return x, y
