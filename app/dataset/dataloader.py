"""DataLoader construction for `TokenDataset`."""

from __future__ import annotations

from torch.utils.data import DataLoader

from app.dataset.token_dataset import TokenDataset


def create_dataloader(
    dataset: TokenDataset,
    batch_size: int,
    shuffle: bool = True,
) -> DataLoader:
    """Wraps a `TokenDataset` in a `DataLoader` that batches (input, target) pairs.

    Args:
        dataset: A `TokenDataset` producing `(block_size,)` id windows.
        batch_size: Number of windows per batch.
        shuffle: Whether to draw windows in random order each epoch. Use
            `True` for training and `False` for reproducible evaluation.

    Returns:
        A `DataLoader` yielding `(input, target)` batches.

    Shape:
        Each batch: input (batch_size, block_size), target (batch_size, block_size).
    """
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
