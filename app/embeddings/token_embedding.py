"""Token embedding layer: maps discrete token ids to dense vectors."""

from __future__ import annotations

import torch
from torch import nn


class TokenEmbedding(nn.Module):
    """Learned lookup table mapping token ids to d_model-dimensional vectors.

    This is the first layer of the model: it has no notion of position or
    context on its own -- every occurrence of the same token id maps to the
    exact same vector, regardless of where it appears in the sequence.
    Positional information is added separately (Module 6).
    """

    def __init__(self, vocab_size: int, d_model: int, init_std: float = 0.02) -> None:
        """Builds the embedding table.

        Args:
            vocab_size: Number of distinct token ids the table covers.
            d_model: Dimensionality of each embedding vector.
            init_std: Standard deviation for the normal-distribution weight
                initialization. GPT-2 uses 0.02, which keeps initial
                embedding magnitudes small enough for stable early training.
        """
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=init_std)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Looks up the embedding vector for each token id.

        Args:
            token_ids: Integer token ids.

        Shape:
            token_ids: (batch_size, seq_len)
            output: (batch_size, seq_len, d_model)
        """
        return self.embedding(token_ids)
