"""Output projection: converts final hidden states into vocabulary logits."""

from __future__ import annotations

import torch
from torch import nn


class OutputHead(nn.Module):
    """Projects d_model-dimensional hidden states to vocab_size logits.

    logits = h @ W_out^T

    Optionally ties W_out to a token embedding matrix (weight tying): the
    exact same parameter tensor is used both to look up token embeddings
    (Module 5) and to project hidden states back to vocabulary logits here.
    This is the convention GPT-2 uses -- it saves `vocab_size * d_model`
    parameters (the single largest matrix in a small GPT model) and acts as
    a mild regularizer, since the input and output token representations
    are forced to live in a shared space.
    """

    def __init__(
        self,
        d_model: int,
        vocab_size: int,
        tied_embedding_weight: nn.Parameter | None = None,
    ) -> None:
        """Builds the projection, optionally sharing weights with an embedding.

        Args:
            d_model: Dimensionality of the incoming hidden states.
            vocab_size: Number of tokens in the vocabulary.
            tied_embedding_weight: If provided, the projection reuses this
                exact parameter tensor (shape `(vocab_size, d_model)`,
                typically `TokenEmbedding.embedding.weight`) instead of
                allocating a new one -- weight tying.

        Raises:
            ValueError: If `tied_embedding_weight` is provided with a shape
                that doesn't match `(vocab_size, d_model)`.
        """
        super().__init__()
        self.projection = nn.Linear(d_model, vocab_size, bias=False)
        if tied_embedding_weight is not None:
            if tied_embedding_weight.shape != (vocab_size, d_model):
                raise ValueError(
                    f"tied_embedding_weight has shape {tuple(tied_embedding_weight.shape)}, "
                    f"expected {(vocab_size, d_model)}"
                )
            self.projection.weight = tied_embedding_weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Projects hidden states to logits over the vocabulary.

        Args:
            x: Final hidden states from the transformer stack.

        Shape:
            x: (batch_size, seq_len, d_model)
            output: (batch_size, seq_len, vocab_size)
        """
        return self.projection(x)
