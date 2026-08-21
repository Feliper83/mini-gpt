"""Positional encodings: inject position information into token embeddings.

Token embeddings alone (see `TokenEmbedding`) are position-blind: the same
token id always maps to the same vector, regardless of where it appears in a
sequence. Both encodings here add a position-dependent vector so that
downstream layers (attention) have something to key off of when reasoning
about order.
"""

from __future__ import annotations

import torch
from torch import nn

from app.utils.validation import check_seq_len


class SinusoidalPositionalEncoding(nn.Module):
    """Fixed, non-learned positional encoding using sine/cosine functions.

    PE(pos, 2i)   = sin(pos / 10000^(2i / d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))

    Every dimension pair (2i, 2i+1) oscillates at its own frequency: low
    dimensions change rapidly with position, high dimensions change slowly.
    Because it has no learned parameters, it generalizes to sequence lengths
    longer than any seen during training -- a property `LearnedPositionalEncoding`
    does not have.
    """

    def __init__(self, d_model: int, max_seq_len: int) -> None:
        """Precomputes the encoding table for all positions up to max_seq_len.

        Args:
            d_model: Embedding dimensionality. Must be even (sin/cos come in pairs).
            max_seq_len: Longest sequence length the table is precomputed for.

        Raises:
            ValueError: If `d_model` is odd.
        """
        super().__init__()
        if d_model % 2 != 0:
            raise ValueError(f"d_model must be even for sin/cos pairing, got {d_model}")

        self.max_seq_len = max_seq_len

        position = torch.arange(max_seq_len, dtype=torch.float32).unsqueeze(1)  # (max_seq_len, 1)
        dim_pair_index = torch.arange(0, d_model, 2, dtype=torch.float32)  # the "2i" values
        inverse_frequency = 1.0 / (10000 ** (dim_pair_index / d_model))  # (d_model/2,)

        angles = position * inverse_frequency  # (max_seq_len, d_model/2), broadcast

        pe = torch.zeros(max_seq_len, d_model)
        pe[:, 0::2] = torch.sin(angles)
        pe[:, 1::2] = torch.cos(angles)

        # Registered as a buffer, not a parameter: it is saved/moved with the
        # module (e.g. .to(device)) but is never updated by the optimizer.
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, seq_len: int) -> torch.Tensor:
        """Returns the positional encoding for positions [0, seq_len).

        Args:
            seq_len: Number of positions to return.

        Returns:
            Positional encoding table.

        Shape:
            output: (seq_len, d_model)

        Raises:
            ValueError: If `seq_len` exceeds `max_seq_len`.
        """
        check_seq_len(seq_len, self.max_seq_len, "this positional encoding")
        return self.pe[:seq_len]


class LearnedPositionalEncoding(nn.Module):
    """Learned positional embedding: a lookup table trained like `TokenEmbedding`.

    This is what GPT-2 (and this project's GPT model, Module 9) actually
    uses. It has strictly more representational freedom than the sinusoidal
    encoding, at the cost of not generalizing beyond `max_seq_len` positions
    seen during training.
    """

    def __init__(self, d_model: int, max_seq_len: int, init_std: float = 0.02) -> None:
        """Builds the positional embedding table.

        Args:
            d_model: Embedding dimensionality.
            max_seq_len: Maximum sequence length this model can ever process.
            init_std: Standard deviation for the normal-distribution weight
                initialization (same convention as `TokenEmbedding`).
        """
        super().__init__()
        self.embedding = nn.Embedding(max_seq_len, d_model)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=init_std)
        self.max_seq_len = max_seq_len

    def forward(self, seq_len: int) -> torch.Tensor:
        """Returns the positional embedding for positions [0, seq_len).

        Args:
            seq_len: Number of positions to return.

        Returns:
            Positional embedding table.

        Shape:
            output: (seq_len, d_model)

        Raises:
            ValueError: If `seq_len` exceeds `max_seq_len`.
        """
        check_seq_len(seq_len, self.max_seq_len, "this positional encoding")
        positions = torch.arange(seq_len, device=self.embedding.weight.device)
        return self.embedding(positions)
