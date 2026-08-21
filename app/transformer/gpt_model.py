"""The full GPT model: embeddings, positional encoding, transformer stack,
and output head, assembled into one trainable module."""

from __future__ import annotations

import torch
from torch import nn

from app.embeddings.positional_encoding import LearnedPositionalEncoding
from app.embeddings.token_embedding import TokenEmbedding
from app.transformer.output_head import OutputHead
from app.transformer.transformer_stack import TransformerStack


class GPTModel(nn.Module):
    """End-to-end GPT: token ids in, vocabulary logits out.

    logits = OutputHead(TransformerStack(TokenEmbedding(ids) + PositionalEncoding(ids)))

    This is where every module built since Module 5 (embeddings) through
    Module 14 (output projection) is wired into a single object -- what
    Module 16 (Training loop) will actually optimize.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        max_seq_len: int,
        d_ff: int | None = None,
        dropout: float = 0.0,
        tie_weights: bool = True,
    ) -> None:
        """Builds the full model.

        Args:
            vocab_size: Number of tokens in the vocabulary.
            d_model: Dimensionality of the token representations.
            n_heads: Number of attention heads per block.
            n_layers: Number of decoder blocks.
            max_seq_len: Longest sequence length the model supports.
            d_ff: Hidden dimensionality of each block's feed-forward
                expansion. Defaults to `4 * d_model`.
            dropout: Dropout probability used throughout the model.
            tie_weights: If True (GPT-2's convention), the output
                projection reuses the token embedding matrix instead of
                learning a separate one.
        """
        super().__init__()
        self.token_embedding = TokenEmbedding(vocab_size, d_model)
        self.positional_encoding = LearnedPositionalEncoding(d_model, max_seq_len)
        self.transformer = TransformerStack(
            d_model, n_heads, n_layers, max_seq_len, d_ff=d_ff, dropout=dropout
        )
        tied_weight = self.token_embedding.embedding.weight if tie_weights else None
        self.output_head = OutputHead(d_model, vocab_size, tied_embedding_weight=tied_weight)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Runs the full model: token ids to vocabulary logits.

        Args:
            token_ids: Input token ids.

        Shape:
            token_ids: (batch_size, seq_len)
            output: (batch_size, seq_len, vocab_size)
        """
        _, seq_len = token_ids.shape
        x = self.token_embedding(token_ids) + self.positional_encoding(seq_len)
        x = self.transformer(x)
        return self.output_head(x)
