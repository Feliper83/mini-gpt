"""Stack of N decoder blocks, with a final LayerNorm (GPT-2 style Pre-LN)."""

from __future__ import annotations

import torch
from torch import nn

from app.transformer.decoder_block import DecoderBlock
from app.transformer.layernorm import LayerNorm


class TransformerStack(nn.Module):
    """N stacked decoder blocks, followed by a final LayerNorm.

    Because every block here uses Pre-LN (Module 11: LayerNorm is applied to
    a copy of the residual stream before each sublayer, never to the stream
    itself), the residual stream coming out of the last block has never been
    normalized directly -- it's an unnormalized running sum across all N
    blocks. GPT-2 (and this project) applies one final LayerNorm here,
    right before the output projection (Module 14), to bring that
    accumulated stream back to a consistent scale.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_layers: int,
        max_seq_len: int,
        d_ff: int | None = None,
        dropout: float = 0.0,
    ) -> None:
        """Builds the stack of decoder blocks and the final LayerNorm.

        Args:
            d_model: Dimensionality of the token representations.
            n_heads: Number of attention heads per block.
            n_layers: Number of decoder blocks to stack.
            max_seq_len: Longest sequence length the causal mask is
                precomputed for.
            d_ff: Hidden dimensionality of each block's feed-forward
                expansion. Defaults to `4 * d_model`.
            dropout: Dropout probability used throughout every block.
        """
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                DecoderBlock(d_model, n_heads, max_seq_len, d_ff=d_ff, dropout=dropout)
                for _ in range(n_layers)
            ]
        )
        self.final_norm = LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Runs `x` through every block in sequence, then a final LayerNorm.

        Args:
            x: Input token representations.

        Shape:
            x: (batch_size, seq_len, d_model)
            output: (batch_size, seq_len, d_model), unchanged shape.
        """
        for block in self.blocks:
            x = block(x)
        return self.final_norm(x)
