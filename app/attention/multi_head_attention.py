"""Multi-head causal self-attention: several attention subspaces in parallel."""

from __future__ import annotations

import math

import torch
from torch import nn

from app.utils.validation import check_seq_len


class MultiHeadAttention(nn.Module):
    """Runs several causal self-attention "heads" in parallel and combines them.

    head_i = softmax(Q_i K_i^T / sqrt(d_head) + M) V_i
    MultiHead(X) = Concat(head_1, ..., head_h) W_O

    Each head operates in its own d_head = d_model / n_heads dimensional
    subspace, letting the model learn several distinct attention patterns
    (e.g. one head attending to the previous token, another to long-range
    context) instead of averaging everything into a single pattern, as
    Module 9's single-head SelfAttention does.
    """

    def __init__(
        self, d_model: int, n_heads: int, max_seq_len: int, dropout: float = 0.0
    ) -> None:
        """Builds the layer and precomputes its causal mask.

        Args:
            d_model: Dimensionality of the input/output token representations.
            n_heads: Number of parallel attention heads.
            max_seq_len: Longest sequence length the causal mask is
                precomputed for.
            dropout: Dropout probability applied to the attention weights.

        Raises:
            ValueError: If `d_model` is not evenly divisible by `n_heads`.
        """
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")

        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        # Combined projections for all heads at once; split apart in forward().
        self.query_proj = nn.Linear(d_model, d_model)
        self.key_proj = nn.Linear(d_model, d_model)
        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

        causal_mask = torch.tril(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool))
        self.register_buffer("causal_mask", causal_mask, persistent=False)

    def _split_heads(self, x: torch.Tensor, batch_size: int, seq_len: int) -> torch.Tensor:
        """Reshapes (batch, seq_len, d_model) into (batch, n_heads, seq_len, d_head)."""
        x = x.view(batch_size, seq_len, self.n_heads, self.d_head)
        return x.transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor, batch_size: int, seq_len: int) -> torch.Tensor:
        """Reshapes (batch, n_heads, seq_len, d_head) back into (batch, seq_len, d_model)."""
        x = x.transpose(1, 2).contiguous()
        return x.view(batch_size, seq_len, self.n_heads * self.d_head)

    def forward(
        self, x: torch.Tensor, return_weights: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Applies multi-head causal self-attention.

        Args:
            x: Input token representations.
            return_weights: If True, also return the per-head attention weights.

        Shape:
            x: (batch_size, seq_len, d_model)
            output: (batch_size, seq_len, d_model)
            weights (if returned): (batch_size, n_heads, seq_len, seq_len)

        Raises:
            ValueError: If `seq_len` exceeds the precomputed `max_seq_len`.
        """
        batch_size, seq_len, _ = x.shape
        check_seq_len(seq_len, self.causal_mask.shape[0], "this causal mask")

        q = self._split_heads(self.query_proj(x), batch_size, seq_len)
        k = self._split_heads(self.key_proj(x), batch_size, seq_len)
        v = self._split_heads(self.value_proj(x), batch_size, seq_len)
        # q, k, v: (batch, n_heads, seq_len, d_head)

        scores = q @ k.transpose(-2, -1)  # (batch, n_heads, seq_len, seq_len)
        scores = scores / math.sqrt(self.d_head)

        mask = self.causal_mask[:seq_len, :seq_len]  # broadcasts over batch and n_heads
        scores = scores.masked_fill(~mask, float("-inf"))

        weights = torch.softmax(scores, dim=-1)

        # Cached BEFORE dropout, unconditionally (cheap: just a reference, no
        # extra compute), so callers that don't control this module's call
        # site -- like DecoderBlock, which always calls with
        # return_weights=False -- can still recover the most recent
        # attention pattern afterward. Used by Module 27 (attention
        # visualization) to extract weights from a full stacked model
        # without changing DecoderBlock/TransformerStack at all. Caching
        # AFTER dropout would return a tensor whose rows no longer sum to 1
        # (inverted dropout rescales survivors) whenever dropout > 0 and the
        # module is training -- these cached/returned weights are meant to
        # represent the attention distribution itself, not the (dropout-
        # regularized) tensor actually used in the weighted sum below.
        self.last_attention_weights = weights.detach()

        weights_for_output = self.dropout(weights)
        heads_output = weights_for_output @ v  # (batch, n_heads, seq_len, d_head)
        merged = self._merge_heads(heads_output, batch_size, seq_len)  # (batch, seq_len, d_model)
        output = self.output_proj(merged)

        if return_weights:
            return output, weights
        return output
