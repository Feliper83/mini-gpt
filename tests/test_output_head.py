"""Unit tests for Module 14 (Output projection)."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from app.embeddings.token_embedding import TokenEmbedding
from app.transformer.output_head import OutputHead

D_MODEL = 16
VOCAB_SIZE = 50


def test_output_shape() -> None:
    """Output must have shape (batch_size, seq_len, vocab_size)."""
    head = OutputHead(D_MODEL, VOCAB_SIZE)
    x = torch.randn(2, 5, D_MODEL)
    logits = head(x)
    assert logits.shape == (2, 5, VOCAB_SIZE)


def test_untied_head_has_independent_weight() -> None:
    """Without tying, the projection must own its own fresh weight tensor."""
    embedding = TokenEmbedding(VOCAB_SIZE, D_MODEL)
    head = OutputHead(D_MODEL, VOCAB_SIZE)
    assert head.projection.weight is not embedding.embedding.weight


def test_weight_tying_shares_the_exact_same_tensor() -> None:
    """Tying must reuse the SAME parameter object, not a copy of its values."""
    embedding = TokenEmbedding(VOCAB_SIZE, D_MODEL)
    head = OutputHead(D_MODEL, VOCAB_SIZE, tied_embedding_weight=embedding.embedding.weight)
    assert head.projection.weight is embedding.embedding.weight


def test_weight_tying_shape_mismatch_raises() -> None:
    """A tied weight of the wrong shape must fail loudly, not silently misalign."""
    wrong_shape_weight = nn.Parameter(torch.randn(VOCAB_SIZE, D_MODEL + 1))
    with pytest.raises(ValueError, match="expected"):
        OutputHead(D_MODEL, VOCAB_SIZE, tied_embedding_weight=wrong_shape_weight)


def test_weight_tying_avoids_double_counting_in_a_parent_module() -> None:
    """When both TokenEmbedding and a tied OutputHead are submodules of the
    same parent, the shared weight must be counted only ONCE by
    parameters() -- this is the actual parameter-savings weight tying provides."""

    class TinyModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embedding = TokenEmbedding(VOCAB_SIZE, D_MODEL)
            self.head = OutputHead(
                D_MODEL, VOCAB_SIZE, tied_embedding_weight=self.embedding.embedding.weight
            )

    tied_model = TinyModel()
    tied_param_count = sum(p.numel() for p in tied_model.parameters())
    assert tied_param_count == VOCAB_SIZE * D_MODEL


def test_gradients_flow_to_untied_projection() -> None:
    """Backprop must produce a gradient for an untied projection's weight."""
    head = OutputHead(D_MODEL, VOCAB_SIZE)
    x = torch.randn(2, 5, D_MODEL, requires_grad=True)
    logits = head(x)
    logits.sum().backward()
    assert head.projection.weight.grad is not None
