"""Unit tests for Module 12 (Decoder block)."""

from __future__ import annotations

import torch

from app.transformer.decoder_block import DecoderBlock

D_MODEL = 16
N_HEADS = 4
MAX_SEQ_LEN = 32


def test_output_shape_matches_input() -> None:
    """A decoder block must preserve shape, so blocks can be stacked (Module 13)."""
    block = DecoderBlock(D_MODEL, N_HEADS, MAX_SEQ_LEN)
    x = torch.randn(2, 5, D_MODEL)
    output = block(x)
    assert output.shape == x.shape


def test_attention_and_feedforward_have_independent_layernorms() -> None:
    """Each residual connection must own its own LayerNorm, not a shared one."""
    block = DecoderBlock(D_MODEL, N_HEADS, MAX_SEQ_LEN)
    assert block.attention_residual.norm is not block.feed_forward_residual.norm


def test_causal_masking_preserved_through_the_full_block() -> None:
    """Perturbing only the LAST token must not change any earlier position's
    output, end to end through both sublayers."""
    block = DecoderBlock(D_MODEL, N_HEADS, MAX_SEQ_LEN)
    torch.manual_seed(0)
    x = torch.randn(1, 6, D_MODEL)
    x_perturbed = x.clone()
    x_perturbed[0, -1] = torch.randn(D_MODEL) * 100

    output = block(x)
    output_perturbed = block(x_perturbed)

    for i in range(5):
        assert torch.allclose(output[0, i], output_perturbed[0, i], atol=1e-5)
    assert not torch.allclose(output[0, 5], output_perturbed[0, 5], atol=1e-5)


def test_matches_manual_two_step_composition() -> None:
    """The block's forward must equal manually chaining its own submodules
    in the documented order: attention residual, then feed-forward residual."""
    block = DecoderBlock(D_MODEL, N_HEADS, MAX_SEQ_LEN)
    x = torch.randn(2, 5, D_MODEL)

    output = block(x)

    x1 = x + block.attention_residual.dropout(
        block.attention(block.attention_residual.norm(x))
    )
    expected = x1 + block.feed_forward_residual.dropout(
        block.feed_forward(block.feed_forward_residual.norm(x1))
    )

    assert torch.allclose(output, expected, atol=1e-6)


def test_deep_stack_remains_finite() -> None:
    """Stacking many blocks must not produce NaN/Inf -- a stability sanity
    check ahead of Module 13 (Transformer Stack)."""
    torch.manual_seed(0)
    blocks = [DecoderBlock(D_MODEL, N_HEADS, MAX_SEQ_LEN) for _ in range(10)]
    x = torch.randn(1, 6, D_MODEL)
    for block in blocks:
        x = block(x)
    assert torch.isfinite(x).all()


def test_gradients_flow_to_attention_and_feedforward() -> None:
    """Backprop must produce gradients for both sublayers' parameters."""
    block = DecoderBlock(D_MODEL, N_HEADS, MAX_SEQ_LEN)
    x = torch.randn(2, 5, D_MODEL, requires_grad=True)
    output = block(x)
    output.sum().backward()

    assert block.attention.query_proj.weight.grad is not None
    assert block.feed_forward.linear_1.weight.grad is not None
    assert x.grad is not None


def test_zero_dropout_is_deterministic() -> None:
    """With dropout=0.0, repeated forward passes must match exactly."""
    block = DecoderBlock(D_MODEL, N_HEADS, MAX_SEQ_LEN, dropout=0.0)
    x = torch.randn(1, 4, D_MODEL)
    assert torch.equal(block(x), block(x))
