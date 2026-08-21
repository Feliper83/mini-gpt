"""Unit tests for Module 13 (Transformer stack)."""

from __future__ import annotations

import pytest
import torch

from app.transformer.transformer_stack import TransformerStack

D_MODEL = 16
N_HEADS = 4
MAX_SEQ_LEN = 32


def test_output_shape_matches_input() -> None:
    """The stack must preserve shape regardless of depth."""
    stack = TransformerStack(D_MODEL, N_HEADS, n_layers=4, max_seq_len=MAX_SEQ_LEN)
    x = torch.randn(2, 5, D_MODEL)
    output = stack(x)
    assert output.shape == x.shape


def test_n_layers_creates_the_requested_number_of_blocks() -> None:
    """The stack must contain exactly n_layers independent DecoderBlocks."""
    stack = TransformerStack(D_MODEL, N_HEADS, n_layers=5, max_seq_len=MAX_SEQ_LEN)
    assert len(stack.blocks) == 5


def test_single_layer_matches_manual_composition() -> None:
    """With n_layers=1, the stack must equal final_norm(block(x)) exactly."""
    stack = TransformerStack(D_MODEL, N_HEADS, n_layers=1, max_seq_len=MAX_SEQ_LEN)
    x = torch.randn(2, 5, D_MODEL)
    output = stack(x)
    expected = stack.final_norm(stack.blocks[0](x))
    assert torch.allclose(output, expected, atol=1e-6)


def test_causal_masking_preserved_through_the_full_stack() -> None:
    """Perturbing only the LAST token must not change any earlier position's
    output, even after many stacked blocks."""
    stack = TransformerStack(D_MODEL, N_HEADS, n_layers=8, max_seq_len=MAX_SEQ_LEN)
    torch.manual_seed(0)
    x = torch.randn(1, 6, D_MODEL)
    x_perturbed = x.clone()
    x_perturbed[0, -1] = torch.randn(D_MODEL) * 100

    output = stack(x)
    output_perturbed = stack(x_perturbed)

    for i in range(5):
        assert torch.allclose(output[0, i], output_perturbed[0, i], atol=1e-4)
    assert not torch.allclose(output[0, 5], output_perturbed[0, 5], atol=1e-4)


def test_deep_stack_remains_finite() -> None:
    """A GPT-2-small-depth stack (12 layers) must not produce NaN/Inf."""
    torch.manual_seed(0)
    stack = TransformerStack(D_MODEL, N_HEADS, n_layers=12, max_seq_len=MAX_SEQ_LEN)
    x = torch.randn(2, 6, D_MODEL)
    output = stack(x)
    assert torch.isfinite(output).all()


def test_parameter_count_scales_linearly_with_n_layers() -> None:
    """Doubling n_layers must roughly double the parameter count (the final
    LayerNorm's fixed, tiny cost is the only deviation from exact doubling)."""
    stack_2 = TransformerStack(D_MODEL, N_HEADS, n_layers=2, max_seq_len=MAX_SEQ_LEN)
    stack_4 = TransformerStack(D_MODEL, N_HEADS, n_layers=4, max_seq_len=MAX_SEQ_LEN)

    params_2 = sum(p.numel() for p in stack_2.parameters())
    params_4 = sum(p.numel() for p in stack_4.parameters())

    per_block = (params_4 - params_2) / 2  # each extra layer adds exactly one block's worth
    single_block_params = sum(p.numel() for p in stack_2.blocks[0].parameters())
    assert per_block == pytest.approx(single_block_params)


def test_gradient_flows_to_first_and_last_block() -> None:
    """Backprop must reach both the first and last block's parameters,
    confirming gradient isn't lost somewhere in a deep stack."""
    stack = TransformerStack(D_MODEL, N_HEADS, n_layers=6, max_seq_len=MAX_SEQ_LEN)
    x = torch.randn(2, 5, D_MODEL, requires_grad=True)
    output = stack(x)
    output.sum().backward()

    first_block_grad = stack.blocks[0].attention.query_proj.weight.grad
    last_block_grad = stack.blocks[-1].attention.query_proj.weight.grad

    assert first_block_grad is not None
    assert last_block_grad is not None
    assert not torch.all(first_block_grad == 0)
    assert not torch.all(last_block_grad == 0)


def test_zero_dropout_is_deterministic() -> None:
    """With dropout=0.0, repeated forward passes must match exactly."""
    stack = TransformerStack(D_MODEL, N_HEADS, n_layers=4, max_seq_len=MAX_SEQ_LEN, dropout=0.0)
    x = torch.randn(1, 4, D_MODEL)
    assert torch.equal(stack(x), stack(x))
