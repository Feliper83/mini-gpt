"""Unit tests for the assembled GPTModel (Module 14 synthesis)."""

from __future__ import annotations

import torch

from app.transformer.gpt_model import GPTModel

VOCAB_SIZE = 30
D_MODEL = 16
N_HEADS = 4
N_LAYERS = 2
MAX_SEQ_LEN = 32


def _build_model(tie_weights: bool = True) -> GPTModel:
    return GPTModel(
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
        max_seq_len=MAX_SEQ_LEN,
        tie_weights=tie_weights,
    )


def test_output_shape() -> None:
    """Token ids in, vocabulary logits out: (batch, seq_len) -> (batch, seq_len, vocab_size)."""
    model = _build_model()
    token_ids = torch.randint(0, VOCAB_SIZE, (2, 6))
    logits = model(token_ids)
    assert logits.shape == (2, 6, VOCAB_SIZE)


def test_tied_weights_are_the_same_object() -> None:
    """With tie_weights=True, the output head must reuse the embedding's exact tensor."""
    model = _build_model(tie_weights=True)
    assert model.output_head.projection.weight is model.token_embedding.embedding.weight


def test_untied_weights_are_independent() -> None:
    """With tie_weights=False, the output head must own a separate weight."""
    model = _build_model(tie_weights=False)
    assert model.output_head.projection.weight is not model.token_embedding.embedding.weight


def test_tied_model_has_fewer_parameters_than_untied() -> None:
    """Tying must save exactly vocab_size * d_model parameters."""
    tied_model = _build_model(tie_weights=True)
    untied_model = _build_model(tie_weights=False)

    tied_params = sum(p.numel() for p in tied_model.parameters())
    untied_params = sum(p.numel() for p in untied_model.parameters())

    assert untied_params - tied_params == VOCAB_SIZE * D_MODEL


def test_causal_masking_holds_end_to_end() -> None:
    """Perturbing only the LAST token id must not change any earlier position's
    logits, through the full assembled model."""
    model = _build_model()
    torch.manual_seed(0)
    token_ids = torch.randint(0, VOCAB_SIZE, (1, 6))
    perturbed_ids = token_ids.clone()
    perturbed_ids[0, -1] = (perturbed_ids[0, -1] + 1) % VOCAB_SIZE

    logits = model(token_ids)
    logits_perturbed = model(perturbed_ids)

    for i in range(5):
        assert torch.allclose(logits[0, i], logits_perturbed[0, i], atol=1e-4)
    assert not torch.allclose(logits[0, 5], logits_perturbed[0, 5], atol=1e-4)


def test_full_pipeline_output_is_finite() -> None:
    """A full forward pass must never produce NaN/Inf, even unlabeled/untrained."""
    model = _build_model()
    token_ids = torch.randint(0, VOCAB_SIZE, (2, 8))
    logits = model(token_ids)
    assert torch.isfinite(logits).all()


def test_gradient_flows_through_the_entire_model() -> None:
    """Backprop from the logits must reach the embedding table, an internal
    block, and the output head -- confirming the full pipeline is connected."""
    model = _build_model()
    token_ids = torch.randint(0, VOCAB_SIZE, (2, 5))
    logits = model(token_ids)
    logits.sum().backward()

    assert model.token_embedding.embedding.weight.grad is not None
    assert model.transformer.blocks[0].attention.query_proj.weight.grad is not None
    assert not torch.all(model.token_embedding.embedding.weight.grad == 0)
