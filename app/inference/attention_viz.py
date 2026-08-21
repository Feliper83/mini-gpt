"""Attention weight extraction and visualization."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # non-interactive backend: no display needed to save PNGs
import matplotlib.pyplot as plt
import torch

from app.transformer.gpt_model import GPTModel


def extract_attention_weights(model: GPTModel, token_ids: torch.Tensor) -> list[torch.Tensor]:
    """Runs a forward pass and collects every block's attention weights.

    Relies on `MultiHeadAttention` (Module 10) caching its most recent
    weights as `last_attention_weights` on every forward call -- this lets
    extraction work through the full model (`DecoderBlock` / `TransformerStack`)
    without those layers needing to know anything about visualization.

    Args:
        model: A `GPTModel`.
        token_ids: Input token ids for a single sequence.

    Shape:
        token_ids: (1, seq_len)

    Returns:
        A list of length `n_layers`; each element has shape
        `(n_heads, seq_len, seq_len)`, the attention weights that block
        produced during this forward pass.

    Raises:
        ValueError: If `token_ids` isn't rank-2 (batch, seq_len), or its
            batch size isn't 1.
    """
    if token_ids.ndim != 2 or token_ids.shape[0] != 1:
        raise ValueError(
            "extract_attention_weights expects a (1, seq_len) tensor, got shape "
            f"{tuple(token_ids.shape)}"
        )

    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            model(token_ids)
    finally:
        model.train(was_training)

    return [block.attention.last_attention_weights[0] for block in model.transformer.blocks]


def plot_attention_heatmap(
    weights: torch.Tensor,
    tokens: list[str],
    title: str,
    output_path: str | Path,
) -> None:
    """Renders one head's attention weight matrix as a PNG heatmap.

    Args:
        weights: Attention weights for a single head.
        tokens: Display label for each position, used as axis tick labels.
        title: Plot title.
        output_path: Destination PNG file path. Parent directories are
            created if needed.

    Shape:
        weights: (seq_len, seq_len)
    """
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots(figsize=(6, 6))
    image = axis.imshow(weights.numpy(), cmap="viridis", vmin=0.0, vmax=1.0)
    axis.set_xticks(range(len(tokens)))
    axis.set_yticks(range(len(tokens)))
    axis.set_xticklabels(tokens, rotation=90)
    axis.set_yticklabels(tokens)
    axis.set_xlabel("Key position (attended TO)")
    axis.set_ylabel("Query position (attending FROM)")
    axis.set_title(title)
    figure.colorbar(image, ax=axis, label="attention weight")
    figure.tight_layout()
    figure.savefig(destination, dpi=150)
    plt.close(figure)
