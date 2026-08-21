"""Runnable example for Module 27 (Visualizing attention).

Run with:

    uv run python scripts/attention_viz_demo.py
"""

from __future__ import annotations

from pathlib import Path

import torch

from app.dataset.dataloader import create_dataloader
from app.dataset.token_dataset import TokenDataset
from app.inference.attention_viz import extract_attention_weights, plot_attention_heatmap
from app.tokenizer.char_tokenizer import CharTokenizer
from app.training.optimizer import AdamW, build_param_groups
from app.training.trainer import Trainer
from app.transformer.gpt_model import GPTModel

CORPUS = (
    "to be or not to be that is the question "
    "whether tis nobler in the mind to suffer "
) * 50
BLOCK_SIZE = 32
D_MODEL = 64
N_HEADS = 4
N_LAYERS = 2
TRAIN_STEPS = 300
OUTPUT_DIR = Path("outputs") / "attention"


def main() -> None:
    """Train briefly, extract real attention weights for a prompt, and save
    them as PNG heatmaps -- plus a quick check for "attention sink" behavior."""
    torch.manual_seed(0)
    tokenizer = CharTokenizer.from_text(CORPUS)
    token_ids = tokenizer.encode(CORPUS)

    dataset = TokenDataset(token_ids, block_size=BLOCK_SIZE)
    dataloader = create_dataloader(dataset, batch_size=16, shuffle=True)

    model = GPTModel(tokenizer.vocab_size, D_MODEL, N_HEADS, N_LAYERS, BLOCK_SIZE)
    optimizer = AdamW(build_param_groups(model, weight_decay=0.1), lr=3e-3)
    trainer = Trainer(model, optimizer)

    print(f"Training for {TRAIN_STEPS} steps...")
    history = trainer.fit(dataloader, max_steps=TRAIN_STEPS, log_every=300)
    print(f"Final loss: {history[-1].loss:.4f}\n")

    prompt = "to be or not to be"
    prompt_ids = tokenizer.encode(prompt)
    input_tensor = torch.tensor([prompt_ids])
    tokens = list(prompt)

    all_weights = extract_attention_weights(model, input_tensor)
    layer_shape = tuple(all_weights[0].shape)
    print(f"Extracted weights for {len(all_weights)} layers, each shape {layer_shape}\n")

    for layer_idx, layer_weights in enumerate(all_weights):
        for head_idx in range(N_HEADS):
            output_path = OUTPUT_DIR / f"layer{layer_idx}_head{head_idx}.png"
            plot_attention_heatmap(
                layer_weights[head_idx],
                tokens=tokens,
                title=f"Layer {layer_idx}, Head {head_idx} -- prompt: {prompt!r}",
                output_path=output_path,
            )
    print(f"Saved {N_LAYERS * N_HEADS} heatmap PNGs to {OUTPUT_DIR}/\n")

    print("--- Attention sink check: how much attention lands on position 0? ---")
    for layer_idx, layer_weights in enumerate(all_weights):
        avg_to_pos0 = layer_weights[:, :, 0].mean().item()
        print(f"  Layer {layer_idx}: average weight on position 0 = {avg_to_pos0:.3f}")
    print("  (Position 0 is the only position every later query can always attend to,")
    print("   which is a well-documented reason it sometimes acts as an 'attention sink'.)")


if __name__ == "__main__":
    main()
