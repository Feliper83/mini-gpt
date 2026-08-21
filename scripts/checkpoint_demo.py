"""Runnable example for Module 24 (Saving checkpoints).

Run with:

    uv run python scripts/checkpoint_demo.py
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as torch_functional

from app.tokenizer.char_tokenizer import CharTokenizer
from app.training.checkpoint import ModelConfig, save_checkpoint
from app.training.optimizer import AdamW, build_param_groups
from app.transformer.gpt_model import GPTModel

CORPUS = "to be or not to be that is the question"
CHECKPOINT_PATH = Path("checkpoints") / "demo_checkpoint.pt"


def main() -> None:
    """Train briefly, save a checkpoint, then peek at its raw contents on disk."""
    tokenizer = CharTokenizer.from_text(CORPUS)
    config = ModelConfig(
        vocab_size=tokenizer.vocab_size,
        d_model=32,
        n_heads=4,
        n_layers=2,
        max_seq_len=16,
    )
    # Config field names mirror GPTModel's constructor exactly -- see the
    # docstring in app/training/checkpoint.py.
    model = GPTModel(**vars(config))
    optimizer = AdamW(build_param_groups(model, weight_decay=0.1), lr=1e-3)

    token_ids = tokenizer.encode(CORPUS)
    inputs = torch.tensor([token_ids[:16]])
    targets = torch.tensor([token_ids[1:17]])

    final_loss = 0.0
    for _ in range(5):
        logits = model(inputs)
        loss = torch_functional.cross_entropy(
            logits.reshape(-1, tokenizer.vocab_size), targets.reshape(-1)
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        final_loss = loss.item()

    save_checkpoint(
        CHECKPOINT_PATH,
        model=model,
        optimizer=optimizer,
        step=5,
        config=config,
        extra={"final_loss": final_loss},
    )

    size_kb = CHECKPOINT_PATH.stat().st_size / 1024
    print(f"Saved checkpoint to {CHECKPOINT_PATH} ({size_kb:.1f} KB)\n")

    print("--- Peeking inside the checkpoint file ---")
    raw = torch.load(CHECKPOINT_PATH, weights_only=False)
    print(f"  Keys:   {list(raw.keys())}")
    print(f"  step:   {raw['step']}")
    print(f"  config: {raw['config']}")
    print(f"  extra:  {raw['extra']}")
    print(f"  model_state_dict:     {len(raw['model_state_dict'])} tensors")
    n_optim_states = len(raw["optimizer_state_dict"]["state"])
    print(f"  optimizer_state_dict: {n_optim_states} parameter states (AdamW m/v buffers)")

    first_state = next(iter(raw["optimizer_state_dict"]["state"].values()))
    print(f"\n  Example optimizer state for one parameter: step={first_state['step']}, "
          f"m.shape={tuple(first_state['m'].shape)}, v.shape={tuple(first_state['v'].shape)}")


if __name__ == "__main__":
    main()
