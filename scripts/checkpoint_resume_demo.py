"""Runnable example for Module 25 (Loading checkpoints).

Run with:

    uv run python scripts/checkpoint_resume_demo.py
"""

from __future__ import annotations

from pathlib import Path

import torch

from app.dataset.dataloader import create_dataloader
from app.dataset.token_dataset import TokenDataset
from app.inference.generator import Generator
from app.tokenizer.char_tokenizer import CharTokenizer
from app.training.checkpoint import ModelConfig, load_checkpoint, save_checkpoint
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
CHECKPOINT_PATH = Path("checkpoints") / "resume_demo.pt"


def main() -> None:
    """Train, save; then reconstruct EVERYTHING from scratch and load, proving
    the reloaded model is functionally identical to the one that was saved."""
    torch.manual_seed(0)
    tokenizer = CharTokenizer.from_text(CORPUS)
    token_ids = tokenizer.encode(CORPUS)

    config = ModelConfig(
        vocab_size=tokenizer.vocab_size,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
        max_seq_len=BLOCK_SIZE,
    )
    model = GPTModel(**vars(config))
    optimizer = AdamW(build_param_groups(model, weight_decay=0.1), lr=3e-3)
    trainer = Trainer(model, optimizer)

    dataset = TokenDataset(token_ids, block_size=BLOCK_SIZE)
    dataloader = create_dataloader(dataset, batch_size=16, shuffle=True)

    print("Training original model for 150 steps...")
    trainer.fit(dataloader, max_steps=150, log_every=150)

    prompt = "to be or"
    prompt_ids = tokenizer.encode(prompt)
    original_generator = Generator(model, max_seq_len=BLOCK_SIZE)
    original_output = original_generator.generate(prompt_ids, max_new_tokens=30)
    print(f"\nOriginal model output: {tokenizer.decode(original_output)!r}")

    save_checkpoint(CHECKPOINT_PATH, model, optimizer, step=150, config=config)
    print(f"Saved checkpoint to {CHECKPOINT_PATH}\n")

    print("--- Simulating a fresh process: brand new model, brand new optimizer ---")
    fresh_model = GPTModel(**vars(config))  # freshly random-initialized, untrained
    fresh_optimizer = AdamW(build_param_groups(fresh_model, weight_decay=0.1), lr=3e-3)

    untrained_generator = Generator(fresh_model, max_seq_len=BLOCK_SIZE)
    untrained_output = untrained_generator.generate(prompt_ids, max_new_tokens=30)
    print(f"Fresh (untrained) model output: {tokenizer.decode(untrained_output)!r}")

    # model=fresh_model is required here, not optional: fresh_optimizer was
    # built over fresh_model's parameters, so weights MUST be loaded into
    # that exact model instance -- otherwise the optimizer keeps referencing
    # stale, untouched parameters and "resumed" training silently never
    # updates anything (see Module 25's write-up for the full story).
    loaded = load_checkpoint(CHECKPOINT_PATH, model=fresh_model, optimizer=fresh_optimizer)
    print(f"\nLoaded checkpoint: step={loaded.step}, config={loaded.config}")

    loaded_generator = Generator(loaded.model, max_seq_len=BLOCK_SIZE)
    loaded_output = loaded_generator.generate(prompt_ids, max_new_tokens=30)
    print(f"Reloaded model output:          {tokenizer.decode(loaded_output)!r}")

    print(f"\nOutput matches the original exactly: {original_output == loaded_output}")

    print("\n--- Resuming training from step 150 with the loaded optimizer state ---")
    trainer_resumed = Trainer(loaded.model, fresh_optimizer)
    resume_target = loaded.step + 30
    history = trainer_resumed.fit(
        dataloader, max_steps=resume_target, log_every=30, start_step=loaded.step
    )
    print(f"Loss: {history[0].loss:.4f} -> {history[-1].loss:.4f} (30 more steps, real progress)")


if __name__ == "__main__":
    main()
