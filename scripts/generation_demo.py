"""Runnable example for Module 19 (Text generation).

Run with:

    uv run python scripts/generation_demo.py
"""

from __future__ import annotations

import time

import torch

from app.dataset.dataloader import create_dataloader
from app.dataset.token_dataset import TokenDataset
from app.inference.generator import Generator
from app.tokenizer.char_tokenizer import CharTokenizer
from app.training.trainer import Trainer
from app.transformer.gpt_model import GPTModel

CORPUS = (
    "to be or not to be that is the question "
    "whether tis nobler in the mind to suffer "
) * 50

BLOCK_SIZE = 32
BATCH_SIZE = 16
D_MODEL = 64
N_HEADS = 4
N_LAYERS = 2
TRAIN_STEPS = 250


def main() -> None:
    """Train briefly, generate with the real Generator, then measure how
    generation cost grows with context length (no KV cache)."""
    torch.manual_seed(0)
    tokenizer = CharTokenizer.from_text(CORPUS)
    token_ids = tokenizer.encode(CORPUS)

    dataset = TokenDataset(token_ids, block_size=BLOCK_SIZE)
    dataloader = create_dataloader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = GPTModel(tokenizer.vocab_size, D_MODEL, N_HEADS, N_LAYERS, BLOCK_SIZE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
    trainer = Trainer(model, optimizer)

    print(f"Training for {TRAIN_STEPS} steps...")
    history = trainer.fit(dataloader, max_steps=TRAIN_STEPS, log_every=250)
    print(f"Final loss: {history[-1].loss:.4f}\n")

    print("--- Generating with the real Generator class (greedy) ---")
    generator = Generator(model, max_seq_len=BLOCK_SIZE)
    prompt = "to be or"
    prompt_ids = tokenizer.encode(prompt)
    generated_ids = generator.generate(prompt_ids, max_new_tokens=60)
    print(f"  Prompt:    {prompt!r}")
    print(f"  Generated: {tokenizer.decode(generated_ids)!r}\n")

    print("--- Context truncation: generating past max_seq_len does not crash ---")
    requested = BLOCK_SIZE * 3
    long_ids = generator.generate(prompt_ids, max_new_tokens=requested)
    produced = len(long_ids) - len(prompt_ids)
    print(f"  Requested {requested} tokens (> max_seq_len={BLOCK_SIZE});", end=" ")
    print(f"got {produced}, no crash.\n")

    print("--- Empirical cost: generation slows down as context grows (no KV cache) ---")
    for context_len in [4, BLOCK_SIZE // 2, BLOCK_SIZE]:
        warm_ids = token_ids[:context_len]
        start = time.perf_counter()
        generator.generate(warm_ids, max_new_tokens=20)
        elapsed_ms = (time.perf_counter() - start) * 1000
        print(f"  starting context_len={context_len:3d}: {elapsed_ms:7.2f} ms for 20 new tokens")
    print("  (Every step here recomputes attention over the whole growing sequence --")
    print("   this is the O(seq_len^2) cost from Module 9, paid again at every step.)")


if __name__ == "__main__":
    main()
