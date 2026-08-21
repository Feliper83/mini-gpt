"""Runnable example for Module 16 (Training loop).

Run with:

    uv run python scripts/training_loop_demo.py
"""

from __future__ import annotations

import torch

from app.dataset.dataloader import create_dataloader
from app.dataset.token_dataset import TokenDataset
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
MAX_STEPS = 300


def main() -> None:
    """Train a tiny GPT on repeated text and watch the loss fall, then do a
    crude greedy continuation (NOT the real generator -- that's Module 19)."""
    torch.manual_seed(0)
    tokenizer = CharTokenizer.from_text(CORPUS)
    token_ids = tokenizer.encode(CORPUS)

    dataset = TokenDataset(token_ids, block_size=BLOCK_SIZE)
    dataloader = create_dataloader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = GPTModel(
        vocab_size=tokenizer.vocab_size,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
        max_seq_len=BLOCK_SIZE,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
    trainer = Trainer(model, optimizer)

    print(f"Corpus: {len(CORPUS)} chars, vocab_size={tokenizer.vocab_size}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    print(f"Training for {MAX_STEPS} steps...\n")

    history = trainer.fit(dataloader, max_steps=MAX_STEPS, log_every=50)

    print(f"\nLoss: {history[0].loss:.4f} -> {history[-1].loss:.4f}")

    print("\n--- Crude greedy continuation (NOT the real generator -- that's Module 19) ---")
    model.eval()
    prompt = "to be or"
    ids = tokenizer.encode(prompt)
    with torch.no_grad():
        for _ in range(40):
            context = torch.tensor([ids[-BLOCK_SIZE:]])
            logits = model(context)
            next_id = int(logits[0, -1].argmax().item())
            ids.append(next_id)
    print(f"  Prompt:    {prompt!r}")
    print(f"  Continued: {tokenizer.decode(ids)!r}")


if __name__ == "__main__":
    main()
