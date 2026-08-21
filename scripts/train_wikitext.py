"""Module 29: Train on WikiText -- a larger, messier real-world corpus.

Trains the SAME architecture and step budget as Module 28 (Tiny Shakespeare)
on WikiText-2 (Merity et al.), to compare how a larger, more diverse corpus
behaves under an identical compute budget.

WikiText-2's raw text has ~1000 distinct characters (multilingual named
entities, symbols), 65% of which appear fewer than 5 times in 10.9M
characters -- the exact long-tail vocabulary problem from Module 2's
exercise, now showing up in a real dataset instead of a toy example. Rare
characters are folded into a single fallback token before building the
character vocabulary, so training time and vocab size stay comparable to
Module 28 instead of being dominated by noise.

The corpus itself is downloaded once via Hugging Face `datasets` (see
scripts/train_wikitext.py's companion download step) into
datasets/wikitext2.txt; this script only reads that plain text file.

Run with:

    uv run python scripts/train_wikitext.py
"""

from __future__ import annotations

import time
from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from app.dataset.dataloader import create_dataloader
from app.dataset.token_dataset import TokenDataset
from app.inference.generator import Generator
from app.inference.sampling import sample_from_logits
from app.inference.temperature import apply_temperature
from app.inference.top_p import apply_top_p
from app.tokenizer.char_tokenizer import CharTokenizer
from app.training.checkpoint import ModelConfig, save_checkpoint
from app.training.loss import CrossEntropyLoss
from app.training.optimizer import AdamW, build_param_groups
from app.training.scheduler import WarmupCosineScheduler
from app.training.trainer import Trainer
from app.transformer.gpt_model import GPTModel

DATASET_PATH = Path("datasets") / "wikitext2.txt"
CHECKPOINT_PATH = Path("checkpoints") / "wikitext.pt"
MIN_CHAR_COUNT = 50  # characters rarer than this get folded into a fallback
FALLBACK_CHAR = " "

# Same architecture and step budget as Module 28, for a fair comparison.
BLOCK_SIZE = 64
BATCH_SIZE = 32
D_MODEL = 96
N_HEADS = 4
N_LAYERS = 4
MAX_LR = 3e-3
MIN_LR = 3e-4
WARMUP_STEPS = 100
MAX_STEPS = 800
EVAL_EVERY = 200
EVAL_BATCHES = 20


def load_and_clean_dataset() -> str:
    """Loads WikiText-2 and folds rare characters into a single fallback.

    Raises:
        FileNotFoundError: If the corpus hasn't been downloaded yet.
    """
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"{DATASET_PATH} not found. Download WikiText-2 first (see the "
            "Module 29 write-up for the Hugging Face `datasets` download step)."
        )
    raw_text = DATASET_PATH.read_text(encoding="utf-8")

    counts = Counter(raw_text)
    rare_chars = {char for char, count in counts.items() if count < MIN_CHAR_COUNT}
    print(f"Raw corpus: {len(raw_text)} chars, {len(counts)} unique characters")
    print(f"Folding {len(rare_chars)} rare characters (<{MIN_CHAR_COUNT} occurrences) "
          f"into {FALLBACK_CHAR!r}")

    cleaned = "".join(FALLBACK_CHAR if char in rare_chars else char for char in raw_text)
    return cleaned


@torch.no_grad()
def estimate_loss(
    model: GPTModel, loss_fn: CrossEntropyLoss, dataloader: DataLoader, n_batches: int
) -> float:
    """Averages loss over a handful of batches, in eval mode, without grad."""
    model.eval()
    total = 0.0
    count = 0
    for inputs, targets in dataloader:
        logits = model(inputs)
        total += loss_fn(logits, targets).item()
        count += 1
        if count >= n_batches:
            break
    model.train()
    return total / count


def main() -> None:
    """Train on WikiText-2 with the exact Module 28 budget, then compare results."""
    torch.manual_seed(0)
    text = load_and_clean_dataset()

    tokenizer = CharTokenizer.from_text(text)
    print(f"Cleaned vocab_size={tokenizer.vocab_size} (Tiny Shakespeare's was 65)\n")

    token_ids = tokenizer.encode(text)
    split = int(0.9 * len(token_ids))
    train_ids, val_ids = token_ids[:split], token_ids[split:]
    print(f"train tokens={len(train_ids)}, val tokens={len(val_ids)}\n")

    train_dataset = TokenDataset(train_ids, block_size=BLOCK_SIZE)
    val_dataset = TokenDataset(val_ids, block_size=BLOCK_SIZE)
    train_loader = create_dataloader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = create_dataloader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)

    config = ModelConfig(
        vocab_size=tokenizer.vocab_size,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
        max_seq_len=BLOCK_SIZE,
        dropout=0.1,
    )
    model = GPTModel(**vars(config))
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params}\n")

    optimizer = AdamW(build_param_groups(model, weight_decay=0.1), lr=MAX_LR)
    scheduler = WarmupCosineScheduler(MAX_LR, WARMUP_STEPS, MAX_STEPS, MIN_LR)
    trainer = Trainer(model, optimizer)
    loss_fn = CrossEntropyLoss()

    print(f"Training for {MAX_STEPS} steps...")
    start = time.perf_counter()

    step = 0
    while step < MAX_STEPS:
        for inputs, targets in train_loader:
            if step >= MAX_STEPS:
                break
            scheduler.step_optimizer(optimizer, step)
            loss_value = trainer.train_step(inputs, targets)

            if step % EVAL_EVERY == 0 or step == MAX_STEPS - 1:
                val_loss = estimate_loss(model, loss_fn, val_loader, EVAL_BATCHES)
                current_lr = optimizer.param_groups[0]["lr"]
                elapsed = time.perf_counter() - start
                print(
                    f"step {step:5d} | train loss {loss_value:.4f} | val loss {val_loss:.4f} "
                    f"| lr {current_lr:.6f} | {elapsed:6.1f}s elapsed"
                )
            step += 1

    total_time = time.perf_counter() - start
    print(f"\nTraining finished in {total_time:.1f}s\n")

    save_checkpoint(CHECKPOINT_PATH, model, optimizer, step=MAX_STEPS, config=config)
    print(f"Saved checkpoint to {CHECKPOINT_PATH}\n")

    print("--- Sampling from the trained model (temperature=0.8, top_p=0.9) ---")

    def sample_fn(logits: torch.Tensor) -> int:
        scaled = apply_temperature(logits, temperature=0.8)
        filtered = apply_top_p(scaled, p=0.9)
        return sample_from_logits(filtered)

    generator = Generator(model, max_seq_len=BLOCK_SIZE, sample_fn=sample_fn)
    prompt = " = History = \n The "
    prompt_ids = tokenizer.encode(prompt)
    generated = generator.generate(prompt_ids, max_new_tokens=300)
    print(tokenizer.decode(generated))


if __name__ == "__main__":
    main()
