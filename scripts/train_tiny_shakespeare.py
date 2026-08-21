"""Module 28: Train on Tiny Shakespeare -- the first real, sizeable training run.

Trains a small GPT on the canonical Tiny Shakespeare corpus (Karpathy's
char-rnn dataset, ~1.1MB), with a proper warmup+cosine LR schedule,
train/validation loss tracking, a saved checkpoint, and sampled output.

Uses the character tokenizer (Module 2), not BPE (Module 3): training BPE's
from-scratch, O(N * num_merges) merge loop over a corpus this size would be
far too slow in pure Python -- exactly the tradeoff flagged back in the
Module 3 exercise.

Run with:

    uv run python scripts/train_tiny_shakespeare.py
"""

from __future__ import annotations

import time
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

DATASET_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
)
DATASET_PATH = Path("datasets") / "tinyshakespeare.txt"
CHECKPOINT_PATH = Path("checkpoints") / "tinyshakespeare.pt"
VOCAB_PATH = Path("checkpoints") / "tinyshakespeare_vocab.json"

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


def load_dataset() -> str:
    """Loads the Tiny Shakespeare corpus from disk.

    Raises:
        FileNotFoundError: If the corpus hasn't been downloaded yet. This
            project downloads it via PowerShell's `Invoke-WebRequest` (which
            uses the OS certificate store), not Python's `urllib`, since
            this environment sits behind a TLS-inspecting proxy that breaks
            Python's own certificate verification (see README).
    """
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"{DATASET_PATH} not found. Download it first, e.g. in PowerShell:\n"
            f'  Invoke-WebRequest -Uri "{DATASET_URL}" -OutFile "{DATASET_PATH}"'
        )
    return DATASET_PATH.read_text(encoding="utf-8")


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
    """Train, evaluate periodically, checkpoint, and sample from a real corpus."""
    torch.manual_seed(0)
    text = load_dataset()
    print(f"Corpus: {len(text)} characters\n")

    tokenizer = CharTokenizer.from_text(text)
    print(f"vocab_size={tokenizer.vocab_size}")

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
    tokenizer.save(VOCAB_PATH)
    print(f"Saved checkpoint to {CHECKPOINT_PATH}")
    print(f"Saved tokenizer vocab to {VOCAB_PATH}\n")

    print("--- Sampling from the trained model (temperature=0.8, top_p=0.9) ---")

    def sample_fn(logits: torch.Tensor) -> int:
        scaled = apply_temperature(logits, temperature=0.8)
        filtered = apply_top_p(scaled, p=0.9)
        return sample_from_logits(filtered)

    generator = Generator(model, max_seq_len=BLOCK_SIZE, sample_fn=sample_fn)
    prompt = "ROMEO:"
    prompt_ids = tokenizer.encode(prompt)
    generated = generator.generate(prompt_ids, max_new_tokens=300)
    print(tokenizer.decode(generated))


if __name__ == "__main__":
    main()
