"""Runnable example for Module 26 (Fine-tuning).

Run with:

    uv run python scripts/finetune_demo.py
"""

from __future__ import annotations

from pathlib import Path

import torch

from app.dataset.dataloader import create_dataloader
from app.dataset.token_dataset import TokenDataset
from app.tokenizer.bpe import BPETokenizer
from app.training.checkpoint import ModelConfig, load_checkpoint, save_checkpoint
from app.training.finetune import (
    count_trainable_parameters,
    freeze_all_parameters,
    unfreeze_last_n_blocks,
)
from app.training.optimizer import AdamW, build_param_groups
from app.training.trainer import Trainer
from app.transformer.gpt_model import GPTModel

PRETRAIN_CORPUS = (
    "to be or not to be that is the question "
    "whether tis nobler in the mind to suffer "
) * 50
# A BPE tokenizer (Module 3) can encode this even though it's a totally
# different domain -- unlike the char tokenizer (Module 2), it never hits
# an out-of-vocabulary error.
FINETUNE_CORPUS = (
    "the weather today is warm and sunny across the region "
    "with clear skies expected through the evening "
) * 50

BLOCK_SIZE = 32
D_MODEL = 64
N_HEADS = 4
N_LAYERS = 4  # deep enough that freezing has something meaningful to show
PRETRAIN_STEPS = 200
FINETUNE_STEPS = 100
CHECKPOINT_PATH = Path("checkpoints") / "pretrained_for_finetune.pt"


def main() -> None:
    """Pretrain, checkpoint, freeze most of the model, then fine-tune the
    rest on an unrelated corpus -- confirming frozen weights never move."""
    torch.manual_seed(0)
    tokenizer = BPETokenizer.train(PRETRAIN_CORPUS, vocab_size=300)

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

    pretrain_ids = tokenizer.encode(PRETRAIN_CORPUS)
    pretrain_dataset = TokenDataset(pretrain_ids, block_size=BLOCK_SIZE)
    pretrain_loader = create_dataloader(pretrain_dataset, batch_size=16, shuffle=True)

    print(f"--- Pretraining on Shakespeare-ish text for {PRETRAIN_STEPS} steps ---")
    history = trainer.fit(pretrain_loader, max_steps=PRETRAIN_STEPS, log_every=200)
    print(f"Pretrain loss: {history[0].loss:.4f} -> {history[-1].loss:.4f}\n")

    save_checkpoint(CHECKPOINT_PATH, model, optimizer, step=PRETRAIN_STEPS, config=config)

    print("--- Loading the pretrained checkpoint fresh, then freezing most of it ---")
    loaded = load_checkpoint(CHECKPOINT_PATH)
    finetune_model = loaded.model

    trainable_before, total = count_trainable_parameters(finetune_model)
    pct_before = 100 * trainable_before / total
    print(f"Before freezing: {trainable_before}/{total} trainable ({pct_before:.1f}%)")

    freeze_all_parameters(finetune_model)
    unfreeze_last_n_blocks(finetune_model, n_trainable_blocks=1)

    trainable_after, total = count_trainable_parameters(finetune_model)
    pct_after = 100 * trainable_after / total
    print(f"After freezing:  {trainable_after}/{total} trainable ({pct_after:.1f}%)\n")

    # Smaller lr than pretraining: the model is already in a good region,
    # large steps here risk destroying what pretraining learned.
    finetune_optimizer = AdamW(build_param_groups(finetune_model, weight_decay=0.1), lr=5e-4)
    finetune_trainer = Trainer(finetune_model, finetune_optimizer)

    finetune_ids = tokenizer.encode(FINETUNE_CORPUS)
    finetune_dataset = TokenDataset(finetune_ids, block_size=BLOCK_SIZE)
    finetune_loader = create_dataloader(finetune_dataset, batch_size=16, shuffle=True)

    frozen_block_before = finetune_model.transformer.blocks[0].attention.query_proj.weight.clone()

    print(f"--- Fine-tuning ONLY unfrozen params on weather text for {FINETUNE_STEPS} steps ---")
    ft_history = finetune_trainer.fit(finetune_loader, max_steps=FINETUNE_STEPS, log_every=100)
    print(f"Finetune loss: {ft_history[0].loss:.4f} -> {ft_history[-1].loss:.4f}\n")

    frozen_block_after = finetune_model.transformer.blocks[0].attention.query_proj.weight
    unchanged = torch.equal(frozen_block_before, frozen_block_after)
    print(f"Frozen block[0] weights unchanged after fine-tuning: {unchanged}")


if __name__ == "__main__":
    main()
