"""Runnable example for Module 20 (Sampling).

Run with:

    uv run python scripts/sampling_demo.py
"""

from __future__ import annotations

import torch

from app.dataset.dataloader import create_dataloader
from app.dataset.token_dataset import TokenDataset
from app.inference.generator import Generator
from app.inference.sampling import sample_from_logits
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
# Deliberately lighter than earlier demos: a fully-converged model (loss ~0.07,
# see Module 19) becomes so confident that even stochastic sampling almost
# always agrees with greedy -- there's no meaningful uncertainty left to see.
# Undertraining a bit keeps genuine entropy in the predicted distribution.
TRAIN_STEPS = 80


def main() -> None:
    """Verify sampling matches true probabilities, then contrast greedy vs
    stochastic generation on a real trained model."""
    print("--- Empirical check: sampling frequency matches softmax probability ---")
    torch.manual_seed(0)
    logits = torch.tensor([2.0, 1.0, 0.5, -1.0])
    true_probs = torch.softmax(logits, dim=-1)
    counts = torch.zeros(4)
    n_draws = 5000
    for _ in range(n_draws):
        counts[sample_from_logits(logits)] += 1
    empirical_probs = counts / n_draws

    print(f"  {'class':>6} {'true p':>8} {'empirical p':>12}")
    for i in range(4):
        print(f"  {i:>6} {true_probs[i].item():8.3f} {empirical_probs[i].item():12.3f}")

    print("\n--- Training a tiny model, then comparing greedy vs stochastic generation ---")
    tokenizer = CharTokenizer.from_text(CORPUS)
    token_ids = tokenizer.encode(CORPUS)
    dataset = TokenDataset(token_ids, block_size=BLOCK_SIZE)
    dataloader = create_dataloader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = GPTModel(tokenizer.vocab_size, D_MODEL, N_HEADS, N_LAYERS, BLOCK_SIZE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
    trainer = Trainer(model, optimizer)
    trainer.fit(dataloader, max_steps=TRAIN_STEPS, log_every=250)

    prompt = "to be or"
    prompt_ids = tokenizer.encode(prompt)

    greedy_generator = Generator(model, max_seq_len=BLOCK_SIZE)
    greedy_results = {
        tuple(greedy_generator.generate(prompt_ids, max_new_tokens=20)) for _ in range(8)
    }
    print(f"\nGreedy: {len(greedy_results)} unique completion(s) out of 8 attempts")
    for result in greedy_results:
        print(f"  {tokenizer.decode(list(result))!r}")

    stochastic_generator = Generator(model, max_seq_len=BLOCK_SIZE, sample_fn=sample_from_logits)
    torch.manual_seed(1)
    stochastic_results = [
        stochastic_generator.generate(prompt_ids, max_new_tokens=20) for _ in range(8)
    ]
    unique_stochastic = {tuple(r) for r in stochastic_results}
    print(f"\nStochastic: {len(unique_stochastic)} unique completion(s) out of 8 attempts")
    for result in stochastic_results[:5]:
        print(f"  {tokenizer.decode(result)!r}")


if __name__ == "__main__":
    main()
