"""Runnable example for Module 15 (Loss function).

Run with:

    uv run python scripts/loss_demo.py
"""

from __future__ import annotations

import math

import torch
from torch.nn.functional import cross_entropy as torch_cross_entropy

from app.tokenizer.char_tokenizer import CharTokenizer
from app.training.loss import CrossEntropyLoss
from app.transformer.gpt_model import GPTModel

CORPUS = "to be or not to be that is the question"
D_MODEL = 16
N_HEADS = 4
N_LAYERS = 2
MAX_SEQ_LEN = 32


def main() -> None:
    """Run the log(vocab_size) sanity check on a real untrained model, then
    show loss behavior as predictions become more/less confident and correct."""
    torch.manual_seed(0)
    loss_fn = CrossEntropyLoss()

    print("--- Sanity check: loss on an UNTRAINED model should be close to log(vocab_size) ---")
    tokenizer = CharTokenizer.from_text(CORPUS)
    model = GPTModel(tokenizer.vocab_size, D_MODEL, N_HEADS, N_LAYERS, MAX_SEQ_LEN)

    text = "to be or not to"
    ids = tokenizer.encode(text)
    inputs = torch.tensor([ids[:-1]])
    targets = torch.tensor([ids[1:]])

    logits = model(inputs)
    loss = loss_fn(logits, targets)
    expected_random_loss = math.log(tokenizer.vocab_size)
    print(f"  vocab_size={tokenizer.vocab_size}, log(vocab_size)={expected_random_loss:.4f}")
    print(f"  Actual loss on untrained model:  {loss.item():.4f}")
    print("  (Close values confirm the untrained model's predictions are roughly")
    print("   uniform -- a real sanity check practitioners run before trusting a training run.")
    print("   If loss at init were wildly different from log(vocab_size), that would")
    print("   signal a bug: e.g. logits blowing up, or a broken loss computation.)")

    print("\n--- Loss decreases as confidence in the CORRECT answer increases ---")
    vocab_size = 10
    correct_class = 3
    for confidence in [0.0, 2.0, 5.0, 10.0, 20.0]:
        toy_logits = torch.zeros(1, 1, vocab_size)
        toy_logits[0, 0, correct_class] = confidence
        toy_targets = torch.tensor([[correct_class]])
        toy_loss = loss_fn(toy_logits, toy_targets)
        print(f"  logit at correct class = {confidence:5.1f} -> loss = {toy_loss.item():.6f}")

    print("\n--- Cross-checking against torch.nn.functional.cross_entropy ---")
    check_logits = torch.randn(4, 6, vocab_size)
    check_targets = torch.randint(0, vocab_size, (4, 6))
    ours = loss_fn(check_logits, check_targets)
    reference = torch_cross_entropy(
        check_logits.reshape(-1, vocab_size), check_targets.reshape(-1)
    )
    print(f"  Our loss:       {ours.item():.6f}")
    print(f"  Reference loss: {reference.item():.6f}")
    print(f"  Match: {torch.allclose(ours, reference, atol=1e-5)}")


if __name__ == "__main__":
    main()
