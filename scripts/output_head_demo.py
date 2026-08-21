"""Runnable example for Module 14 (Output projection).

This module completes the architecture: every piece built since Module 5
(embeddings) through Module 13 (transformer stack) now assembles into a
single GPTModel that goes from raw text to vocabulary logits, end to end.

Run with:

    uv run python scripts/output_head_demo.py
"""

from __future__ import annotations

import torch

from app.tokenizer.char_tokenizer import CharTokenizer
from app.transformer.gpt_model import GPTModel

CORPUS = "to be or not to be that is the question"
D_MODEL = 16
N_HEADS = 4
N_LAYERS = 2
MAX_SEQ_LEN = 32


def main() -> None:
    """Demonstrate weight tying, then run the first full text -> logits pass."""
    torch.manual_seed(0)
    tokenizer = CharTokenizer.from_text(CORPUS)

    print("--- Weight tying: savings only show up at the FULL MODEL level ---")
    print("(Looking at OutputHead in isolation is misleading: it always reports just")
    print(" its one weight tensor, tied or not. The savings come from that tensor")
    print(" being the SAME object the embedding table already counted.)\n")

    tied_model = GPTModel(
        vocab_size=tokenizer.vocab_size,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
        max_seq_len=MAX_SEQ_LEN,
        tie_weights=True,
    )
    untied_model = GPTModel(
        vocab_size=tokenizer.vocab_size,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
        max_seq_len=MAX_SEQ_LEN,
        tie_weights=False,
    )

    tied_is_same = (
        tied_model.output_head.projection.weight is tied_model.token_embedding.embedding.weight
    )
    print(f"  tied_model:   output_head weight IS embedding weight: {tied_is_same}")
    untied_is_same = (
        untied_model.output_head.projection.weight
        is untied_model.token_embedding.embedding.weight
    )
    print(f"  untied_model: output_head weight IS embedding weight: {untied_is_same}")

    tied_total = sum(p.numel() for p in tied_model.parameters())
    untied_total = sum(p.numel() for p in untied_model.parameters())
    print(f"\n  vocab_size={tokenizer.vocab_size}, d_model={D_MODEL}")
    print(f"  tied_model total parameters:   {tied_total}")
    print(f"  untied_model total parameters: {untied_total}")
    print(f"  Savings from tying: {untied_total - tied_total} (= vocab_size * d_model)")

    print("\n--- Full pipeline, for the first time: text -> logits ---")
    model = tied_model

    text = "to be or not"
    token_ids = torch.tensor([tokenizer.encode(text)])
    print(f"  Input text: {text!r} -> ids {token_ids.tolist()}")

    logits = model(token_ids)
    print(f"  Output logits shape: {tuple(logits.shape)}  (batch, seq_len, vocab_size)")

    probs = torch.softmax(logits[0, -1], dim=-1)
    print(f"  Probabilities for the NEXT token after {text!r} sum to: {probs.sum().item():.4f}")
    top_id = int(probs.argmax().item())
    top_char = tokenizer.decode([top_id])
    print(f"  Most likely next char (untrained, so meaningless): {top_char!r}")

    print("\n--- Causal masking holds end-to-end through the full model ---")
    ids_perturbed = token_ids.clone()
    ids_perturbed[0, -1] = (ids_perturbed[0, -1] + 1) % tokenizer.vocab_size

    logits_a = model(token_ids)
    logits_b = model(ids_perturbed)
    for i in range(token_ids.shape[1] - 1):
        unchanged = torch.allclose(logits_a[0, i], logits_b[0, i], atol=1e-4)
        print(f"  Position {i} logits unchanged: {unchanged}")


if __name__ == "__main__":
    main()
