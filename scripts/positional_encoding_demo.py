"""Runnable example for Module 6 (Positional encoding).

Reuses the exact repeated-token scenario from the Module 5 demo, but now
adds positional encoding to show that the previously identical vectors
become distinct.

Run with:

    uv run python scripts/positional_encoding_demo.py
"""

from __future__ import annotations

import torch

from app.dataset.token_dataset import TokenDataset
from app.embeddings.positional_encoding import (
    LearnedPositionalEncoding,
    SinusoidalPositionalEncoding,
)
from app.embeddings.token_embedding import TokenEmbedding
from app.tokenizer.char_tokenizer import CharTokenizer

CORPUS = "to be or not to be that is the question"
BLOCK_SIZE = 8
D_MODEL = 6


def main() -> None:
    """Show the sinusoidal PE table, then confirm it breaks position blindness."""
    tokenizer = CharTokenizer.from_text(CORPUS)
    token_ids = tokenizer.encode(CORPUS)
    dataset = TokenDataset(token_ids, block_size=BLOCK_SIZE)

    token_embedding = TokenEmbedding(vocab_size=tokenizer.vocab_size, d_model=D_MODEL)
    sinusoidal_pe = SinusoidalPositionalEncoding(d_model=D_MODEL, max_seq_len=32)
    learned_pe = LearnedPositionalEncoding(d_model=D_MODEL, max_seq_len=32)

    x, _ = dataset[0]
    batch = x.unsqueeze(0)
    seq_len = batch.shape[1]
    print(f"Decoded text: {tokenizer.decode(x.tolist())!r}")

    token_vectors = token_embedding(batch)

    print("\n--- Sinusoidal PE table (first 4 positions) ---")
    pe_table = sinusoidal_pe(seq_len)
    for pos in range(min(4, seq_len)):
        print(f"  pos {pos}: {[round(v, 3) for v in pe_table[pos].tolist()]}")

    combined_sinusoidal = token_vectors + sinusoidal_pe(seq_len)
    combined_learned = token_vectors + learned_pe(seq_len)

    ids = x.tolist()
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            if ids[i] != ids[j]:
                continue
            char = tokenizer.decode([ids[i]])
            print(f"\nToken {char!r} repeats at position {i} and position {j}.")

            raw_identical = torch.equal(token_vectors[0, i], token_vectors[0, j])
            sin_identical = torch.equal(combined_sinusoidal[0, i], combined_sinusoidal[0, j])
            learned_identical = torch.equal(combined_learned[0, i], combined_learned[0, j])

            print(f"  Raw token embeddings identical:             {raw_identical}")
            print(f"  + sinusoidal positional encoding identical: {sin_identical}")
            print(f"  + learned positional encoding identical:    {learned_identical}")
            return


if __name__ == "__main__":
    main()
