"""Runnable example for Module 5 (Embedding layer).

Run with:

    uv run python scripts/embedding_demo.py
"""

from __future__ import annotations

import torch

from app.dataset.token_dataset import TokenDataset
from app.embeddings.token_embedding import TokenEmbedding
from app.tokenizer.char_tokenizer import CharTokenizer

CORPUS = "to be or not to be that is the question"
BLOCK_SIZE = 8
D_MODEL = 6


def main() -> None:
    """Embed a window of token ids and show the lookup-table property."""
    tokenizer = CharTokenizer.from_text(CORPUS)
    token_ids = tokenizer.encode(CORPUS)
    dataset = TokenDataset(token_ids, block_size=BLOCK_SIZE)

    embedding = TokenEmbedding(vocab_size=tokenizer.vocab_size, d_model=D_MODEL)

    x, _ = dataset[0]
    batch = x.unsqueeze(0)  # (1, block_size) -- add a batch dimension
    print(f"Input ids:    {batch.tolist()} shape={tuple(batch.shape)}")
    print(f"Decoded text: {tokenizer.decode(x.tolist())!r}")

    vectors = embedding(batch)
    print(f"Output shape: {tuple(vectors.shape)}  (batch, seq_len, d_model)")

    ids = x.tolist()
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            if ids[i] == ids[j]:
                identical = torch.equal(vectors[0, i], vectors[0, j])
                char = tokenizer.decode([ids[i]])
                print(f"\nToken {char!r} appears at position {i} and position {j}.")
                print(f"Their embedding vectors are identical: {identical}")
                print(f"  vector @ pos {i}: {[round(v, 4) for v in vectors[0, i].tolist()]}")
                print(f"  vector @ pos {j}: {[round(v, 4) for v in vectors[0, j].tolist()]}")
                print("\n(This is exactly the 'position blindness' Module 6 will fix.)")
                return
    print("\nNo repeated token found in this window to demonstrate the property.")


if __name__ == "__main__":
    main()
