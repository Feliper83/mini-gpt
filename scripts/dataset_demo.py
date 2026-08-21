"""Runnable example for Module 4 (Dataset loader).

Run with:

    uv run python scripts/dataset_demo.py
"""

from __future__ import annotations

from app.dataset.dataloader import create_dataloader
from app.dataset.token_dataset import TokenDataset
from app.tokenizer.char_tokenizer import CharTokenizer

CORPUS = (
    "To be, or not to be, that is the question: "
    "Whether 'tis nobler in the mind to suffer "
    "The slings and arrows of outrageous fortune, "
    "Or to take arms against a sea of troubles "
) * 5

BLOCK_SIZE = 16
BATCH_SIZE = 4


def main() -> None:
    """Tokenize a corpus, window it, and inspect a single example and a batch."""
    tokenizer = CharTokenizer.from_text(CORPUS)
    token_ids = tokenizer.encode(CORPUS)
    print(f"Corpus length: {len(CORPUS)} chars -> {len(token_ids)} tokens")

    dataset = TokenDataset(token_ids, block_size=BLOCK_SIZE)
    print(f"Dataset size (valid windows): {len(dataset)}")

    x0, y0 = dataset[0]
    print("\nFirst window:")
    print(f"  x: {x0.tolist()}")
    print(f"     -> {tokenizer.decode(x0.tolist())!r}")
    print(f"  y: {y0.tolist()}")
    print(f"     -> {tokenizer.decode(y0.tolist())!r}")
    shift_holds = bool((y0[:-1] == x0[1:]).all().item())
    print(f"  y is x shifted by one position (y[:-1] == x[1:]): {shift_holds}")

    dataloader = create_dataloader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    batch_x, batch_y = next(iter(dataloader))
    print(f"\nOne batch shapes: x={tuple(batch_x.shape)}, y={tuple(batch_y.shape)}")
    print(f"Expected: ({BATCH_SIZE}, {BLOCK_SIZE}) for both.")


if __name__ == "__main__":
    main()
