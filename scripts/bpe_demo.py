"""Runnable example for Module 3 (BPE tokenizer).

Trains a byte-level BPE tokenizer and compares it against the Module 2
character tokenizer, to make the seq_len compression argument (from the
Module 2 exercise) concrete with real numbers.

Run with:

    uv run python scripts/bpe_demo.py
"""

from __future__ import annotations

import sys

from app.tokenizer.bpe import BPETokenizer
from app.tokenizer.char_tokenizer import CharTokenizer

MEMORIZED_SENTENCE = "To be, or not to be, that is the question."

TRAINING_TEXT = (MEMORIZED_SENTENCE + " Whether 'tis nobler in the mind to suffer ") * 20

# Never appears in TRAINING_TEXT: a fair test of how well BPE generalizes,
# since MEMORIZED_SENTENCE was seen verbatim (and would be memorized as a
# handful of merged tokens, which overstates the typical compression ratio).
UNSEEN_SENTENCE = "or to take arms against a sea of troubles and by opposing end them"


def compare_tokenizers(
    label: str,
    text: str,
    char_tokenizer: CharTokenizer,
    bpe_tokenizer: BPETokenizer,
) -> None:
    """Prints token counts for both tokenizers on the same text.

    The char tokenizer can legitimately fail here (any character outside
    its training vocabulary raises `ValueError`) -- that failure mode
    itself is part of what this comparison is demonstrating.
    """
    print(f"[{label}] {text!r}")

    try:
        char_ids = char_tokenizer.encode(text)
        print(f"  CharTokenizer: {len(char_ids)} tokens")
    except ValueError as error:
        char_ids = None
        print(f"  CharTokenizer: FAILED -- {error}")

    bpe_ids = bpe_tokenizer.encode(text)
    print(f"  BPETokenizer:  {len(bpe_ids)} tokens")

    if char_ids is not None:
        compression = len(char_ids) / len(bpe_ids)
        cheaper = compression**2
        print(f"  Compression: {compression:.2f}x (attention cost ~{cheaper:.1f}x cheaper)")
    print()


def main() -> None:
    """Train BPE, then compare token counts and vocab sizes against CharTokenizer."""
    # Reconfigure stdout to UTF-8 so printing non-Latin text (emoji, Japanese)
    # doesn't crash on Windows consoles whose default codepage can't represent it.
    sys.stdout.reconfigure(encoding="utf-8")

    char_tokenizer = CharTokenizer.from_text(TRAINING_TEXT)
    bpe_tokenizer = BPETokenizer.train(TRAINING_TEXT, vocab_size=300)

    num_merges = len(bpe_tokenizer.merges)
    print(f"CharTokenizer vocab_size: {char_tokenizer.vocab_size}")
    print(f"BPETokenizer  vocab_size: {bpe_tokenizer.vocab_size} ({num_merges} merges)\n")

    compare_tokenizers(
        "seen verbatim in training -- overstates compression",
        MEMORIZED_SENTENCE,
        char_tokenizer,
        bpe_tokenizer,
    )
    compare_tokenizers(
        "never seen, has an out-of-vocab char for CharTokenizer",
        UNSEEN_SENTENCE,
        char_tokenizer,
        bpe_tokenizer,
    )

    encoded = bpe_tokenizer.encode(MEMORIZED_SENTENCE)
    assert bpe_tokenizer.decode(encoded) == MEMORIZED_SENTENCE
    print("Round-trip OK: decode(encode(text)) == text\n")

    print("A few learned merges (as decoded byte strings):")
    learned_tokens = sorted(bpe_tokenizer.merges.items(), key=lambda item: item[1])[:8]
    for (a, b), new_id in learned_tokens:
        merged_bytes = bpe_tokenizer.vocab[new_id]
        print(f"  {new_id}: {a!r}+{b!r} -> {merged_bytes!r}")

    print("\nUnlike CharTokenizer, BPE never fails on unseen characters:")
    exotic_text = "emoji test \U0001f680 and japanese こんにちは"
    exotic_ids = bpe_tokenizer.encode(exotic_text)
    decoded_exotic = bpe_tokenizer.decode(exotic_ids)
    print(f"  {exotic_text!r} -> {len(exotic_ids)} tokens -> decodes back to: {decoded_exotic!r}")


if __name__ == "__main__":
    main()
