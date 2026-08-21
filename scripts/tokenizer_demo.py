"""Runnable example for Module 2 (Character tokenizer).

Run with:

    uv run python scripts/tokenizer_demo.py
"""

from __future__ import annotations

from app.tokenizer.char_tokenizer import CharTokenizer

SAMPLE_TEXT = "To be, or not to be, that is the question."


def main() -> None:
    """Build a tokenizer from a sample sentence and round-trip it."""
    tokenizer = CharTokenizer.from_text(SAMPLE_TEXT)

    print(f"Corpus:     {SAMPLE_TEXT!r}")
    print(f"Vocab size: {tokenizer.vocab_size}")
    print(f"Vocabulary: {sorted(tokenizer.stoi)}")

    encoded = tokenizer.encode(SAMPLE_TEXT)
    print(f"Encoded ({len(encoded)} ids): {encoded}")

    decoded = tokenizer.decode(encoded)
    print(f"Decoded:    {decoded!r}")
    assert decoded == SAMPLE_TEXT, "round-trip failed"
    print("Round-trip OK: decode(encode(text)) == text")

    try:
        tokenizer.encode("this has a § character")
    except ValueError as error:
        print(f"\nExpected failure on out-of-vocabulary input: {error}")


if __name__ == "__main__":
    main()
