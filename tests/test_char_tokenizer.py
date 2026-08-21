"""Unit tests for Module 2 (Character tokenizer)."""

from __future__ import annotations

import pytest

from app.tokenizer.char_tokenizer import CharTokenizer


def test_vocab_contains_exactly_the_unique_characters() -> None:
    """The vocabulary must match the set of distinct characters in the corpus."""
    tokenizer = CharTokenizer.from_text("abcabc")
    assert set(tokenizer.stoi.keys()) == {"a", "b", "c"}
    assert tokenizer.vocab_size == 3


def test_vocab_assignment_is_deterministic() -> None:
    """Building from the same text twice must yield identical id assignments."""
    text = "the quick brown fox jumps over the lazy dog"
    tokenizer_a = CharTokenizer.from_text(text)
    tokenizer_b = CharTokenizer.from_text(text)
    assert tokenizer_a.stoi == tokenizer_b.stoi


def test_encode_decode_roundtrip() -> None:
    """decode(encode(x)) must reconstruct x exactly for in-vocabulary text."""
    text = "To be, or not to be, that is the question."
    tokenizer = CharTokenizer.from_text(text)
    encoded = tokenizer.encode(text)
    assert tokenizer.decode(encoded) == text


def test_encode_produces_one_id_per_character() -> None:
    """Character tokenization has no compression: len(encode(x)) == len(x)."""
    text = "hello world"
    tokenizer = CharTokenizer.from_text(text)
    assert len(tokenizer.encode(text)) == len(text)


def test_encode_unknown_character_raises() -> None:
    """Encoding a character absent from the vocabulary must fail loudly."""
    tokenizer = CharTokenizer.from_text("abc")
    with pytest.raises(ValueError, match="not in the tokenizer's vocabulary"):
        tokenizer.encode("abcz")


def test_decode_out_of_range_id_raises() -> None:
    """Decoding an id outside [0, vocab_size) must fail loudly."""
    tokenizer = CharTokenizer.from_text("abc")
    with pytest.raises(ValueError, match="out of range"):
        tokenizer.decode([0, 1, 999])


def test_encode_empty_string_returns_empty_list() -> None:
    """The empty string is a valid input and encodes to an empty list."""
    tokenizer = CharTokenizer.from_text("abc")
    assert tokenizer.encode("") == []


def test_decode_empty_list_returns_empty_string() -> None:
    """An empty id list is valid and decodes to the empty string."""
    tokenizer = CharTokenizer.from_text("abc")
    assert tokenizer.decode([]) == ""


def test_save_and_load_roundtrip(tmp_path) -> None:
    """A tokenizer saved to disk and reloaded must behave identically."""
    text = "hello, world!"
    original = CharTokenizer.from_text(text)
    vocab_path = tmp_path / "vocab.json"

    original.save(vocab_path)
    restored = CharTokenizer.load(vocab_path)

    assert restored.stoi == original.stoi
    assert restored.vocab_size == original.vocab_size
    assert restored.decode(restored.encode(text)) == text
