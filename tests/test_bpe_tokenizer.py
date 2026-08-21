"""Unit tests for Module 3 (BPE tokenizer)."""

from __future__ import annotations

import pytest

from app.tokenizer.bpe import BPETokenizer

REPETITIVE_TEXT = (
    "To be, or not to be, that is the question: "
    "Whether 'tis nobler in the mind to suffer "
) * 20


def test_train_rejects_vocab_size_below_256() -> None:
    """The byte base vocabulary means vocab_size can never be smaller than 256."""
    with pytest.raises(ValueError, match="must be >= 256"):
        BPETokenizer.train(REPETITIVE_TEXT, vocab_size=100)


def test_train_with_no_merges_is_a_raw_byte_tokenizer() -> None:
    """vocab_size=256 (zero merges) must behave exactly like raw UTF-8 bytes."""
    tokenizer = BPETokenizer.train(REPETITIVE_TEXT, vocab_size=256)
    assert tokenizer.vocab_size == 256
    assert tokenizer.encode("hello") == list(b"hello")


def test_vocab_size_matches_requested_size() -> None:
    """Training for N merges must grow the vocabulary by exactly N tokens."""
    tokenizer = BPETokenizer.train(REPETITIVE_TEXT, vocab_size=300)
    assert tokenizer.vocab_size == 300
    assert len(tokenizer.merges) == 300 - 256


def test_training_is_deterministic() -> None:
    """Training twice on identical text must produce identical merge rules."""
    tokenizer_a = BPETokenizer.train(REPETITIVE_TEXT, vocab_size=280)
    tokenizer_b = BPETokenizer.train(REPETITIVE_TEXT, vocab_size=280)
    assert tokenizer_a.merges == tokenizer_b.merges


def test_encode_decode_roundtrip_on_training_text() -> None:
    """decode(encode(x)) must reconstruct x exactly for text seen in training."""
    tokenizer = BPETokenizer.train(REPETITIVE_TEXT, vocab_size=300)
    encoded = tokenizer.encode(REPETITIVE_TEXT)
    assert tokenizer.decode(encoded) == REPETITIVE_TEXT


def test_encode_decode_roundtrip_on_unseen_unicode_text() -> None:
    """Byte-level BPE must losslessly round-trip text never seen in training,
    including characters outside the training corpus entirely (unlike CharTokenizer)."""
    tokenizer = BPETokenizer.train(REPETITIVE_TEXT, vocab_size=300)
    unseen_text = "こんにちは! Emoji test: 🚀🔥, and some café résumé accents."
    encoded = tokenizer.encode(unseen_text)
    assert tokenizer.decode(encoded) == unseen_text


def test_bpe_compresses_repetitive_text_vs_raw_bytes() -> None:
    """Merging frequent pairs must strictly reduce token count on repetitive text."""
    tokenizer = BPETokenizer.train(REPETITIVE_TEXT, vocab_size=300)
    raw_byte_count = len(REPETITIVE_TEXT.encode("utf-8"))
    encoded_count = len(tokenizer.encode(REPETITIVE_TEXT))
    assert encoded_count < raw_byte_count


def test_decode_out_of_range_id_raises() -> None:
    """Decoding an id outside the vocabulary must fail loudly."""
    tokenizer = BPETokenizer.train(REPETITIVE_TEXT, vocab_size=260)
    with pytest.raises(ValueError, match="out of range"):
        tokenizer.decode([0, 1, 999_999])


def test_decode_of_id_subsequence_splitting_a_multibyte_char_raises_value_error() -> None:
    """Decoding a partial slice that cuts through a multi-byte UTF-8
    character must raise ValueError (not an unhandled UnicodeDecodeError).

    Uses vocab_size=256 (zero merges, one byte per token) so the split is
    deterministic and doesn't depend on which merges training happens to
    learn: "é" is exactly the 2-byte UTF-8 sequence 0xC3 0xA9, so decoding
    only its first byte can never be valid UTF-8 on its own."""
    tokenizer = BPETokenizer.train("hello world", vocab_size=256)
    ids = tokenizer.encode("é")
    assert len(ids) == 2
    with pytest.raises(ValueError, match="do not form valid UTF-8"):
        tokenizer.decode(ids[:1])


def test_save_and_load_roundtrip(tmp_path) -> None:
    """A tokenizer saved to disk and reloaded must behave identically."""
    original = BPETokenizer.train(REPETITIVE_TEXT, vocab_size=300)
    merges_path = tmp_path / "merges.json"

    original.save(merges_path)
    restored = BPETokenizer.load(merges_path)

    assert restored.merges == original.merges
    assert restored.vocab_size == original.vocab_size
    assert restored.decode(restored.encode(REPETITIVE_TEXT)) == REPETITIVE_TEXT
