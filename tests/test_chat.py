"""Unit tests for Module 30 (Chat interface)."""

from __future__ import annotations

import pytest

from app.inference.chat import ChatSession, SamplingConfig
from app.tokenizer.char_tokenizer import CharTokenizer
from app.transformer.gpt_model import GPTModel

CORPUS = "to be or not to be that is the question"
D_MODEL = 16
N_HEADS = 4
N_LAYERS = 2
MAX_SEQ_LEN = 32


def _build_session() -> ChatSession:
    tokenizer = CharTokenizer.from_text(CORPUS)
    model = GPTModel(tokenizer.vocab_size, D_MODEL, N_HEADS, N_LAYERS, MAX_SEQ_LEN)
    return ChatSession(model, tokenizer, max_seq_len=MAX_SEQ_LEN)


def test_sampling_config_has_sensible_defaults() -> None:
    """Defaults must be temperature=0.8, top_p=0.9, top_k disabled."""
    config = SamplingConfig()
    assert config.temperature == 0.8
    assert config.top_p == 0.9
    assert config.top_k is None


def test_send_appends_user_text_and_continuation_to_transcript() -> None:
    """After send(), the transcript must contain the user text followed by
    whatever was generated."""
    session = _build_session()
    continuation = session.send("to be", max_new_tokens=5)
    assert session.transcript == "to be" + continuation


def test_send_returns_only_the_new_continuation() -> None:
    """The return value must be just the newly generated text, not the prompt."""
    session = _build_session()
    continuation = session.send("to be", max_new_tokens=5)
    assert not continuation.startswith("to be")
    assert len(session.tokenizer.encode(continuation)) == 5


def test_multi_turn_conversation_grows_the_transcript() -> None:
    """Each send() call must append onto the existing transcript, not replace it."""
    session = _build_session()
    session.send("to be", max_new_tokens=3)
    length_after_first = len(session.transcript)
    session.send(" or", max_new_tokens=3)
    assert len(session.transcript) > length_after_first


def test_reset_clears_the_transcript() -> None:
    """reset() must return the session to an empty transcript."""
    session = _build_session()
    session.send("to be", max_new_tokens=3)
    session.reset()
    assert session.transcript == ""


def test_send_raises_on_out_of_vocabulary_character() -> None:
    """A character outside the tokenizer's vocabulary must fail loudly,
    inherited directly from CharTokenizer's behavior (Module 2)."""
    session = _build_session()
    with pytest.raises(ValueError, match="not in the tokenizer's vocabulary"):
        session.send("this has a § character never seen in training")


def test_rejected_send_does_not_poison_the_transcript() -> None:
    """Regression test: a send() that raises must leave the transcript
    unchanged, so a later valid send() is NOT re-encoding (and re-failing
    on) the previously rejected text."""
    session = _build_session()
    session.send("to be", max_new_tokens=3)
    transcript_before = session.transcript

    with pytest.raises(ValueError):
        session.send("§ bad char")
    assert session.transcript == transcript_before

    # A perfectly valid follow-up must succeed, not raise again.
    session.send("or not", max_new_tokens=3)


def test_context_truncation_does_not_crash_on_a_long_transcript() -> None:
    """Repeated turns eventually exceed max_seq_len; Generator's truncation
    (Module 19) must keep this working without error."""
    session = _build_session()
    for _ in range(10):
        session.send("to be or not", max_new_tokens=10)
    assert len(session.transcript) > MAX_SEQ_LEN


def test_custom_sampling_config_is_used() -> None:
    """A session built with a custom SamplingConfig must use it, not the defaults."""
    tokenizer = CharTokenizer.from_text(CORPUS)
    model = GPTModel(tokenizer.vocab_size, D_MODEL, N_HEADS, N_LAYERS, MAX_SEQ_LEN)
    custom = SamplingConfig(temperature=1.5, top_k=3, top_p=None)
    session = ChatSession(model, tokenizer, max_seq_len=MAX_SEQ_LEN, sampling=custom)

    assert session.sampling.temperature == 1.5
    assert session.sampling.top_k == 3
    assert session.sampling.top_p is None
    session.send("to be", max_new_tokens=3)  # must not raise with this config
