"""Shared interface implemented by every tokenizer in this project."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol


class Tokenizer(Protocol):
    """Common contract for text <-> integer-id conversion.

    Every tokenizer (character-level, BPE, ...) implements this interface so
    that downstream code (the dataset pipeline, the generator) can depend on
    `Tokenizer` without knowing which concrete implementation it is using.
    """

    @property
    def vocab_size(self) -> int:
        """Number of distinct token ids this tokenizer can produce."""
        ...

    def encode(self, text: str) -> list[int]:
        """Convert a string into a list of token ids."""
        ...

    def decode(self, ids: Sequence[int]) -> str:
        """Convert a sequence of token ids back into a string."""
        ...
