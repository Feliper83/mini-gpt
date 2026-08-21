"""Character-level tokenizer.

Builds a vocabulary from the distinct characters observed in a corpus and
converts between text and lists of integer token ids using that vocabulary.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class CharTokenizer:
    """Maps text to integer ids one character at a time.

    Attributes:
        stoi: Mapping from character to token id ("string-to-index").
        itos: Mapping from token id to character ("index-to-string"). Derived
            from `stoi` and kept in sync with it.
    """

    stoi: dict[str, int]
    itos: dict[int, str] = field(init=False)

    def __post_init__(self) -> None:
        self.itos = {index: char for char, index in self.stoi.items()}

    @classmethod
    def from_text(cls, text: str) -> CharTokenizer:
        """Builds a tokenizer whose vocabulary is every distinct character in `text`.

        Args:
            text: Corpus used to derive the vocabulary. Every character that
                appears here becomes part of the vocabulary; characters that
                never appear cannot be encoded later.

        Returns:
            A `CharTokenizer` with a vocabulary sorted for determinism: the
            same input text always produces the same id assignment.
        """
        unique_chars = sorted(set(text))
        stoi = {char: index for index, char in enumerate(unique_chars)}
        return cls(stoi=stoi)

    @property
    def vocab_size(self) -> int:
        """Number of distinct characters in the vocabulary."""
        return len(self.stoi)

    def encode(self, text: str) -> list[int]:
        """Converts a string into a list of token ids, one per character.

        Args:
            text: Input string. Every character must exist in the vocabulary.

        Returns:
            A list of token ids with the same length as `text`.

        Raises:
            ValueError: If `text` contains a character outside the vocabulary.
        """
        try:
            return [self.stoi[char] for char in text]
        except KeyError as error:
            unknown_char = error.args[0]
            raise ValueError(
                f"Character {unknown_char!r} (U+{ord(unknown_char):04X}) is not in the "
                f"tokenizer's vocabulary of {self.vocab_size} characters."
            ) from error

    def decode(self, ids: Sequence[int]) -> str:
        """Converts a sequence of token ids back into a string.

        Args:
            ids: Token ids to decode. Every id must be within `[0, vocab_size)`.

        Returns:
            The concatenated characters corresponding to `ids`.

        Raises:
            ValueError: If an id has no corresponding character.
        """
        try:
            return "".join(self.itos[token_id] for token_id in ids)
        except KeyError as error:
            unknown_id = error.args[0]
            raise ValueError(
                f"Token id {unknown_id} is out of range for a vocabulary of "
                f"size {self.vocab_size}."
            ) from error

    def save(self, path: str | Path) -> None:
        """Persists the vocabulary to a JSON file.

        Args:
            path: Destination file path. Parent directories must already exist.
        """
        Path(path).write_text(json.dumps(self.stoi, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> CharTokenizer:
        """Loads a tokenizer from a vocabulary JSON file written by `save`.

        Args:
            path: Path to a JSON file containing a `str -> int` mapping.

        Returns:
            A `CharTokenizer` restored with the exact same id assignments.
        """
        stoi = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(stoi=stoi)
