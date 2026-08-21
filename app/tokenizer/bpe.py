"""Byte-level Byte Pair Encoding (BPE) tokenizer, trained from scratch.

Unlike `CharTokenizer`, the base vocabulary is the 256 possible byte values,
so this tokenizer can always encode and decode any valid UTF-8 text without
ever hitting an out-of-vocabulary error.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

Pair = tuple[int, int]


@dataclass
class BPETokenizer:
    """Byte-level BPE tokenizer.

    Attributes:
        merges: Ordered mapping from a byte-pair to the new token id it was
            merged into. The order (by increasing id) is the order in which
            merges were learned during training, and encoding must replay
            them in that same order.
        vocab: Mapping from token id to the raw bytes it represents. Derived
            from `merges` and kept in sync with it.
    """

    merges: dict[Pair, int]
    vocab: dict[int, bytes] = field(init=False)

    def __post_init__(self) -> None:
        self.vocab = _build_vocab(self.merges)

    @property
    def vocab_size(self) -> int:
        """Total number of tokens: 256 base bytes plus every learned merge."""
        return len(self.vocab)

    @classmethod
    def train(cls, text: str, vocab_size: int) -> BPETokenizer:
        """Learns merge rules from `text` until the vocabulary reaches `vocab_size`.

        Args:
            text: Training corpus.
            vocab_size: Desired final vocabulary size. Must be at least 256,
                since the 256 raw byte values are always the base vocabulary.

        Returns:
            A trained `BPETokenizer`.

        Raises:
            ValueError: If `vocab_size` is smaller than 256.
        """
        if vocab_size < 256:
            raise ValueError(
                f"vocab_size must be >= 256 (the byte base vocabulary), got {vocab_size}"
            )

        num_merges = vocab_size - 256
        ids = list(text.encode("utf-8"))
        merges: dict[Pair, int] = {}

        for merge_index in range(num_merges):
            pair_counts = _count_pairs(ids)
            if not pair_counts:
                break  # Sequence has collapsed to a single token (or is empty).

            best_pair = max(pair_counts, key=lambda pair: pair_counts[pair])
            new_id = 256 + merge_index
            ids = _merge_pair(ids, best_pair, new_id)
            merges[best_pair] = new_id

        return cls(merges=merges)

    def encode(self, text: str) -> list[int]:
        """Encodes text into token ids by replaying learned merges in order.

        Args:
            text: Input string. Any valid Unicode text is accepted.

        Returns:
            A list of token ids.
        """
        ids = list(text.encode("utf-8"))

        while len(ids) >= 2:
            pair_counts = _count_pairs(ids)
            mergeable_pairs = [pair for pair in pair_counts if pair in self.merges]
            if not mergeable_pairs:
                break

            # Earliest-learned merge (smallest assigned id) wins, matching
            # the greedy order the merges were produced in during training.
            best_pair = min(mergeable_pairs, key=lambda pair: self.merges[pair])
            ids = _merge_pair(ids, best_pair, self.merges[best_pair])

        return ids

    def decode(self, ids: Sequence[int]) -> str:
        """Decodes token ids back into text.

        Args:
            ids: Token ids produced by `encode` (or otherwise valid ids).

        Returns:
            The decoded string.

        Raises:
            ValueError: If an id has no corresponding vocabulary entry, or if
                `ids` is a subsequence whose concatenated bytes split a
                multi-byte UTF-8 character (e.g. decoding only part of a
                previously encoded sequence).
        """
        try:
            raw_bytes = b"".join(self.vocab[token_id] for token_id in ids)
        except KeyError as error:
            unknown_id = error.args[0]
            raise ValueError(
                f"Token id {unknown_id} is out of range for a vocabulary of "
                f"size {self.vocab_size}."
            ) from error

        try:
            return raw_bytes.decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValueError(
                "Cannot decode this id sequence: its bytes do not form valid "
                "UTF-8 on their own. This usually means `ids` is a partial "
                "slice of a longer encoded sequence that split a multi-byte "
                "character -- decode the full sequence, or a slice aligned "
                "on token boundaries that don't cut through one."
            ) from error

    def save(self, path: str | Path) -> None:
        """Persists the learned merges to a JSON file.

        Args:
            path: Destination file path. Parent directories must already exist.
        """
        ordered_merges = sorted(self.merges.items(), key=lambda item: item[1])
        payload = [[a, b, new_id] for (a, b), new_id in ordered_merges]
        Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> BPETokenizer:
        """Loads a tokenizer from a merges JSON file written by `save`.

        Args:
            path: Path to a JSON file containing `[a, b, new_id]` triples.

        Returns:
            A `BPETokenizer` restored with the exact same merge rules.
        """
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        merges = {(a, b): new_id for a, b, new_id in payload}
        return cls(merges=merges)


def _count_pairs(ids: list[int]) -> Counter[Pair]:
    """Counts adjacent-pair frequencies in a token id sequence."""
    return Counter(zip(ids, ids[1:], strict=False))


def _merge_pair(ids: list[int], pair: Pair, new_id: int) -> list[int]:
    """Replaces every non-overlapping occurrence of `pair` in `ids` with `new_id`."""
    merged: list[int] = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and (ids[i], ids[i + 1]) == pair:
            merged.append(new_id)
            i += 2
        else:
            merged.append(ids[i])
            i += 1
    return merged


def _build_vocab(merges: dict[Pair, int]) -> dict[int, bytes]:
    """Reconstructs id -> bytes for every base byte and every learned merge."""
    vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    for (a, b), new_id in sorted(merges.items(), key=lambda item: item[1]):
        vocab[new_id] = vocab[a] + vocab[b]
    return vocab
