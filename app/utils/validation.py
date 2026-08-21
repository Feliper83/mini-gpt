"""Shared validation helpers."""

from __future__ import annotations


def check_seq_len(seq_len: int, max_seq_len: int, context: str) -> None:
    """Raises `ValueError` if `seq_len` exceeds `max_seq_len`.

    Several modules (causal attention masks, positional encodings) precompute
    a buffer sized for a maximum sequence length and must reject any request
    longer than that instead of silently truncating or misaligning.

    Args:
        seq_len: The sequence length actually being requested.
        max_seq_len: The longest sequence length that was precomputed for.
        context: A short phrase describing what was precomputed (e.g.
            "this causal mask", "this positional encoding"), used to build
            a clear, specific error message.

    Raises:
        ValueError: If `seq_len > max_seq_len`.
    """
    if seq_len > max_seq_len:
        raise ValueError(
            f"seq_len ({seq_len}) exceeds max_seq_len ({max_seq_len}) that {context} "
            "was precomputed for."
        )
