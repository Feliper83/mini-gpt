"""Unit tests for the shared app.utils.validation helper."""

from __future__ import annotations

import pytest

from app.utils.validation import check_seq_len


def test_seq_len_within_bounds_does_not_raise() -> None:
    check_seq_len(seq_len=5, max_seq_len=10, context="this thing")
    check_seq_len(seq_len=10, max_seq_len=10, context="this thing")  # boundary: equal is fine


def test_seq_len_exceeding_bounds_raises_with_context() -> None:
    with pytest.raises(ValueError, match="exceeds max_seq_len") as exc_info:
        check_seq_len(seq_len=11, max_seq_len=10, context="this widget")
    assert "this widget" in str(exc_info.value)
