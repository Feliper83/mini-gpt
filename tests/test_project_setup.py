"""Sanity tests for Module 1 (Project setup).

These tests don't exercise any model logic yet — they only confirm that the
package structure is importable and that the PyTorch installation works,
which is the entire scope of this module.
"""

from __future__ import annotations

import torch

import app


def test_package_is_importable() -> None:
    """The top-level `app` package must expose a version string."""
    assert hasattr(app, "__version__")
    assert isinstance(app.__version__, str)


def test_subpackages_are_importable() -> None:
    """Every subpackage declared in the architecture must be importable."""
    import app.attention
    import app.dataset
    import app.embeddings
    import app.inference
    import app.tokenizer
    import app.training
    import app.transformer
    import app.utils  # noqa: F401


def test_torch_matmul_sanity() -> None:
    """A basic tensor operation must run and produce the expected shape."""
    x = torch.randn(2, 3)
    y = torch.randn(3, 4)
    z = x @ y
    assert z.shape == (2, 4)
