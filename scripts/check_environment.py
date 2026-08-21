"""Runnable example for Module 1 (Project setup).

Verifies that the project is installed correctly and that PyTorch can see
the expected compute devices. Run with:

    uv run python scripts/check_environment.py
"""

from __future__ import annotations

import sys

import torch

import app


def main() -> None:
    """Print environment diagnostics and confirm the package is importable."""
    print(f"mini-gpt version: {app.__version__}")
    print(f"Python version:   {sys.version.split()[0]}")
    print(f"PyTorch version:  {torch.__version__}")
    print(f"CUDA available:   {torch.cuda.is_available()}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = torch.randn(2, 3, device=device)
    y = torch.randn(3, 4, device=device)
    z = x @ y

    print(f"Sanity matmul on '{device}': {tuple(x.shape)} @ {tuple(y.shape)} -> {tuple(z.shape)}")
    print("Environment OK.")


if __name__ == "__main__":
    main()
