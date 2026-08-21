"""Checkpoint saving and loading for model + optimizer + training state."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from app.transformer.gpt_model import GPTModel


@dataclass
class ModelConfig:
    """Architecture hyperparameters needed to reconstruct a `GPTModel`.

    A `state_dict` alone is just a mapping of tensor names to values -- it
    doesn't encode which constructor arguments (`vocab_size`, `d_model`,
    ...) were used to build the model in the first place. These field names
    intentionally mirror `GPTModel.__init__`'s keyword arguments exactly, so
    a config can be unpacked straight into it: `GPTModel(**vars(config))`.
    """

    vocab_size: int
    d_model: int
    n_heads: int
    n_layers: int
    max_seq_len: int
    d_ff: int | None = None
    dropout: float = 0.0
    tie_weights: bool = True


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    config: ModelConfig,
    extra: dict[str, Any] | None = None,
) -> None:
    """Saves everything needed to exactly resume training or reload the model.

    Args:
        path: Destination file path. Parent directories are created if
            they don't already exist.
        model: The model whose weights will be saved.
        optimizer: The optimizer whose state (e.g. AdamW's m/v buffers,
            Module 17) will be saved -- without this, resuming would
            restart the optimizer's gradient-history statistics from
            scratch, causing a discontinuity in training dynamics.
        step: The current training step, so a resumed run knows where to
            continue the learning rate schedule (Module 18) from.
        config: Architecture hyperparameters, needed to reconstruct the
            model class before its weights can be loaded back in (Module 25).
        extra: Optional additional metadata to store alongside the
            checkpoint (e.g. loss history). Defaults to an empty dict.
    """
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "step": step,
        "config": asdict(config),
        "extra": extra or {},
    }
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, destination)


@dataclass
class LoadedCheckpoint:
    """Everything reconstructed from a saved checkpoint."""

    model: GPTModel
    step: int
    config: ModelConfig
    extra: dict[str, Any]


def load_checkpoint(
    path: str | Path,
    model: GPTModel | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    map_location: str | torch.device = "cpu",
) -> LoadedCheckpoint:
    """Reconstructs a GPTModel (and optionally an optimizer's state) from a checkpoint.

    Args:
        path: Path to a checkpoint written by `save_checkpoint`.
        model: If provided, weights are loaded into this EXISTING model
            instance in place, rather than constructing a new one. This is
            REQUIRED when also passing `optimizer`: an optimizer's param
            groups hold references to specific parameter tensor objects, so
            resuming training only works if weights are loaded into the
            exact model instance the optimizer was already built over. If
            `model` is `None` and `optimizer` is also `None`, a fresh model
            is constructed from the saved config -- convenient for
            inference-only use (Module 19's `Generator`), where there is no
            optimizer to keep in sync.
        optimizer: If provided, must already be constructed over `model`'s
            parameters -- its state (e.g. AdamW's m/v buffers, Module 17)
            will be loaded into it in place.
        map_location: Device to load tensors onto.

    Returns:
        A `LoadedCheckpoint` with the model (the passed-in `model` if given,
        otherwise a freshly constructed one), the saved step, config, and
        any extra metadata.

    Raises:
        ValueError: If `optimizer` is given without `model`. Constructing a
            fresh model internally in that case would leave `optimizer`
            referencing a different model's parameters than the one
            actually used for training afterward -- backward() would
            populate gradients on the fresh model returned here, while
            `optimizer.step()` would silently update nothing, since it only
            knows about the caller's original (untouched) parameters.
    """
    if optimizer is not None and model is None:
        raise ValueError(
            "load_checkpoint(optimizer=...) also requires model=...: resuming "
            "training needs weights loaded into the SAME model instance the "
            "optimizer was built over. Build the model from the checkpoint's "
            "config, construct the optimizer over it, then pass both here."
        )

    raw = torch.load(path, map_location=map_location, weights_only=False)
    config = ModelConfig(**raw["config"])

    if model is None:
        model = GPTModel(**vars(config))
    model.load_state_dict(raw["model_state_dict"])

    if optimizer is not None:
        optimizer.load_state_dict(raw["optimizer_state_dict"])

    return LoadedCheckpoint(model=model, step=raw["step"], config=config, extra=raw["extra"])
