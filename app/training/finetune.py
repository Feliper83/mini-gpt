"""Fine-tuning utilities: freezing layers of a pretrained model."""

from __future__ import annotations

from torch import nn

from app.transformer.gpt_model import GPTModel


def freeze_all_parameters(model: nn.Module) -> None:
    """Freezes every parameter in `model` (sets requires_grad=False).

    A frozen parameter receives no gradient during backward() and is never
    updated by the optimizer -- `build_param_groups` (Module 17) already
    excludes non-trainable parameters, and this project's `AdamW.step`
    (Module 17) skips any parameter whose `.grad` is `None`.

    Args:
        model: The model whose parameters will all be frozen.
    """
    for param in model.parameters():
        param.requires_grad = False


def unfreeze_last_n_blocks(model: GPTModel, n_trainable_blocks: int) -> None:
    """Unfreezes the last `n_trainable_blocks` decoder blocks and the output head.

    A common fine-tuning strategy: keep the token embedding and EARLY
    transformer blocks frozen (they tend to capture general, broadly
    reusable patterns from pretraining), and let only the LAST few blocks
    plus the output head adapt to the new task or domain.

    Note: if `model` was built with `tie_weights=True` (Module 14), the
    output head's projection weight IS the token embedding's weight -- the
    same `Parameter` object. Unfreezing the output head therefore also
    unfreezes the token embedding, even with `n_trainable_blocks=0`. Use
    `tie_weights=False` if you need to freeze the embedding independently
    of the output head.

    Args:
        model: A `GPTModel`, typically already fully frozen via
            `freeze_all_parameters`.
        n_trainable_blocks: Number of decoder blocks, counting from the
            END of the stack, to unfreeze.

    Raises:
        ValueError: If `n_trainable_blocks` is negative or exceeds the
            number of blocks in the model.
    """
    total_blocks = len(model.transformer.blocks)
    if not (0 <= n_trainable_blocks <= total_blocks):
        raise ValueError(
            f"n_trainable_blocks must be in [0, {total_blocks}], got {n_trainable_blocks}"
        )

    for block in model.transformer.blocks[total_blocks - n_trainable_blocks :]:
        for param in block.parameters():
            param.requires_grad = True

    for param in model.transformer.final_norm.parameters():
        param.requires_grad = True
    for param in model.output_head.parameters():
        param.requires_grad = True


def count_trainable_parameters(model: nn.Module) -> tuple[int, int]:
    """Counts trainable vs. total parameters.

    Args:
        model: The model to inspect.

    Returns:
        A tuple `(trainable_count, total_count)`.
    """
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total
