"""Runnable example for Module 17 (Optimizer).

Run with:

    uv run python scripts/optimizer_demo.py
"""

from __future__ import annotations

import torch
from torch import nn

from app.training.optimizer import AdamW, build_param_groups
from app.transformer.gpt_model import GPTModel


def main() -> None:
    """Sanity-check convergence, cross-check against PyTorch, then show
    GPT-2 style parameter grouping on a real model."""
    print("--- Convergence check: minimize f(x) = (x - 3)^2 ---")
    x = nn.Parameter(torch.tensor([0.0]))
    optimizer = AdamW([x], lr=0.1)
    for step in range(100):
        loss = (x - 3.0) ** 2
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 20 == 0:
            print(f"  step {step:3d}: x={x.item():.4f}, loss={loss.item():.6f}")
    print(f"  final x = {x.item():.4f} (target: 3.0)\n")

    print("--- Cross-check against torch.optim.AdamW over 20 steps ---")
    torch.manual_seed(0)
    param_ours = nn.Parameter(torch.randn(10))
    param_ref = nn.Parameter(param_ours.data.clone())
    ours = AdamW([param_ours], lr=1e-2, weight_decay=0.01)
    reference = torch.optim.AdamW([param_ref], lr=1e-2, weight_decay=0.01)
    for _ in range(20):
        grad = torch.randn(10)
        param_ours.grad = grad.clone()
        param_ref.grad = grad.clone()
        ours.step()
        reference.step()
    max_diff = (param_ours - param_ref).abs().max().item()
    print(f"  Max absolute difference vs torch.optim.AdamW: {max_diff:.2e}\n")

    print("--- GPT-2 style parameter grouping on a real GPTModel ---")
    model = GPTModel(vocab_size=50, d_model=32, n_heads=4, n_layers=2, max_seq_len=16)
    decay_group, no_decay_group = build_param_groups(model, weight_decay=0.1)
    decay_count = sum(p.numel() for p in decay_group["params"])
    no_decay_count = sum(p.numel() for p in no_decay_group["params"])
    decay_wd = decay_group["weight_decay"]
    no_decay_wd = no_decay_group["weight_decay"]
    print(f"  Weight-decay group:    {decay_count:6d} params (weight_decay={decay_wd})")
    print(f"  No-weight-decay group: {no_decay_count:6d} params (weight_decay={no_decay_wd})")

    print("\n  Examples of parameters in each group:")
    examples = (
        "token_embedding.embedding.weight",
        "transformer.blocks.0.attention.query_proj.weight",
        "transformer.blocks.0.attention.query_proj.bias",
        "transformer.blocks.0.attention_residual.norm.gamma",
    )
    for name, param in model.named_parameters():
        if name in examples:
            group = "DECAY" if param.dim() >= 2 else "no-decay"
            print(f"    {group:>8}  {name:50s} shape={tuple(param.shape)}")


if __name__ == "__main__":
    main()
