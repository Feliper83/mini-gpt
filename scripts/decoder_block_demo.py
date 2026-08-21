"""Runnable example for Module 12 (Decoder block).

Run with:

    uv run python scripts/decoder_block_demo.py
"""

from __future__ import annotations

import torch

from app.transformer.decoder_block import DecoderBlock

D_MODEL = 32
N_HEADS = 4
MAX_SEQ_LEN = 16
N_BLOCKS = 6  # a preview of Module 13 (Transformer Stack)


def main() -> None:
    """Inspect a single block's parameter breakdown, then stack several and
    verify shape, numerical stability, and causal masking survive the stack."""
    torch.manual_seed(0)
    block = DecoderBlock(D_MODEL, N_HEADS, MAX_SEQ_LEN)

    x = torch.randn(2, 5, D_MODEL)
    output = block(x)
    print(f"Input shape:  {tuple(x.shape)}")
    print(f"Output shape: {tuple(output.shape)}  (unchanged -- required for stacking)")

    total_params = sum(p.numel() for p in block.parameters())
    attn_params = sum(p.numel() for p in block.attention.parameters())
    ffn_params = sum(p.numel() for p in block.feed_forward.parameters())
    print(f"\nOne DecoderBlock parameter count: {total_params}")
    print(f"  attention:    {attn_params:5d}  ({100 * attn_params / total_params:.1f}%)")
    print(f"  feed_forward: {ffn_params:5d}  ({100 * ffn_params / total_params:.1f}%)")

    print(f"\n--- Stacking {N_BLOCKS} blocks (a preview of Module 13) ---")
    blocks = [DecoderBlock(D_MODEL, N_HEADS, MAX_SEQ_LEN) for _ in range(N_BLOCKS)]
    h = torch.randn(1, 6, D_MODEL)
    for i, blk in enumerate(blocks):
        h = blk(h)
        is_finite = torch.isfinite(h).all().item()
        print(f"  after block {i}: shape={tuple(h.shape)}, finite={is_finite}")

    print("\n--- Causal masking preserved through the FULL stack ---")
    h0 = torch.randn(1, 6, D_MODEL)
    h0_perturbed = h0.clone()
    h0_perturbed[0, -1] = torch.randn(D_MODEL) * 100

    out_a, out_b = h0, h0_perturbed
    for blk in blocks:
        out_a = blk(out_a)
        out_b = blk(out_b)

    for i in range(5):
        unchanged = torch.allclose(out_a[0, i], out_b[0, i], atol=1e-4)
        print(f"  Position {i} unchanged after {N_BLOCKS} blocks: {unchanged}")
    changed = not torch.allclose(out_a[0, 5], out_b[0, 5], atol=1e-4)
    print(f"  Position 5 (perturbed) changed:      {changed}")


if __name__ == "__main__":
    main()
