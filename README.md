<div align="center">

# 🧠 mini-gpt

**A GPT-style decoder-only transformer, built from scratch in PyTorch — one module at a time.**

No Hugging Face `transformers`. No copied nanoGPT code. Every layer — tokenizer, attention, training loop, sampling, checkpointing, fine-tuning, chat — written and explained from first principles.

![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.4+-EE4C2C?logo=pytorch&logoColor=white)
![Tests](https://img.shields.io/badge/tests-235%20passing-brightgreen)
![Built from scratch](https://img.shields.io/badge/built-from%20scratch-blueviolet)
![Style: ruff](https://img.shields.io/badge/style-ruff-black)

</div>

---

## ✨ Highlights

- **30 modules, end to end** — from a character tokenizer all the way to a working chat REPL, each in its own `app/` module with a runnable `scripts/` demo.
- **Everything from scratch** — including **AdamW** and the **cosine LR schedule**, both cross-checked against PyTorch's own implementations in tests.
- **235 tests** — reference cross-checks (LayerNorm / GELU / AdamW / cross-entropy), end-to-end causal-masking integration tests, and the classic "overfit a tiny batch" training sanity check.
- **Pre-LN GPT-2-style blocks** with weight tying between the embedding and output projection.
- **Theory included** — the math and tensor-shape derivations behind each module live in [`docs/guia_mini_gpt.pdf`](docs/guia_mini_gpt.pdf).
- **Trainable on a laptop** — a ~460K-param char-level model trains on Tiny Shakespeare in ~9 min on CPU.

## Setup

```
uv sync
```

> **Note (Windows / corporate networks):** if `uv sync` fails with
> `invalid peer certificate: UnknownIssuer`, your network has TLS
> inspection whose root CA isn't in uv's bundled cert store. Either run
> `uv sync --system-certs`, or set `UV_NATIVE_TLS=true` once (as a user
> environment variable) so every `uv` invocation uses the OS certificate
> store automatically. The same issue affects any script here that talks to
> the network (`scripts/train_wikitext.py`'s dataset download): it works
> around it with the `truststore` package, which patches Python's `ssl`
> module to use the OS certificate store directly.

## Running tests

```
uv run pytest
```

235 tests across every module, including reference-implementation
cross-checks (LayerNorm/GELU/AdamW/cross-entropy vs. PyTorch's own), causal
masking end-to-end integration tests, and the classic "overfit a tiny batch"
sanity test for the training loop.

## Linting

```
uv run ruff check .
```

## Project layout

```
app/
    tokenizer/      # character + BPE tokenizers (Modules 2-3)
    dataset/        # tokenized dataset windowing / batching (Module 4)
    embeddings/     # token + positional embeddings (Modules 5-6)
    attention/      # self-attention, multi-head attention (Modules 9-10)
    transformer/    # LayerNorm, FFN, residuals, decoder block, full
                     # stack, output head, and the assembled GPTModel
                     # (Modules 7-8, 11-14)
    training/       # loss, training loop, optimizer, LR scheduler,
                     # checkpointing, fine-tuning (Modules 15-18, 24-26)
    inference/      # generation, sampling strategies, attention
                     # visualization, chat session (Modules 19-23, 27, 30)
    utils/          # reserved, currently unused
configs/            # reserved for YAML configs; not used by the current
                     # scripts (they hardcode hyperparameters directly)
tests/              # pytest unit + integration tests, one file per module
checkpoints/        # saved model checkpoints + tokenizer vocabs (gitignored)
datasets/           # raw/processed text data (gitignored)
outputs/            # generated artifacts, e.g. attention heatmap PNGs (gitignored)
scripts/            # thin, runnable entrypoints — one demo per module below
```

## Module index

Every module has: a theory + math write-up (see [`docs/guia_mini_gpt.pdf`](docs/guia_mini_gpt.pdf)),
implementation under `app/`, tests under `tests/`, and a runnable demo under
`scripts/`.

| # | Module | Library code | Demo script |
|---|--------|--------------|-------------|
| 1 | Project setup | — | `scripts/check_environment.py` |
| 2 | Character tokenizer | `app/tokenizer/char_tokenizer.py` | `scripts/tokenizer_demo.py` |
| 3 | BPE tokenizer | `app/tokenizer/bpe.py` | `scripts/bpe_demo.py` |
| 4 | Dataset loader | `app/dataset/` | `scripts/dataset_demo.py` |
| 5 | Embedding layer | `app/embeddings/token_embedding.py` | `scripts/embedding_demo.py` |
| 6 | Positional encoding | `app/embeddings/positional_encoding.py` | `scripts/positional_encoding_demo.py` |
| 7 | LayerNorm | `app/transformer/layernorm.py` | `scripts/layernorm_demo.py` |
| 8 | Feed forward network | `app/transformer/feedforward.py` | `scripts/feedforward_demo.py` |
| 9 | Self attention | `app/attention/self_attention.py` | `scripts/self_attention_demo.py` |
| 10 | Multi-head attention | `app/attention/multi_head_attention.py` | `scripts/multi_head_attention_demo.py` |
| 11 | Residual connections | `app/transformer/residual.py` | `scripts/residual_connection_demo.py` |
| 12 | Decoder block | `app/transformer/decoder_block.py` | `scripts/decoder_block_demo.py` |
| 13 | Transformer stack | `app/transformer/transformer_stack.py` | `scripts/transformer_stack_demo.py` |
| 14 | Output projection | `app/transformer/output_head.py`, `app/transformer/gpt_model.py` | `scripts/output_head_demo.py` |
| 15 | Loss function | `app/training/loss.py` | `scripts/loss_demo.py` |
| 16 | Training loop | `app/training/trainer.py` | `scripts/training_loop_demo.py` |
| 17 | Optimizer | `app/training/optimizer.py` | `scripts/optimizer_demo.py` |
| 18 | LR scheduler | `app/training/scheduler.py` | `scripts/scheduler_demo.py` |
| 19 | Text generation | `app/inference/generator.py` | `scripts/generation_demo.py` |
| 20 | Sampling | `app/inference/sampling.py` | `scripts/sampling_demo.py` |
| 21 | Temperature | `app/inference/temperature.py` | `scripts/temperature_demo.py` |
| 22 | Top-K sampling | `app/inference/top_k.py` | `scripts/top_k_demo.py` |
| 23 | Top-P sampling | `app/inference/top_p.py` | `scripts/top_p_demo.py` |
| 24 | Saving checkpoints | `app/training/checkpoint.py` | `scripts/checkpoint_demo.py` |
| 25 | Loading checkpoints | `app/training/checkpoint.py` | `scripts/checkpoint_resume_demo.py` |
| 26 | Fine-tuning | `app/training/finetune.py` | `scripts/finetune_demo.py` |
| 27 | Visualizing attention | `app/inference/attention_viz.py` | `scripts/attention_viz_demo.py` |
| 28 | Train on Tiny Shakespeare | — | `scripts/train_tiny_shakespeare.py` |
| 29 | Train on WikiText | — | `scripts/train_wikitext.py` |
| 30 | Chat interface | `app/inference/chat.py` | `scripts/chat.py` |

## Try the trained model

`scripts/train_tiny_shakespeare.py` trains a small char-level GPT
(~460K params) on the canonical Tiny Shakespeare corpus (~9 min on CPU) and
saves a checkpoint + tokenizer vocab to `checkpoints/`. Once that's run once:

```
uv run python scripts/chat.py
```

This is a raw completion model, not an instruction-tuned assistant — typing
text gets you a Shakespeare-style *continuation* of it, not an answer.
Type `/help` inside the REPL for sampling controls (temperature, top-k, top-p).

## Design notes

- **Pre-LN transformer blocks** (GPT-2 style): `x = x + Sublayer(LayerNorm(x))`,
  with one final LayerNorm after the last block (`TransformerStack.final_norm`).
- **Weight tying** (`GPTModel(tie_weights=True)`, the default): the output
  projection reuses the token embedding matrix, saving `vocab_size * d_model`
  parameters.
- **AdamW and the cosine LR schedule are implemented from scratch** (Modules
  17-18), not `torch.optim.AdamW` — both are cross-checked against PyTorch's
  own implementations in tests.
- **No KV cache**: `Generator` (Module 19) recomputes the full forward pass
  at every generation step — simple and correct, but `O(new_tokens * seq_len²)`.
  Noted as the natural next optimization, not implemented, to keep the
  codebase readable.
