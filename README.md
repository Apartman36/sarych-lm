# SARYCH-LM

SARYCH-LM is Antoniy Sarychev's from-scratch small language model project. It is a hands-on PyTorch systems project for building a decoder-only language model training stack one stage at a time: model code, tokenizer, memmap data, checkpointing, logging, resume, and generation.

This repository is not a production model, not a chatbot, and not instruction tuned. The current target is a small English story-continuation base model trained on TinyStories.

## Milestones

- v0.1 `sarych-5m-sanity`: synthetic-token sanity training for model, optimizer, checkpoint, resume, JSONL logging, and environment reporting.
- v0.2 `sarych-5m-tinystories-smoke`: byte-level BPE tokenizer, text-to-memmap data pipeline, short TinyStories-style smoke training, and tokenizer-backed generation.
- v0.3 `sarych-30m-tinystories-base`: 30M-class TinyStories base model training stage with vocab size 8192 and context length 512.

## Current v0.3 Target

`sarych-30m-tinystories-base` is the first serious SARYCH-LM base model attempt. It is intended to train on `TinyStories-train.txt`, validate on `TinyStories-valid.txt`, and produce English story continuation text after enough training.

Default architecture:

- Decoder-only Transformer
- RMSNorm, RoPE, SwiGLU, causal self-attention
- Tied token embeddings
- 10 layers, 8 heads, width 448, SwiGLU width 1344
- Byte-level BPE tokenizer with vocab size 8192
- Context length 512
- Approximately 30M trainable parameters

## Hardware Target

The main development target is Windows 11 Pro with WSL2 Ubuntu and a single NVIDIA GeForce RTX 5060 Ti 16GB. The known working CUDA stack is PyTorch 2.9.1+cu128, CUDA 12.8, compute capability 12.0, with BF16 supported.

CPU execution is useful for tests and tiny smoke checks, but the v0.3 model is intended for single-GPU WSL training.

## Example Generated Text

This section is intentionally a placeholder until a medium or long v0.3 run has produced representative samples.

```text
Prompt: Once upon a time
Sample: [pending v0.3 training run]
```

## Documentation

- Engineering setup and commands: [docs/dev_guide.md](docs/dev_guide.md)
- v0.3 model card: [docs/model_card_v0_3.md](docs/model_card_v0_3.md)
- Earlier design notes: [docs/v0_1_design.md](docs/v0_1_design.md), [docs/v0_2_design.md](docs/v0_2_design.md)

## Repository Hygiene

Downloaded data, processed memmaps, generated tokenizers, checkpoints, runs, caches, and virtual environments are intentionally ignored. The repo keeps only source code, configs, docs, tests, `.gitkeep` placeholders, and the tiny committed sample in `data/samples/tiny_stories_sample.txt`.
