# SARYCH-LM

SARYCH-LM is Antoniy Sarychev's from-scratch small language model project. It is a hands-on PyTorch systems project for building a decoder-only language model training stack one stage at a time: model code, tokenizer, memmap data, checkpointing, logging, resume, and generation.

This repository is not a production model and is not a safe or factual assistant. The current base model is a small English story model trained on TinyStories; v0.4 adds infrastructure for a Xiaomi-generated instruction-tuning pass.

## Milestones

- v0.1 `sarych-5m-sanity`: synthetic-token sanity training for model, optimizer, checkpoint, resume, JSONL logging, and environment reporting.
- v0.2 `sarych-5m-tinystories-smoke`: byte-level BPE tokenizer, text-to-memmap data pipeline, short TinyStories-style smoke training, and tokenizer-backed generation.
- v0.3 `sarych-30m-tinystories-base`: 29,770,944-parameter TinyStories base model trained for 10,000 steps.
- v0.4 `sarych-30m-instruct-xiaomi`: SFT infrastructure for Xiaomi-generated synthetic instruction data. This milestone prepares filtering, splitting, output-only SFT training, instruct generation, and manual base-vs-instruct evaluation.

## Current v0.3 Target

`sarych-30m-tinystories-base` is the first completed SARYCH-LM base model milestone. It trained on `TinyStories-train.txt`, validated on `TinyStories-valid.txt`, and produces coherent simple English story text.

Default architecture:

- Decoder-only Transformer
- RMSNorm, RoPE, SwiGLU, causal self-attention
- Tied token embeddings
- 10 layers, 8 heads, width 448, SwiGLU width 1344
- Byte-level BPE tokenizer with vocab size 8192
- Context length 512
- Approximately 30M trainable parameters

Completed run:

- Training steps: 10,000
- Tokens processed: 327,680,000
- Best validation loss: 1.493360996246338
- Final train loss: about 5.924356
- Peak CUDA memory: about 3.53 GB

## Hardware Target

The main development target is Windows 11 Pro with WSL2 Ubuntu and a single NVIDIA GeForce RTX 5060 Ti 16GB. The known working CUDA stack is PyTorch 2.9.1+cu128, CUDA 12.8, compute capability 12.0, with BF16 supported.

CPU execution is useful for tests and tiny smoke checks, but the v0.3 model is intended for single-GPU WSL training.

## v0.4 Xiaomi SFT

v0.4 keeps generation separate from training. Xiaomi is used only as a future teacher/judge for synthetic examples; SARYCH-LM does not call Xiaomi, OpenCode, or any external API during dataset building, training, tests, or evaluation.

The v0.4 SFT format uses literal chat markers without expanding the tokenizer vocabulary:

- `<|user|>`
- `<|assistant|>`
- `<|endoftext|>`

The SFT trainer masks loss on the user marker, instruction, optional input, assistant marker, and padding. Loss is computed only on assistant output tokens plus `<|endoftext|>`.

## Documentation

- Engineering setup and commands: [docs/dev_guide.md](docs/dev_guide.md)
- v0.3 model card: [docs/model_card_v0_3.md](docs/model_card_v0_3.md)
- v0.4 model card: [docs/model_card_v0_4.md](docs/model_card_v0_4.md)
- Xiaomi SFT data guide: [docs/xiaomi_sft_data.md](docs/xiaomi_sft_data.md)
- v0.4 evaluation guide: [docs/eval_v0_4.md](docs/eval_v0_4.md)
- Earlier design notes: [docs/v0_1_design.md](docs/v0_1_design.md), [docs/v0_2_design.md](docs/v0_2_design.md)

## Repository Hygiene

Downloaded data, processed memmaps, generated tokenizers, checkpoints, runs, caches, and virtual environments are intentionally ignored. The repo keeps only source code, configs, docs, tests, `.gitkeep` placeholders, and the tiny committed sample in `data/samples/tiny_stories_sample.txt`.
