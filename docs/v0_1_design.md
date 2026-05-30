# SARYCH-LM v0.1 Design

## Purpose

v0.1 exists to validate the engineering path before any real language data is introduced. The goal is a small, reproducible training pipeline that proves model construction, forward pass, loss computation, backward pass, AdamW updates, BF16 autocast, FP32 fallback, checkpoint/resume, synthetic data, logging, sampling, and environment reporting.

The model is intentionally not useful. It is a sanity-check system for future SARYCH-LM work.

## Why Synthetic Data First

Synthetic data removes tokenizer, dataset download, preprocessing, and licensing variables. The mixed generator includes random chunks, short motifs, and arithmetic sequences so loss can improve and tiny overfit tests can pass. That validates training mechanics without claiming language understanding.

## Model Architecture

`sarych-5m-sanity` is a decoder-only Transformer configured from YAML:

- vocabulary size 1024
- context length 128
- 4 layers
- 4 attention heads
- embedding width 256
- feed-forward width 768
- RMSNorm
- RoPE on query/key tensors
- SwiGLU feed-forward blocks
- tied token embedding and LM head weights
- dropout configurable, default 0.0
- bias disabled by default

Attention uses PyTorch `scaled_dot_product_attention` when available with `is_causal=True`. A manual masked attention path remains in the module for older PyTorch versions.

## Training Loop

The trainer loads YAML configuration, sets seeds, selects CUDA if available, uses BF16 autocast only when CUDA BF16 is supported, and otherwise runs FP32. It uses AdamW, gradient accumulation, gradient clipping, warmup plus cosine decay, periodic evaluation, JSONL logging, sample generation, and checkpoints.

`torch.compile` is exposed only as an optional config flag and defaults to false.

## Checkpoint And Resume

Checkpoints are plain `.pt` files saved with `torch.save`. Each checkpoint stores:

- model state dict
- optimizer state dict
- simple scheduler metadata
- current completed optimizer step
- best validation loss
- full config
- Python, NumPy, Torch CPU, and CUDA RNG state when available
- timestamp
- git commit when available
- parameter count
- environment metadata

The trainer writes step checkpoints, `checkpoint_latest.pt`, and `checkpoint_best.pt` when validation improves. Resume loads `checkpoint_latest.pt`, restores optimizer and RNG state, and continues from the saved step. Checkpoints are saved only after completed optimizer steps.

RNG restore normalizes saved PyTorch RNG states back to CPU `uint8` ByteTensors before calling PyTorch restore APIs, which keeps resume compatible with PyTorch 2.9/Python 3.12 serialization behavior. Restore failures raise a clear error unless `train.strict_rng_restore` is explicitly set false.

## Environment Assumptions

The target training environment is WSL2 Ubuntu on the RTX 5060 Ti system. Blackwell support requires PyTorch 2.7+ with CUDA 12.8+ wheels. The code degrades gracefully on CPU for tests and smoke runs.

## Intentional Exclusions

v0.1 excludes TinyStories, Hugging Face datasets, Hugging Face transformers, tokenizers, Qwen distillation, instruction tuning, APIs, UIs, Docker, DeepSpeed, FSDP, DDP, FlashAttention, Triton kernels, custom CUDA, LoRA, quantization, MoE, and cloud training.

## Future Additions

v0.2 adds a local text sample, byte-level BPE tokenizer, memmap token dataset, optional TinyStories download, and a text smoke model. v1 can introduce the first real `sarych-30m-base` training path. Those should be added only after v0.2 proves reliable on the target WSL/CUDA setup.
