# SARYCH-LM v0.2 Model Card

## Model

`sarych-5m-tinystories-smoke` is a small decoder-only Transformer smoke model with a byte-level BPE text pipeline.

## Intended Use

Engineering validation of real text ingestion: tokenizer training, tokenized memmap datasets, checkpointable text training, and tokenizer-decoded generation.

## Training Data

The default workflow uses `data/samples/tiny_stories_sample.txt`, a tiny original English story sample. Optional TinyStories raw text can be downloaded manually into `data/raw/`, but it is not downloaded automatically and is not committed.

## Architecture

Default config: vocab size 4096, context length 256, 6 layers, 8 heads, embedding width 256, SwiGLU width 512, RMSNorm, RoPE, tied embeddings.

## Limitations

v0.2 is not a useful language model. The committed sample is far too small for language quality, and short smoke runs only prove that the pipeline executes. This is not the future v1 30M base model.

## Exclusions

No instruction tuning, Qwen distillation, API/UI, MoE, DeepSpeed/FSDP/DDP, FlashAttention, custom CUDA, or default `torch.compile`.
