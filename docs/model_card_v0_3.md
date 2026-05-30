# SARYCH-LM v0.3 Model Card

## Name

`sarych-30m-tinystories-base`

## Stage

v0.3 / completed base model milestone. This is a 30M-class TinyStories base model, not a polished v1 release and not an instruction-tuned assistant.

## Architecture

- Decoder-only Transformer
- RMSNorm
- RoPE position encoding
- SwiGLU MLP
- Causal self-attention
- Tied token embeddings
- Vocab size: 8192
- Context length: 512
- Layers: 10
- Attention heads: 8
- Embedding width: 448
- SwiGLU width: 1344
- Dropout: 0.0
- Bias: false

The default config reports 29,770,944 trainable parameters.

## Tokenizer

Byte-level BPE tokenizer trained with the local `sarych.tokenizer_bpe` wrapper and the `tokenizers` library.

- Target vocabulary size: 8192
- Special tokens: `<|endoftext|>`, `<|pad|>`
- Default path: `data/tokenizers/sarych_bpe_8192_tinystories.json`

Generated tokenizer files are local artifacts and are ignored by Git by default.

## Data

Training data:

```text
data/raw/TinyStories-train.txt
```

Validation data:

```text
data/raw/TinyStories-valid.txt
```

Prepared memmaps:

```text
data/processed/v0_3_tinystories_8192/train.bin
data/processed/v0_3_tinystories_8192/val.bin
```

The v0.3 preparation path uses the train file only for training and the valid file only for validation when `--val-input` is provided.

## Intended Use

Story continuation and base language model experimentation on short English children's-story-style text.

## Not Intended Use

- Chatbot behavior
- Instruction following
- Factual assistant use
- Production use
- Safety-critical or user-facing deployment

## Training Hardware

Target hardware is a single NVIDIA GeForce RTX 5060 Ti 16GB under WSL2 with PyTorch 2.9.1+cu128, CUDA 12.8, compute capability 12.0, and BF16 support.

## Evaluation

Primary engineering evaluation:

- Validation loss from `runs/v0_3_30m_tinystories_base/train_log.jsonl`
- Generated story-continuation samples from checkpoints
- Checkpoint resume behavior
- Throughput and CUDA memory metrics

Completed run summary:

- Training steps: 10,000
- Tokens processed: 327,680,000
- Best validation loss: 1.493360996246338
- Final train loss: about 5.924356
- Peak CUDA memory: about 3.53 GB
- Generated text quality: coherent simple English story text

No benchmark claims are made at this stage.

## Limitations

This is a small base LM trained on a narrow story corpus. It is expected to hallucinate, continue prompts unreliably during short runs, and lack instruction-following behavior. It has no retrieval, no safety tuning, no chat template, and no factual grounding.
