# SARYCH-LM v0.1 Model Card

## Model

`sarych-5m-sanity` is a tiny decoder-only Transformer used to validate the SARYCH-LM training stack.

## Intended Use

Engineering sanity only: forward/loss/backward, AdamW updates, BF16 autocast on CUDA, FP32 fallback, checkpoint/resume, JSONL logging, and environment reporting.

## Training Data

Synthetic integer-token sequences generated locally. No real language data is used.

## Limitations

This is not a useful language model, chatbot, or assistant. Generated samples are synthetic token labels such as `tok_123`.

## Safety And Scope

v0.1 excludes instruction tuning, distillation, API/UI serving, MoE, distributed training, FlashAttention, custom CUDA, and external downloads.
