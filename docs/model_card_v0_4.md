# SARYCH-LM v0.4 Model Card

## Name

`sarych-30m-instruct`

## Stage

v0.4 SFT infrastructure. This pass prepares instruction fine-tuning for the completed v0.3 TinyStories base model. It does not include a trained v0.4 checkpoint in Git.

## Base Model

- Base checkpoint: `runs/v0_3_30m_tinystories_base/checkpoints/checkpoint_latest.pt`
- Base model: `sarych-30m-tinystories-base`
- Parameters: 29,770,944
- Context length: 512
- Tokenizer: byte-level BPE, vocab size 8192

## Architecture

The v0.4 model keeps the v0.3 architecture unchanged:

- Decoder-only Transformer
- RMSNorm
- RoPE
- SwiGLU
- Tied embeddings
- 10 layers, 8 heads, embedding width 448, MLP width 1344
- Vocab size remains 8192

The tokenizer vocabulary is not expanded in this pass. The markers `<|user|>` and `<|assistant|>` are literal byte-level strings. `<|endoftext|>` is the existing end token.

## Training Data

Expected SFT data comes from manually generated Xiaomi synthetic instruction examples. Generation lives outside this repository at:

```text
C:\Users\hustlePC\PycharmProjects\sft-examples
```

SARYCH-LM imports filtered data into ignored local paths under `data/xiaomi/`.

Target raw dataset size: 30,000 examples.

Category mix:

- 70% child-friendly/simple English tasks
- 30% general simple tasks

No code tasks are included in v0.4.

The v0.4.3 experimentation pass added a local-only lite replay recipe:

- Dolly-lite conversion from already downloaded Dolly raw JSONL with aggressive filtering.
- TinyStories replay rows to preserve the base model's simple narrative behavior.
- Source-capped mixing across everyday, Dolly-lite, Xiaomi seeded, and replay data.
- Source-aware SFT filtering so replay rows are not rejected merely because they reuse story-writing or continuation instruction templates.

This is an experiment recipe, not a committed dataset or final checkpoint.

The v0.4.4 experimentation pass adds instruction-lite tooling:

- fixed 40-prompt eval suite for identity, explanations, lists, QA, stories, reasoning, and safety/kindness
- deterministic seed generation for a 1500-example external-teacher target
- validator for teacher outputs with category-specific filters
- replay + instruction-lite + small everyday mix builder
- conservative LR configs at 1e-5, 5e-6, and 3e-6

The target dataset is English-only, child-simple, and intentionally excludes code, broad professional assistant behavior, politics, adult, medical, legal, and financial tasks.

## Training Objective

Supervised fine-tuning with output-only loss masking:

- Mask `<|user|>`
- Mask instruction text
- Mask optional input text
- Mask `<|assistant|>`
- Mask padding
- Train on assistant output tokens and `<|endoftext|>` only

## Intended Use

Engineering validation of a small instruction-tuning pipeline for simple English prompts, especially child-friendly story, explanation, QA, dialogue, summarization, simple reasoning, and structured-output tasks.

## Not Intended Use

- Strong general assistant use
- Factual answering
- Production use
- Safety-critical use
- Code generation
- Political, adult, medical, legal, or financial advice
- Any claim that Xiaomi provides logits or direct model weights

Xiaomi is a teacher/judge for synthetic data only.

## Evaluation

Use `scripts/eval_instruction_lite_v0_4.py` with `evals/v0_4_instruction_lite_prompts.jsonl` for the v0.4.4 fixed category suite. It produces generated outputs, a JSON summary, heuristic metrics, and a markdown report with blank manual-score fields. No Xiaomi judge is called in this pass.

## Limitations

This remains a 30M-parameter model trained from a narrow story base. It should be expected to hallucinate, struggle with facts, fail complex instructions, and require manual review before any downstream use.

Earlier v0.4 SFT attempts showed repetition and degraded language quality at longer runs. Lower learning rates and replay data are intended to reduce destructive fine-tuning, but they do not remove the fundamental capacity and data-quality limits.
