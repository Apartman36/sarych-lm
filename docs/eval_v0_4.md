# v0.4 Evaluation Guide

## Prompt Set

Fixed manual evaluation prompts live in:

```text
eval/prompts_v0_4.jsonl
```

The set contains 50 prompts across:

- `story_writing`
- `story_continuation`
- `explanation_for_children`
- `simple_qa`
- `dialogue`
- `summarization`
- `simple_reasoning`
- `structured_output`

## Run Evaluation

Before comparing full eval outputs, inspect the processed SFT data and prompt logits:

```bash
python scripts/diagnose_sft_v0_4.py \
  --train data/xiaomi/processed/sft/train.jsonl \
  --val data/xiaomi/processed/sft/val.jsonl \
  --tokenizer data/tokenizers/sarych_bpe_8192_tinystories.json \
  --base-checkpoint runs/v0_3_30m_tinystories_base/checkpoints/checkpoint_latest.pt \
  --sft-checkpoint runs/v0_4_30m_instruct_xiaomi/checkpoints/checkpoint_latest.pt \
  --top-k 10
```

Generation diagnostics can show whether EOS is the first predicted token after `<|assistant|>`:

```bash
python scripts/generate_instruct_v0_4.py \
  --checkpoint runs/v0_4_30m_instruct_xiaomi/checkpoints/checkpoint_latest.pt \
  --tokenizer data/tokenizers/sarych_bpe_8192_tinystories.json \
  --instruction "Write a short story for young children about a turtle who asks for help." \
  --max-new-tokens 120 \
  --temperature 0.8 \
  --top-k 50 \
  --debug-top-k 10 \
  --no-print-prompt
```

`--min-new-tokens-before-eos` and `--suppress-eos-for-first-n-tokens` are diagnostic options only. They can reveal whether the model has learned useful continuation tokens behind an early EOS preference, but they should not be used to hide a failing checkpoint.

```bash
python scripts/eval_sarych.py \
  --prompts eval/prompts_v0_4.jsonl \
  --tokenizer data/tokenizers/sarych_bpe_8192_tinystories.json \
  --base-checkpoint runs/v0_3_30m_tinystories_base/checkpoints/checkpoint_latest.pt \
  --instruct-checkpoint runs/v0_4_30m_instruct_xiaomi/checkpoints/checkpoint_latest.pt \
  --base-out eval/results/base_v0_3_outputs.jsonl \
  --instruct-out eval/results/instruct_v0_4_outputs.jsonl
```

Outputs are ignored by Git:

```text
eval/results/base_v0_3_outputs.jsonl
eval/results/instruct_v0_4_outputs.jsonl
```

## Manual Comparison

Compare base and instruct outputs for:

- Following the requested task type
- Staying in simple English
- Avoiding code-like content
- Completing the answer instead of only continuing the prompt
- Basic coherence
- Child-friendly tone where requested

No Xiaomi judge is called in this pass.
