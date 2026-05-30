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
