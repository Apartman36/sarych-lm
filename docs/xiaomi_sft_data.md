# Xiaomi SFT Data Guide

## Separation Boundary

Generation stays outside SARYCH-LM. The external generation folder is:

```text
C:\Users\hustlePC\PycharmProjects\sft-examples
```

Expected layout:

```text
sft-examples/
  seeds/
  prompts/
  raw/
  scored/
  rejected/
  logs/
  manifests/
```

Generated data must not be committed to SARYCH-LM. The repo only keeps schema, filtering, training, evaluation code, docs, tests, and `.gitkeep` placeholders.

## Raw JSONL Schema

Each raw SFT line is one JSON object:

```json
{
  "id": "xm_sft_000001",
  "source": "xiaomi_mimo_v2_5_pro",
  "task_type": "story_writing",
  "instruction": "Write a short story for young children about a rabbit who learns to share.",
  "input": "",
  "output": "Once there was a little rabbit named Pip...",
  "language": "en",
  "metadata": {
    "created_at": "2026-05-30T12:00:00Z",
    "generator": "opencode|direct_xiaomi|manual",
    "model": "mimo-v2.5-pro",
    "temperature": 0.7,
    "max_tokens": 512,
    "prompt_template": "sft_v1"
  }
}
```

Allowed `task_type` values:

- `story_writing`
- `story_continuation`
- `explanation_for_children`
- `simple_qa`
- `dialogue`
- `summarization`
- `simple_reasoning`
- `structured_output`
- `creative_generation`

Do not include code tasks in v0.4.

## Scored JSONL Schema

```json
{
  "id": "xm_score_000001",
  "example_id": "xm_sft_000001",
  "scores": {
    "instruction_following": 4,
    "coherence": 5,
    "safety": 5,
    "age_appropriateness": 5,
    "english_quality": 4
  },
  "judge": {
    "source": "xiaomi_mimo_v2_5_pro",
    "model": "mimo-v2.5-pro",
    "rubric": "sarych_sft_judge_v1"
  },
  "notes": "Clear, simple, coherent."
}
```

Safety score must be 5 when present. Other present scores should be at least 4, or the average score must be at least 4.0.

## Data Policy

- English only.
- Target raw size: 30,000 examples.
- 70% child-friendly/simple English tasks.
- 30% general simple tasks.
- Generation should not use web browsing.
- Generated raw, scored, rejected, processed, manifest, and log files should not be committed.

## Import Command

```bash
python scripts/build_sft_dataset.py \
  --raw /mnt/c/Users/hustlePC/PycharmProjects/sft-examples/raw/sft_30000.jsonl \
  --scored /mnt/c/Users/hustlePC/PycharmProjects/sft-examples/scored/sft_30000_scores.jsonl \
  --tokenizer data/tokenizers/sarych_bpe_8192_tinystories.json \
  --train-out data/xiaomi/processed/sft/train.jsonl \
  --val-out data/xiaomi/processed/sft/val.jsonl \
  --rejected-dir data/xiaomi/rejected \
  --manifest data/xiaomi/manifests/sft_v0_4_manifest.json \
  --val-ratio 0.02 \
  --seed 1337
```

The build script rejects malformed rows, too-short rows, non-English-looking text, repeated or low-diversity output, duplicates, examples over 512 tokens, and low-scored examples.
