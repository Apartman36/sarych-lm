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

## Seeded Generation Workflow

Do not ask the external generator to invent large batches freely. First create deterministic seed tasks, then ask OpenCode/Xiaomi manually to expand each seed into exactly one SFT row.

Generate a 1000-row seed file:

```bash
python scripts/make_sft_seed_prompts.py \
  --out /mnt/c/Users/hustlePC/PycharmProjects/sft-examples/seeds/sft_seeds_1000.jsonl \
  --count 1000 \
  --start-id 1 \
  --profile v0_4_child_simple \
  --seed 1337
```

Generate 10k seeds in 1000-row shards:

```bash
python scripts/make_sft_seed_prompts.py \
  --count 10000 \
  --start-id 1 \
  --profile v0_4_child_simple \
  --seed 1337 \
  --shard-size 1000 \
  --out-dir /mnt/c/Users/hustlePC/PycharmProjects/sft-examples/seeds/shards
```

Use `prompts/xiaomi_sft_from_seeds_v1.md` as the manual OpenCode/Xiaomi prompt template for each seed shard. The template requires strict JSONL, no web browsing, preserved `target_sft_id`, preserved `task_type`, and one final SFT row per seed row.

After manual generation, merge raw shards:

```bash
python scripts/merge_sft_shards.py \
  --input-dir /mnt/c/Users/hustlePC/PycharmProjects/sft-examples/raw/shards \
  --out /mnt/c/Users/hustlePC/PycharmProjects/sft-examples/raw/sft_10000_seeded.jsonl \
  --manifest /mnt/c/Users/hustlePC/PycharmProjects/sft-examples/manifests/sft_10000_seeded_merge_manifest.json
```

Analyze the raw merged file before building filtered splits:

```bash
python scripts/analyze_sft_jsonl.py \
  /mnt/c/Users/hustlePC/PycharmProjects/sft-examples/raw/sft_10000_seeded.jsonl \
  --out /mnt/c/Users/hustlePC/PycharmProjects/sft-examples/manifests/sft_10000_seeded_analysis.json
```

## Import Command

```bash
python scripts/build_sft_dataset.py \
  --raw /mnt/c/Users/hustlePC/PycharmProjects/sft-examples/raw/sft_10000_seeded.jsonl \
  --scored /mnt/c/Users/hustlePC/PycharmProjects/sft-examples/scored/sft_10000_scores.jsonl \
  --tokenizer data/tokenizers/sarych_bpe_8192_tinystories.json \
  --train-out data/xiaomi/processed/sft/train.jsonl \
  --val-out data/xiaomi/processed/sft/val.jsonl \
  --rejected-dir data/xiaomi/rejected \
  --manifest data/xiaomi/manifests/sft_v0_4_manifest.json \
  --val-ratio 0.02 \
  --seed 1337
```

The build script rejects malformed rows, too-short rows, non-English-looking text, repeated or low-diversity output, duplicates, examples over 512 tokens, and low-scored examples.

Run SFT diagnostics on the processed splits before training:

```bash
python scripts/diagnose_sft_v0_4.py \
  --train data/xiaomi/processed/sft/train.jsonl \
  --val data/xiaomi/processed/sft/val.jsonl \
  --tokenizer data/tokenizers/sarych_bpe_8192_tinystories.json \
  --base-checkpoint runs/v0_3_30m_tinystories_base/checkpoints/checkpoint_latest.pt \
  --max-seq-len 512 \
  --out /mnt/c/Users/hustlePC/PycharmProjects/sft-examples/manifests/sft_diagnostics.json
```
