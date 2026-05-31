# Instruction-Lite v0.4.4

## Purpose

v0.4.3 fixed forgetting symptoms by making TinyStories replay usable again. Replay-heavy SFT restored story fluency, but it did not teach stable assistant behavior: identity prompts still collapsed into stories, and explanation/list prompts stayed weak. The bottleneck is now data distribution and curriculum, not model size or training length.

v0.4.4 adds infrastructure for an English-only, child-simple instruction-lite dataset and a fixed eval suite. It does not call Xiaomi/OpenCode, generate teacher outputs, train a model, or commit generated data.

## Scope Choices

Dolly, OASST, Alpaca, and UltraChat remain out for v0.4 because they add broad assistant behavior, adult/professional domains, code patterns, and register drift. DPO, RL, LoRA, a 100M model, and long training on the current replay-heavy data are also deferred.

The target behavior is narrow:

- identity/chat
- simple explanations
- simple lists
- simple QA
- simple reasoning
- gentle safety/refusal
- emotional support/kindness
- explicit story writing and continuation

## Fixed Eval Suite

The fixed prompt set is:

```text
evals/v0_4_instruction_lite_prompts.jsonl
```

It has 40 prompts:

- identity_chat: 8
- simple_explanation: 8
- simple_list: 6
- simple_qa: 5
- story_request: 6
- simple_reasoning: 4
- safety_kindness: 3

Run eval on a selected checkpoint:

```bash
python scripts/eval_instruction_lite_v0_4.py \
  --checkpoint runs/v0_4/checkpoints/checkpoint_latest.pt \
  --tokenizer data/tokenizers/sarych_bpe_8192_tinystories.json \
  --prompts evals/v0_4_instruction_lite_prompts.jsonl \
  --out-dir artifacts/evals/v0_4/run_name \
  --temperature 0.5 \
  --top-k 20 \
  --max-new-tokens 160
```

The eval writes `outputs.jsonl`, `summary.json`, and `report.md`. The markdown report includes heuristic flags and a blank manual score column.

Interpretation:

- `story_collapse_rate_non_story`: non-story prompts that drift into TinyStories openers.
- `list_format_fail_rate`: list prompts that answer in prose instead of clear items.
- `identity_fail_rate`: identity prompts that do not produce a first-person helper/self-description.
- `loop_rate`: repeated sentences or repeated n-grams.
- story fluency preservation: story_request outputs should remain coherent, simple, and non-looping while non-story categories stop becoming stories.

## Seed Generation

Create seed prompts for an external teacher:

```bash
python scripts/make_instruction_lite_seeds_v0_4.py \
  --out data/xiaomi/seeds/instruction_lite_v0_4_seeds.jsonl \
  --manifest data/xiaomi/manifests/instruction_lite_v0_4_seeds_manifest.json \
  --seed 1337
```

Default seed counts total 1500:

- identity_chat: 170
- simple_explanation: 315
- simple_list: 255
- simple_qa: 145
- simple_reasoning: 85
- story_request: 170
- story_continuation: 145
- safety_refusal: 70
- summarization_rewrite: 45
- emotional_support_kindness: 100

Generated seed JSONL is ignored by Git. Hand this file to the teacher workflow outside this repository.

## Teacher Output Schema

Expected teacher rows:

```json
{
  "id": "instr_lite_000001",
  "seed_id": "instr_lite_seed_000001",
  "source": "xiaomi_instruction_lite_v0_4",
  "task_type": "explanation_for_children",
  "instruction": "Explain why plants need sunlight in simple words.",
  "input": "",
  "output": "Plants need sunlight because it helps them make food...",
  "language": "en",
  "metadata": {
    "category": "simple_explanation",
    "teacher_model": "teacher-name",
    "seed_topic": "plants need sunlight",
    "notes": ""
  }
}
```

Category to task mapping:

- identity_chat -> dialogue
- simple_explanation -> explanation_for_children
- simple_list -> structured_output
- simple_qa -> simple_qa
- simple_reasoning -> simple_reasoning
- story_request -> story_writing
- story_continuation -> story_continuation
- safety_refusal -> dialogue
- summarization_rewrite -> summarization
- emotional_support_kindness -> dialogue

## Instruction-Lite Data Factory (v0.4.5)

For shard-based teacher generation, validation, repairs, and merge, see:

```text
docs/instruction_lite_factory_v0_4.md
```

Quick entry:

```bash
python scripts/run_instruction_lite_factory.py prepare-shards --help
```

## Validation

Validate teacher output (single file or post-factory merge):

```bash
python scripts/validate_instruction_lite_sft.py \
  --input /mnt/c/Users/hustlePC/PycharmProjects/sft-examples/raw/instruction_lite_v0_4.jsonl \
  --accepted /mnt/c/Users/hustlePC/PycharmProjects/sft-examples/processed/instruction_lite_v0_4_accepted.jsonl \
  --rejected /mnt/c/Users/hustlePC/PycharmProjects/sft-examples/rejected/instruction_lite_v0_4_rejected.jsonl \
  --manifest /mnt/c/Users/hustlePC/PycharmProjects/sft-examples/manifests/instruction_lite_v0_4_manifest.json \
  --strictness standard
```

The validator checks schema, language, allowed task type, empty fields, "As an AI", chatty openers, blocked adult/political/medical/legal/financial/code terms, URLs, markdown tables, loops, repeated sentences, weird nonwords, category length windows, exact duplicates, and near-duplicate output overlap (calibrated by category and `--strictness`).

Category checks enforce identity helper terms, direct causal explanations, list markers or clear separators, required input for continuation/rewrite, safety redirection, and warm direct support.

## Mix Recipe

Create the experiment mix:

```bash
python scripts/make_v0_4_instruction_lite_mix.py \
  --replay data/xiaomi/processed/replay/tinystories_replay_sft_v0_4.jsonl \
  --instruction-lite /mnt/c/Users/hustlePC/PycharmProjects/sft-examples/processed/instruction_lite_v0_4_accepted.jsonl \
  --everyday /mnt/c/Users/hustlePC/PycharmProjects/sft-examples/public_converted/everyday_single_sft_v0_4.jsonl \
  --out /mnt/c/Users/hustlePC/PycharmProjects/sft-examples/public_converted/mixed_instruction_lite_v0_4.jsonl \
  --manifest /mnt/c/Users/hustlePC/PycharmProjects/sft-examples/manifests/mixed_instruction_lite_v0_4_manifest.json \
  --replay-cap 900 \
  --instruction-lite-cap 1500 \
  --everyday-cap 400 \
  --seed 1337
```

Old Xiaomi seeded data and Dolly-lite default to zero. The manifest reports selected rows by source, task type, and category.

## Build, Grid, Eval

Build final SFT train/val splits with replay flags:

```bash
python scripts/build_sft_dataset.py \
  --raw /mnt/c/Users/hustlePC/PycharmProjects/sft-examples/public_converted/mixed_instruction_lite_v0_4.jsonl \
  --tokenizer data/tokenizers/sarych_bpe_8192_tinystories.json \
  --train-out data/xiaomi/processed/sft/train.jsonl \
  --val-out data/xiaomi/processed/sft/val.jsonl \
  --rejected-dir data/xiaomi/rejected \
  --manifest data/xiaomi/manifests/sft_v0_4_instruction_lite_manifest.json \
  --val-ratio 0.05 \
  --seed 1337 \
  --replay-source-prefix tinystories_replay \
  --keep-replay-duplicates \
  --replay-dedup-mode output_hash \
  --disable-replay-low-diversity-filter
```

Run grid:

```bash
python scripts/run_sft_experiment_grid.py \
  --config configs/v0_4_30m_instruct_instruction_lite_lr1e5.yaml \
  --config configs/v0_4_30m_instruct_instruction_lite_lr5e6.yaml \
  --config configs/v0_4_30m_instruct_instruction_lite_lr3e6.yaml \
  --steps 500 1000
```

Then run the fixed eval on selected checkpoints. The grid can also use the fixed prompt file for qualitative samples:

```bash
python scripts/run_sft_experiment_grid.py \
  --prompts-file evals/v0_4_instruction_lite_prompts.jsonl \
  --dry-run
```

No generated seeds, accepted/rejected teacher data, mixed SFT files, checkpoints, runs, or eval artifacts should be committed.
