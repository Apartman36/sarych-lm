# Instruction-Lite Data Factory (v0.4.5)

## Why one-shot generation failed

The first OpenCode pass generated **1500 rows** in a single prompt. Only **284** passed validation (**1216** rejected). Main failure modes:

| Reason | Count |
|--------|------:|
| too_short | 409 |
| near_duplicate_output | 374 |
| duplicate_instruction_output | 220 |
| weak_explanation | 100 |
| weak_reasoning | 46 |
| identity_fail | 47 |

A monolithic prompt encourages repeated openings, similar story shapes, and length drift. Manual review also showed **false rejects** (valid short lists, safe medicine refusals, causal explanations without the word “because”). v0.4.5 fixes validation calibration and replaces one-shot generation with a **shard-based factory**.

## Why shard-based generation is better

- **50–100 seeds per prompt** — easier for the teacher to follow rules.
- **Category-aware instructions** — only rules for categories present in the shard.
- **Explicit Windows output paths** — place files where the factory expects them.
- **Per-shard validation reports** — see which shards/categories need repair.
- **Repair packs** — rejected rows get targeted rewrite prompts with the rejection reason.

## Factory layout

```text
sft-examples/factory/instruction_lite_v0_4/
  manifests/
    shards_manifest.json
    validation_summary.json
  reports/
    shard_preparation.md
    validation_summary.md
    merge_accepted.md
  shards/
    seeds/shard_0001.jsonl
    prompts/shard_0001_prompt.md
    raw/shard_0001.jsonl          # you place teacher output here
    accepted/shard_0001_accepted.jsonl
    rejected/shard_0001_rejected.jsonl
  repairs/
    round_1/
      seeds/repair_shard_0001.jsonl
      prompts/repair_shard_0001_prompt.md
      raw/repair_shard_0001.jsonl
      accepted/...
```

## Commands

All subcommands live in `scripts/run_instruction_lite_factory.py`.

### 1. Prepare shards

```bash
python scripts/run_instruction_lite_factory.py prepare-shards \
  --seeds C:\Users\hustlePC\PycharmProjects\sft-examples\seeds\instruction_lite_v0_4_seeds.jsonl \
  --out-dir C:\Users\hustlePC\PycharmProjects\sft-examples\factory\instruction_lite_v0_4 \
  --shard-size 100 \
  --seed 1337
```

Pilot (3 × 50):

```bash
python scripts/run_instruction_lite_factory.py prepare-shards \
  --seeds C:\Users\hustlePC\PycharmProjects\sft-examples\seeds\instruction_lite_v0_4_seeds.jsonl \
  --out-dir C:\Users\hustlePC\PycharmProjects\sft-examples\factory\instruction_lite_v0_4_pilot \
  --shard-size 50 \
  --seed 1337
```

Take only the first three prompt files if you want a minimal pilot.

### 2. Hand prompts to OpenCode / Xiaomi

For each `shards/prompts/shard_XXXX_prompt.md`:

1. Open the prompt in OpenCode (Windows).
2. Let the teacher write **only** to the path shown in the prompt, e.g.  
   `C:\Users\hustlePC\PycharmProjects\sft-examples\factory\instruction_lite_v0_4\shards\raw\shard_0001.jsonl`
3. One JSON object per line, no markdown fences.

### 3. Validate shards

```bash
python scripts/run_instruction_lite_factory.py validate-shards \
  --factory-dir C:\Users\hustlePC\PycharmProjects\sft-examples\factory\instruction_lite_v0_4 \
  --strictness standard
```

Read `reports/validation_summary.md` for acceptance by shard/category and suggested next actions.

Strictness: `strict` | `standard` (default) | `lenient` (exploration only).

### 4. Repair pack

```bash
python scripts/run_instruction_lite_factory.py make-repair-pack \
  --factory-dir C:\Users\hustlePC\PycharmProjects\sft-examples\factory\instruction_lite_v0_4 \
  --max-per-shard 50 \
  --round 1
```

Hand `repairs/round_1/prompts/*_prompt.md` to the teacher; outputs go to `repairs/round_1/raw/`.

### 5. Validate repairs

```bash
python scripts/run_instruction_lite_factory.py validate-repairs \
  --factory-dir C:\Users\hustlePC\PycharmProjects\sft-examples\factory\instruction_lite_v0_4 \
  --round 1 \
  --strictness standard
```

### 6. Merge accepted

```bash
python scripts/run_instruction_lite_factory.py merge-accepted \
  --factory-dir C:\Users\hustlePC\PycharmProjects\sft-examples\factory\instruction_lite_v0_4 \
  --out C:\Users\hustlePC\PycharmProjects\sft-examples\processed\instruction_lite_v0_4_factory_accepted.jsonl \
  --manifest C:\Users\hustlePC\PycharmProjects\sft-examples\manifests\instruction_lite_v0_4_factory_manifest.json
```

### 7. Audit sample

```bash
python scripts/run_instruction_lite_factory.py audit-sample \
  --input C:\Users\hustlePC\PycharmProjects\sft-examples\processed\instruction_lite_v0_4_factory_accepted.jsonl \
  --out C:\Users\hustlePC\PycharmProjects\sft-examples\reports\instruction_lite_v0_4_audit_sample.md \
  --per-category 5 \
  --seed 1337
```

## Acceptance targets

| Stage | Target |
|-------|--------|
| Pilot (3 × 50) | ≥ 50% acceptance per shard |
| Full (1500 seeds) | 800–1200 accepted after one repair round |
| Train gate | **Do not train** if merged accepted &lt; 500 |

After acceptance looks good: `make_v0_4_instruction_lite_mix.py` → `build_sft_dataset.py` → grid configs (see `docs/instruction_lite_v0_4.md`).

## Validator calibration (v0.4.5)

Category-specific improvements in `scripts/validate_instruction_lite_sft.py`:

- **simple_list** — short outputs OK if ≥ 2 list markers/items
- **simple_qa** — very short direct answers OK
- **summarization_rewrite** — min 5 words (standard)
- **simple_explanation** — causal via so/helps/turns/makes, not only “because”
- **simple_reasoning** — practical reasoning (“keeps you warm”)
- **safety_refusal** — “medicine” allowed in safe grown-up redirect
- **identity_chat** — broader I/me/helper/friend patterns
- **near_duplicate_output** — softer for story categories; lenient mode warns instead of rejecting

Do not commit generated JSONL, factory outputs, checkpoints, or runs.
