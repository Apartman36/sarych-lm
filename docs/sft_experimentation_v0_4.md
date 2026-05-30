# v0.4 SFT Experimentation

## Why v0.4.3 Exists

The first balanced v0.4 SFT mix was too aggressive for a 30M TinyStories base. Runs around 25-50 steps started to answer instructions but repeated; longer runs showed more repetition and odd words. The likely causes were a general/adult Dolly-heavy mix, too little TinyStories-style replay, and an SFT learning rate that moved the small base too far from its story distribution.

v0.4.3 keeps the model architecture and base checkpoint unchanged. It fixes the replay data path: TinyStories replay rows reuse a few instruction templates by design, so the SFT builder must not reject them as duplicate instruction examples. The source-aware replay policy deduplicates replay by output text when requested, keeps stricter duplicate filtering for non-replay SFT rows, and reports accept/reject counts by source.

## Data Strategy

Dolly-lite is stricter than the earlier Dolly-filtered pass. It keeps only simple, child-compatible Dolly categories and rejects rows with context, code/programming terms, adult/political/medical/legal/financial terms, long outputs, long instructions, URLs, table-like markdown, formal phrasing, or too many rare-looking long words. The output schema is the normal SARYCH SFT raw schema.

TinyStories replay is mixed into SFT to protect the base model's narrative ability. Replay rows ask the model to write or continue short simple stories using local TinyStories text only. Repeated replay instructions are acceptable because the target behavior is preserving the output distribution, not teaching many distinct instruction phrasings. These generated replay JSONL files are ignored by Git.

The build manifest now includes `accepted_by_source`, `rejected_by_source`, `accepted_by_task_type`, `rejected_by_task_type`, and `rejected_reason_by_source`. For replay debugging, check that `accepted_by_source.tinystories_replay` is nonzero and close to the replay cap after malformed, too-long, and output-duplicate rows are removed.

## Commands

Convert Dolly-lite from the already downloaded local raw file:

```bash
python scripts/convert_dolly_lite_to_sft.py \
  --input /mnt/c/Users/hustlePC/PycharmProjects/sft-examples/public_raw/dolly_15k_train.jsonl \
  --out /mnt/c/Users/hustlePC/PycharmProjects/sft-examples/public_converted/dolly_lite_sft_v0_4.jsonl \
  --manifest /mnt/c/Users/hustlePC/PycharmProjects/sft-examples/manifests/dolly_lite_sft_v0_4_manifest.json
```

Create TinyStories replay:

```bash
python scripts/make_tinystories_replay_sft.py \
  --input data/raw/TinyStories-valid.txt \
  --out data/xiaomi/processed/replay/tinystories_replay_sft_v0_4.jsonl \
  --count 1000 \
  --tokenizer data/tokenizers/sarych_bpe_8192_tinystories.json \
  --mode mixed \
  --min-words 40 \
  --max-words 140 \
  --seed 1337
```

Mix local sources:

```bash
python scripts/mix_sft_sources.py \
  --source everyday=/mnt/c/Users/hustlePC/PycharmProjects/sft-examples/public_converted/everyday_single_sft_v0_4.jsonl:cap=1500 \
  --source dolly_lite=/mnt/c/Users/hustlePC/PycharmProjects/sft-examples/public_converted/dolly_lite_sft_v0_4.jsonl:cap=2500 \
  --source xiaomi=/mnt/c/Users/hustlePC/PycharmProjects/sft-examples/raw/sft_1000_seeded.jsonl:cap=800 \
  --source replay=data/xiaomi/processed/replay/tinystories_replay_sft_v0_4.jsonl:cap=1200 \
  --out /mnt/c/Users/hustlePC/PycharmProjects/sft-examples/public_converted/mixed_lite_replay_v0_4.jsonl \
  --manifest /mnt/c/Users/hustlePC/PycharmProjects/sft-examples/manifests/mixed_lite_replay_v0_4_manifest.json \
  --seed 1337
```

Build filtered train/val splits:

```bash
python scripts/build_sft_dataset.py \
  --raw /mnt/c/Users/hustlePC/PycharmProjects/sft-examples/public_converted/mixed_lite_replay_v0_4.jsonl \
  --tokenizer data/tokenizers/sarych_bpe_8192_tinystories.json \
  --train-out data/xiaomi/processed/sft/train.jsonl \
  --val-out data/xiaomi/processed/sft/val.jsonl \
  --rejected-dir data/xiaomi/rejected \
  --manifest data/xiaomi/manifests/sft_v0_4_lite_replay_manifest.json \
  --val-ratio 0.05 \
  --seed 1337 \
  --replay-source-prefix tinystories_replay \
  --keep-replay-duplicates \
  --replay-dedup-mode output_hash \
  --disable-replay-low-diversity-filter
```

Run the short grid:

```bash
python scripts/run_sft_experiment_grid.py \
  --config configs/v0_4_30m_instruct_lite_replay_lr1e5.yaml \
  --config configs/v0_4_30m_instruct_lite_replay_lr5e6.yaml \
  --steps 100 200 300
```

Use `--run-dir` when launching a single short experiment to keep probes isolated:

```bash
python scripts/train_sft_v0_4.py \
  --config configs/v0_4_30m_instruct_lite_replay_lr5e6.yaml \
  --max-steps 100 \
  --no-resume \
  --run-dir runs/v0_4_30m_instruct_lite_replay_lr5e6_smoke_100
```

Layer freezing was intentionally not added in v0.4.3. The model ties token embeddings and `lm_head`, so freezing embeddings while leaving the head trainable would require a more careful untie-or-mask design than this replay-filtering change should carry.

Dry-run command construction:

```bash
python scripts/run_sft_experiment_grid.py --dry-run
```

## Git Hygiene

Do not commit generated data, replay JSONL/manifests, checkpoints, runs, logs, or grid artifacts. The committed surface should stay limited to scripts, configs, docs, tests, and ignore rules.
