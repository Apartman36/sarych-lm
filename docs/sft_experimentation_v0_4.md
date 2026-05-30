# v0.4 SFT Experimentation

## Why v0.4.2 Exists

The first balanced v0.4 SFT mix was too aggressive for a 30M TinyStories base. Runs around 25-50 steps started to answer instructions but repeated; longer runs showed more repetition and odd words. The likely causes were a general/adult Dolly-heavy mix, too little TinyStories-style replay, and an SFT learning rate that moved the small base too far from its story distribution.

v0.4.2 keeps the model architecture and base checkpoint unchanged. It adds stricter local data tools, TinyStories replay rows, lower learning-rate configs, and a short grid runner so experiments can compare damage and instruction following without committing generated data or checkpoints.

## Data Strategy

Dolly-lite is stricter than the earlier Dolly-filtered pass. It keeps only simple, child-compatible Dolly categories and rejects rows with context, code/programming terms, adult/political/medical/legal/financial terms, long outputs, long instructions, URLs, table-like markdown, formal phrasing, or too many rare-looking long words. The output schema is the normal SARYCH SFT raw schema.

TinyStories replay is mixed into SFT to protect the base model's narrative ability. Replay rows ask the model to write or continue short simple stories using local TinyStories text only. These generated replay JSONL files are ignored by Git.

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
  --seed 1337
```

Run the short grid:

```bash
python scripts/run_sft_experiment_grid.py \
  --config configs/v0_4_30m_instruct_lite_lr2e5.yaml \
  --config configs/v0_4_30m_instruct_lite_lr1e5.yaml \
  --steps 100 200 300
```

Dry-run command construction:

```bash
python scripts/run_sft_experiment_grid.py --dry-run
```

## Git Hygiene

Do not commit generated data, replay JSONL/manifests, checkpoints, runs, logs, or grid artifacts. The committed surface should stay limited to scripts, configs, docs, tests, and ignore rules.
