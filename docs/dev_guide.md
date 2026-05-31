# SARYCH-LM Engineering Guide

This guide is for local development and training. It keeps operational commands separate from the public README.

## WSL Setup

From WSL:

```bash
cd ~/projects/sarych-lm
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

# RTX 50-series / Blackwell target.
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

Verify CUDA:

```bash
python - <<'PY'
import torch
print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
print(torch.cuda.get_device_capability(0) if torch.cuda.is_available() else None)
print(torch.cuda.is_bf16_supported() if torch.cuda.is_available() else None)
PY
```

Run tests:

```bash
pytest -q
```

## Dataset Files

TinyStories raw text is expected at:

```text
data/raw/TinyStories-train.txt
data/raw/TinyStories-valid.txt
```

The repository does not download TinyStories automatically during training. If needed, use the explicit download script:

```bash
python scripts/download_tinystories.py --valid
python scripts/download_tinystories.py --train
```

Raw data under `data/raw/` is ignored by Git.

## v0.1 Synthetic Sanity

```bash
python scripts/train_v0_1.py --config configs/v0_1_synthetic_sanity.yaml --max-steps 20 --no-resume
python scripts/train_v0_1.py --config configs/v0_1_synthetic_sanity.yaml --resume
python scripts/generate_v0_1.py --checkpoint runs/v0_1_synthetic_sanity/checkpoints/checkpoint_latest.pt --max-new-tokens 80
```

## v0.2 Text Smoke

Train the sample tokenizer:

```bash
python scripts/train_tokenizer_v0_2.py \
  --input data/samples/tiny_stories_sample.txt \
  --output data/tokenizers/sarych_bpe_4096_sample.json \
  --vocab-size 4096
```

Prepare the sample memmap dataset:

```bash
python scripts/prepare_text_dataset_v0_2.py \
  --input data/samples/tiny_stories_sample.txt \
  --tokenizer data/tokenizers/sarych_bpe_4096_sample.json \
  --output-dir data/processed/v0_2_sample \
  --block-size 256 \
  --val-fraction 0.1
```

Train and generate:

```bash
python scripts/train_v0_2.py \
  --config configs/v0_2_tinystories_smoke.yaml \
  --max-steps 50 \
  --no-resume

python scripts/generate_v0_2.py \
  --checkpoint runs/v0_2_tinystories_smoke/checkpoints/checkpoint_latest.pt \
  --tokenizer data/tokenizers/sarych_bpe_4096_sample.json \
  --prompt "Once upon a time" \
  --max-new-tokens 80
```

## v0.3 30M TinyStories Base

Train the 8192-token byte-level BPE tokenizer on TinyStories train:

```bash
python scripts/train_tokenizer_v0_2.py \
  --input data/raw/TinyStories-train.txt \
  --output data/tokenizers/sarych_bpe_8192_tinystories.json \
  --vocab-size 8192
```

The script prints the realized vocabulary size and the special token IDs. Keep generated tokenizer JSON files out of Git unless you intentionally decide to version one.

Prepare train and validation memmaps from separate TinyStories files:

```bash
python scripts/prepare_text_dataset_v0_2.py \
  --input data/raw/TinyStories-train.txt \
  --val-input data/raw/TinyStories-valid.txt \
  --tokenizer data/tokenizers/sarych_bpe_8192_tinystories.json \
  --output-dir data/processed/v0_3_tinystories_8192 \
  --block-size 512 \
  --val-fraction 0.1
```

When `--val-input` is provided, `--val-fraction` is recorded in metadata but ignored for splitting. Train tokens come only from `--input`; validation tokens come only from `--val-input`.

Smoke train for 500 steps:

```bash
python scripts/train_v0_3.py \
  --config configs/v0_3_30m_tinystories_base.yaml \
  --max-steps 500 \
  --no-resume
```

Resume to a medium run:

```bash
python scripts/train_v0_3.py \
  --config configs/v0_3_30m_tinystories_base.yaml \
  --max-steps 5000 \
  --resume
```

Resume to a long run:

```bash
python scripts/train_v0_3.py \
  --config configs/v0_3_30m_tinystories_base.yaml \
  --max-steps 10000 \
  --resume
```

Generate from the latest checkpoint:

```bash
python scripts/generate_v0_3.py \
  --checkpoint runs/v0_3_30m_tinystories_base/checkpoints/checkpoint_latest.pt \
  --tokenizer data/tokenizers/sarych_bpe_8192_tinystories.json \
  --prompt "Once upon a time" \
  --max-new-tokens 160 \
  --temperature 0.8 \
  --top-k 50
```

Read recent logs:

```bash
tail -n 20 runs/v0_3_30m_tinystories_base/train_log.jsonl
```

Each JSONL record includes step, train loss, validation loss when evaluated, learning rate, tokens processed, throughput, device, dtype, parameter count, and CUDA memory allocation/reservation/peak fields.

## v0.4 Xiaomi SFT Infrastructure

SARYCH-LM does not generate Xiaomi data itself. Put manually generated examples outside this repository:

```text
C:\Users\hustlePC\PycharmProjects\sft-examples
```

Expected external layout:

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

Create deterministic seed shards before any external generation:

```bash
python scripts/make_sft_seed_prompts.py \
  --count 10000 \
  --start-id 1 \
  --profile v0_4_child_simple \
  --seed 1337 \
  --shard-size 1000 \
  --out-dir /mnt/c/Users/hustlePC/PycharmProjects/sft-examples/seeds/shards
```

Use `prompts/xiaomi_sft_from_seeds_v1.md` manually with OpenCode/Xiaomi for each shard. SARYCH-LM must not call Xiaomi/OpenCode directly.

Merge and analyze generated raw shards:

```bash
python scripts/merge_sft_shards.py \
  --input-dir /mnt/c/Users/hustlePC/PycharmProjects/sft-examples/raw/shards \
  --out /mnt/c/Users/hustlePC/PycharmProjects/sft-examples/raw/sft_10000_seeded.jsonl \
  --manifest /mnt/c/Users/hustlePC/PycharmProjects/sft-examples/manifests/sft_10000_seeded_merge_manifest.json

python scripts/analyze_sft_jsonl.py \
  /mnt/c/Users/hustlePC/PycharmProjects/sft-examples/raw/sft_10000_seeded.jsonl \
  --out /mnt/c/Users/hustlePC/PycharmProjects/sft-examples/manifests/sft_10000_seeded_analysis.json
```

Build filtered train/val splits:

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

Diagnose processed SFT data before training:

```bash
python scripts/diagnose_sft_v0_4.py \
  --train data/xiaomi/processed/sft/train.jsonl \
  --val data/xiaomi/processed/sft/val.jsonl \
  --tokenizer data/tokenizers/sarych_bpe_8192_tinystories.json \
  --base-checkpoint runs/v0_3_30m_tinystories_base/checkpoints/checkpoint_latest.pt \
  --max-seq-len 512
```

Run a tiny smoke only to verify the pipeline, not quality:

```bash
python scripts/train_sft_v0_4.py \
  --config configs/v0_4_30m_instruct_xiaomi.yaml \
  --max-steps 100 \
  --no-resume
```

Generate an instruct sample:

```bash
python scripts/generate_instruct_v0_4.py \
  --checkpoint runs/v0_4_30m_instruct_xiaomi/checkpoints/checkpoint_latest.pt \
  --tokenizer data/tokenizers/sarych_bpe_8192_tinystories.json \
  --instruction "Write a short story about a careful fox who learns to ask for help." \
  --max-new-tokens 160 \
  --temperature 0.8 \
  --top-k 50 \
  --debug-top-k 10 \
  --no-print-prompt
```

Training step guidance:

- Tiny smoke: 50-100 max steps only.
- Around 1k accepted examples: 300-500 max steps.
- 10k+ accepted examples: 1000-2000 max steps.
- If overfitting or EOS bias appears, try `lr: 0.00005`.
- Do not judge model quality from tiny SFT runs; use them only for alignment and pipeline checks.

Train a larger SFT run from the v0.3 base:

```bash
python scripts/train_sft_v0_4.py \
  --config configs/v0_4_30m_instruct_xiaomi.yaml \
  --max-steps 2000 \
  --resume
```

Evaluate base vs instruct without Xiaomi judging:

```bash
python scripts/eval_sarych.py \
  --prompts eval/prompts_v0_4.jsonl \
  --tokenizer data/tokenizers/sarych_bpe_8192_tinystories.json \
  --base-checkpoint runs/v0_3_30m_tinystories_base/checkpoints/checkpoint_latest.pt \
  --instruct-checkpoint runs/v0_4_30m_instruct_xiaomi/checkpoints/checkpoint_latest.pt
```

The v0.4 SFT trainer uses output-only next-token loss masking. Labels are `-100` for prompt content and padding. The final prompt token predicts the first assistant output token, then output tokens predict the next output token through `<|endoftext|>`.

## v0.4.3 Lite Replay Experiments

Use `docs/sft_experimentation_v0_4.md` for the Dolly-lite, TinyStories replay, source mixing, source-aware SFT build, and lower-LR grid workflow. Replay builds should pass:

```bash
--replay-source-prefix tinystories_replay \
--keep-replay-duplicates \
--replay-dedup-mode output_hash \
--disable-replay-low-diversity-filter
```

This keeps repeated TinyStories replay instruction templates from being rejected by the non-replay duplicate-instruction policy. The manifest fields `accepted_by_source`, `rejected_by_source`, and `rejected_reason_by_source` are the first checks when replay acceptance looks wrong.

The two conservative replay configs are:

```text
configs/v0_4_30m_instruct_lite_replay_lr1e5.yaml
configs/v0_4_30m_instruct_lite_replay_lr5e6.yaml
```

Run a dry grid check without training:

```bash
python scripts/run_sft_experiment_grid.py --dry-run
```

Run the short grid only after processed SFT splits are rebuilt:

```bash
python scripts/run_sft_experiment_grid.py \
  --config configs/v0_4_30m_instruct_lite_replay_lr1e5.yaml \
  --config configs/v0_4_30m_instruct_lite_replay_lr5e6.yaml \
  --steps 100 200 300
```

Grid output is written under `artifacts/sft_grid_v0_4/` and must not be committed.

## v0.4.4 Instruction-Lite Tooling

Use `docs/instruction_lite_v0_4.md` for the current instruction-lite pipeline. This workflow keeps teacher generation outside the repo and adds local-only tooling for seeds, validation, mixing, and fixed eval.

### v0.4.5 shard factory

Prefer the data factory over one-shot 1500-row teacher prompts. See `docs/instruction_lite_factory_v0_4.md`.

```bash
python scripts/run_instruction_lite_factory.py prepare-shards \
  --seeds /mnt/c/Users/hustlePC/PycharmProjects/sft-examples/seeds/instruction_lite_v0_4_seeds.jsonl \
  --out-dir /mnt/c/Users/hustlePC/PycharmProjects/sft-examples/factory/instruction_lite_v0_4 \
  --shard-size 100 --seed 1337

# After placing teacher outputs in shards/raw/
python scripts/run_instruction_lite_factory.py validate-shards \
  --factory-dir /mnt/c/Users/hustlePC/PycharmProjects/sft-examples/factory/instruction_lite_v0_4
```

Create external-teacher seeds:

```bash
python scripts/make_instruction_lite_seeds_v0_4.py \
  --out data/xiaomi/seeds/instruction_lite_v0_4_seeds.jsonl \
  --manifest data/xiaomi/manifests/instruction_lite_v0_4_seeds_manifest.json
```

Validate imported teacher rows:

```bash
python scripts/validate_instruction_lite_sft.py \
  --input /mnt/c/Users/hustlePC/PycharmProjects/sft-examples/raw/instruction_lite_v0_4.jsonl \
  --accepted /mnt/c/Users/hustlePC/PycharmProjects/sft-examples/processed/instruction_lite_v0_4_accepted.jsonl \
  --rejected /mnt/c/Users/hustlePC/PycharmProjects/sft-examples/rejected/instruction_lite_v0_4_rejected.jsonl \
  --manifest /mnt/c/Users/hustlePC/PycharmProjects/sft-examples/manifests/instruction_lite_v0_4_manifest.json \
  --strictness standard
```

Run the fixed eval on a selected checkpoint:

```bash
python scripts/eval_instruction_lite_v0_4.py \
  --checkpoint runs/v0_4_30m_instruct_instruction_lite_lr5e6/checkpoints/checkpoint_latest.pt \
  --tokenizer data/tokenizers/sarych_bpe_8192_tinystories.json \
  --prompts evals/v0_4_instruction_lite_prompts.jsonl \
  --out-dir artifacts/evals/v0_4/lr5e6_steps1000
```

Generated seeds, accepted/rejected teacher JSONL, mixed SFT JSONL, manifests, eval artifacts, runs, and checkpoints remain ignored.

## OOM Fallback

The v0.3 default uses:

```yaml
micro_batch_size: 16
grad_accumulation_steps: 4
```

If CUDA OOM occurs, edit the config or pass an adjusted config with:

```yaml
micro_batch_size: 8
grad_accumulation_steps: 8
```

This keeps the effective batch size similar without silently changing the hidden training setup.

## Resume Behavior

Training resumes from `checkpoint_latest.pt` when `train.resume` is true or `--resume` is passed. Checkpoints include model state, optimizer state, scheduler state, step, config, parameter count, RNG state, environment metadata, and dataset RNG state. `strict_rng_restore: true` keeps resume failures explicit.

Use `--no-resume` for a fresh smoke run.

## Sync From Windows To WSL

From PowerShell in the Windows project folder:

```powershell
.\scripts\sync_to_wsl.ps1
```

From WSL:

```bash
bash /mnt/c/Users/hustlePC/PycharmProjects/sarych-lm/scripts/sync_to_wsl.sh
```

The sync scripts exclude heavy and generated artifacts. They do not delete WSL checkpoints.

## Git Hygiene

Do not commit:

- `.venv/`
- `runs/`
- checkpoints such as `*.pt`
- token memmaps such as `*.bin`
- NumPy arrays such as `*.npy` and `*.npz`
- `data/raw/*`
- `data/processed/*`
- generated tokenizer JSON files under `data/tokenizers/`
- Xiaomi raw/scored/processed/rejected JSONL and manifests under `data/xiaomi/`
- Xiaomi logs under `logs/xiaomi/`
- evaluation result JSONL files under `eval/results/`
- Python, pytest, mypy, ruff, and pip caches

Keep:

- `data/raw/.gitkeep`
- `data/processed/.gitkeep`
- `data/tokenizers/.gitkeep`
- `data/samples/tiny_stories_sample.txt`
