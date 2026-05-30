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

Build a tiny 100-example dataset:

```bash
python scripts/build_sft_dataset.py \
  --raw /mnt/c/Users/hustlePC/PycharmProjects/sft-examples/raw/sft_100.jsonl \
  --scored /mnt/c/Users/hustlePC/PycharmProjects/sft-examples/scored/sft_100_scores.jsonl \
  --tokenizer data/tokenizers/sarych_bpe_8192_tinystories.json \
  --train-out data/xiaomi/processed/sft/train.jsonl \
  --val-out data/xiaomi/processed/sft/val.jsonl \
  --rejected-dir data/xiaomi/rejected \
  --manifest data/xiaomi/manifests/sft_v0_4_manifest.json \
  --val-ratio 0.05 \
  --seed 1337
```

Run a 100-step SFT smoke:

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
  --top-k 50
```

Build the full 30k dataset:

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

Train the 2000-step SFT run:

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

The v0.4 SFT trainer uses output-only loss masking. Labels are `-100` for prompt and padding tokens, and token IDs only for assistant output plus `<|endoftext|>`.

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
