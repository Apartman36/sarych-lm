# SARYCH-LM

SARYCH-LM is Antoniy Sarychev's personal small-language-model engineering project. v0.1 is named `sarych-5m-sanity` and validates a clean single-GPU PyTorch training pipeline. v0.2 is named `sarych-5m-tinystories-smoke` and adds the first real text/tokenizer/data smoke pipeline.

This project is not a chatbot, not an assistant, and not a useful language model yet. v0.1 trains on synthetic integer-token data. v0.2 can train briefly on a tiny original local story sample or on manually downloaded TinyStories text.

## What v0.1 Does

- Instantiates a small decoder-only Transformer with RMSNorm, RoPE, SwiGLU, causal attention, and tied embeddings.
- Trains on synthetic random and structured token sequences only.
- Uses BF16 autocast when CUDA and BF16 are available; otherwise uses FP32.
- Saves and resumes `.pt` checkpoints with model, optimizer, step, config, RNG state, and metadata.
- Logs JSONL metrics including loss, learning rate, throughput, elapsed time, and CUDA memory when available.
- Provides pytest coverage for model forward/backward, synthetic data, checkpoint restore, tiny overfit, and environment reporting.

## What v0.2 Adds

- Byte-level BPE tokenizer training through the `tokenizers` library, without `transformers`.
- A small original sample at `data/samples/tiny_stories_sample.txt` for offline tests and smoke runs.
- Optional TinyStories raw-text download scripts that write only to `data/raw/`.
- Tokenized `uint16` memmap datasets in `data/processed/`.
- `sarych-5m-tinystories-smoke`, a decoder-only Transformer config with vocab size 4096, context length 256, 6 layers, 8 heads, width 256, SwiGLU width 512, RMSNorm, RoPE, and tied embeddings.
- Text generation that encodes prompts and decodes generated token IDs through the trained tokenizer.

## What This Does Not Do

v0.1 does not use TinyStories, Hugging Face datasets, tokenizers, transformers, Qwen distillation, instruction tuning, an API server, a UI, MoE, quantization, DeepSpeed, FSDP, DDP, FlashAttention, Triton kernels, or external downloads.

v0.2 still does not implement v1 30M training, instruction tuning, Qwen distillation, API/UI, MoE, DeepSpeed/FSDP/DDP, FlashAttention, custom CUDA, or default `torch.compile`.

## Hardware Target

The target training machine is Windows 11 Pro with WSL2 Ubuntu, Ryzen 7 8700F, 32GB RAM, and NVIDIA GeForce RTX 5060 Ti 16GB. For Blackwell / `sm_120`, use PyTorch 2.7+ with CUDA 12.8+ wheels inside WSL.

## WSL Setup

From WSL:

```bash
cd ~/projects/sarych-lm
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

# For RTX 50-series / Blackwell, use a PyTorch CUDA 12.8+ wheel.
pip install torch --index-url https://download.pytorch.org/whl/cu128

pip install -r requirements.txt
```

Verify PyTorch and CUDA:

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

## Run Checks

```bash
python scripts/env_report.py --output runs/v0_1_synthetic_sanity/env_report.txt
pytest -q
```

## Train v0.1

```bash
python scripts/train_v0_1.py --config configs/v0_1_synthetic_sanity.yaml
python scripts/train_v0_1.py --config configs/v0_1_synthetic_sanity.yaml --max-steps 20
python scripts/train_v0_1.py --config configs/v0_1_synthetic_sanity.yaml --resume
python scripts/train_v0_1.py --config configs/v0_1_synthetic_sanity.yaml --no-resume
python scripts/generate_v0_1.py --checkpoint runs/v0_1_synthetic_sanity/checkpoints/checkpoint_latest.pt --max-new-tokens 80
```

## v0.2 Text Smoke Pipeline

The default v0.2 path uses the committed local sample, so it does not need internet:

```bash
python scripts/train_tokenizer_v0_2.py \
  --input data/samples/tiny_stories_sample.txt \
  --output data/tokenizers/sarych_bpe_4096_sample.json \
  --vocab-size 4096

python scripts/prepare_text_dataset_v0_2.py \
  --input data/samples/tiny_stories_sample.txt \
  --tokenizer data/tokenizers/sarych_bpe_4096_sample.json \
  --output-dir data/processed/v0_2_sample \
  --block-size 256 \
  --val-fraction 0.1

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

The sample corpus is tiny. If BPE learns fewer than 4096 realized entries from it, the model still uses a 4096-logit head and generation masks to the tokenizer's actual realized vocabulary before decoding.

Optional TinyStories download, only when you explicitly want raw files:

```bash
python scripts/download_tinystories.py --valid
python scripts/download_tinystories.py --train
```

Downloaded raw files stay under `data/raw/` and prepared token files stay under `data/processed/`; both are ignored by Git.

## Sync From Windows To WSL

From PowerShell in the Windows project folder:

```powershell
.\scripts\sync_to_wsl.ps1
```

From WSL:

```bash
bash /mnt/c/Users/hustlePC/PycharmProjects/sarych-lm/scripts/sync_to_wsl.sh
```

Both scripts copy code into `~/projects/sarych-lm` and exclude heavy/cache artifacts such as `.venv/`, `runs/`, `.git/`, `__pycache__/`, `*.pt`, `*.npy`, and `*.npz`. They do not delete WSL checkpoints.

## Repository Layout

```text
configs/v0_1_synthetic_sanity.yaml   # v0.1 config
configs/v0_2_tinystories_smoke.yaml  # v0.2 text smoke config
sarych/model.py                      # decoder-only Transformer
sarych/data_synthetic.py             # synthetic integer-token data
sarych/data_text.py                  # uint16 memmap token batches
sarych/tokenizer_bpe.py              # byte-level BPE tokenizer wrapper
sarych/train.py                      # training and resume loop
sarych/checkpoint.py                 # .pt checkpoint save/load
sarych/reporting.py                  # JSONL and memory/throughput reporting
sarych/env_report.py                 # environment report
scripts/                             # CLI entry points and sync helpers
data/samples/tiny_stories_sample.txt # committed offline sample
tests/                               # pytest sanity coverage
docs/v0_1_design.md                  # v0.1 design notes
docs/v0_2_design.md                  # v0.2 design notes
```

## Git Hygiene

Do not commit `.venv/`, `runs/`, checkpoints, downloaded TinyStories files, tokenized memmaps, NumPy arrays, caches, or full datasets. The generated sample tokenizer JSON is small enough to commit if deliberately wanted, but the normal workflow regenerates it from `data/samples/tiny_stories_sample.txt`.
