# SARYCH-LM v0.2 Design

## Purpose

v0.2 introduces the first real text pipeline for SARYCH-LM. It is named `sarych-5m-tinystories-smoke`. It is a smoke model and data path, not the future 30M base model and not a useful language model.

The default path uses `data/samples/tiny_stories_sample.txt`, an original tiny English story sample committed to the repo. This keeps tests and smoke runs offline and deterministic. TinyStories download is optional and must be run manually.

## Model

The default config keeps the same decoder-only Transformer family as v0.1:

- vocab size 4096
- context length 256
- 6 layers
- 8 attention heads
- embedding width 256
- SwiGLU feed-forward width 512
- RMSNorm
- RoPE
- tied embeddings
- BF16 autocast on CUDA when supported, FP32 fallback
- AdamW with warmup and cosine decay

The actual parameter count is logged at runtime.

## Tokenizer

`sarych/tokenizer_bpe.py` wraps the Hugging Face `tokenizers` package directly. It trains a byte-level BPE tokenizer with `<|endoftext|>` and `<|pad|>` special tokens. It does not use `transformers`.

Train the sample tokenizer:

```bash
python scripts/train_tokenizer_v0_2.py \
  --input data/samples/tiny_stories_sample.txt \
  --output data/tokenizers/sarych_bpe_4096_sample.json \
  --vocab-size 4096
```

On the tiny sample, the realized tokenizer vocabulary can be smaller than 4096 because BPE cannot invent useful merges that do not appear in the corpus. The model config still uses a 4096-token output head; v0.2 generation masks logits to the tokenizer's realized vocabulary before decoding.

## Dataset Preparation

`scripts/prepare_text_dataset_v0_2.py` encodes text into `uint16` token IDs and writes:

- `train.bin`
- `val.bin`
- `metadata.json`

The local sample is tiny, so preparation repeats undersized inputs enough to satisfy the requested block size. The repeat factor is recorded in metadata. Full TinyStories files should not need repetition.

```bash
python scripts/prepare_text_dataset_v0_2.py \
  --input data/samples/tiny_stories_sample.txt \
  --tokenizer data/tokenizers/sarych_bpe_4096_sample.json \
  --output-dir data/processed/v0_2_sample \
  --block-size 256 \
  --val-fraction 0.1
```

`sarych/data_text.py` loads these files via NumPy memmap and returns `(x, y)` batches where `y` is shifted by one token. The dataset has an RNG state dict for checkpoint/resume.

## Train And Generate

Short training smoke:

```bash
python scripts/train_v0_2.py \
  --config configs/v0_2_tinystories_smoke.yaml \
  --max-steps 50 \
  --no-resume
```

Generate from the latest checkpoint:

```bash
python scripts/generate_v0_2.py \
  --checkpoint runs/v0_2_tinystories_smoke/checkpoints/checkpoint_latest.pt \
  --tokenizer data/tokenizers/sarych_bpe_4096_sample.json \
  --prompt "Once upon a time" \
  --max-new-tokens 80
```

## Optional TinyStories

The downloader is never run by tests or training. Run it explicitly when ready:

```bash
python scripts/download_tinystories.py --valid
python scripts/download_tinystories.py --train
```

Files are written to `data/raw/`. Interrupted downloads resume from `.part` files when the server honors byte ranges. Use `--force` to restart.

## Git Hygiene

Commit `data/samples/tiny_stories_sample.txt` and `.gitkeep` placeholders. Do not commit:

- `.venv/`
- `runs/`
- checkpoints such as `*.pt`
- tokenized `*.bin`, `*.npy`, `*.npz`
- `data/raw/*`
- `data/processed/*`
- cache directories

The sample tokenizer JSON is small and reproducible. It can be committed intentionally, but the documented default is to regenerate it from the sample.
