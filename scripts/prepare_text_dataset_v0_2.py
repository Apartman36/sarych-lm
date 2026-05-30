from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sarych.tokenizer_bpe import SarychBPETokenizer


EOT_TOKEN = "<|endoftext|>"


def _read_text_with_eot(input_path: Path) -> str:
    text = input_path.read_text(encoding="utf-8")
    if EOT_TOKEN in text:
        return text
    stories = [part.strip() for part in text.split("\n\n") if part.strip()]
    if not stories:
        return EOT_TOKEN
    return f"\n{EOT_TOKEN}\n".join(stories) + f"\n{EOT_TOKEN}\n"


def _write_uint16(path: Path, token_ids: list[int]) -> None:
    array = np.asarray(token_ids, dtype=np.uint16)
    array.tofile(path)


def prepare_text_dataset(
    *,
    input_path: str | Path,
    tokenizer_path: str | Path,
    output_dir: str | Path,
    block_size: int,
    val_fraction: float = 0.1,
    val_input_path: str | Path | None = None,
) -> dict[str, Any]:
    input_path = Path(input_path)
    tokenizer_path = Path(tokenizer_path)
    output_dir = Path(output_dir)
    block_size = int(block_size)
    if not (0.0 < float(val_fraction) < 1.0):
        raise ValueError("val_fraction must be between 0 and 1.")
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = SarychBPETokenizer.from_file(tokenizer_path)
    train_text = _read_text_with_eot(input_path)
    token_ids = tokenizer.encode(train_text)
    if not token_ids:
        raise ValueError("No tokens produced from input text.")
    max_token = max(token_ids)
    if max_token >= tokenizer.vocab_size:
        raise ValueError(f"Token id {max_token} exceeds tokenizer vocab size {tokenizer.vocab_size}.")
    min_total_tokens = 2 * (block_size + 2) + 1
    repeat_factor = 1
    if len(token_ids) < min_total_tokens:
        repeat_factor = int(math.ceil(min_total_tokens / len(token_ids)))
        token_ids = token_ids * repeat_factor

    if val_input_path is not None:
        val_ids = tokenizer.encode(_read_text_with_eot(Path(val_input_path)))
        train_ids = token_ids
    else:
        val_count = max(block_size + 2, int(len(token_ids) * float(val_fraction)))
        if len(token_ids) - val_count <= block_size + 1:
            val_count = max(1, min(len(token_ids) // 5, len(token_ids) - block_size - 2))
        if val_count <= block_size + 1 or len(token_ids) - val_count <= block_size + 1:
            raise ValueError("Input text is too small for the requested block_size and validation split.")
        train_ids = token_ids[:-val_count]
        val_ids = token_ids[-val_count:]

    for split, ids in {"train": train_ids, "val": val_ids}.items():
        if len(ids) <= block_size + 1:
            raise ValueError(f"{split} split is too small for block_size={block_size}.")
        split_max = max(ids)
        if split_max >= tokenizer.vocab_size:
            raise ValueError(f"{split} token id {split_max} exceeds tokenizer vocab size {tokenizer.vocab_size}.")

    train_bin = output_dir / "train.bin"
    val_bin = output_dir / "val.bin"
    _write_uint16(train_bin, train_ids)
    _write_uint16(val_bin, val_ids)

    metadata = {
        "input_path": str(input_path),
        "val_input_path": str(val_input_path) if val_input_path is not None else None,
        "tokenizer_path": str(tokenizer_path),
        "vocab_size": tokenizer.vocab_size,
        "block_size": block_size,
        "dtype": "uint16",
        "train_tokens": len(train_ids),
        "val_tokens": len(val_ids),
        "repeat_factor": repeat_factor,
        "train_bin": str(train_bin),
        "val_bin": str(val_bin),
        "disk_bytes": train_bin.stat().st_size + val_bin.stat().st_size,
    }
    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare SARYCH-LM v0.2 token memmap dataset.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--val-input", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata = prepare_text_dataset(
        input_path=args.input,
        tokenizer_path=args.tokenizer,
        output_dir=args.output_dir,
        block_size=args.block_size,
        val_fraction=args.val_fraction,
        val_input_path=args.val_input,
    )
    print(f"Train tokens: {metadata['train_tokens']}")
    print(f"Val tokens: {metadata['val_tokens']}")
    print(f"Disk usage: {metadata['disk_bytes']} bytes")
    print(f"Metadata: {Path(args.output_dir) / 'metadata.json'}")


if __name__ == "__main__":
    main()
