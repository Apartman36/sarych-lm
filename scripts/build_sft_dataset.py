from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sarych.sft import build_sft_splits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate, filter, deduplicate, and split Xiaomi SFT JSONL data.")
    parser.add_argument("--raw", required=True, help="Raw Xiaomi SFT JSONL path.")
    parser.add_argument("--scored", default=None, help="Optional Xiaomi score JSONL path.")
    parser.add_argument("--tokenizer", default="data/tokenizers/sarych_bpe_8192_tinystories.json")
    parser.add_argument("--train-out", default="data/xiaomi/processed/sft/train.jsonl")
    parser.add_argument("--val-out", default="data/xiaomi/processed/sft/val.jsonl")
    parser.add_argument("--rejected-dir", default="data/xiaomi/rejected")
    parser.add_argument("--manifest", default="data/xiaomi/manifests/sft_v0_4_manifest.json")
    parser.add_argument("--val-ratio", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--max-seq-len", type=int, default=512)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = build_sft_splits(
        raw_path=args.raw,
        scored_path=args.scored,
        tokenizer_path=args.tokenizer,
        train_path=args.train_out,
        val_path=args.val_out,
        rejected_dir=args.rejected_dir,
        val_ratio=args.val_ratio,
        seed=args.seed,
        max_seq_len=args.max_seq_len,
        manifest_path=args.manifest,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
