from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sarych.tokenizer_bpe import DEFAULT_SPECIAL_TOKENS, train_byte_bpe_tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SARYCH-LM v0.2 byte-level BPE tokenizer.")
    parser.add_argument("--input", required=True, nargs="+", help="Input text file(s).")
    parser.add_argument("--output", required=True, help="Output tokenizer JSON path.")
    parser.add_argument("--vocab-size", type=int, default=4096)
    parser.add_argument("--limit-lines", type=int, default=None)
    parser.add_argument("--special-token", action="append", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    special_tokens = args.special_token if args.special_token is not None else DEFAULT_SPECIAL_TOKENS
    tokenizer = train_byte_bpe_tokenizer(
        input_paths=args.input,
        output_path=args.output,
        vocab_size=args.vocab_size,
        special_tokens=special_tokens,
        limit_lines=args.limit_lines,
    )
    print(f"Saved tokenizer: {args.output}")
    print(f"Vocab size: {tokenizer.vocab_size}")
    for token in special_tokens:
        print(f"{token}: {tokenizer.token_to_id(token)}")


if __name__ == "__main__":
    main()
