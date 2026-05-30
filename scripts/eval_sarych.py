from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.generate_instruct_v0_4 import generate_instruct_text


def _read_prompts(path: Path) -> list[dict]:
    prompts = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                prompts.append(json.loads(line))
    return prompts


def _write_outputs(
    *,
    prompts: list[dict],
    checkpoint_path: str,
    tokenizer_path: str,
    output_path: Path,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    device: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for prompt in prompts:
            text = generate_instruct_text(
                checkpoint_path=checkpoint_path,
                tokenizer_path=tokenizer_path,
                instruction=prompt["instruction"],
                input_text=prompt.get("input", ""),
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                device=device,
            )
            record = {
                "id": prompt["id"],
                "category": prompt["category"],
                "instruction": prompt["instruction"],
                "input": prompt.get("input", ""),
                "output": text,
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate manual base-vs-instruct v0.4 evaluation outputs.")
    parser.add_argument("--prompts", default="eval/prompts_v0_4.jsonl")
    parser.add_argument("--tokenizer", default="data/tokenizers/sarych_bpe_8192_tinystories.json")
    parser.add_argument("--base-checkpoint", default="runs/v0_3_30m_tinystories_base/checkpoints/checkpoint_latest.pt")
    parser.add_argument("--instruct-checkpoint", default="runs/v0_4_30m_instruct_xiaomi/checkpoints/checkpoint_latest.pt")
    parser.add_argument("--base-out", default="eval/results/base_v0_3_outputs.jsonl")
    parser.add_argument("--instruct-out", default="eval/results/instruct_v0_4_outputs.jsonl")
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prompts = _read_prompts(Path(args.prompts))
    _write_outputs(
        prompts=prompts,
        checkpoint_path=args.base_checkpoint,
        tokenizer_path=args.tokenizer,
        output_path=Path(args.base_out),
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        device=args.device,
    )
    _write_outputs(
        prompts=prompts,
        checkpoint_path=args.instruct_checkpoint,
        tokenizer_path=args.tokenizer,
        output_path=Path(args.instruct_out),
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        device=args.device,
    )
    print(f"Wrote base outputs to {args.base_out}")
    print(f"Wrote instruct outputs to {args.instruct_out}")
    print("No Xiaomi judge was called. Compare the JSONL outputs manually.")


if __name__ == "__main__":
    main()
