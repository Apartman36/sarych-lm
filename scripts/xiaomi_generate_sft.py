from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

from pathlib import Path

TASK_TYPES = [
    "story_writing",
    "story_continuation",
    "explanation_for_children",
    "simple_qa",
    "dialogue",
    "summarization",
    "simple_reasoning",
    "structured_output",
    "creative_generation",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Safe Xiaomi SFT generation template. Default is dry-run; it never calls Xiaomi automatically."
    )
    parser.add_argument("--output", default="C:/Users/hustlePC/PycharmProjects/sft-examples/raw/sft_preview.jsonl")
    parser.add_argument("--count", type=int, default=3)
    parser.add_argument("--dry-run", action="store_true", default=True)
    parser.add_argument(
        "--allow-external-call",
        action="store_true",
        help="Reserved for a future manual integration. This pass still exits without calling Xiaomi.",
    )
    return parser.parse_args()


def _template_row(index: int) -> dict:
    task_type = TASK_TYPES[index % len(TASK_TYPES)]
    return {
        "id": f"xm_sft_{index + 1:06d}",
        "source": "xiaomi_mimo_v2_5_pro",
        "task_type": task_type,
        "instruction": "Write a short story for young children about a rabbit who learns to share.",
        "input": "",
        "output": "Once there was a little rabbit named Pip who had two apples and one hungry friend.",
        "language": "en",
        "metadata": {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "generator": "manual",
            "model": "mimo-v2.5-pro",
            "temperature": 0.7,
            "max_tokens": 512,
            "prompt_template": "sft_v1",
        },
    }


def main() -> None:
    args = parse_args()
    if args.allow_external_call:
        raise SystemExit("External Xiaomi calls are intentionally disabled in SARYCH-LM v0.4 infrastructure.")
    rows = [_template_row(i) for i in range(args.count)]
    print("Dry-run Xiaomi SFT generation template. No external API was called.")
    print(f"Example output target: {Path(args.output)}")
    for row in rows:
        print(json.dumps(row, ensure_ascii=False))


if __name__ == "__main__":
    main()
