from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Safe Xiaomi SFT scoring template. Default is dry-run; it never calls Xiaomi automatically."
    )
    parser.add_argument("--input", default="C:/Users/hustlePC/PycharmProjects/sft-examples/raw/sft.jsonl")
    parser.add_argument("--output", default="C:/Users/hustlePC/PycharmProjects/sft-examples/scored/sft_scores.jsonl")
    parser.add_argument("--dry-run", action="store_true", default=True)
    parser.add_argument(
        "--allow-external-call",
        action="store_true",
        help="Reserved for a future manual integration. This pass still exits without calling Xiaomi.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.allow_external_call:
        raise SystemExit("External Xiaomi judge calls are intentionally disabled in SARYCH-LM v0.4 infrastructure.")
    template = {
        "id": "xm_score_000001",
        "example_id": "xm_sft_000001",
        "scores": {
            "instruction_following": 4,
            "coherence": 5,
            "safety": 5,
            "age_appropriateness": 5,
            "english_quality": 4,
        },
        "judge": {"source": "xiaomi_mimo_v2_5_pro", "model": "mimo-v2.5-pro", "rubric": "sarych_sft_judge_v1"},
        "notes": "Clear, simple, coherent.",
    }
    print("Dry-run Xiaomi SFT scoring template. No external API was called.")
    print(f"Input target: {Path(args.input)}")
    print(f"Output target: {Path(args.output)}")
    print(json.dumps(template, ensure_ascii=False))


if __name__ == "__main__":
    main()
