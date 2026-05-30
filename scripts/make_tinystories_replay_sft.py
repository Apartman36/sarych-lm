from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sarych.sft import format_instruct_prompt
from sarych.tokenizer_bpe import SarychBPETokenizer

DEFAULT_INSTRUCTIONS = [
    "Write a short simple story.",
    "Write a short story for children.",
    "Tell a gentle story with a clear ending.",
]
CONTINUATION_INSTRUCTION = "Continue this simple story."


def _word_count(text: str) -> int:
    return len(re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", text))


def _split_sentences(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+", text.strip()) if part.strip()]


def _split_stories(text: str) -> list[str]:
    if "<|endoftext|>" in text:
        parts = text.split("<|endoftext|>")
    else:
        parts = re.split(r"\n\s*\n", text)
    stories: list[str] = []
    for part in parts:
        story = re.sub(r"\s+", " ", part).strip()
        if story:
            stories.append(story)
    return stories


def _fits_token_budget(
    *,
    instruction: str,
    input_text: str,
    output: str,
    tokenizer: SarychBPETokenizer | None,
    max_seq_len: int,
) -> bool:
    if tokenizer is None:
        return _word_count(instruction) + _word_count(input_text) + _word_count(output) <= max_seq_len
    text = format_instruct_prompt(instruction, input_text) + output.strip() + "<|endoftext|>"
    return len(tokenizer.encode(text)) <= max_seq_len


def _make_row(index: int, task_type: str, instruction: str, input_text: str, output: str) -> dict[str, Any]:
    return {
        "id": f"tinystories_replay_{index:06d}",
        "source": "tinystories_replay",
        "task_type": task_type,
        "instruction": instruction,
        "input": input_text,
        "output": output.strip(),
        "language": "en",
        "metadata": {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "generator": "make_tinystories_replay_sft",
        },
    }


def make_tinystories_replay(
    input_path: str | Path,
    output_path: str | Path,
    *,
    count: int = 1000,
    tokenizer_path: str | Path | None = None,
    seed: int = 1337,
    manifest_path: str | Path | None = None,
    max_seq_len: int = 512,
) -> dict[str, Any]:
    input_path = Path(input_path)
    output_path = Path(output_path)
    tokenizer = SarychBPETokenizer.from_file(tokenizer_path) if tokenizer_path else None
    rng = random.Random(seed)
    stories = _split_stories(input_path.read_text(encoding="utf-8"))

    candidates: list[dict[str, Any]] = []
    for story in stories:
        wc = _word_count(story)
        if 40 <= wc <= 140:
            instruction = rng.choice(DEFAULT_INSTRUCTIONS)
            if _fits_token_budget(
                instruction=instruction,
                input_text="",
                output=story,
                tokenizer=tokenizer,
                max_seq_len=max_seq_len,
            ):
                candidates.append(
                    {
                        "task_type": "story_writing",
                        "instruction": instruction,
                        "input": "",
                        "output": story,
                    }
                )

        sentences = _split_sentences(story)
        if len(sentences) >= 3:
            prefix_count = 1 if len(sentences) <= 4 else rng.choice([1, 2])
            input_text = " ".join(sentences[:prefix_count])
            continuation = " ".join(sentences[prefix_count:]).strip()
            continuation_wc = _word_count(continuation)
            if 40 <= continuation_wc <= 140 and _fits_token_budget(
                instruction=CONTINUATION_INSTRUCTION,
                input_text=input_text,
                output=continuation,
                tokenizer=tokenizer,
                max_seq_len=max_seq_len,
            ):
                candidates.append(
                    {
                        "task_type": "story_continuation",
                        "instruction": CONTINUATION_INSTRUCTION,
                        "input": input_text,
                        "output": continuation,
                    }
                )

    rng.shuffle(candidates)
    selected = candidates[: max(0, count)]
    rows = [
        _make_row(i, row["task_type"], row["instruction"], row["input"], row["output"])
        for i, row in enumerate(selected, start=1)
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    task_counts = Counter(row["task_type"] for row in rows)
    manifest = {
        "converter": "make_tinystories_replay_sft",
        "source_path": str(input_path),
        "output_path": str(output_path),
        "raw_stories": len(stories),
        "candidate_rows": len(candidates),
        "written_rows": len(rows),
        "task_type_counts": dict(sorted(task_counts.items())),
        "count_limit": count,
        "tokenizer_path": str(tokenizer_path) if tokenizer_path else None,
        "max_seq_len": max_seq_len,
        "seed": seed,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    if manifest_path is not None:
        manifest_path = Path(manifest_path)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def _default_input_path() -> str:
    train = Path("data/raw/TinyStories-train.txt")
    if train.exists():
        return str(train)
    return "data/raw/TinyStories-valid.txt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create SARYCH SFT replay rows from local TinyStories text.")
    parser.add_argument("--input", default=_default_input_path())
    parser.add_argument("--out", default="data/xiaomi/processed/replay/tinystories_replay_sft_v0_4.jsonl")
    parser.add_argument("--count", type=int, default=1000)
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--manifest", default="data/xiaomi/processed/replay/tinystories_replay_sft_v0_4_manifest.json")
    parser.add_argument("--max-seq-len", type=int, default=512)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = make_tinystories_replay(
        args.input,
        args.out,
        count=args.count,
        tokenizer_path=args.tokenizer,
        seed=args.seed,
        manifest_path=args.manifest,
        max_seq_len=args.max_seq_len,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
