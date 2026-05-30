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
    "Write a tiny story about kind friends.",
    "Tell a simple bedtime story.",
    "Write a warm story with a happy ending.",
]
CONTINUATION_INSTRUCTIONS = [
    "Continue this simple story.",
    "Finish this gentle story.",
    "Continue the story in simple English.",
    "Write what happens next in the story.",
]


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


def _is_clean_story_span(text: str) -> bool:
    stripped = text.strip()
    if not stripped or stripped[-1] not in ".!?":
        return False
    sentences = _split_sentences(stripped)
    if len(sentences) < 2:
        return False
    return all(sentence[0].isupper() or sentence[0].isdigit() for sentence in sentences if sentence)


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
    mode: str = "mixed",
    min_words: int = 40,
    max_words: int = 140,
    unique_instructions: bool = False,
) -> dict[str, Any]:
    if mode not in {"story_writing", "story_continuation", "mixed"}:
        raise ValueError(f"Unsupported replay mode: {mode}")
    if min_words <= 0 or max_words < min_words:
        raise ValueError("min_words must be positive and max_words must be greater than or equal to min_words.")

    input_path = Path(input_path)
    output_path = Path(output_path)
    tokenizer = SarychBPETokenizer.from_file(tokenizer_path) if tokenizer_path else None
    rng = random.Random(seed)
    stories = _split_stories(input_path.read_text(encoding="utf-8"))

    candidates: list[dict[str, Any]] = []
    skipped_short = 0
    skipped_long = 0
    skipped_unclean = 0
    tokenizer_rejected_too_long = 0
    for story_index, story in enumerate(stories, start=1):
        wc = _word_count(story)
        clean_story = _is_clean_story_span(story)
        if not clean_story:
            skipped_unclean += 1
        if mode in {"story_writing", "mixed"} and clean_story:
            instruction = rng.choice(DEFAULT_INSTRUCTIONS)
            if unique_instructions:
                instruction = f"{instruction} Replay story {story_index}."
            if wc < min_words:
                skipped_short += 1
            elif wc > max_words:
                skipped_long += 1
            elif _fits_token_budget(
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
            else:
                tokenizer_rejected_too_long += 1

        sentences = _split_sentences(story)
        if mode in {"story_continuation", "mixed"} and clean_story and len(sentences) >= 3:
            prefix_count = 1 if len(sentences) <= 4 else rng.choice([1, 2])
            input_text = " ".join(sentences[:prefix_count])
            continuation = " ".join(sentences[prefix_count:]).strip()
            continuation_wc = _word_count(continuation)
            instruction = rng.choice(CONTINUATION_INSTRUCTIONS)
            if unique_instructions:
                instruction = f"{instruction} Replay continuation {story_index}."
            if continuation_wc < min_words:
                skipped_short += 1
            elif continuation_wc > max_words:
                skipped_long += 1
            elif not _is_clean_story_span(continuation):
                skipped_unclean += 1
            elif _fits_token_budget(
                instruction=instruction,
                input_text=input_text,
                output=continuation,
                tokenizer=tokenizer,
                max_seq_len=max_seq_len,
            ):
                candidates.append(
                    {
                        "task_type": "story_continuation",
                        "instruction": instruction,
                        "input": input_text,
                        "output": continuation,
                    }
                )
            else:
                tokenizer_rejected_too_long += 1

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
        "stories_read": len(stories),
        "rows_written": len(rows),
        "story_writing_count": task_counts.get("story_writing", 0),
        "story_continuation_count": task_counts.get("story_continuation", 0),
        "skipped_short": skipped_short,
        "skipped_long": skipped_long,
        "skipped_unclean": skipped_unclean,
        "tokenizer_rejected_too_long": tokenizer_rejected_too_long,
        "mode": mode,
        "min_words": min_words,
        "max_words": max_words,
        "unique_instructions": unique_instructions,
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
    parser.add_argument("--mode", choices=["story_writing", "story_continuation", "mixed"], default="mixed")
    parser.add_argument("--min-words", type=int, default=40)
    parser.add_argument("--max-words", type=int, default=140)
    parser.add_argument("--unique-instructions", action="store_true")
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
        mode=args.mode,
        min_words=args.min_words,
        max_words=args.max_words,
        unique_instructions=args.unique_instructions,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
