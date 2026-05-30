from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sarych.sft import ALLOWED_SFT_TASK_TYPES

KEEP_CATEGORIES = {
    "creative_writing",
    "generation",
    "brainstorming",
    "open_qa",
    "classification",
}

DISCARD_CATEGORIES = {
    "closed_qa",
    "information_extraction",
    "summarization",
}

CODE_TERMS = {
    "python", "javascript", "java", "c++", "sql", "function", "code",
    "algorithm", "script", "compile", "debug", "programming", "runtime",
    "syntax", "debugger", "compiler", "repository", "github", "api",
}

ADULT_TERMS = {
    "election", "president", "gun", "abortion", "war", "murder", "sex",
    "alcohol", "drug", "investment", "stock", "lawyer", "lawsuit",
    "diagnosis", "medication", "weapon", "terrorism", "suicide",
    "assault", "trafficking", "cartel", "genocide", "bomb",
}

FORMAL_TONE_PATTERNS = re.compile(
    r"\b(therefore|moreover|in conclusion|furthermore|aforementioned|"
    r"notwithstanding|hereby|whereas|consequently|subsequently)\b",
    re.IGNORECASE,
)

URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
MARKDOWN_TABLE_PATTERN = re.compile(r"\|.*\|.*\|")
AS_AI_PATTERN = re.compile(r"\bas an AI\b", re.IGNORECASE)
DIGIT_PATTERN = re.compile(r"\d")
WHO_WHEN_WHERE_PATTERN = re.compile(
    r"^(who|when|where|which|what year|what date|what time)\b", re.IGNORECASE
)

OUTPUT_MIN_WORDS = 20
OUTPUT_MAX_WORDS = 90
INSTRUCTION_MAX_WORDS = 40
MAX_LONG_WORDS = 4
LONG_WORD_MIN_LEN = 12
MAX_DIGITS = 10
WHO_WHEN_WHERE_OUTPUT_MAX_WORDS = 50


def _word_count(text: str) -> int:
    return len(re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", text))


def _contains_any(text: str, terms: set[str]) -> bool:
    lower = text.lower()
    words = set(re.findall(r"[a-z]+", lower))
    return bool(words & terms)


def _count_long_words(text: str) -> int:
    return sum(1 for w in re.findall(r"[A-Za-z]+", text) if len(w) >= LONG_WORD_MIN_LEN)


def _count_digits(text: str) -> int:
    return len(DIGIT_PATTERN.findall(text))


def _map_task_type(dolly_category: str, instruction: str, output: str) -> str | None:
    instr_lower = instruction.lower()
    output_lower = output.lower()
    combined = instr_lower + " " + output_lower

    if dolly_category == "creative_writing":
        story_keywords = {"story", "tale", "character", "adventure", "fairy", "dragon", "prince", "princess"}
        if any(kw in combined for kw in story_keywords):
            return "story_writing"
        return "creative_generation"

    if dolly_category == "generation":
        structured_hints = {"list", "bullet", "steps", "numbered", "format", "table"}
        if any(kw in instr_lower for kw in structured_hints):
            return "structured_output"
        return "creative_generation"

    if dolly_category == "brainstorming":
        return "structured_output"

    if dolly_category == "classification":
        return "simple_reasoning"

    if dolly_category == "open_qa":
        explain_patterns = re.compile(
            r"^(why|how|explain|what is|what are|describe)\b", re.IGNORECASE
        )
        if explain_patterns.search(instr_lower):
            return "explanation_for_children"
        return "simple_qa"

    return None


def _passes_filters(instruction: str, context: str, output: str) -> str | None:
    if context.strip():
        return "has_context"

    if "```" in output or "`" in output:
        return "output_has_code_block"

    if _contains_any(instruction, CODE_TERMS) or _contains_any(output, CODE_TERMS):
        return "contains_code_terms"

    if _contains_any(instruction, ADULT_TERMS) or _contains_any(output, ADULT_TERMS):
        return "contains_adult_terms"

    output_wc = _word_count(output)
    if output_wc < OUTPUT_MIN_WORDS:
        return "output_too_short"
    if output_wc > OUTPUT_MAX_WORDS:
        return "output_too_long"

    instr_wc = _word_count(instruction)
    if instr_wc > INSTRUCTION_MAX_WORDS:
        return "instruction_too_long"

    if WHO_WHEN_WHERE_PATTERN.search(instruction):
        if output_wc > WHO_WHEN_WHERE_OUTPUT_MAX_WORDS:
            return "factual_qa_too_long"

    if _count_long_words(output) > MAX_LONG_WORDS:
        return "too_many_long_words"

    if _count_digits(output) > MAX_DIGITS:
        return "too_many_digits"

    if URL_PATTERN.search(output):
        return "output_has_url"

    if MARKDOWN_TABLE_PATTERN.search(output):
        return "output_has_markdown_table"

    if FORMAL_TONE_PATTERNS.search(output):
        return "formal_tone"

    if AS_AI_PATTERN.search(output):
        return "as_an_ai"

    return None


def _format_row(index: int, instruction: str, output: str, task_type: str, dolly_category: str) -> dict[str, Any]:
    return {
        "id": f"dolly_lite_{index:06d}",
        "source": "databricks_dolly_15k_lite",
        "task_type": task_type,
        "instruction": instruction.strip(),
        "input": "",
        "output": output.strip(),
        "language": "en",
        "metadata": {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "generator": "convert_dolly_lite_to_sft",
            "dolly_category": dolly_category,
        },
    }


def convert_dolly_lite(
    input_path: str | Path,
    output_path: str | Path,
    manifest_path: str | Path | None = None,
) -> dict[str, Any]:
    input_path = Path(input_path)
    output_path = Path(output_path)

    raw_rows: list[dict[str, Any]] = []
    with input_path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            line = line.strip()
            if line:
                raw_rows.append(json.loads(line))

    category_input_counts: Counter[str] = Counter()
    task_type_counts: Counter[str] = Counter()
    filter_reason_counts: Counter[str] = Counter()
    kept_rows: list[dict[str, Any]] = []
    kept_index = 0

    for row in raw_rows:
        category = str(row.get("category", ""))
        category_input_counts[category] += 1

        if category in DISCARD_CATEGORIES:
            filter_reason_counts["discard_category"] += 1
            continue

        if category not in KEEP_CATEGORIES:
            filter_reason_counts["unknown_category"] += 1
            continue

        instruction = str(row.get("instruction", ""))
        context = str(row.get("context", ""))
        output = str(row.get("response", ""))

        filter_reason = _passes_filters(instruction, context, output)
        if filter_reason is not None:
            filter_reason_counts[filter_reason] += 1
            continue

        task_type = _map_task_type(category, instruction, output)
        if task_type is None:
            filter_reason_counts["unmappable_task_type"] += 1
            continue

        if task_type not in ALLOWED_SFT_TASK_TYPES:
            filter_reason_counts["disallowed_task_type"] += 1
            continue

        kept_index += 1
        kept_rows.append(_format_row(kept_index, instruction, output, task_type, category))
        task_type_counts[task_type] += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in kept_rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    manifest: dict[str, Any] = {
        "converter": "convert_dolly_lite_to_sft",
        "source_path": str(input_path),
        "output_path": str(output_path),
        "raw_rows": len(raw_rows),
        "kept_rows": len(kept_rows),
        "filtered_rows": len(raw_rows) - len(kept_rows),
        "category_input_counts": dict(sorted(category_input_counts.items())),
        "output_task_type_counts": dict(sorted(task_type_counts.items())),
        "filter_reason_counts": dict(sorted(filter_reason_counts.items())),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    if manifest_path is not None:
        manifest_path = Path(manifest_path)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    return manifest


def _external_default(relative_path: str) -> str:
    if os.name == "nt":
        return str(Path(r"C:\Users\hustlePC\PycharmProjects\sft-examples") / relative_path)
    return str(Path("/mnt/c/Users/hustlePC/PycharmProjects/sft-examples") / relative_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert Dolly raw JSONL to SARYCH SFT schema with strict filtering.")
    parser.add_argument(
        "--input",
        default=_external_default("public_raw/dolly_15k_train.jsonl"),
    )
    parser.add_argument(
        "--out",
        default=_external_default("public_converted/dolly_lite_sft_v0_4.jsonl"),
    )
    parser.add_argument(
        "--manifest",
        default=_external_default("manifests/dolly_lite_sft_v0_4_manifest.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = convert_dolly_lite(
        input_path=args.input,
        output_path=args.out,
        manifest_path=args.manifest,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
