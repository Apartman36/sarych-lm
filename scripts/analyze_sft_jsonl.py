from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _read_jsonl(path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append({"line": line_number, "reason": "json_decode_error", "detail": exc.msg})
                continue
            if isinstance(row, dict):
                rows.append(row)
            else:
                errors.append({"line": line_number, "reason": "not_object"})
    return rows, errors


def _words(text: str) -> list[str]:
    return re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", text)


def _stats(values: list[int]) -> dict[str, float | int | None]:
    if not values:
        return {"min": None, "max": None, "mean": None, "median": None}
    return {"min": min(values), "max": max(values), "mean": round(mean(values), 3), "median": round(median(values), 3)}


def _near_duplicate_clusters(rows: list[dict[str, Any]], threshold: float = 0.86) -> list[dict[str, Any]]:
    token_sets: list[tuple[str, set[str]]] = []
    clusters: list[dict[str, Any]] = []
    for row in rows:
        instruction = str(row.get("instruction", ""))
        tokens = set(word.lower() for word in _words(instruction))
        if len(tokens) < 4:
            continue
        matches = []
        for other_id, other_tokens in token_sets:
            union = tokens | other_tokens
            score = len(tokens & other_tokens) / len(union) if union else 0.0
            if score >= threshold:
                matches.append({"id": other_id, "jaccard": round(score, 3)})
        if matches:
            clusters.append({"id": str(row.get("id", "")), "instruction": instruction, "matches": matches[:5]})
        token_sets.append((str(row.get("id", "")), tokens))
    return clusters[:50]


def analyze_sft_jsonl(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    rows, errors = _read_jsonl(path)
    instructions = [str(row.get("instruction", "")) for row in rows]
    outputs = [str(row.get("output", "")) for row in rows]
    instruction_counts = Counter(text.strip().lower() for text in instructions if text.strip())
    duplicate_instructions = {text: count for text, count in instruction_counts.items() if count > 1}
    first_three_outputs = Counter(" ".join(_words(output)[:3]).lower() for output in outputs if _words(output))
    opening_phrases = Counter(" ".join(_words(output)[:4]).lower() for output in outputs if len(_words(output)) >= 4)
    common_names = Counter()
    for text in instructions + outputs:
        for match in re.findall(r"\b[A-Z][a-z]{2,}\b", text):
            if match not in {"The", "This", "That", "Then", "When", "Write", "Explain", "Create", "Make"}:
                common_names[match] += 1
    likely_templated = []
    for row in rows:
        output_words = [word.lower() for word in _words(str(row.get("output", "")))]
        if len(output_words) >= 20 and len(set(output_words)) / len(output_words) < 0.32:
            likely_templated.append({"id": str(row.get("id", "")), "reason": "low_unique_word_ratio"})
        elif output_words[:3] and first_three_outputs[" ".join(output_words[:3])] >= 5:
            likely_templated.append({"id": str(row.get("id", "")), "reason": "common_output_opening"})
    by_category: dict[str, list[int]] = defaultdict(list)
    for row in rows:
        by_category[str(row.get("task_type", "<missing>"))].append(len(_words(str(row.get("output", "")))))
    return {
        "path": str(path),
        "total_rows": len(rows),
        "malformed_rows": len(errors),
        "malformed_samples": errors[:10],
        "category_counts": dict(sorted(Counter(str(row.get("task_type", "<missing>")) for row in rows).items())),
        "instruction_word_stats": _stats([len(_words(text)) for text in instructions]),
        "output_word_stats": _stats([len(_words(text)) for text in outputs]),
        "output_word_stats_by_category": {key: _stats(values) for key, values in sorted(by_category.items())},
        "duplicate_instruction_count": sum(count - 1 for count in duplicate_instructions.values()),
        "duplicate_instruction_samples": list(duplicate_instructions.items())[:20],
        "near_duplicate_instruction_clusters": _near_duplicate_clusters(rows),
        "repeated_opening_phrases": opening_phrases.most_common(20),
        "most_common_names": common_names.most_common(30),
        "most_common_first_3_output_words": first_three_outputs.most_common(20),
        "likely_templated_examples": likely_templated[:50],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze raw or processed SFT JSONL data for duplicates and templating.")
    parser.add_argument("path")
    parser.add_argument("--out", default=None, help="Optional JSON report path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = analyze_sft_jsonl(args.path)
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
