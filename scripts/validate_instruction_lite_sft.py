from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


TASK_TYPE_BY_CATEGORY = {
    "identity_chat": "dialogue",
    "simple_explanation": "explanation_for_children",
    "simple_list": "structured_output",
    "simple_qa": "simple_qa",
    "simple_reasoning": "simple_reasoning",
    "story_request": "story_writing",
    "story_continuation": "story_continuation",
    "safety_refusal": "dialogue",
    "summarization_rewrite": "summarization",
    "emotional_support_kindness": "dialogue",
}

REQUIRED_FIELDS = {"id", "seed_id", "source", "task_type", "instruction", "input", "output", "language", "metadata"}
STORY_STARTS = ("once upon a time", "once there was", "one day", "long ago")
BLOCKED_DOMAIN_TERMS = {
    "adult",
    "casino",
    "gambling",
    "politics",
    "election",
    "president",
    "diagnosis",
    "medicine dose",
    "lawyer",
    "lawsuit",
    "investment",
    "stock market",
    "mortgage",
    "python",
    "javascript",
    "code",
    "script",
    "function",
    "algorithm",
}

WORD_LIMITS = {
    "identity_chat": (8, 50),
    "simple_explanation": (15, 80),
    "simple_list": (15, 90),
    "simple_qa": (3, 40),
    "simple_reasoning": (10, 80),
    "story_request": (40, 160),
    "story_continuation": (30, 140),
    "safety_refusal": (10, 60),
    "emotional_support_kindness": (10, 80),
    "summarization_rewrite": (10, 80),
}


def _read_jsonl_with_errors(path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append({"id": None, "reason": "invalid_json", "detail": f"line {line_number}: {exc.msg}", "record": line.rstrip("\n")})
                continue
            if not isinstance(value, dict):
                errors.append({"id": None, "reason": "not_object", "detail": f"line {line_number}", "record": value})
            else:
                rows.append(value)
    return rows, errors


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _words(text: str) -> list[str]:
    return re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", text.lower())


def _word_count(text: str) -> int:
    return len(_words(text))


def _sentences(text: str) -> list[str]:
    return [re.sub(r"\s+", " ", item).strip().lower() for item in re.split(r"[.!?]+", text) if item.strip()]


def _starts_with_story(text: str) -> bool:
    lowered = text.strip().lower()
    return any(lowered.startswith(phrase) for phrase in STORY_STARTS)


def _has_as_ai(text: str) -> bool:
    return "as an ai" in text.lower()


def _has_chatty_opener(text: str, category: str) -> bool:
    return category not in {"identity_chat", "emotional_support_kindness"} and bool(
        re.match(r"^\s*(sure!?|of course!?)\b", text.lower())
    )


def _has_blocked_domain_terms(*parts: str) -> bool:
    text = " ".join(parts).lower()
    return any(re.search(rf"\b{re.escape(term)}\b", text) for term in BLOCKED_DOMAIN_TERMS)


def _has_url(text: str) -> bool:
    return bool(re.search(r"https?://|www\.", text.lower()))


def _has_markdown_table(text: str) -> bool:
    lines = [line for line in text.splitlines() if line.strip()]
    return any("|" in line for line in lines) and any(re.search(r"\|\s*-{3,}\s*\|", line) for line in lines)


def _has_repeated_sentence(text: str) -> bool:
    sentences = _sentences(text)
    counts = Counter(sentences)
    return any(count >= 2 for sentence, count in counts.items() if len(sentence.split()) >= 3)


def _has_loop(text: str) -> bool:
    words = _words(text)
    if len(words) < 10:
        return False
    for width in (3, 4, 5):
        grams = Counter(tuple(words[index : index + width]) for index in range(len(words) - width + 1))
        if any(count >= 3 for count in grams.values()):
            return True
    return False


def _has_weird_nonword(text: str) -> bool:
    for word in re.findall(r"[A-Za-z]+", text):
        lowered = word.lower()
        if len(lowered) >= 14:
            return True
        if re.search(r"[bcdfghjklmnpqrstvwxyz]{5,}", lowered):
            return True
    return False


def _has_list_format(text: str) -> bool:
    markers = re.findall(r"(?m)^\s*(?:[-*]|\d+[.)])\s+", text)
    if len(markers) >= 2:
        return True
    if text.count(";") >= 1 or text.count("\n") >= 1:
        return len([part for part in re.split(r"[;\n]", text) if part.strip()]) >= 2
    if text.count(",") >= 2:
        return True
    return False


def _identity_ok(text: str) -> bool:
    lowered = text.lower()
    return bool(re.search(r"\b(i am|i'm|my name is|i can|i do not|i don't)\b", lowered)) and any(
        term in lowered for term in ("sarych", "helper", "computer", "help", "answer")
    )


def _causal_explanation_ok(text: str) -> bool:
    first_sentence = _sentences(text)[:1]
    if not first_sentence:
        return False
    first = first_sentence[0]
    causal = ("because", "so", "helps", "need", "needs", "means", "makes")
    return any(word in first for word in causal) and not _starts_with_story(first)


def _reasoning_ok(text: str) -> bool:
    lowered = text.lower()
    return any(term in lowered for term in ("so", "because", "left", "bigger", "yes", "no", "first", "then"))


def _safety_ok(text: str) -> bool:
    lowered = text.lower()
    return bool(re.search(r"\b(can't|cannot|do not|don't|not safe|ask)\b", lowered)) and any(
        term in lowered for term in ("grown-up", "grown up", "trusted", "adult", "safer", "safe")
    )


def _support_ok(text: str) -> bool:
    lowered = text.lower()
    warm = any(term in lowered for term in ("sorry", "sad", "breath", "trusted", "kind", "feel", "friend"))
    return warm and not _starts_with_story(text)


def _ngram_set(text: str, width: int = 5) -> set[tuple[str, ...]]:
    words = _words(text)
    return {tuple(words[index : index + width]) for index in range(max(0, len(words) - width + 1))}


def _near_duplicate(text: str, previous: list[set[tuple[str, ...]]]) -> bool:
    grams = _ngram_set(text)
    if len(grams) < 4:
        return False
    for seen in previous:
        union = grams | seen
        if union and len(grams & seen) / len(union) >= 0.82:
            return True
    return False


def _reject(row: Any, reason: str, detail: str | None = None) -> dict[str, Any]:
    return {
        "id": row.get("id") if isinstance(row, dict) else None,
        "seed_id": row.get("seed_id") if isinstance(row, dict) else None,
        "source": row.get("source") if isinstance(row, dict) else None,
        "task_type": row.get("task_type") if isinstance(row, dict) else None,
        "category": row.get("metadata", {}).get("category") if isinstance(row, dict) and isinstance(row.get("metadata"), dict) else None,
        "reason": reason,
        "detail": detail,
        "record": row,
    }


def validate_row(row: dict[str, Any], *, seen_exact: set[str], seen_category_ngrams: dict[str, list[set[tuple[str, ...]]]]) -> tuple[bool, str | None, str | None]:
    missing = sorted(REQUIRED_FIELDS - set(row))
    if missing:
        return False, "missing_fields", ",".join(missing)
    if row.get("language") != "en":
        return False, "invalid_language", None
    if not isinstance(row.get("metadata"), dict):
        return False, "invalid_metadata", None
    category = str(row["metadata"].get("category", ""))
    if category not in TASK_TYPE_BY_CATEGORY:
        return False, "invalid_category", category
    if row.get("task_type") != TASK_TYPE_BY_CATEGORY[category]:
        return False, "invalid_task_type", f"expected {TASK_TYPE_BY_CATEGORY[category]}"
    instruction = str(row.get("instruction", "")).strip()
    input_text = str(row.get("input", ""))
    output = str(row.get("output", "")).strip()
    if not instruction:
        return False, "empty_instruction", None
    if not output:
        return False, "empty_output", None
    if _has_as_ai(output):
        return False, "as_ai_phrase", None
    if _has_chatty_opener(output, category):
        return False, "chatty_opener", None
    if _has_blocked_domain_terms(instruction, input_text, output):
        return False, "blocked_domain_term", None
    if _has_url(output) or _has_url(instruction):
        return False, "url", None
    if _has_markdown_table(output):
        return False, "markdown_table", None
    if _has_repeated_sentence(output):
        return False, "repeated_sentence", None
    if _has_loop(output):
        return False, "obvious_loop", None
    if _has_weird_nonword(output):
        return False, "weird_nonword", None
    minimum, maximum = WORD_LIMITS[category]
    count = _word_count(output)
    if count < minimum:
        return False, "too_short", f"{count} < {minimum}"
    if count > maximum:
        return False, "too_long", f"{count} > {maximum}"
    exact_key = re.sub(r"\s+", " ", instruction.lower()) + "\n" + re.sub(r"\s+", " ", output.lower())
    if exact_key in seen_exact:
        return False, "duplicate_instruction_output", None
    if _near_duplicate(output, seen_category_ngrams[category]):
        return False, "near_duplicate_output", None

    if category == "identity_chat":
        if _starts_with_story(output):
            return False, "story_collapse", None
        if not _identity_ok(output):
            return False, "identity_fail", None
    elif category == "simple_explanation":
        if _starts_with_story(output):
            return False, "story_collapse", None
        if not _causal_explanation_ok(output):
            return False, "weak_explanation", None
    elif category == "simple_list":
        if _starts_with_story(output):
            return False, "story_collapse", None
        if not _has_list_format(output):
            return False, "list_format_fail", None
    elif category == "story_continuation":
        if not str(row.get("input", "")).strip():
            return False, "missing_input", None
    elif category == "safety_refusal":
        if not _safety_ok(output):
            return False, "unsafe_refusal", None
    elif category == "emotional_support_kindness":
        if not _support_ok(output):
            return False, "weak_support", None
    elif category == "summarization_rewrite":
        if not str(row.get("input", "")).strip():
            return False, "missing_input", None
    elif category == "simple_reasoning":
        if _starts_with_story(output):
            return False, "story_collapse", None
        if not _reasoning_ok(output):
            return False, "weak_reasoning", None
    elif category == "simple_qa":
        if _starts_with_story(output):
            return False, "story_collapse", None

    return True, None, None


def validate_instruction_lite_sft(
    input_path: str | Path,
    accepted_path: str | Path,
    rejected_path: str | Path,
    manifest_path: str | Path,
) -> dict[str, Any]:
    input_path = Path(input_path)
    accepted_path = Path(accepted_path)
    rejected_path = Path(rejected_path)
    manifest_path = Path(manifest_path)
    rows, parse_rejections = _read_jsonl_with_errors(input_path)
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = list(parse_rejections)
    seen_exact: set[str] = set()
    seen_category_ngrams: dict[str, list[set[tuple[str, ...]]]] = defaultdict(list)
    reason_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    category_counts: Counter[str] = Counter()

    for rejection in parse_rejections:
        reason_counts[str(rejection["reason"])] += 1

    for row in rows:
        ok, reason, detail = validate_row(row, seen_exact=seen_exact, seen_category_ngrams=seen_category_ngrams)
        if not ok:
            rejected.append(_reject(row, reason or "rejected", detail))
            reason_counts[reason or "rejected"] += 1
            continue
        instruction = str(row["instruction"]).strip()
        output = str(row["output"]).strip()
        category = str(row["metadata"]["category"])
        seen_exact.add(re.sub(r"\s+", " ", instruction.lower()) + "\n" + re.sub(r"\s+", " ", output.lower()))
        seen_category_ngrams[category].append(_ngram_set(output))
        accepted.append(row)
        source_counts[str(row["source"])] += 1
        category_counts[category] += 1

    _write_jsonl(accepted_path, accepted)
    _write_jsonl(rejected_path, rejected)
    manifest = {
        "validator": "validate_instruction_lite_sft_v0_4",
        "input_path": str(input_path),
        "accepted_path": str(accepted_path),
        "rejected_path": str(rejected_path),
        "total_rows": len(rows) + len(parse_rejections),
        "accepted_rows": len(accepted),
        "rejected_rows": len(rejected),
        "category_counts": dict(sorted(category_counts.items())),
        "source_counts": dict(sorted(source_counts.items())),
        "rejection_reasons": dict(sorted(reason_counts.items())),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate teacher-generated v0.4 instruction-lite SFT JSONL.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--accepted", required=True)
    parser.add_argument("--rejected", required=True)
    parser.add_argument("--manifest", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = validate_instruction_lite_sft(args.input, args.accepted, args.rejected, args.manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
