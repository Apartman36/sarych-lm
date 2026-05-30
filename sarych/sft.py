from __future__ import annotations

import hashlib
import json
import random
import re
import string
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from sarych.tokenizer_bpe import SarychBPETokenizer
from sarych.utils import ensure_dir, get_git_commit

USER_MARKER = "<|user|>"
ASSISTANT_MARKER = "<|assistant|>"
EOT_TOKEN = "<|endoftext|>"
IGNORE_INDEX = -100

ALLOWED_SFT_TASK_TYPES = {
    "story_writing",
    "story_continuation",
    "explanation_for_children",
    "simple_qa",
    "dialogue",
    "summarization",
    "simple_reasoning",
    "structured_output",
    "creative_generation",
}

REQUIRED_RAW_FIELDS = {"id", "source", "task_type", "instruction", "input", "output", "language", "metadata"}
REJECT_FILES = {
    "malformed": "malformed.jsonl",
    "filtered": "filtered.jsonl",
    "duplicates": "duplicates.jsonl",
    "too_long": "too_long.jsonl",
}


@dataclass(frozen=True)
class ValidationResult:
    ok: bool
    reason: str | None = None
    detail: str | None = None


@dataclass(frozen=True)
class SFTFeatures:
    input_ids: list[int]
    labels: list[int]


def validate_raw_sft_row(row: Any) -> ValidationResult:
    if not isinstance(row, dict):
        return ValidationResult(False, "not_object")
    missing = sorted(REQUIRED_RAW_FIELDS - set(row))
    if missing:
        return ValidationResult(False, "missing_fields", ",".join(missing))
    if row["task_type"] not in ALLOWED_SFT_TASK_TYPES:
        return ValidationResult(False, "invalid_task_type")
    if row["language"] != "en":
        return ValidationResult(False, "invalid_language")
    if not isinstance(row["instruction"], str) or not row["instruction"].strip():
        return ValidationResult(False, "empty_instruction")
    if not isinstance(row["input"], str):
        return ValidationResult(False, "invalid_input")
    if not isinstance(row["output"], str) or not row["output"].strip():
        return ValidationResult(False, "empty_output")
    if not isinstance(row["metadata"], dict):
        return ValidationResult(False, "invalid_metadata")
    return ValidationResult(True)


def validate_scored_sft_row(row: Any) -> ValidationResult:
    if not isinstance(row, dict):
        return ValidationResult(False, "not_object")
    for field in ("id", "example_id", "scores", "judge"):
        if field not in row:
            return ValidationResult(False, "missing_fields", field)
    if not isinstance(row["scores"], dict):
        return ValidationResult(False, "invalid_scores")
    for name, value in row["scores"].items():
        if not isinstance(name, str) or not isinstance(value, (int, float)):
            return ValidationResult(False, "invalid_score_value")
    if not isinstance(row["judge"], dict):
        return ValidationResult(False, "invalid_judge")
    return ValidationResult(True)


def format_instruct_prompt(instruction: str, input_text: str = "") -> str:
    body = instruction.strip()
    cleaned_input = input_text.strip()
    if cleaned_input:
        body = f"{body}\n\n{cleaned_input}"
    return f"{USER_MARKER}\n{body}\n\n{ASSISTANT_MARKER}\n"


def format_sft_text(row: dict[str, Any], *, include_output: bool = True) -> str:
    prompt = format_instruct_prompt(str(row["instruction"]), str(row.get("input", "")))
    if not include_output:
        return prompt
    return prompt + str(row["output"]).strip() + EOT_TOKEN


def build_sft_features(row: dict[str, Any], tokenizer: SarychBPETokenizer, *, max_seq_len: int) -> SFTFeatures:
    prompt_ids = tokenizer.encode(format_sft_text(row, include_output=False))
    output_ids = tokenizer.encode(str(row["output"]).strip() + EOT_TOKEN)
    if not prompt_ids:
        raise ValueError(f"SFT example {row.get('id', '<unknown>')} produced an empty prompt.")
    if not output_ids:
        raise ValueError(f"SFT example {row.get('id', '<unknown>')} produced an empty output.")
    input_ids = prompt_ids + output_ids[:-1]
    if len(input_ids) > max_seq_len:
        raise ValueError(f"SFT example {row.get('id', '<unknown>')} has {len(input_ids)} tokens; max is {max_seq_len}.")
    labels = [IGNORE_INDEX] * (len(prompt_ids) - 1) + output_ids
    return SFTFeatures(input_ids=input_ids, labels=labels)


class SFTJsonlDataset:
    def __init__(
        self,
        path: str | Path,
        tokenizer: SarychBPETokenizer,
        *,
        max_seq_len: int,
        seed: int = 1337,
    ) -> None:
        self.path = Path(path)
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.rng = random.Random(seed)
        self.pad_id = tokenizer.token_to_id("<|pad|>")
        if self.pad_id is None:
            self.pad_id = tokenizer.token_to_id(EOT_TOKEN) or 0
        self.examples = self._load_examples()
        if not self.examples:
            raise ValueError(f"SFT dataset has no usable examples: {self.path}")

    def _load_examples(self) -> list[SFTFeatures]:
        examples: list[SFTFeatures] = []
        with self.path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                validation = validate_raw_sft_row(row)
                if not validation.ok:
                    raise ValueError(f"Invalid SFT row in {self.path}:{line_number}: {validation.reason}")
                examples.append(build_sft_features(row, self.tokenizer, max_seq_len=self.max_seq_len))
        return examples

    def state_dict(self) -> dict[str, Any]:
        return {"rng_state": self.rng.getstate()}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if state and "rng_state" in state:
            self.rng.setstate(state["rng_state"])

    def get_batch(self, *, batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        input_batch: list[list[int]] = []
        label_batch: list[list[int]] = []
        for _ in range(batch_size):
            features = self.examples[self.rng.randrange(len(self.examples))]
            input_ids = features.input_ids[: self.max_seq_len]
            labels = features.labels[: self.max_seq_len]
            pad_count = self.max_seq_len - len(input_ids)
            input_batch.append(input_ids + [self.pad_id] * pad_count)
            label_batch.append(labels + [IGNORE_INDEX] * pad_count)
        x = torch.tensor(input_batch, dtype=torch.long, device=device)
        y = torch.tensor(label_batch, dtype=torch.long, device=device)
        return x, y


def _read_jsonl(path: Path) -> tuple[list[dict[str, Any]], list[tuple[str, str]]]:
    rows: list[dict[str, Any]] = []
    errors: list[tuple[str, str]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append((line.rstrip("\n"), f"line_{line_number}:json_decode:{exc.msg}"))
                continue
            rows.append(row)
    return rows, errors


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _word_count(text: str) -> int:
    return len(re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", text))


def _printable_ascii_ok(text: str) -> bool:
    if not text:
        return False
    printable = set(string.printable)
    printable_ratio = sum(ch in printable or ch.isspace() for ch in text) / len(text)
    ascii_ratio = sum(ord(ch) < 128 for ch in text) / len(text)
    return printable_ratio >= 0.98 and ascii_ratio >= 0.90


def _has_repeated_exact_sentence(text: str) -> bool:
    sentences = [s.strip().lower() for s in re.split(r"[.!?]+", text) if s.strip()]
    return len(sentences) != len(set(sentences))


def _has_obvious_loop(words: list[str]) -> bool:
    if len(words) < 12:
        return False
    for width in (2, 3, 4):
        seen: Counter[tuple[str, ...]] = Counter(tuple(words[i : i + width]) for i in range(0, len(words) - width + 1))
        if seen and max(seen.values()) >= 5:
            return True
    return False


def _passes_content_filters(row: dict[str, Any]) -> ValidationResult:
    instruction = str(row["instruction"])
    output = str(row["output"])
    if _word_count(instruction) < 4:
        return ValidationResult(False, "instruction_too_short")
    output_words = re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", output.lower())
    if len(output_words) < 20:
        return ValidationResult(False, "output_too_short")
    if not _printable_ascii_ok(instruction + "\n" + str(row.get("input", "")) + "\n" + output):
        return ValidationResult(False, "not_english_printable")
    if _has_repeated_exact_sentence(output):
        return ValidationResult(False, "repeated_sentence")
    if len(output_words) >= 20 and len(set(output_words)) / len(output_words) < 0.25:
        return ValidationResult(False, "low_unique_word_ratio")
    if _has_obvious_loop(output_words):
        return ValidationResult(False, "obvious_loop")
    return ValidationResult(True)


def _normalize_for_dedup(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _dedup_key(row: dict[str, Any]) -> str:
    normalized = _normalize_for_dedup(str(row["instruction"])) + "\n" + _normalize_for_dedup(str(row["output"]))
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _instruction_tokens(row: dict[str, Any]) -> set[str]:
    return set(re.findall(r"[a-z]+", str(row["instruction"]).lower()))


def _is_near_duplicate_instruction(row: dict[str, Any], seen_instructions: list[set[str]]) -> bool:
    tokens = _instruction_tokens(row)
    if len(tokens) < 4:
        return False
    for seen in seen_instructions:
        union = tokens | seen
        if union and len(tokens & seen) / len(union) >= 1.0:
            return True
    return False


def _read_scores(scored_path: str | Path | None) -> dict[str, dict[str, Any]]:
    if scored_path is None:
        return {}
    path = Path(scored_path)
    if not path.exists():
        return {}
    rows, _ = _read_jsonl(path)
    scores: dict[str, dict[str, Any]] = {}
    for row in rows:
        if validate_scored_sft_row(row).ok:
            scores[str(row["example_id"])] = row
    return scores


def _passes_score_filter(row: dict[str, Any], scores_by_example_id: dict[str, dict[str, Any]]) -> ValidationResult:
    score_row = scores_by_example_id.get(str(row["id"]))
    if score_row is None:
        return ValidationResult(True)
    scores = score_row["scores"]
    safety = scores.get("safety")
    if safety is not None and safety != 5:
        return ValidationResult(False, "score_safety_below_5")
    values = [float(value) for value in scores.values()]
    if not values:
        return ValidationResult(False, "score_missing_values")
    if all(value >= 4.0 for value in values) or (sum(values) / len(values) >= 4.0):
        return ValidationResult(True)
    return ValidationResult(False, "score_below_threshold")


def _reject_record(row: Any, reason: str, detail: str | None = None) -> dict[str, Any]:
    return {"reason": reason, "detail": detail, "row": row}


def _split_stratified(rows: list[dict[str, Any]], *, val_ratio: float, seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["task_type"])].append(row)
    train_rows: list[dict[str, Any]] = []
    val_rows: list[dict[str, Any]] = []
    rng = random.Random(seed)
    for task_type in sorted(grouped):
        group = list(grouped[task_type])
        rng.shuffle(group)
        if len(group) <= 1 or val_ratio <= 0:
            val_count = 0
        else:
            val_count = min(len(group) - 1, max(1, round(len(group) * val_ratio)))
        val_rows.extend(group[:val_count])
        train_rows.extend(group[val_count:])
    train_rows.sort(key=lambda row: str(row["id"]))
    val_rows.sort(key=lambda row: str(row["id"]))
    return train_rows, val_rows


def _category_distribution(train_rows: list[dict[str, Any]], val_rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    distribution: dict[str, dict[str, int]] = {}
    for task_type in sorted(ALLOWED_SFT_TASK_TYPES):
        train_count = sum(1 for row in train_rows if row["task_type"] == task_type)
        val_count = sum(1 for row in val_rows if row["task_type"] == task_type)
        if train_count or val_count:
            distribution[task_type] = {"train": train_count, "val": val_count, "total": train_count + val_count}
    return distribution


def build_sft_splits(
    *,
    raw_path: str | Path,
    scored_path: str | Path | None,
    tokenizer_path: str | Path,
    train_path: str | Path,
    val_path: str | Path,
    rejected_dir: str | Path,
    val_ratio: float,
    seed: int,
    max_seq_len: int,
    manifest_path: str | Path | None = None,
) -> dict[str, Any]:
    raw_path = Path(raw_path)
    tokenizer_path = Path(tokenizer_path)
    train_path = Path(train_path)
    val_path = Path(val_path)
    rejected_dir = ensure_dir(rejected_dir)
    manifest_path = Path(manifest_path) if manifest_path is not None else None

    tokenizer = SarychBPETokenizer.from_file(tokenizer_path)
    rows, parse_errors = _read_jsonl(raw_path)
    scores_by_example_id = _read_scores(scored_path)

    rejected: dict[str, list[dict[str, Any]]] = {key: [] for key in REJECT_FILES}
    for raw_line, detail in parse_errors:
        rejected["malformed"].append(_reject_record(raw_line, "json_decode_error", detail))

    accepted: list[dict[str, Any]] = []
    seen_hashes: set[str] = set()
    seen_instruction_sets: list[set[str]] = []

    for row in rows:
        structural = validate_raw_sft_row(row)
        if not structural.ok:
            rejected["malformed"].append(_reject_record(row, structural.reason or "malformed", structural.detail))
            continue

        try:
            build_sft_features(row, tokenizer, max_seq_len=max_seq_len)
        except ValueError as exc:
            rejected["too_long"].append(_reject_record(row, "too_long", str(exc)))
            continue

        content = _passes_content_filters(row)
        if not content.ok:
            rejected["filtered"].append(_reject_record(row, content.reason or "filtered", content.detail))
            continue

        key = _dedup_key(row)
        if key in seen_hashes or _is_near_duplicate_instruction(row, seen_instruction_sets):
            rejected["duplicates"].append(_reject_record(row, "duplicate"))
            continue

        score_result = _passes_score_filter(row, scores_by_example_id)
        if not score_result.ok:
            rejected["filtered"].append(_reject_record(row, score_result.reason or "score_filtered", score_result.detail))
            continue

        seen_hashes.add(key)
        seen_instruction_sets.append(_instruction_tokens(row))
        accepted.append(row)

    train_rows, val_rows = _split_stratified(accepted, val_ratio=val_ratio, seed=seed)
    _write_jsonl(train_path, train_rows)
    _write_jsonl(val_path, val_rows)
    for reject_type, filename in REJECT_FILES.items():
        _write_jsonl(rejected_dir / filename, rejected[reject_type])

    manifest = {
        "input_paths": {
            "raw_path": str(raw_path),
            "scored_path": str(scored_path) if scored_path is not None else None,
        },
        "output_paths": {
            "train_path": str(train_path),
            "val_path": str(val_path),
            "rejected_dir": str(rejected_dir),
        },
        "total_raw_rows": len(rows) + len(parse_errors),
        "accepted_rows": len(accepted),
        "rejected_counts": {key: len(value) for key, value in rejected.items()},
        "train_count": len(train_rows),
        "val_count": len(val_rows),
        "category_distribution": _category_distribution(train_rows, val_rows),
        "tokenizer_path": str(tokenizer_path),
        "max_sequence_length": max_seq_len,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": get_git_commit(),
        "seed": seed,
        "val_ratio": val_ratio,
    }
    if manifest_path is not None:
        ensure_dir(manifest_path.parent)
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest
