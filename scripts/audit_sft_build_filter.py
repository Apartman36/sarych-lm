from __future__ import annotations

import argparse
import json
import random
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


INSTRUCTION_LITE_SOURCE = "xiaomi_instruction_lite_v0_4"
REJECTION_FILES = ("duplicates", "filtered", "malformed", "too_long")
TARGET_CATEGORIES = (
    "summarization",
    "simple_qa",
    "structured_output",
    "simple_reasoning",
    "safety_refusal",
    "identity_chat",
    "story_request",
    "story_continuation",
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            if not line.strip():
                continue
            rows.append(json.loads(line))
    return rows


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def _row_from_rejection(row: dict[str, Any]) -> dict[str, Any]:
    record = row.get("record")
    return record if isinstance(record, dict) else row


def _category(row: dict[str, Any]) -> str:
    metadata = row.get("metadata")
    if isinstance(metadata, dict):
        value = metadata.get("category")
        if isinstance(value, str) and value.strip():
            return value.strip()
    return "UNKNOWN"


def _source(row: dict[str, Any]) -> str:
    return str(row.get("source", "UNKNOWN"))


def _task_type(row: dict[str, Any]) -> str:
    return str(row.get("task_type", "UNKNOWN"))


def _row_id(row: dict[str, Any]) -> str:
    return str(row.get("id", ""))


def _content_fingerprint(row: dict[str, Any]) -> str:
    parts = [
        _source(row),
        _task_type(row),
        _normalize_key(str(row.get("instruction", ""))),
        _normalize_key(str(row.get("input", ""))),
        _normalize_key(str(row.get("output", ""))),
    ]
    return "\n".join(parts)


def _normalize_key(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _normalize_display(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def _word_count(text: str) -> int:
    return len(re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", text))


def _percentiles(values: list[int]) -> dict[str, int | None]:
    if not values:
        return {"min": None, "median": None, "p90": None}
    ordered = sorted(values)
    p90_index = min(len(ordered) - 1, int(0.9 * (len(ordered) - 1)))
    return {"min": ordered[0], "median": int(statistics.median(ordered)), "p90": ordered[p90_index]}


def _counts(counter: Counter[str]) -> dict[str, int]:
    return dict(sorted(counter.items()))


def _count_by(rows: list[dict[str, Any]], getter) -> dict[str, int]:
    return _counts(Counter(getter(row) for row in rows))


def _load_rejections(rejected_dir: Path) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    by_file: dict[str, list[dict[str, Any]]] = {}
    flattened: list[dict[str, Any]] = []
    for name in REJECTION_FILES:
        rows = _read_jsonl(rejected_dir / f"{name}.jsonl")
        by_file[name] = rows
        for wrapper in rows:
            row = _row_from_rejection(wrapper)
            flattened.append({"file": name, "wrapper": wrapper, "record": row})
    return by_file, flattened


def _duplicate_counts(rows: list[dict[str, Any]]) -> tuple[dict[str, dict[str, int]], dict[str, list[dict[str, Any]]]]:
    specs = {
        "instruction_only": lambda row: str(row.get("instruction", "")),
        "output_only": lambda row: str(row.get("output", "")),
        "instruction_output_pair": lambda row: f"{row.get('instruction', '')}\n{row.get('output', '')}",
        "output_within_task_type": lambda row: f"{_task_type(row)}\n{row.get('output', '')}",
        "output_across_all_tasks": lambda row: str(row.get("output", "")),
    }
    counts: dict[str, Counter[str]] = {name: Counter() for name in specs}
    display: dict[str, dict[str, str]] = {name: {} for name in specs}
    members: dict[str, dict[str, list[dict[str, Any]]]] = {name: defaultdict(list) for name in specs}

    for row in rows:
        for name, getter in specs.items():
            raw = getter(row)
            key = _normalize_key(raw)
            if not key:
                continue
            counts[name][key] += 1
            if name == "output_within_task_type":
                display[name].setdefault(key, f"{_task_type(row)} :: {_normalize_display(str(row.get('output', '')))}")
            elif name == "instruction_output_pair":
                display[name].setdefault(
                    key,
                    f"{_normalize_display(str(row.get('instruction', '')))} || {_normalize_display(str(row.get('output', '')))}",
                )
            else:
                display[name].setdefault(key, _normalize_display(str(raw)))
            members[name][key].append(row)

    duplicate_maps: dict[str, dict[str, int]] = {}
    for name, counter in counts.items():
        duplicate_maps[name] = {
            display[name][key]: value for key, value in sorted(counter.items(), key=lambda item: (-item[1], display[name][item[0]])) if value > 1
        }
    return duplicate_maps, members["output_across_all_tasks"]


def _top_duplicate_clusters(members: dict[str, list[dict[str, Any]]], limit: int = 10) -> list[dict[str, Any]]:
    clusters: list[dict[str, Any]] = []
    for key, rows in members.items():
        if len(rows) <= 1:
            continue
        clusters.append(
            {
                "normalized_text": _normalize_display(str(rows[0].get("output", ""))),
                "count": len(rows),
                "task_type_distribution": _count_by(rows, _task_type),
                "category_distribution": _count_by(rows, _category),
                "samples": [_sample_row(row) for row in rows[:3]],
            }
        )
    clusters.sort(key=lambda item: (-int(item["count"]), str(item["normalized_text"])))
    return clusters[:limit]


def _sample_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": row.get("id"),
        "source": row.get("source"),
        "task_type": row.get("task_type"),
        "category": _category(row),
        "instruction": row.get("instruction"),
        "output": row.get("output"),
    }


def _sample_rejections(rows: list[dict[str, Any]], *, sample_per_reason: int, seed: int) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in rows:
        reason = str(item["wrapper"].get("reason") or item["file"])
        grouped[reason].append(item["record"])
    rng = random.Random(seed)
    samples: dict[str, list[dict[str, Any]]] = {}
    for reason in sorted(grouped):
        group = list(grouped[reason])
        rng.shuffle(group)
        samples[reason] = [_sample_row(row) for row in group[:sample_per_reason]]
    return samples


def _length_distributions(rows: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, int | None]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[_task_type(row)].append(row)
    result: dict[str, dict[str, dict[str, int | None]]] = {}
    for task_type, group in sorted(grouped.items()):
        result[task_type] = {
            "instruction_words": _percentiles([_word_count(str(row.get("instruction", ""))) for row in group]),
            "output_words": _percentiles([_word_count(str(row.get("output", ""))) for row in group]),
        }
    return result


def _jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def _char_ngrams(text: str, n: int = 5) -> set[str]:
    normalized = _normalize_key(text)
    if len(normalized) <= n:
        return {normalized} if normalized else set()
    return {normalized[index : index + n] for index in range(len(normalized) - n + 1)}


def _lexical_similarity(rows: list[dict[str, Any]], threshold: float = 0.82) -> dict[str, Any]:
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_task[_task_type(row)].append(row)
    summary: dict[str, Any] = {}
    for task_type, group in sorted(by_task.items()):
        token_pairs = 0
        char_pairs = 0
        for index, left in enumerate(group):
            left_tokens = set(re.findall(r"[a-z]+", str(left.get("output", "")).lower()))
            left_grams = _char_ngrams(str(left.get("output", "")))
            for right in group[index + 1 :]:
                right_tokens = set(re.findall(r"[a-z]+", str(right.get("output", "")).lower()))
                right_grams = _char_ngrams(str(right.get("output", "")))
                if _jaccard(left_tokens, right_tokens) >= threshold:
                    token_pairs += 1
                if _jaccard(left_grams, right_grams) >= threshold:
                    char_pairs += 1
        summary[task_type] = {"token_jaccard_pairs_ge_0_82": token_pairs, "char_5gram_jaccard_pairs_ge_0_82": char_pairs}
    openings = Counter(_normalize_key(str(row.get("output", "")))[:48] for row in rows if row.get("output"))
    summary["common_openings"] = {key: count for key, count in openings.most_common(10) if count > 1}
    return summary


def _semantic_similarity(rows: list[dict[str, Any]], mode: str, model_name: str, threshold: float) -> dict[str, Any]:
    if mode == "none":
        return {"enabled": False, "reason": "semantic embeddings disabled"}
    try:
        from sentence_transformers import SentenceTransformer, util  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency path
        return {"enabled": False, "reason": f"sentence-transformers unavailable: {exc}"}

    model = SentenceTransformer(model_name)
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_task[_task_type(row)].append(row)
    result: dict[str, Any] = {"enabled": True, "model": model_name, "threshold": threshold, "by_task_type": {}}
    for task_type, group in sorted(by_task.items()):  # pragma: no cover - optional dependency path
        texts = [str(row.get("output", "")) for row in group]
        embeddings = model.encode(texts, convert_to_tensor=True, normalize_embeddings=True)
        pairs = util.semantic_search(embeddings, embeddings, top_k=min(6, len(texts)))
        seen: set[tuple[int, int]] = set()
        matches: list[dict[str, Any]] = []
        for left_index, hits in enumerate(pairs):
            for hit in hits:
                right_index = int(hit["corpus_id"])
                if left_index >= right_index or float(hit["score"]) < threshold:
                    continue
                key = (left_index, right_index)
                if key in seen:
                    continue
                seen.add(key)
                matches.append(
                    {
                        "score": float(hit["score"]),
                        "left": _sample_row(group[left_index]),
                        "right": _sample_row(group[right_index]),
                    }
                )
        result["by_task_type"][task_type] = {"pair_count": len(matches), "top_pairs": matches[:10]}
    return result


def _conclusion(factory_rows: list[dict[str, Any]], rejected_instruction_lite: list[dict[str, Any]], duplicate_maps: dict[str, dict[str, int]]) -> str:
    if not rejected_instruction_lite:
        return "Mixed evidence"
    rejected_ratio = len(rejected_instruction_lite) / max(1, len(factory_rows))
    exact_pair_duplicates = sum(duplicate_maps["normalized_instruction_output_pair"].values())
    output_duplicates = sum(duplicate_maps["normalized_output_across_all_tasks"].values())
    if rejected_ratio > 0.50 and exact_pair_duplicates < len(rejected_instruction_lite) * 0.25:
        return "Evidence suggests build filter is over-aggressive"
    if exact_pair_duplicates > len(rejected_instruction_lite) * 0.50 or output_duplicates > len(factory_rows):
        return "Evidence suggests data is bad"
    return "Mixed evidence"


def _write_markdown(path: Path, result: dict[str, Any]) -> None:
    lines = [
        "# SFT Build Filter Audit",
        "",
        "## Inputs",
        f"- Raw rows: {sum(result['raw_rows_by_source'].values())}",
        f"- Factory accepted rows: {sum(result['factory_accepted_by_source'].values())}",
        f"- Rejected files: duplicates.jsonl={result['rejected_rows_by_file']['duplicates']}, filtered.jsonl={result['rejected_rows_by_file']['filtered']}, malformed.jsonl={result['rejected_rows_by_file']['malformed']}, too_long.jsonl={result['rejected_rows_by_file']['too_long']}",
        "",
        "## Overlap",
        f"- Factory accepted and build rejected: {result['overlap']['factory_accepted_and_build_rejected']}",
        f"- Every rejected instruction-lite row in factory accepted: {result['overlap']['all_rejected_instruction_lite_in_factory_accepted']}",
        "",
        "## Duplicate Clusters",
    ]
    clusters = result["top_duplicate_clusters"]["output_across_all_tasks"]
    if clusters:
        for cluster in clusters[:10]:
            lines.extend(
                [
                    "",
                    f"### Count {cluster['count']}",
                    f"- Text: {cluster['normalized_text']}",
                    f"- Task types: {cluster['task_type_distribution']}",
                    f"- Categories: {cluster['category_distribution']}",
                ]
            )
            for sample in cluster["samples"]:
                lines.append(f"- Sample {sample['id']}: {sample['instruction']} => {sample['output']}")
    else:
        lines.append("- No duplicate output clusters above count 1.")
    lines.extend(["", "## Samples"])
    for reason, samples in result["samples"].items():
        lines.append(f"### {reason}")
        for sample in samples:
            lines.append(
                f"- {sample['id']} | {sample['source']} | {sample['task_type']} | {sample['category']} | {sample['instruction']} => {sample['output']}"
            )
    lines.extend(["", "## Conclusion", result["conclusion"]])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def audit_sft_build_filter(
    *,
    raw_path: str | Path,
    factory_accepted_path: str | Path,
    build_manifest_path: str | Path,
    rejected_dir: str | Path,
    out_md_path: str | Path,
    out_json_path: str | Path,
    sample_per_reason: int = 10,
    seed: int = 1337,
    semantic_embeddings: str = "none",
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    semantic_threshold: float = 0.92,
) -> dict[str, Any]:
    raw_path = Path(raw_path)
    factory_accepted_path = Path(factory_accepted_path)
    build_manifest_path = Path(build_manifest_path)
    rejected_dir = Path(rejected_dir)

    raw_rows = _read_jsonl(raw_path)
    factory_rows = _read_jsonl(factory_accepted_path)
    manifest = json.loads(build_manifest_path.read_text(encoding="utf-8")) if build_manifest_path.exists() else {}
    rejected_by_file, rejected_items = _load_rejections(rejected_dir)
    rejected_records = [item["record"] for item in rejected_items if isinstance(item["record"], dict)]
    rejected_instruction_lite_items = [item for item in rejected_items if isinstance(item["record"], dict) and _source(item["record"]) == INSTRUCTION_LITE_SOURCE]
    rejected_instruction_lite = [item["record"] for item in rejected_instruction_lite_items]

    factory_ids = {_row_id(row) for row in factory_rows}
    rejected_instruction_lite_ids = {_row_id(row) for row in rejected_instruction_lite}
    factory_fingerprints = {_content_fingerprint(row) for row in factory_rows}
    rejected_instruction_lite_fingerprints = {_content_fingerprint(row) for row in rejected_instruction_lite}
    duplicate_maps, output_members = _duplicate_counts(factory_rows)
    accepted_rows_for_lengths = [
        row
        for row in raw_rows
        if _source(row) == INSTRUCTION_LITE_SOURCE and _content_fingerprint(row) not in rejected_instruction_lite_fingerprints
    ]
    destroyed: dict[str, dict[str, int]] = {}
    for key in TARGET_CATEGORIES:
        rejected_count = sum(1 for row in rejected_instruction_lite if _task_type(row) == key or _category(row) == key)
        factory_count = sum(1 for row in factory_rows if _task_type(row) == key or _category(row) == key)
        if factory_count or rejected_count:
            destroyed[key] = {"factory_accepted": factory_count, "build_rejected": rejected_count}

    result: dict[str, Any] = {
        "paths": {
            "raw": str(raw_path),
            "factory_accepted": str(factory_accepted_path),
            "build_manifest": str(build_manifest_path),
            "rejected_dir": str(rejected_dir),
        },
        "raw_rows_by_source": _count_by(raw_rows, _source),
        "raw_rows_by_task_type": _count_by(raw_rows, _task_type),
        "factory_accepted_by_source": _count_by(factory_rows, _source),
        "factory_accepted_by_task_type": _count_by(factory_rows, _task_type),
        "factory_accepted_by_category": _count_by(factory_rows, _category),
        "build_manifest": {
            "accepted_by_source": manifest.get("accepted_by_source", {}),
            "accepted_by_task_type": manifest.get("accepted_by_task_type", {}),
            "accepted_by_category": manifest.get("accepted_by_category", {}),
        },
        "rejected_rows_by_file": {name: len(rejected_by_file[name]) for name in REJECTION_FILES},
        "rejected_instruction_lite_by_task_type": _count_by(rejected_instruction_lite, _task_type),
        "rejected_instruction_lite_by_category": _count_by(rejected_instruction_lite, _category),
        "overlap": {
            "factory_accepted_and_build_rejected": len(factory_fingerprints & rejected_instruction_lite_fingerprints),
            "factory_accepted_and_build_rejected_by_id": len(factory_ids & rejected_instruction_lite_ids),
            "build_rejected_instruction_lite_rows": len(rejected_instruction_lite_ids),
            "all_rejected_instruction_lite_in_factory_accepted": rejected_instruction_lite_fingerprints <= factory_fingerprints,
        },
        "duplicates": {
            "instruction_only": duplicate_maps["instruction_only"],
            "output_only": duplicate_maps["output_only"],
            "normalized_instruction_output_pair": duplicate_maps["instruction_output_pair"],
            "normalized_output_within_same_task_type": duplicate_maps["output_within_task_type"],
            "normalized_output_across_all_tasks": duplicate_maps["output_across_all_tasks"],
        },
        "top_duplicate_clusters": {"output_across_all_tasks": _top_duplicate_clusters(output_members)},
        "samples": _sample_rejections(rejected_instruction_lite_items, sample_per_reason=sample_per_reason, seed=seed),
        "length_distributions": {
            "accepted_instruction_lite": _length_distributions(accepted_rows_for_lengths),
            "rejected_instruction_lite": _length_distributions(rejected_instruction_lite),
        },
        "destroyed_categories_or_task_types": destroyed,
        "lexical_similarity": _lexical_similarity(factory_rows),
        "semantic_similarity": _semantic_similarity(factory_rows, semantic_embeddings, embedding_model, semantic_threshold),
    }
    result["conclusion"] = _conclusion(factory_rows, rejected_instruction_lite, result["duplicates"])
    _write_markdown(Path(out_md_path), result)
    _write_json(Path(out_json_path), result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit why build_sft_dataset rejects factory-accepted SFT rows.")
    parser.add_argument("--raw", required=True)
    parser.add_argument("--factory-accepted", required=True)
    parser.add_argument("--build-manifest", required=True)
    parser.add_argument("--rejected-dir", required=True)
    parser.add_argument("--out-md", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--sample-per-reason", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--semantic-embeddings", choices=["none", "sentence-transformers"], default="none")
    parser.add_argument("--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--semantic-threshold", type=float, default=0.92)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = audit_sft_build_filter(
        raw_path=args.raw,
        factory_accepted_path=args.factory_accepted,
        build_manifest_path=args.build_manifest,
        rejected_dir=args.rejected_dir,
        out_md_path=args.out_md,
        out_json_path=args.out_json,
        sample_per_reason=args.sample_per_reason,
        seed=args.seed,
        semantic_embeddings=args.semantic_embeddings,
        embedding_model=args.embedding_model,
        semantic_threshold=args.semantic_threshold,
    )
    print(json.dumps({"conclusion": result["conclusion"], "overlap": result["overlap"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
