from __future__ import annotations

import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from scripts.make_instruction_lite_seeds_v0_4 import FINAL_CATEGORY_COUNTS

from sarych.instruction_lite_factory.io_utils import read_jsonl, write_json, write_jsonl
from sarych.instruction_lite_factory.paths import factory_layout
from sarych.instruction_lite_factory.reports import write_merge_report


def _exact_key(row: dict[str, Any]) -> str:
    instruction = str(row.get("instruction", "")).strip().lower()
    output = str(row.get("output", "")).strip().lower()
    return re.sub(r"\s+", " ", instruction) + "\n" + re.sub(r"\s+", " ", output)


def _collect_accepted_paths(factory_dir: Path) -> list[tuple[str, Path]]:
    layout = factory_layout(factory_dir)
    found: list[tuple[str, Path]] = []
    for path in sorted(layout["shards_accepted"].glob("shard_*_accepted.jsonl")):
        found.append(("initial", path))
    repairs_root = layout["repairs"]
    if repairs_root.exists():
        for round_dir in sorted(repairs_root.glob("round_*")):
            accepted_dir = round_dir / "accepted"
            if not accepted_dir.exists():
                continue
            round_name = round_dir.name
            for path in sorted(accepted_dir.glob("repair_shard_*_accepted.jsonl")):
                found.append((round_name, path))
    return found


def merge_accepted(
    *,
    factory_dir: str | Path,
    out_path: str | Path,
    manifest_path: str | Path,
) -> dict[str, Any]:
    factory_dir = Path(factory_dir)
    out_path = Path(out_path)
    manifest_path = Path(manifest_path)
    layout = factory_layout(factory_dir)

    merged: list[dict[str, Any]] = []
    seen: set[str] = set()
    duplicate_removals = 0
    accepted_by_round: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    category_counts: Counter[str] = Counter()
    task_type_counts: Counter[str] = Counter()

    for round_name, path in _collect_accepted_paths(factory_dir):
        rows = read_jsonl(path)
        accepted_by_round[round_name] += len(rows)
        for row in rows:
            key = _exact_key(row)
            if key in seen:
                duplicate_removals += 1
                continue
            seen.add(key)
            merged.append(row)
            source_counts[str(row.get("source", "unknown"))] += 1
            category = str((row.get("metadata") or {}).get("category", "unknown"))
            category_counts[category] += 1
            task_type_counts[str(row.get("task_type", "unknown"))] += 1

    write_jsonl(out_path, merged)

    unresolved: Counter[str] = Counter()
    for rejected_path in layout["shards_rejected"].glob("shard_*_rejected.jsonl"):
        for item in read_jsonl(rejected_path):
            unresolved[str(item.get("reason", "unknown"))] += 1
    repairs_root = layout["repairs"]
    if repairs_root.exists():
        for round_dir in repairs_root.glob("round_*"):
            rejected_dir = round_dir / "rejected"
            if rejected_dir.exists():
                for rejected_path in rejected_dir.glob("*_rejected.jsonl"):
                    for item in read_jsonl(rejected_path):
                        unresolved[str(item.get("reason", "unknown"))] += 1

    total_target = sum(FINAL_CATEGORY_COUNTS.values())
    if len(merged) >= 800:
        recommendation = "Enough accepted rows for mix/build; proceed to make_v0_4_instruction_lite_mix.py."
    elif len(merged) >= 500:
        recommendation = "Moderate acceptance; consider one more repair round before training."
    else:
        recommendation = "Do not train yet (<500 accepted). Revise teacher prompts or run another repair round."

    manifest = {
        "factory_version": "v0_4_5",
        "factory_dir": str(factory_dir),
        "output_path": str(out_path),
        "total_accepted": len(merged),
        "duplicate_removals": duplicate_removals,
        "category_counts": dict(sorted(category_counts.items())),
        "task_type_counts": dict(sorted(task_type_counts.items())),
        "source_counts": dict(sorted(source_counts.items())),
        "accepted_by_round": dict(sorted(accepted_by_round.items())),
        "target_category_counts": dict(sorted(FINAL_CATEGORY_COUNTS.items())),
        "unresolved_rejection_reasons": dict(sorted(unresolved.items())),
        "recommended_next_action": recommendation,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    write_json(manifest_path, manifest)
    write_merge_report(layout["reports"] / "merge_accepted.md", manifest)
    return manifest
