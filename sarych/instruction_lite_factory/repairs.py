from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from scripts.validate_instruction_lite_sft import Strictness, validate_instruction_lite_sft

from sarych.instruction_lite_factory.io_utils import read_jsonl, write_json, write_jsonl
from sarych.instruction_lite_factory.paths import factory_layout, to_windows_path
from sarych.instruction_lite_factory.prompt_templates import render_repair_prompt
from sarych.instruction_lite_factory.reports import aggregate_validation, write_repair_pack_report, write_validation_summary_report


def _repair_round_dir(factory_dir: Path, round_number: int) -> Path:
    return factory_dir / "repairs" / f"round_{round_number}"


def make_repair_pack(
    *,
    factory_dir: str | Path,
    max_per_shard: int = 50,
    round_number: int = 1,
) -> dict[str, Any]:
    factory_dir = Path(factory_dir)
    layout = factory_layout(factory_dir)
    round_dir = _repair_round_dir(factory_dir, round_number)
    seeds_dir = round_dir / "seeds"
    prompts_dir = round_dir / "prompts"
    seeds_dir.mkdir(parents=True, exist_ok=True)
    prompts_dir.mkdir(parents=True, exist_ok=True)

    rejected_files = sorted(layout["shards_rejected"].glob("shard_*_rejected.jsonl"))
    repair_shards: list[dict[str, Any]] = []
    repair_index = 0
    for rejected_path in rejected_files:
        rejected_rows = read_jsonl(rejected_path)
        if not rejected_rows:
            continue
        repair_index += 1
        shard_id = rejected_path.stem.replace("_rejected", "")
        limited = rejected_rows[:max_per_shard]
        repair_rows: list[dict[str, Any]] = []
        for item in limited:
            record = item.get("record") or {}
            seed_id = item.get("seed_id") or record.get("seed_id")
            category = item.get("category") or (record.get("metadata") or {}).get("category")
            repair_rows.append(
                {
                    "repair_id": f"repair_{round_number}_{repair_index:04d}_{len(repair_rows)+1:03d}",
                    "shard_id": shard_id,
                    "seed_id": seed_id,
                    "category": category,
                    "rejection_reason": item.get("reason"),
                    "rejection_detail": item.get("detail"),
                    "original_seed": _lookup_seed(layout, seed_id),
                    "rejected_record": record,
                    "fix_instructions": _fix_hint(str(item.get("reason")), str(category)),
                }
            )
        repair_shard_id = f"repair_shard_{repair_index:04d}"
        seed_file = seeds_dir / f"{repair_shard_id}.jsonl"
        prompt_file = prompts_dir / f"{repair_shard_id}_prompt.md"
        raw_out = round_dir / "raw" / f"{repair_shard_id}.jsonl"
        write_jsonl(seed_file, repair_rows)
        prompt = render_repair_prompt(
            repair_index=repair_index,
            repair_rows=repair_rows,
            seeds_path_windows=to_windows_path(seed_file),
            output_path_windows=to_windows_path(raw_out),
            round_number=round_number,
        )
        prompt_file.write_text(prompt, encoding="utf-8")
        repair_shards.append(
            {
                "repair_shard_id": repair_shard_id,
                "source_shard_id": shard_id,
                "row_count": len(repair_rows),
                "prompt_path": str(prompt_file),
                "seeds_path": str(seed_file),
                "expected_raw_output": str(raw_out),
            }
        )

    manifest = {
        "factory_version": "v0_4_5",
        "round": round_number,
        "repair_shard_count": len(repair_shards),
        "total_repair_seeds": sum(shard["row_count"] for shard in repair_shards),
        "max_per_shard": max_per_shard,
        "shards": repair_shards,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    write_json(round_dir / "manifest.json", manifest)
    write_repair_pack_report(round_dir / "reports" / "repair_pack.md", manifest)
    return manifest


def _lookup_seed(layout: dict[str, Path], seed_id: str | None) -> dict[str, Any] | None:
    if not seed_id:
        return None
    for seed_file in layout["shards_seeds"].glob("shard_*.jsonl"):
        for row in read_jsonl(seed_file):
            if row.get("id") == seed_id:
                return row
    return None


def _fix_hint(reason: str | None, category: str | None) -> str:
    hints = {
        "too_short": "Make the output longer while staying child-simple and on-category.",
        "too_long": "Shorten the output; remove filler.",
        "near_duplicate_output": "Rewrite with different wording and opening; avoid repeated phrases.",
        "duplicate_instruction_output": "Change the output completely; must not match any prior row.",
        "weak_explanation": "Start with a direct answer using because/so/helps/needs/makes/means.",
        "weak_reasoning": "Add one clear step or practical reason.",
        "weak_support": "Give warm comfort plus one simple action (breath, talk to grown-up, etc.).",
        "identity_fail": "Use first-person helper voice (I/me/my) and mention help/helper/Sarych.",
        "unsafe_refusal": "Gently refuse and redirect to a grown-up/trusted person.",
        "list_format_fail": "Use numbered or bullet list with at least 2 items.",
        "story_collapse": "Do not use story openers for this non-story category.",
    }
    return hints.get(reason or "", f"Fix rejection `{reason}` for category `{category}`.")


def validate_repairs(
    *,
    factory_dir: str | Path,
    round_number: int = 1,
    strictness: Strictness | str = Strictness.STANDARD,
) -> dict[str, Any]:
    factory_dir = Path(factory_dir)
    round_dir = _repair_round_dir(factory_dir, round_number)
    if isinstance(strictness, str):
        strictness = Strictness(strictness.lower())

    raw_dir = round_dir / "raw"
    accepted_dir = round_dir / "accepted"
    rejected_dir = round_dir / "rejected"
    manifests_dir = round_dir / "manifests"
    for path in (accepted_dir, rejected_dir, manifests_dir, round_dir / "reports"):
        path.mkdir(parents=True, exist_ok=True)

    shard_results: list[dict[str, Any]] = []
    for raw_path in sorted(raw_dir.glob("repair_shard_*.jsonl")):
        shard_id = raw_path.stem
        accepted_path = accepted_dir / f"{shard_id}_accepted.jsonl"
        rejected_path = rejected_dir / f"{shard_id}_rejected.jsonl"
        manifest_path = manifests_dir / f"{shard_id}_manifest.json"
        manifest = validate_instruction_lite_sft(
            raw_path,
            accepted_path,
            rejected_path,
            manifest_path,
            strictness=strictness,
        )
        input_rows = len(read_jsonl(raw_path))
        shard_results.append(
            {
                "shard_id": shard_id,
                "input_rows": input_rows,
                "accepted_rows": manifest["accepted_rows"],
                "rejected_rows": manifest["rejected_rows"],
                "rejection_reasons": manifest.get("rejection_reasons", {}),
                "category_counts": manifest.get("category_counts", {}),
            }
        )

    summary = aggregate_validation(shard_results)
    summary["strictness"] = strictness.value
    summary["round"] = round_number
    summary["timestamp"] = datetime.now(timezone.utc).isoformat()
    write_json(manifests_dir / "validation_summary.json", summary)
    write_validation_summary_report(
        round_dir / "reports" / "validation_summary.md",
        title=f"Validation summary (repairs round {round_number})",
        summary=summary,
        shard_results=shard_results,
    )
    return summary
