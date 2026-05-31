from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from scripts.validate_instruction_lite_sft import Strictness, validate_instruction_lite_sft

from sarych.instruction_lite_factory.io_utils import read_jsonl, write_json
from sarych.instruction_lite_factory.paths import factory_layout
from sarych.instruction_lite_factory.reports import aggregate_validation, write_validation_summary_report


def _shard_ids(layout: dict[str, Path], raw_dir: Path) -> list[str]:
    if raw_dir.exists():
        return sorted(path.stem for path in raw_dir.glob("shard_*.jsonl"))
    manifest_path = layout["manifests"] / "shards_manifest.json"
    if manifest_path.exists():
        import json

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        return [shard["shard_id"] for shard in manifest["shards"]]
    return []


def validate_shards(
    *,
    factory_dir: str | Path,
    strictness: Strictness | str = Strictness.STANDARD,
) -> dict[str, Any]:
    factory_dir = Path(factory_dir)
    layout = factory_layout(factory_dir)
    if isinstance(strictness, str):
        strictness = Strictness(strictness.lower())

    raw_dir = layout["shards_raw"]
    shard_results: list[dict[str, Any]] = []
    for shard_id in _shard_ids(layout, raw_dir):
        raw_path = raw_dir / f"{shard_id}.jsonl"
        if not raw_path.exists():
            continue
        accepted_path = layout["shards_accepted"] / f"{shard_id}_accepted.jsonl"
        rejected_path = layout["shards_rejected"] / f"{shard_id}_rejected.jsonl"
        manifest_path = layout["shards_manifests"] / f"{shard_id}_manifest.json"
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
                "warnings": manifest.get("warnings", {}),
                "manifest_path": str(manifest_path),
            }
        )

    summary = aggregate_validation(shard_results)
    summary["strictness"] = strictness.value
    summary["shard_results"] = shard_results
    summary["timestamp"] = datetime.now(timezone.utc).isoformat()
    summary_path = layout["manifests"] / "validation_summary.json"
    write_json(summary_path, summary)
    write_validation_summary_report(
        layout["reports"] / "validation_summary.md",
        title="Validation summary (shards)",
        summary=summary,
        shard_results=shard_results,
    )
    return summary
