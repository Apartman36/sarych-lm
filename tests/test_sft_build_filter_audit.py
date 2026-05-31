from __future__ import annotations

import json
from pathlib import Path

from scripts.audit_sft_build_filter import audit_sft_build_filter


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def _row(row_id: str, instruction: str, output: str, *, category: str = "simple_qa") -> dict:
    return {
        "id": row_id,
        "source": "xiaomi_instruction_lite_v0_4",
        "task_type": "simple_qa",
        "instruction": instruction,
        "input": "",
        "output": output,
        "language": "en",
        "metadata": {"category": category},
    }


def _reject(row: dict, reason: str) -> dict:
    return {
        "reason": reason,
        "detail": None,
        "source": row["source"],
        "task_type": row["task_type"],
        "id": row["id"],
        "record": row,
    }


def test_audit_reads_duplicate_and_filtered_rejections_and_writes_reports(tmp_path):
    first = _row("a", "What is rain?", "Rain is water from clouds.")
    duplicate = _row("b", "Tell me what rain is.", "Rain is water from clouds.")
    filtered = _row("c", "Say hi kindly.", "Hi!")
    accepted_other = _row("d", "What is wind?", "Wind is moving air.")
    accepted_other["source"] = "other_source"

    raw_path = tmp_path / "raw.jsonl"
    factory_path = tmp_path / "factory.jsonl"
    rejected_dir = tmp_path / "rejected"
    manifest_path = tmp_path / "manifest.json"
    out_md = tmp_path / "audit.md"
    out_json = tmp_path / "audit.json"
    _write_jsonl(raw_path, [first, duplicate, filtered, accepted_other])
    _write_jsonl(factory_path, [first, duplicate, filtered])
    _write_jsonl(rejected_dir / "duplicates.jsonl", [_reject(duplicate, "duplicate")])
    _write_jsonl(rejected_dir / "filtered.jsonl", [_reject(filtered, "output_too_short")])
    _write_jsonl(rejected_dir / "malformed.jsonl", [])
    _write_jsonl(rejected_dir / "too_long.jsonl", [])
    manifest_path.write_text(
        json.dumps(
            {
                "accepted_by_source": {"xiaomi_instruction_lite_v0_4": 1},
                "accepted_by_task_type": {"simple_qa": 1},
            }
        ),
        encoding="utf-8",
    )

    result = audit_sft_build_filter(
        raw_path=raw_path,
        factory_accepted_path=factory_path,
        build_manifest_path=manifest_path,
        rejected_dir=rejected_dir,
        out_md_path=out_md,
        out_json_path=out_json,
        sample_per_reason=2,
        seed=123,
    )
    result_again = audit_sft_build_filter(
        raw_path=raw_path,
        factory_accepted_path=factory_path,
        build_manifest_path=manifest_path,
        rejected_dir=rejected_dir,
        out_md_path=tmp_path / "audit_again.md",
        out_json_path=tmp_path / "audit_again.json",
        sample_per_reason=2,
        seed=123,
    )

    assert result["rejected_rows_by_file"] == {"duplicates": 1, "filtered": 1, "malformed": 0, "too_long": 0}
    assert result["overlap"]["factory_accepted_and_build_rejected"] == 2
    assert result["overlap"]["all_rejected_instruction_lite_in_factory_accepted"] is True
    assert result["duplicates"]["normalized_output_across_all_tasks"]["Rain is water from clouds."] == 2
    assert result["top_duplicate_clusters"]["output_across_all_tasks"][0]["count"] == 2
    assert result["samples"] == result_again["samples"]
    assert "duplicates.jsonl" in out_md.read_text(encoding="utf-8")
    assert "Conclusion" in out_md.read_text(encoding="utf-8")
    assert json.loads(out_json.read_text(encoding="utf-8"))["samples"] == result["samples"]
