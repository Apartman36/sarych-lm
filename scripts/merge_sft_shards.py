from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sarych.sft import ALLOWED_SFT_TASK_TYPES, validate_raw_sft_row
from sarych.utils import ensure_dir


def _id_sort_key(row: dict[str, Any]) -> tuple[int, str]:
    row_id = str(row.get("id", ""))
    match = re.search(r"(\d+)$", row_id)
    return (int(match.group(1)) if match else 10**12, row_id)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def merge_sft_shards(*, input_dir: str | Path, out: str | Path, manifest: str | Path, rejected: str | Path | None = None) -> dict[str, Any]:
    input_dir = Path(input_dir)
    out = Path(out)
    manifest = Path(manifest)
    rejected = Path(rejected) if rejected is not None else manifest.with_name(f"{manifest.stem}_rejected.jsonl")
    files = sorted(input_dir.glob("*.jsonl"))
    if not files:
        raise FileNotFoundError(f"No *.jsonl files found in {input_dir}")

    rows: list[dict[str, Any]] = []
    rejected_rows: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    duplicate_ids: list[str] = []

    for path in files:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    rejected_rows.append({"file": str(path), "line": line_number, "reason": "json_decode_error", "detail": exc.msg, "raw": line.rstrip("\n")})
                    continue
                validation = validate_raw_sft_row(row)
                if not validation.ok:
                    rejected_rows.append({"file": str(path), "line": line_number, "reason": validation.reason, "detail": validation.detail, "row": row})
                    continue
                row_id = str(row["id"])
                if row_id in seen_ids:
                    duplicate_ids.append(row_id)
                    rejected_rows.append({"file": str(path), "line": line_number, "reason": "duplicate_id", "row": row})
                    continue
                seen_ids.add(row_id)
                rows.append(row)

    rows.sort(key=_id_sort_key)
    category_counts = Counter(str(row["task_type"]) for row in rows)
    unknown_categories = sorted(set(category_counts) - ALLOWED_SFT_TASK_TYPES)
    _write_jsonl(out, rows)
    _write_jsonl(rejected, rejected_rows)
    summary = {
        "input_dir": str(input_dir),
        "input_files": [str(path) for path in files],
        "output_path": str(out),
        "rejected_path": str(rejected),
        "total_input_lines": len(rows) + len(rejected_rows),
        "accepted_rows": len(rows),
        "rejected_rows": len(rejected_rows),
        "duplicate_ids": sorted(set(duplicate_ids)),
        "allowed_task_types": sorted(ALLOWED_SFT_TASK_TYPES),
        "unknown_categories": unknown_categories,
        "category_counts": dict(sorted(category_counts.items())),
    }
    ensure_dir(manifest.parent)
    manifest.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge generated SFT JSONL shards into one raw JSONL file.")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--rejected", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = merge_sft_shards(input_dir=args.input_dir, out=args.out, manifest=args.manifest, rejected=args.rejected)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
