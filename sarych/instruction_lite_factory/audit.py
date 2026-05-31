from __future__ import annotations

import random
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def audit_sample(
    *,
    input_path: str | Path,
    out_path: str | Path,
    per_category: int = 5,
    seed: int = 1337,
) -> dict[str, Any]:
    from sarych.instruction_lite_factory.io_utils import read_jsonl

    input_path = Path(input_path)
    out_path = Path(out_path)
    rows = read_jsonl(input_path)
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        category = str((row.get("metadata") or {}).get("category", "unknown"))
        by_category[category].append(row)

    rng = random.Random(seed)
    lines = [
        "# Instruction-lite audit sample",
        "",
        f"- Source: `{input_path}`",
        f"- Total rows: {len(rows)}",
        f"- Per category: {per_category}",
        f"- Seed: {seed}",
        f"- Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
    ]
    sampled_total = 0
    for category in sorted(by_category):
        bucket = by_category[category]
        rng.shuffle(bucket)
        sample = bucket[:per_category]
        sampled_total += len(sample)
        lines.append(f"## {category} ({len(sample)} samples)")
        lines.append("")
        for index, row in enumerate(sample, start=1):
            instruction = str(row.get("instruction", "")).strip()
            input_text = str(row.get("input", "")).strip()
            output = str(row.get("output", "")).strip()
            lines.append(f"### Sample {index}")
            lines.append(f"- id: `{row.get('id')}`")
            lines.append(f"- seed_id: `{row.get('seed_id')}`")
            lines.append("")
            lines.append("**Instruction**")
            lines.append("")
            lines.append(instruction)
            lines.append("")
            if input_text:
                lines.append("**Input**")
                lines.append("")
                lines.append(input_text)
                lines.append("")
            lines.append("**Output**")
            lines.append("")
            lines.append(output)
            lines.append("")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "input_path": str(input_path),
        "out_path": str(out_path),
        "total_rows": len(rows),
        "sampled_rows": sampled_total,
        "categories": {category: min(per_category, len(bucket)) for category, bucket in sorted(by_category.items())},
    }
