from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _dedup_key(row: dict[str, Any]) -> str:
    instruction = re.sub(r"\s+", " ", str(row.get("instruction", "")).strip().lower())
    output = re.sub(r"\s+", " ", str(row.get("output", "")).strip().lower())
    return instruction + "\n" + output


def _category(row: dict[str, Any]) -> str:
    metadata = row.get("metadata")
    if isinstance(metadata, dict) and metadata.get("category"):
        return str(metadata["category"])
    return str(row.get("task_type", "unknown"))


def _sample(rows: list[dict[str, Any]], cap: int, rng: random.Random) -> list[dict[str, Any]]:
    shuffled = list(rows)
    rng.shuffle(shuffled)
    if cap <= 0:
        return []
    return shuffled[: min(cap, len(shuffled))]


def make_instruction_lite_mix(
    *,
    replay_path: str | Path,
    instruction_lite_path: str | Path,
    everyday_path: str | Path,
    output_path: str | Path,
    manifest_path: str | Path,
    replay_cap: int = 900,
    instruction_lite_cap: int = 1500,
    everyday_cap: int = 400,
    old_xiaomi_seeded_path: str | Path | None = None,
    old_xiaomi_seeded_cap: int = 0,
    dolly_lite_path: str | Path | None = None,
    dolly_lite_cap: int = 0,
    seed: int = 1337,
) -> dict[str, Any]:
    rng = random.Random(seed)
    specs: list[tuple[str, Path, int]] = [
        ("replay", Path(replay_path), replay_cap),
        ("instruction_lite", Path(instruction_lite_path), instruction_lite_cap),
        ("everyday", Path(everyday_path), everyday_cap),
    ]
    if old_xiaomi_seeded_path is not None:
        specs.append(("old_xiaomi_seeded", Path(old_xiaomi_seeded_path), old_xiaomi_seeded_cap))
    if dolly_lite_path is not None:
        specs.append(("dolly_lite", Path(dolly_lite_path), dolly_lite_cap))

    input_counts: Counter[str] = Counter()
    selected_by_source: Counter[str] = Counter()
    selected_by_task_type: Counter[str] = Counter()
    selected_by_category: Counter[str] = Counter()
    duplicate_rows = 0
    seen: set[str] = set()
    mixed: list[dict[str, Any]] = []

    for source_name, path, cap in specs:
        rows = _read_jsonl(path)
        input_counts[source_name] = len(rows)
        for index, row in enumerate(_sample(rows, cap, rng), start=1):
            key = _dedup_key(row)
            if key in seen:
                duplicate_rows += 1
                continue
            seen.add(key)
            rewritten = dict(row)
            rewritten["id"] = f"{source_name}_{index:06d}"
            metadata = dict(rewritten.get("metadata", {})) if isinstance(rewritten.get("metadata"), dict) else {}
            metadata["mix_source"] = source_name
            metadata["original_id"] = row.get("id")
            rewritten["metadata"] = metadata
            mixed.append(rewritten)
            selected_by_source[source_name] += 1
            selected_by_task_type[str(rewritten.get("task_type", "unknown"))] += 1
            selected_by_category[_category(rewritten)] += 1

    rng.shuffle(mixed)
    output_path = Path(output_path)
    manifest_path = Path(manifest_path)
    _write_jsonl(output_path, mixed)
    manifest = {
        "builder": "make_v0_4_instruction_lite_mix",
        "output_path": str(output_path),
        "input_counts_by_source": dict(sorted(input_counts.items())),
        "selected_by_source": dict(sorted(selected_by_source.items())),
        "selected_by_task_type": dict(sorted(selected_by_task_type.items())),
        "selected_by_category": dict(sorted(selected_by_category.items())),
        "duplicate_rows_removed": duplicate_rows,
        "written_rows": len(mixed),
        "seed": seed,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a v0.4 instruction-lite SFT mix from replay, instruction-lite, and everyday sources.")
    parser.add_argument("--replay", required=True)
    parser.add_argument("--instruction-lite", required=True)
    parser.add_argument("--everyday", required=True)
    parser.add_argument("--old-xiaomi-seeded", default=None)
    parser.add_argument("--dolly-lite", default=None)
    parser.add_argument("--out", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--replay-cap", type=int, default=900)
    parser.add_argument("--instruction-lite-cap", type=int, default=1500)
    parser.add_argument("--everyday-cap", type=int, default=400)
    parser.add_argument("--old-xiaomi-seeded-cap", type=int, default=0)
    parser.add_argument("--dolly-lite-cap", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1337)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = make_instruction_lite_mix(
        replay_path=args.replay,
        instruction_lite_path=args.instruction_lite,
        everyday_path=args.everyday,
        old_xiaomi_seeded_path=args.old_xiaomi_seeded,
        dolly_lite_path=args.dolly_lite,
        output_path=args.out,
        manifest_path=args.manifest,
        replay_cap=args.replay_cap,
        instruction_lite_cap=args.instruction_lite_cap,
        everyday_cap=args.everyday_cap,
        old_xiaomi_seeded_cap=args.old_xiaomi_seeded_cap,
        dolly_lite_cap=args.dolly_lite_cap,
        seed=args.seed,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
