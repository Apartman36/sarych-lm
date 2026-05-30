from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SourceSpec:
    name: str
    path: Path
    cap: int | None = None
    task_caps: dict[str, int] = field(default_factory=dict)


def parse_source_spec(spec: str) -> SourceSpec:
    if "=" not in spec:
        raise ValueError(f"Source spec must be name=path[:cap=N][:task=type:N]: {spec}")
    name, rest = spec.split("=", 1)
    matches = list(re.finditer(r":(cap=\d+|task=[^:=]+=\d+)", rest))
    path_text = rest[: matches[0].start()] if matches else rest
    path = Path(path_text)
    cap: int | None = None
    task_caps: dict[str, int] = {}
    for match in matches:
        option = match.group(1)
        if not option:
            continue
        if option.startswith("cap="):
            cap = int(option.removeprefix("cap="))
        elif option.startswith("task="):
            task_name, value = option.removeprefix("task=").split("=", 1)
            task_caps[task_name] = int(value)
        else:
            raise ValueError(f"Unknown source option in {spec}: {option}")
    return SourceSpec(name=name, path=path, cap=cap, task_caps=task_caps)


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


def _select_rows(rows: list[dict[str, Any]], spec: SourceSpec, rng: random.Random) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    task_counts: Counter[str] = Counter()
    shuffled = list(rows)
    rng.shuffle(shuffled)
    if spec.cap is None and not spec.task_caps:
        return shuffled
    for row in shuffled:
        task_type = str(row.get("task_type", ""))
        if spec.task_caps and task_counts[task_type] >= spec.task_caps.get(task_type, len(rows)):
            continue
        if spec.cap is not None and len(selected) >= spec.cap:
            break
        selected.append(row)
        task_counts[task_type] += 1
    return selected


def mix_sft_sources(
    sources: list[SourceSpec],
    output_path: str | Path,
    *,
    manifest_path: str | Path | None = None,
    seed: int = 1337,
) -> dict[str, Any]:
    output_path = Path(output_path)
    rng = random.Random(seed)
    input_counts: dict[str, int] = {}
    selected_counts: Counter[str] = Counter()
    task_counts: Counter[str] = Counter()
    seen_keys: set[str] = set()
    mixed_rows: list[dict[str, Any]] = []
    duplicate_count = 0

    for spec in sources:
        rows = _read_jsonl(spec.path)
        input_counts[spec.name] = len(rows)
        for source_index, row in enumerate(_select_rows(rows, spec, rng), start=1):
            key = _dedup_key(row)
            if key in seen_keys:
                duplicate_count += 1
                continue
            seen_keys.add(key)
            rewritten = dict(row)
            rewritten["id"] = f"{spec.name}_{source_index:06d}"
            metadata = dict(rewritten.get("metadata", {}))
            metadata["mix_source"] = spec.name
            metadata["original_id"] = row.get("id")
            rewritten["metadata"] = metadata
            mixed_rows.append(rewritten)
            selected_counts[spec.name] += 1
            task_counts[str(rewritten.get("task_type", ""))] += 1

    rng.shuffle(mixed_rows)
    _write_jsonl(output_path, mixed_rows)
    manifest = {
        "converter": "mix_sft_sources",
        "output_path": str(output_path),
        "input_counts_by_source": dict(sorted(input_counts.items())),
        "selected_counts_by_source": dict(sorted(selected_counts.items())),
        "selected_counts_by_task_type": dict(sorted(task_counts.items())),
        "rejected_duplicate_rows": duplicate_count,
        "written_rows": len(mixed_rows),
        "seed": seed,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    if manifest_path is not None:
        manifest_path = Path(manifest_path)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mix SARYCH SFT JSONL sources with deterministic caps.")
    parser.add_argument("--source", action="append", required=True, help="name=path:cap=N[:task=task_type=N]")
    parser.add_argument("--out", required=True)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--seed", type=int, default=1337)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = mix_sft_sources(
        [parse_source_spec(source) for source in args.source],
        args.out,
        manifest_path=args.manifest,
        seed=args.seed,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
