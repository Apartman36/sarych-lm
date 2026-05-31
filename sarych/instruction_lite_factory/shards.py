from __future__ import annotations

import random
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sarych.instruction_lite_factory.io_utils import read_jsonl, write_json, write_jsonl
from sarych.instruction_lite_factory.paths import factory_layout, to_windows_path
from sarych.instruction_lite_factory.prompt_templates import render_shard_prompt
from sarych.instruction_lite_factory.reports import write_shard_preparation_report


def _balanced_shard_order(rows: list[dict[str, Any]], *, shard_size: int, rng: random.Random) -> list[list[dict[str, Any]]]:
    """Round-robin by category for balance, then chunk into shards."""
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_category[str(row["category"])].append(row)
    for bucket in by_category.values():
        rng.shuffle(bucket)
    categories = sorted(by_category)
    ordered: list[dict[str, Any]] = []
    index = 0
    while True:
        added = False
        for category in categories:
            if index < len(by_category[category]):
                ordered.append(by_category[category][index])
                added = True
        if not added:
            break
        index += 1
    shards: list[list[dict[str, Any]]] = []
    for start in range(0, len(ordered), shard_size):
        shards.append(ordered[start : start + shard_size])
    return shards


def prepare_shards(
    *,
    seeds_path: str | Path,
    out_dir: str | Path,
    shard_size: int = 100,
    seed: int = 1337,
) -> dict[str, Any]:
    seeds_path = Path(seeds_path)
    out_dir = Path(out_dir)
    layout = factory_layout(out_dir)
    for key in ("shards_seeds", "shards_prompts", "manifests", "reports"):
        layout[key].mkdir(parents=True, exist_ok=True)

    rows = read_jsonl(seeds_path)
    rng = random.Random(seed)
    shards = _balanced_shard_order(rows, shard_size=shard_size, rng=rng)

    shard_entries: list[dict[str, Any]] = []
    for index, shard_rows in enumerate(shards, start=1):
        shard_id = f"shard_{index:04d}"
        seed_file = layout["shards_seeds"] / f"{shard_id}.jsonl"
        prompt_file = layout["shards_prompts"] / f"{shard_id}_prompt.md"
        raw_out = layout["shards_raw"] / f"{shard_id}.jsonl"
        write_jsonl(seed_file, shard_rows)
        prompt = render_shard_prompt(
            shard_index=index,
            shard_rows=shard_rows,
            seeds_path_windows=to_windows_path(seed_file),
            output_path_windows=to_windows_path(raw_out),
        )
        prompt_file.write_text(prompt, encoding="utf-8")
        category_counts = dict(sorted(Counter(str(r["category"]) for r in shard_rows).items()))
        shard_entries.append(
            {
                "shard_id": shard_id,
                "index": index,
                "row_count": len(shard_rows),
                "category_counts": category_counts,
                "seeds_path": str(seed_file),
                "prompt_path": str(prompt_file),
                "expected_raw_output": str(raw_out),
            }
        )

    manifest = {
        "factory_version": "v0_4_5",
        "total_seeds": len(rows),
        "shard_size": shard_size,
        "shard_count": len(shards),
        "random_seed": seed,
        "seeds_path": str(seeds_path),
        "factory_dir": str(out_dir),
        "shards": shard_entries,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    manifest_path = layout["manifests"] / "shards_manifest.json"
    write_json(manifest_path, manifest)
    write_shard_preparation_report(layout["reports"] / "shard_preparation.md", manifest)
    return manifest
