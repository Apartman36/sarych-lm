from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any


def _write_md(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_shard_preparation_report(path: Path, manifest: dict[str, Any]) -> None:
    lines = [
        "# Shard preparation report",
        "",
        f"- Total seeds: {manifest['total_seeds']}",
        f"- Shard size: {manifest['shard_size']}",
        f"- Shard count: {manifest['shard_count']}",
        f"- Random seed: {manifest['random_seed']}",
        "",
        "## Shards",
    ]
    for shard in manifest["shards"]:
        lines.append(
            f"- `{shard['shard_id']}`: {shard['row_count']} rows — "
            + ", ".join(f"{k}={v}" for k, v in shard["category_counts"].items())
        )
    lines.extend(["", "## Next step", "Hand each `shards/prompts/*_prompt.md` to OpenCode/Xiaomi, then place outputs in `shards/raw/`."])
    _write_md(path, lines)


def write_validation_summary_report(
    path: Path,
    *,
    title: str,
    summary: dict[str, Any],
    shard_results: list[dict[str, Any]],
) -> None:
    total_in = summary.get("total_input_rows", 0)
    total_acc = summary.get("total_accepted", 0)
    rate = (100.0 * total_acc / total_in) if total_in else 0.0
    lines = [
        f"# {title}",
        "",
        f"- Overall acceptance: **{total_acc}/{total_in}** ({rate:.1f}%)",
        f"- Strictness: `{summary.get('strictness', 'standard')}`",
        "",
        "## By shard",
        "| Shard | In | Accepted | Rate |",
        "|-------|-----|----------|------|",
    ]
    worst: list[tuple[float, str]] = []
    for item in shard_results:
        shard_rate = (100.0 * item["accepted_rows"] / item["input_rows"]) if item["input_rows"] else 0.0
        lines.append(
            f"| {item['shard_id']} | {item['input_rows']} | {item['accepted_rows']} | {shard_rate:.1f}% |"
        )
        worst.append((shard_rate, item["shard_id"]))
    lines.extend(["", "## Rejection reasons (all shards)"])
    for reason, count in sorted(summary.get("rejection_reasons", {}).items(), key=lambda x: (-x[1], x[0])):
        lines.append(f"- `{reason}`: {count}")
    lines.extend(["", "## Acceptance by category"])
    for category, count in sorted(summary.get("category_counts", {}).items()):
        lines.append(f"- `{category}`: {count}")
    worst.sort()
    lines.extend(["", "## Suggested next action"])
    if rate >= 50:
        lines.append("- Acceptance is healthy for a pilot; consider merge or full factory run.")
    else:
        lines.append("- Low acceptance: revise teacher prompts, run repair pack, or calibrate validator strictness.")
    if worst and worst[0][0] < 40:
        lines.append(f"- Worst shard: `{worst[0][1]}` ({worst[0][0]:.1f}%) — regenerate or repair.")
    reasons = summary.get("rejection_reasons", {})
    if reasons.get("near_duplicate_output", 0) + reasons.get("duplicate_instruction_output", 0) > 20:
        lines.append("- High duplicates: tighten prompt diversity requirements.")
    if reasons.get("too_short", 0) > 30:
        lines.append("- High too_short: ask teacher for longer outputs or use category-specific minima.")
    _write_md(path, lines)


def write_repair_pack_report(path: Path, manifest: dict[str, Any]) -> None:
    lines = [
        "# Repair pack report",
        "",
        f"- Round: {manifest['round']}",
        f"- Repair shards: {manifest['repair_shard_count']}",
        f"- Total repair seeds: {manifest['total_repair_seeds']}",
        "",
        "## Files",
    ]
    for shard in manifest["shards"]:
        lines.append(f"- `{shard['repair_shard_id']}`: {shard['row_count']} rows ({shard['prompt_path']})")
    lines.extend(["", "## Next step", "Hand repair prompts to teacher; place outputs in repairs round raw folder."])
    _write_md(path, lines)


def write_merge_report(path: Path, manifest: dict[str, Any]) -> None:
    lines = [
        "# Merge accepted report",
        "",
        f"- Total accepted: {manifest['total_accepted']}",
        f"- Duplicate removals: {manifest['duplicate_removals']}",
        "",
        "## By category",
    ]
    for category, count in sorted(manifest.get("category_counts", {}).items()):
        target = manifest.get("target_category_counts", {}).get(category)
        suffix = f" (target {target})" if target is not None else ""
        lines.append(f"- `{category}`: {count}{suffix}")
    lines.extend(["", "## By source", ""])
    for source, count in sorted(manifest.get("source_counts", {}).items()):
        lines.append(f"- `{source}`: {count}")
    lines.extend(["", "## Accepted by round", ""])
    for round_name, count in sorted(manifest.get("accepted_by_round", {}).items()):
        lines.append(f"- {round_name}: {count}")
    if manifest.get("unresolved_rejection_reasons"):
        lines.extend(["", "## Unresolved rejections (still rejected)"])
        for reason, count in sorted(manifest["unresolved_rejection_reasons"].items(), key=lambda x: (-x[1], x[0])):
            lines.append(f"- `{reason}`: {count}")
    rec = manifest.get("recommended_next_action", "")
    if rec:
        lines.extend(["", "## Recommended next action", rec])
    _write_md(path, lines)


def aggregate_validation(shard_results: list[dict[str, Any]]) -> dict[str, Any]:
    total_in = sum(item["input_rows"] for item in shard_results)
    total_acc = sum(item["accepted_rows"] for item in shard_results)
    reasons: Counter[str] = Counter()
    categories: Counter[str] = Counter()
    for item in shard_results:
        for reason, count in item.get("rejection_reasons", {}).items():
            reasons[reason] += count
        for category, count in item.get("category_counts", {}).items():
            categories[category] += count
    return {
        "total_input_rows": total_in,
        "total_accepted": total_acc,
        "acceptance_rate": (total_acc / total_in) if total_in else 0.0,
        "rejection_reasons": dict(sorted(reasons.items())),
        "category_counts": dict(sorted(categories.items())),
    }
