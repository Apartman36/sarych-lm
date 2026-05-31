from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


DECISION_RANK = {"PASS_DEMO": 3, "NEEDS_TARGETED_CORRECTION": 2, "NO_GO": 1}


def _load_metrics(eval_dir: Path) -> dict[str, Any]:
    metrics_path = eval_dir / "metrics.json"
    if not metrics_path.exists():
        summary_path = eval_dir / "summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        return dict(summary["metrics"]["strict"])
    return json.loads(metrics_path.read_text(encoding="utf-8"))


def _candidate_sort_key(item: dict[str, Any]) -> tuple[int, int, int, float]:
    metrics = item["metrics"]
    return (
        DECISION_RANK.get(str(metrics.get("decision")), 0),
        -int(metrics.get("critical_safety_fail_count", 0)),
        int(metrics.get("total_score", 0)),
        float(metrics.get("average_score", 0.0)),
    )


def compare_eval_dirs(*, eval_dirs: list[str | Path], out_md: str | Path, out_json: str | Path) -> dict[str, Any]:
    candidates = [{"eval_dir": str(Path(eval_dir)), "metrics": _load_metrics(Path(eval_dir))} for eval_dir in eval_dirs]
    ranked = sorted(candidates, key=_candidate_sort_key, reverse=True)
    best = ranked[0] if ranked else None
    backup = ranked[1] if len(ranked) > 1 else None
    safe_demo = bool(best and best["metrics"].get("decision") == "PASS_DEMO")
    trainable = next(
        (
            candidate
            for candidate in ranked
            if int(candidate["metrics"].get("critical_safety_fail_count", 0)) == 0
            and candidate["metrics"].get("decision") in {"PASS_DEMO", "NEEDS_TARGETED_CORRECTION"}
        ),
        None,
    )
    flag_counts: Counter[str] = Counter()
    for candidate in candidates:
        flag_counts.update(candidate["metrics"].get("flag_counts", {}))

    result = {
        "candidates": candidates,
        "best_candidate": best["eval_dir"] if best else None,
        "backup_candidate": backup["eval_dir"] if backup else None,
        "safe_to_release_demo": safe_demo,
        "safe_to_continue_training_from": trainable["eval_dir"] if trainable else None,
        "combined_flag_counts": dict(sorted(flag_counts.items())),
    }

    out_json = Path(out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_markdown(Path(out_md), result)
    return result


def _write_markdown(path: Path, result: dict[str, Any]) -> None:
    lines = [
        "# Strict Instruction-Lite Eval Comparison",
        "",
        f"- best candidate: {result['best_candidate']}",
        f"- backup candidate: {result['backup_candidate']}",
        f"- safe to release as demo: {result['safe_to_release_demo']}",
        f"- safe to continue training from: {result['safe_to_continue_training_from']}",
        "",
        "## Candidates",
        "",
        "| eval dir | decision | total score | critical safety fails |",
        "|---|---|---:|---:|",
    ]
    for candidate in result["candidates"]:
        metrics = candidate["metrics"]
        lines.append(
            f"| {candidate['eval_dir']} | {metrics.get('decision')} | "
            f"{metrics.get('total_score')} | {metrics.get('critical_safety_fail_count')} |"
        )
    lines.extend(["", "## Category Scores", ""])
    for candidate in result["candidates"]:
        lines.extend([f"### {candidate['eval_dir']}", "", "| category | avg score | fail rate |", "|---|---:|---:|"])
        for category, summary in sorted(candidate["metrics"].get("category_summary", {}).items()):
            lines.append(f"| {category} | {summary.get('average_score', 0):.3f} | {summary.get('fail_rate', 0):.3f} |")
        lines.append("")
    lines.extend(["## Combined Top Flags", ""])
    for flag, count in Counter(result["combined_flag_counts"]).most_common(20):
        lines.append(f"- {flag}: {count}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare strict v0.4 instruction-lite eval directories.")
    parser.add_argument("--eval-dir", action="append", required=True, help="Strict eval directory containing metrics.json")
    parser.add_argument("--out-md", required=True)
    parser.add_argument("--out-json", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = compare_eval_dirs(eval_dirs=args.eval_dir, out_md=args.out_md, out_json=args.out_json)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
