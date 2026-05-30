from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


DEFAULT_PROMPTS = [
    "Hi, who are you?",
    "Explain why plants need sunlight in simple words.",
    "Write a short story about a careful fox who learns to ask for help.",
    "List three kind things a friend can do.",
    "Write one sentence about a happy dog.",
    "Continue this story: A little turtle found a shiny key under a leaf.",
]


@dataclass(frozen=True)
class ExperimentCommand:
    config: str
    steps: int
    run_dir: Path
    train_cmd: list[str]


def _config_slug(config: str) -> str:
    return Path(config).stem.replace("v0_4_30m_instruct_", "")


def _load_prompts(prompts_file: str | Path | None) -> list[str]:
    if prompts_file is None:
        return list(DEFAULT_PROMPTS)
    path = Path(prompts_file)
    prompts: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("{"):
            prompts.append(str(json.loads(line)["instruction"]))
        else:
            prompts.append(line)
    return prompts


def build_experiment_commands(
    *,
    configs: list[str],
    steps: list[int],
    run_root: str | Path,
    device: str | None = None,
) -> list[ExperimentCommand]:
    run_root = Path(run_root)
    commands: list[ExperimentCommand] = []
    for config in configs:
        for step_count in steps:
            run_dir = run_root / f"{_config_slug(config)}_steps{step_count}"
            cmd = [
                "python",
                "scripts/train_sft_v0_4.py",
                "--config",
                config,
                "--max-steps",
                str(step_count),
                "--no-resume",
                "--run-dir",
                str(run_dir),
            ]
            if device is not None:
                cmd.extend(["--device", device])
            commands.append(ExperimentCommand(config=config, steps=step_count, run_dir=run_dir, train_cmd=cmd))
    return commands


def _read_log_summary(log_path: Path) -> dict[str, Any]:
    if not log_path.exists():
        return {"final_train_loss": None, "best_val_loss": None, "tail": []}
    rows = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    val_losses = [row["val_loss"] for row in rows if row.get("val_loss") is not None]
    return {
        "final_train_loss": rows[-1].get("train_loss") if rows else None,
        "best_val_loss": min(val_losses) if val_losses else None,
        "tail": rows[-5:],
    }


def _run_generations(
    *,
    run_dir: Path,
    prompts: list[str],
    console,
    device: str | None,
    tokenizer: str,
    max_new_tokens: int,
) -> list[dict[str, Any]]:
    checkpoint = run_dir / "checkpoints" / "checkpoint_latest.pt"
    generations: list[dict[str, Any]] = []
    generations_path = run_dir / "generations.jsonl"
    generations_path.parent.mkdir(parents=True, exist_ok=True)
    with generations_path.open("w", encoding="utf-8") as handle:
        for prompt in prompts:
            cmd = [
                "python",
                "scripts/generate_instruct_v0_4.py",
                "--checkpoint",
                str(checkpoint),
                "--tokenizer",
                tokenizer,
                "--instruction",
                prompt,
                "--max-new-tokens",
                str(max_new_tokens),
                "--no-print-prompt",
            ]
            if device is not None:
                cmd.extend(["--device", device])
            process = subprocess.run(
                cmd,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
            )
            console.write("\nGEN: " + " ".join(cmd) + "\n")
            console.write(process.stdout)
            record = {
                "prompt": prompt,
                "returncode": process.returncode,
                "output": process.stdout.strip(),
                "cmd": cmd,
            }
            generations.append(record)
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    return generations


def _write_report(run_dir: Path, results: list[dict[str, Any]], *, dry_run: bool) -> None:
    lines = ["# SFT Grid v0.4 Report", ""]
    if dry_run:
        lines.extend(["DRY RUN: commands were constructed but not executed.", ""])
    lines.extend(["| config | steps | status | best_val_loss | final_train_loss |", "|---|---:|---|---:|---:|"])
    for result in results:
        lines.append(
            "| {config} | {steps} | {status} | {best_val_loss} | {final_train_loss} |".format(
                config=result["config"],
                steps=result["steps"],
                status=result["status"],
                best_val_loss=result.get("best_val_loss"),
                final_train_loss=result.get("final_train_loss"),
            )
        )
    lines.extend(["", "## Commands", ""])
    for result in results:
        lines.extend(["```bash", " ".join(result["train_cmd"]), "```", ""])
    (run_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def run_sft_experiment_grid(
    *,
    configs: list[str],
    steps: list[int],
    output_root: str | Path = "artifacts/sft_grid_v0_4",
    prompts_file: str | Path | None = None,
    dry_run: bool = False,
    timestamp: str | None = None,
    device: str | None = None,
    tokenizer: str = "data/tokenizers/sarych_bpe_8192_tinystories.json",
    max_new_tokens: int = 160,
) -> dict[str, Any]:
    timestamp = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(output_root) / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    console_log = run_dir / "console.log"
    prompts = _load_prompts(prompts_file)
    commands = build_experiment_commands(configs=configs, steps=steps, run_root=run_dir / "runs", device=device)
    results: list[dict[str, Any]] = []

    with console_log.open("w", encoding="utf-8") as console:
        for command in commands:
            result: dict[str, Any] = {
                "config": command.config,
                "steps": command.steps,
                "run_dir": str(command.run_dir),
                "train_cmd": command.train_cmd,
            }
            if dry_run:
                result.update({"status": "dry_run", "best_val_loss": None, "final_train_loss": None})
                console.write("DRY RUN: " + " ".join(command.train_cmd) + "\n")
            else:
                process = subprocess.run(
                    command.train_cmd,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
                console.write(process.stdout)
                summary = _read_log_summary(command.run_dir / "train_log.jsonl")
                generations = []
                if process.returncode == 0:
                    generations = _run_generations(
                        run_dir=command.run_dir,
                        prompts=prompts,
                        console=console,
                        device=device,
                        tokenizer=tokenizer,
                        max_new_tokens=max_new_tokens,
                    )
                result.update(
                    {
                        "status": "ok" if process.returncode == 0 else f"failed:{process.returncode}",
                        "best_val_loss": summary["best_val_loss"],
                        "final_train_loss": summary["final_train_loss"],
                        "train_log_tail": summary["tail"],
                        "generations": generations,
                    }
                )
            results.append(result)

    results_path = run_dir / "results.jsonl"
    with results_path.open("w", encoding="utf-8") as handle:
        for result in results:
            handle.write(json.dumps(result, ensure_ascii=False, sort_keys=True) + "\n")
    _write_report(run_dir, results, dry_run=dry_run)
    return {"run_dir": str(run_dir), "results_path": str(results_path), "console_log": str(console_log)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run short SARYCH v0.4 SFT grid experiments.")
    parser.add_argument(
        "--config",
        action="append",
        dest="configs",
        default=None,
        help="Config path. Repeatable.",
    )
    parser.add_argument("--steps", nargs="+", type=int, default=[100, 200, 300])
    parser.add_argument("--prompts-file", default=None)
    parser.add_argument("--out-root", default="artifacts/sft_grid_v0_4")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--device", default=None)
    parser.add_argument("--tokenizer", default="data/tokenizers/sarych_bpe_8192_tinystories.json")
    parser.add_argument("--max-new-tokens", type=int, default=160)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configs = args.configs or [
        "configs/v0_4_30m_instruct_lite_lr2e5.yaml",
        "configs/v0_4_30m_instruct_lite_lr1e5.yaml",
    ]
    result = run_sft_experiment_grid(
        configs=configs,
        steps=args.steps,
        output_root=args.out_root,
        prompts_file=args.prompts_file,
        dry_run=args.dry_run,
        device=args.device,
        tokenizer=args.tokenizer,
        max_new_tokens=args.max_new_tokens,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
