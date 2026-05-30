from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from sarych.model import SarychConfig


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Config file must contain a mapping: {path}")
    return data


def save_yaml_config(config: dict[str, Any], path: str | Path) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)


def model_config_from_dict(config: dict[str, Any]) -> SarychConfig:
    model_data = deepcopy(config["model"])
    return SarychConfig(**model_data)


def apply_cli_overrides(
    config: dict[str, Any],
    *,
    resume: bool | None = None,
    max_steps: int | None = None,
    device: str | None = None,
    run_dir: str | None = None,
) -> dict[str, Any]:
    updated = deepcopy(config)
    if resume is not None:
        updated["train"]["resume"] = resume
    if max_steps is not None:
        updated["train"]["max_steps"] = max_steps
    if device is not None:
        updated["train"]["device"] = device
    if run_dir is not None:
        updated["paths"]["run_dir"] = run_dir
        updated["paths"]["checkpoint_dir"] = str(Path(run_dir) / "checkpoints")
        updated["paths"]["log_path"] = str(Path(run_dir) / "train_log.jsonl")
        updated["paths"]["env_report_path"] = str(Path(run_dir) / "env_report.txt")
        updated["paths"]["sample_dir"] = str(Path(run_dir) / "samples")
    return updated
