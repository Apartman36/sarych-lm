from __future__ import annotations

import os
import random
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def choose_device(requested: str = "auto") -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false.")
    return device


def choose_dtype(device: torch.device, requested: str = "auto") -> tuple[torch.dtype, bool]:
    if requested == "auto":
        if device.type == "cuda" and torch.cuda.is_bf16_supported():
            return torch.bfloat16, True
        return torch.float32, False
    normalized = requested.lower()
    if normalized in {"bf16", "bfloat16"}:
        if device.type != "cuda" or not torch.cuda.is_bf16_supported():
            raise RuntimeError("BF16 was requested, but CUDA BF16 support is unavailable.")
        return torch.bfloat16, True
    if normalized in {"fp32", "float32"}:
        return torch.float32, False
    raise ValueError(f"Unsupported dtype setting: {requested}")


def get_git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=os.getcwd(),
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    commit = result.stdout.strip()
    return commit if result.returncode == 0 and commit else None


def detach_for_json(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().item() if value.ndim == 0 else value.detach().cpu().tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value
