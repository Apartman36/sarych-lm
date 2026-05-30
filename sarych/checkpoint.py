from __future__ import annotations

import random
import shutil
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from sarych.utils import ensure_dir, get_git_commit


def _rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python_random": random.getstate(),
        "numpy_random": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def _as_cpu_byte_tensor(value: Any, *, name: str) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        tensor = value.detach()
        if tensor.dtype != torch.uint8:
            tensor = tensor.to(dtype=torch.uint8)
        return tensor.cpu().contiguous()
    try:
        return torch.as_tensor(value, dtype=torch.uint8, device="cpu").contiguous()
    except Exception as exc:
        raise TypeError(f"{name} must be convertible to a CPU torch.uint8 ByteTensor.") from exc


def restore_rng_state(state: dict[str, Any] | None, *, strict: bool = True) -> None:
    if not state:
        return
    try:
        random.setstate(state["python_random"])
        np.random.set_state(state["numpy_random"])
        torch.set_rng_state(_as_cpu_byte_tensor(state["torch_cpu"], name="torch_cpu RNG state"))
        if torch.cuda.is_available() and "torch_cuda" in state:
            cuda_states = [
                _as_cpu_byte_tensor(cuda_state, name=f"torch_cuda RNG state {index}")
                for index, cuda_state in enumerate(state["torch_cuda"])
            ]
            torch.cuda.set_rng_state_all(cuda_states)
    except Exception as exc:
        message = (
            "Failed to restore checkpoint RNG state. This makes resumed training nondeterministic. "
            "Set train.strict_rng_restore=false only when you intentionally accept this."
        )
        if strict:
            raise RuntimeError(message) from exc
        warnings.warn(f"{message} Original error: {exc}", RuntimeWarning, stacklevel=2)


def save_checkpoint(
    *,
    checkpoint_dir: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler_state: dict[str, Any] | None,
    step: int,
    best_val_loss: float | None,
    config: dict[str, Any],
    parameter_count: int,
    is_best: bool = False,
    environment: dict[str, Any] | None = None,
    extra_state: dict[str, Any] | None = None,
) -> Path:
    checkpoint_dir = ensure_dir(checkpoint_dir)
    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler_state or {},
        "step": step,
        "best_val_loss": best_val_loss,
        "config": config,
        "rng_state": _rng_state(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": get_git_commit(),
        "parameter_count": parameter_count,
        "environment": environment or {},
        "extra_state": extra_state or {},
    }
    step_path = checkpoint_dir / f"checkpoint_step_{step:07d}.pt"
    latest_path = checkpoint_dir / "checkpoint_latest.pt"
    torch.save(payload, step_path)
    shutil.copy2(step_path, latest_path)
    if is_best:
        shutil.copy2(step_path, checkpoint_dir / "checkpoint_best.pt")
    return step_path


def latest_checkpoint(checkpoint_dir: str | Path) -> Path | None:
    path = Path(checkpoint_dir) / "checkpoint_latest.pt"
    return path if path.exists() else None


def load_checkpoint(
    checkpoint_path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    map_location: str | torch.device = "cpu",
    restore_rng: bool = True,
    strict_rng_restore: bool = True,
) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if restore_rng:
        restore_rng_state(checkpoint.get("rng_state"), strict=strict_rng_restore)
    return checkpoint
