from __future__ import annotations

import os
import platform
import subprocess
import sys
from importlib import metadata
from pathlib import Path


def _safe_run(command: list[str], timeout: int = 15) -> str:
    try:
        result = subprocess.run(command, check=False, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.SubprocessError) as exc:
        return f"Unavailable: {exc}"
    text = (result.stdout or result.stderr).strip()
    return text if text else f"No output (exit code {result.returncode})"


def _package_version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "not installed"


def _git_commit() -> str:
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], check=False, capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.SubprocessError) as exc:
        return f"Unavailable: {exc}"
    commit = result.stdout.strip()
    if result.returncode == 0 and commit:
        return commit
    error = result.stderr.strip().splitlines()[0] if result.stderr.strip() else ""
    return f"Unavailable: {error or 'not a git repository with commits'}"


def collect_env_report() -> str:
    lines: list[str] = []
    lines.append("SARYCH-LM environment report")
    lines.append("=" * 32)
    lines.append(f"Platform: {platform.platform()}")
    lines.append(f"OS: {os.name}")
    lines.append(f"Python: {sys.version.replace(chr(10), ' ')}")
    lines.append(f"Python executable: {sys.executable}")
    lines.append(f"Current working directory: {os.getcwd()}")
    lines.append(f"Git commit: {_git_commit()}")

    try:
        import torch

        lines.append("")
        lines.append("PyTorch")
        lines.append(f"torch.__version__: {torch.__version__}")
        lines.append(f"torch.version.cuda: {torch.version.cuda}")
        lines.append(f"CUDA available: {torch.cuda.is_available()}")
        lines.append(f"CUDA device count: {torch.cuda.device_count() if torch.cuda.is_available() else 0}")
        if torch.cuda.is_available():
            index = torch.cuda.current_device()
            props = torch.cuda.get_device_properties(index)
            lines.append(f"GPU name: {torch.cuda.get_device_name(index)}")
            lines.append(f"Compute capability: {torch.cuda.get_device_capability(index)}")
            lines.append(f"BF16 support: {torch.cuda.is_bf16_supported()}")
            lines.append(f"CUDA memory total GB: {props.total_memory / (1024**3):.2f}")
    except Exception as exc:
        lines.append("")
        lines.append(f"PyTorch unavailable or failed to import: {exc}")

    lines.append("")
    lines.append("nvidia-smi")
    lines.append(_safe_run(["nvidia-smi"], timeout=15))

    lines.append("")
    lines.append("Key packages")
    for package in ["torch", "numpy", "PyYAML", "tqdm", "pytest"]:
        lines.append(f"{package}: {_package_version(package)}")

    return "\n".join(lines) + "\n"


def write_env_report(path: str | Path) -> str:
    text = collect_env_report()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return text
