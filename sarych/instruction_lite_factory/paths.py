from __future__ import annotations

from pathlib import Path


def to_windows_path(path: Path | str) -> str:
    """Format a path for teacher prompts (OpenCode on Windows)."""
    text = str(Path(path))
    if text.startswith("\\\\") or (len(text) > 1 and text[1] == ":"):
        return text.replace("/", "\\")
    # WSL / POSIX -> best-effort Windows drive mapping
    if text.startswith("/mnt/c/"):
        return "C:\\" + text[len("/mnt/c/") :].replace("/", "\\")
    if text.startswith("/mnt/"):
        parts = text.split("/")
        if len(parts) >= 4 and len(parts[2]) == 1:
            drive = parts[2].upper()
            rest = "\\".join(parts[3:])
            return f"{drive}:\\{rest}"
    return text.replace("/", "\\")


def factory_layout(factory_dir: Path) -> dict[str, Path]:
    root = factory_dir
    return {
        "root": root,
        "shards_seeds": root / "shards" / "seeds",
        "shards_prompts": root / "shards" / "prompts",
        "shards_raw": root / "shards" / "raw",
        "shards_accepted": root / "shards" / "accepted",
        "shards_rejected": root / "shards" / "rejected",
        "shards_manifests": root / "shards" / "manifests",
        "manifests": root / "manifests",
        "reports": root / "reports",
        "repairs": root / "repairs",
    }
