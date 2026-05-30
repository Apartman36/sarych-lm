from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import torch

from sarych.utils import detach_for_json, ensure_dir


def cuda_memory_report() -> dict[str, float | None]:
    if not torch.cuda.is_available():
        return {
            "cuda_memory_allocated_gb": None,
            "cuda_memory_reserved_gb": None,
            "cuda_memory_peak_gb": None,
        }
    gb = 1024**3
    return {
        "cuda_memory_allocated_gb": torch.cuda.memory_allocated() / gb,
        "cuda_memory_reserved_gb": torch.cuda.memory_reserved() / gb,
        "cuda_memory_peak_gb": torch.cuda.max_memory_allocated() / gb,
    }


class JsonlLogger:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        ensure_dir(self.path.parent)

    def write(self, record: dict[str, Any]) -> None:
        clean = {key: detach_for_json(value) for key, value in record.items()}
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(clean, sort_keys=True) + "\n")


class ThroughputMeter:
    def __init__(self) -> None:
        self.start_time = time.perf_counter()
        self.last_time = self.start_time
        self.last_tokens = 0

    def update(self, tokens_processed: int) -> tuple[float, float]:
        now = time.perf_counter()
        elapsed = now - self.start_time
        delta_time = max(now - self.last_time, 1e-9)
        delta_tokens = tokens_processed - self.last_tokens
        self.last_time = now
        self.last_tokens = tokens_processed
        return delta_tokens / delta_time, elapsed
