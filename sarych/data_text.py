from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


class MemmapTokenDataset:
    def __init__(
        self,
        bin_path: str | Path,
        *,
        block_size: int,
        seed: int = 1337,
        vocab_size: int | None = None,
    ) -> None:
        self.bin_path = Path(bin_path)
        self.block_size = int(block_size)
        self.vocab_size = vocab_size
        if self.block_size <= 0:
            raise ValueError("block_size must be positive.")
        if not self.bin_path.exists():
            raise FileNotFoundError(f"Token bin file not found: {self.bin_path}")
        self.tokens = np.memmap(self.bin_path, dtype=np.uint16, mode="r")
        if len(self.tokens) <= self.block_size + 1:
            raise ValueError("Token bin must contain more than block_size + 1 tokens.")
        if self.vocab_size is not None:
            max_token = int(np.max(self.tokens))
            if max_token >= self.vocab_size:
                raise ValueError(f"Token id {max_token} exceeds vocab_size {self.vocab_size}.")
        self._rng = np.random.default_rng(seed)

    def get_batch(self, batch_size: int, device: str | torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        max_start = len(self.tokens) - self.block_size - 1
        starts = self._rng.integers(0, max_start + 1, size=int(batch_size))
        x_np = np.stack([np.asarray(self.tokens[start : start + self.block_size], dtype=np.int64) for start in starts])
        y_np = np.stack(
            [np.asarray(self.tokens[start + 1 : start + self.block_size + 1], dtype=np.int64) for start in starts]
        )
        x = torch.from_numpy(x_np).long()
        y = torch.from_numpy(y_np).long()
        return x.to(device=device, non_blocking=True), y.to(device=device, non_blocking=True)

    def state_dict(self) -> dict:
        return {"rng_state": self._rng.bit_generator.state}

    def load_state_dict(self, state: dict) -> None:
        self._rng.bit_generator.state = state["rng_state"]
