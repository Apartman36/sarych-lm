from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class SyntheticTokenDataset:
    total_tokens: int
    vocab_size: int
    block_size: int
    pattern_mode: str = "mixed"
    seed: int = 1337

    def __post_init__(self) -> None:
        if self.total_tokens <= self.block_size + 1:
            raise ValueError("total_tokens must be larger than block_size + 1.")
        self._rng = np.random.default_rng(self.seed)
        self.tokens = torch.tensor(self._generate_tokens(self.total_tokens + 1), dtype=torch.long)

    def _generate_tokens(self, length: int) -> np.ndarray:
        mode = self.pattern_mode.lower()
        if mode == "random":
            return self._rng.integers(0, self.vocab_size, size=length, dtype=np.int64)
        if mode == "arithmetic":
            start = int(self._rng.integers(0, self.vocab_size))
            step = int(self._rng.choice([1, 2, 3, 5]))
            return (start + step * np.arange(length, dtype=np.int64)) % self.vocab_size
        if mode == "motif":
            motif_len = int(self._rng.integers(3, 9))
            motif = self._rng.integers(0, self.vocab_size, size=motif_len, dtype=np.int64)
            return np.resize(motif, length)
        if mode != "mixed":
            raise ValueError(f"Unsupported synthetic pattern_mode: {self.pattern_mode}")

        chunks: list[np.ndarray] = []
        produced = 0
        while produced < length:
            chunk_len = min(int(self._rng.integers(self.block_size, self.block_size * 4 + 1)), length - produced)
            choice = float(self._rng.random())
            if choice < 0.50:
                chunk = self._rng.integers(0, self.vocab_size, size=chunk_len, dtype=np.int64)
            elif choice < 0.75:
                motif_len = int(self._rng.integers(3, 12))
                motif = self._rng.integers(0, min(self.vocab_size, 128), size=motif_len, dtype=np.int64)
                chunk = np.resize(motif, chunk_len)
            else:
                start = int(self._rng.integers(0, self.vocab_size))
                step = int(self._rng.choice([1, 2, 3, 5, 8]))
                chunk = (start + step * np.arange(chunk_len, dtype=np.int64)) % self.vocab_size
            chunks.append(chunk)
            produced += chunk_len
        return np.concatenate(chunks)[:length]

    def get_batch(self, batch_size: int, device: str | torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        max_start = len(self.tokens) - self.block_size - 1
        starts = self._rng.integers(0, max_start + 1, size=batch_size)
        x = torch.stack([self.tokens[start : start + self.block_size] for start in starts])
        y = torch.stack([self.tokens[start + 1 : start + self.block_size + 1] for start in starts])
        return x.to(device=device, non_blocking=True), y.to(device=device, non_blocking=True)

    def state_dict(self) -> dict:
        return {"rng_state": self._rng.bit_generator.state}

    def load_state_dict(self, state: dict) -> None:
        self._rng.bit_generator.state = state["rng_state"]


class FixedBatchDataset:
    def __init__(self, batch_size: int, block_size: int, vocab_size: int, device: str | torch.device = "cpu") -> None:
        base = torch.arange(block_size + 1, dtype=torch.long)
        rows = [((base + offset) % vocab_size) for offset in range(batch_size)]
        tokens = torch.stack(rows).to(device)
        self.x = tokens[:, :-1].contiguous()
        self.y = tokens[:, 1:].contiguous()

    def get_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.x, self.y
