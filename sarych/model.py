from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class SarychConfig:
    vocab_size: int = 1024
    block_size: int = 128
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 256
    d_ff: int = 768
    dropout: float = 0.0
    bias: bool = False
    norm: str = "rmsnorm"
    activation: str = "swiglu"
    position_encoding: str = "rope"
    tie_embeddings: bool = True

    def __post_init__(self) -> None:
        if self.n_embd % self.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head.")
        if (self.n_embd // self.n_head) % 2 != 0:
            raise ValueError("RoPE requires an even head dimension.")
        if self.norm != "rmsnorm":
            raise ValueError("v0.1 only supports RMSNorm.")
        if self.activation != "swiglu":
            raise ValueError("v0.1 only supports SwiGLU.")
        if self.position_encoding != "rope":
            raise ValueError("v0.1 only supports RoPE position encoding.")


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normed = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return normed * self.weight


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, block_size: int, base: float = 10_000.0) -> None:
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError("RoPE head_dim must be even.")
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        positions = torch.arange(block_size, dtype=torch.float32)
        freqs = torch.outer(positions, inv_freq)
        self.register_buffer("cos", freqs.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin", freqs.sin()[None, None, :, :], persistent=False)

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        time = q.size(-2)
        cos = self.cos[:, :, :time, :].to(device=q.device, dtype=q.dtype)
        sin = self.sin[:, :, :time, :].to(device=q.device, dtype=q.dtype)
        return apply_rope(q, cos, sin), apply_rope(k, cos, sin)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    rotated = torch.stack((x_even * cos - x_odd * sin, x_even * sin + x_odd * cos), dim=-1)
    return rotated.flatten(-2)


class CausalSelfAttention(nn.Module):
    def __init__(self, config: SarychConfig) -> None:
        super().__init__()
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.rope = RotaryEmbedding(self.head_dim, config.block_size)
        mask = torch.tril(torch.ones(config.block_size, config.block_size, dtype=torch.bool))
        self.register_buffer("causal_mask", mask.view(1, 1, config.block_size, config.block_size), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, time, channels = x.shape
        qkv = self.c_attn(x)
        q, k, v = qkv.split(channels, dim=2)
        q = q.view(batch, time, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(batch, time, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(batch, time, self.n_head, self.head_dim).transpose(1, 2)
        q, k = self.rope(q, k)

        dropout_p = self.attn_dropout.p if self.training else 0.0
        if hasattr(F, "scaled_dot_product_attention"):
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=dropout_p, is_causal=True)
        else:
            att = (q @ k.transpose(-2, -1)) * (self.head_dim**-0.5)
            mask = self.causal_mask[:, :, :time, :time]
            att = att.masked_fill(~mask, torch.finfo(att.dtype).min)
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v

        y = y.transpose(1, 2).contiguous().view(batch, time, channels)
        return self.resid_dropout(self.c_proj(y))


class SwiGLU(nn.Module):
    def __init__(self, config: SarychConfig) -> None:
        super().__init__()
        self.gate = nn.Linear(config.n_embd, config.d_ff, bias=config.bias)
        self.up = nn.Linear(config.n_embd, config.d_ff, bias=config.bias)
        self.down = nn.Linear(config.d_ff, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.down(F.silu(self.gate(x)) * self.up(x)))


class Block(nn.Module):
    def __init__(self, config: SarychConfig) -> None:
        super().__init__()
        self.ln_1 = RMSNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = RMSNorm(config.n_embd)
        self.mlp = SwiGLU(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class SarychLM(nn.Module):
    def __init__(self, config: SarychConfig) -> None:
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(
            {
                "wte": nn.Embedding(config.vocab_size, config.n_embd),
                "drop": nn.Dropout(config.dropout),
                "h": nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                "ln_f": RMSNorm(config.n_embd),
            }
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        if config.tie_embeddings:
            self.lm_head.weight = self.transformer["wte"].weight
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor, targets: torch.Tensor | None = None):
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape (batch, time).")
        batch, time = input_ids.shape
        if time > self.config.block_size:
            raise ValueError(f"Sequence length {time} exceeds block_size {self.config.block_size}.")

        x = self.transformer["wte"](input_ids)
        x = self.transformer["drop"](x)
        for block in self.transformer["h"]:
            x = block(x)
        x = self.transformer["ln_f"](x)
        logits = self.lm_head(x)

        if targets is None:
            return logits
        if targets.shape != input_ids.shape:
            raise ValueError("targets must have the same shape as input_ids.")
        loss = F.cross_entropy(logits.reshape(batch * time, -1), targets.reshape(batch * time))
        return logits, loss

    def count_parameters(self, trainable_only: bool = True) -> int:
        params = self.parameters()
        if trainable_only:
            return sum(p.numel() for p in params if p.requires_grad)
        return sum(p.numel() for p in params)

    def estimate_model_size_mb(self) -> float:
        total_bytes = sum(p.numel() * p.element_size() for p in self.parameters())
        total_bytes += sum(b.numel() * b.element_size() for b in self.buffers())
        return total_bytes / (1024**2)


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    if hasattr(model, "count_parameters"):
        return int(model.count_parameters(trainable_only=trainable_only))
    return sum(p.numel() for p in model.parameters() if p.requires_grad or not trainable_only)


def estimate_model_size(model: nn.Module) -> float:
    total_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    total_bytes += sum(b.numel() * b.element_size() for b in model.buffers())
    return total_bytes / (1024**2)
