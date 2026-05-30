from __future__ import annotations

import torch
from torch.nn import functional as F


@torch.no_grad()
def generate_tokens(
    model,
    input_ids: torch.Tensor,
    *,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int | None = None,
    vocab_size_limit: int | None = None,
) -> torch.Tensor:
    model.eval()
    for _ in range(max_new_tokens):
        context = input_ids[:, -model.config.block_size :]
        logits = model(context)
        next_logits = logits[:, -1, :]
        if vocab_size_limit is not None:
            next_logits = next_logits[:, :vocab_size_limit]
        if temperature <= 0:
            next_token = torch.argmax(next_logits, dim=-1, keepdim=True)
        else:
            next_logits = next_logits / temperature
            if top_k is not None and top_k > 0:
                values, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
                next_logits = next_logits.masked_fill(next_logits < values[:, [-1]], -float("inf"))
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
        input_ids = torch.cat((input_ids, next_token), dim=1)
    return input_ids


def decode_token_ids(token_ids: torch.Tensor) -> str:
    return " ".join(f"tok_{int(token)}" for token in token_ids.detach().cpu().flatten())
