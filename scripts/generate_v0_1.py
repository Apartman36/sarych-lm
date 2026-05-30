from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sarych.checkpoint import load_checkpoint
from sarych.config import model_config_from_dict
from sarych.model import SarychLM
from sarych.sampling import decode_token_ids, generate_tokens
from sarych.utils import choose_device, choose_dtype


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate integer-token samples from a v0.1 checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = raw["config"]
    device = choose_device(args.device)
    _, use_bf16_autocast = choose_dtype(device, "auto")
    model = SarychLM(model_config_from_dict(config)).to(device)
    load_checkpoint(args.checkpoint, model=model, map_location=device, restore_rng=False)
    prompt = torch.zeros((1, min(8, model.config.block_size)), dtype=torch.long, device=device)
    if use_bf16_autocast:
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = generate_tokens(
                model,
                prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
            )
    else:
        output = generate_tokens(
            model,
            prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
        )
    print(decode_token_ids(output[0]))


if __name__ == "__main__":
    main()
