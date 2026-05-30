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
from sarych.sampling import generate_tokens
from sarych.tokenizer_bpe import SarychBPETokenizer
from sarych.utils import choose_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate text with a SARYCH-LM v0.2 checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = checkpoint["config"]
    model = SarychLM(model_config_from_dict(config)).to(device)
    load_checkpoint(args.checkpoint, model=model, optimizer=None, map_location=device, restore_rng=False)

    tokenizer = SarychBPETokenizer.from_file(args.tokenizer)
    prompt_ids = tokenizer.encode(args.prompt)
    if not prompt_ids:
        prompt_ids = [tokenizer.token_to_id("<|endoftext|>") or 0]
    input_ids = torch.tensor([prompt_ids[-model.config.block_size :]], dtype=torch.long, device=device)
    output = generate_tokens(
        model,
        input_ids,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        vocab_size_limit=tokenizer.vocab_size,
    )
    print(tokenizer.decode(output[0].detach().cpu().tolist()))


if __name__ == "__main__":
    main()
