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
from sarych.sft import format_instruct_prompt
from sarych.tokenizer_bpe import SarychBPETokenizer
from sarych.utils import choose_device


def generate_instruct_text(
    *,
    checkpoint_path: str | Path,
    tokenizer_path: str | Path,
    instruction: str,
    input_text: str = "",
    max_new_tokens: int = 160,
    temperature: float = 0.8,
    top_k: int = 50,
    device: str = "auto",
) -> str:
    resolved_device = choose_device(device)
    checkpoint = torch.load(checkpoint_path, map_location=resolved_device, weights_only=False)
    model = SarychLM(model_config_from_dict(checkpoint["config"])).to(resolved_device)
    load_checkpoint(checkpoint_path, model=model, optimizer=None, map_location=resolved_device, restore_rng=False)
    tokenizer = SarychBPETokenizer.from_file(tokenizer_path)
    prompt = format_instruct_prompt(instruction, input_text)
    prompt_ids = tokenizer.encode(prompt)
    if not prompt_ids:
        prompt_ids = [tokenizer.token_to_id("<|endoftext|>") or 0]
    input_ids = torch.tensor([prompt_ids[-model.config.block_size :]], dtype=torch.long, device=resolved_device)
    output = generate_tokens(
        model,
        input_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        vocab_size_limit=tokenizer.vocab_size,
    )
    return tokenizer.decode(output[0].detach().cpu().tolist())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate from a SARYCH-LM v0.4 instruct checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokenizer", default="data/tokenizers/sarych_bpe_8192_tinystories.json")
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--input", default="")
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(
        generate_instruct_text(
            checkpoint_path=args.checkpoint,
            tokenizer_path=args.tokenizer,
            instruction=args.instruction,
            input_text=args.input,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            device=args.device,
        )
    )


if __name__ == "__main__":
    main()
