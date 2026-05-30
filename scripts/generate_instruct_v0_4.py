from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sarych.checkpoint import load_checkpoint
from sarych.config import model_config_from_dict
from sarych.model import SarychLM
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
    min_new_tokens_before_eos: int = 0,
    suppress_eos_for_first_n_tokens: int = 0,
    print_prompt: bool = True,
    debug_top_k: int = 0,
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
    eos_id = tokenizer.token_to_id("<|endoftext|>")
    generated: list[int] = []
    if debug_top_k > 0:
        with torch.no_grad():
            logits = model(input_ids)[:, -1, : tokenizer.vocab_size][0]
            values, indices = torch.topk(logits, min(debug_top_k, logits.numel()))
            debug_rows = []
            probs = F.softmax(logits, dim=-1)
            for value, index in zip(values, indices):
                token_id = int(index)
                debug_rows.append(
                    {
                        "token_id": token_id,
                        "token": tokenizer.decode([token_id]),
                        "logit": float(value.detach().cpu()),
                        "prob": float(probs[token_id].detach().cpu()),
                    }
                )
            print("next_token_debug=" + json.dumps(debug_rows, ensure_ascii=False), file=sys.stderr)
    for step in range(max_new_tokens):
        context = input_ids[:, -model.config.block_size :]
        with torch.no_grad():
            logits = model(context)[:, -1, : tokenizer.vocab_size]
        if eos_id is not None and step < max(min_new_tokens_before_eos, suppress_eos_for_first_n_tokens):
            logits[:, eos_id] = -float("inf")
        if temperature <= 0:
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
        else:
            logits = logits / temperature
            if top_k is not None and top_k > 0:
                values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits = logits.masked_fill(logits < values[:, [-1]], -float("inf"))
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
        token_id = int(next_token[0, 0].detach().cpu())
        generated.append(token_id)
        input_ids = torch.cat((input_ids, next_token), dim=1)
        if eos_id is not None and token_id == eos_id:
            if step == 0:
                print("warning: first generated token is <|endoftext|>", file=sys.stderr)
            break
    token_ids = prompt_ids + generated if print_prompt else generated
    return tokenizer.decode(token_ids)


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
    parser.add_argument("--min-new-tokens-before-eos", type=int, default=0)
    parser.add_argument("--suppress-eos-for-first-n-tokens", type=int, default=0)
    prompt_group = parser.add_mutually_exclusive_group()
    prompt_group.add_argument("--print-prompt", action="store_true", dest="print_prompt", default=True)
    prompt_group.add_argument("--no-print-prompt", action="store_false", dest="print_prompt")
    parser.add_argument("--debug-top-k", type=int, default=0, help="Print top-k next-token diagnostics after the prompt to stderr.")
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
            min_new_tokens_before_eos=args.min_new_tokens_before_eos,
            suppress_eos_for_first_n_tokens=args.suppress_eos_for_first_n_tokens,
            print_prompt=args.print_prompt,
            debug_top_k=args.debug_top_k,
        )
    )


if __name__ == "__main__":
    main()
