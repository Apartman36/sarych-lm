from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from statistics import mean, median
from typing import Any

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sarych.checkpoint import load_checkpoint
from sarych.config import model_config_from_dict
from sarych.model import SarychLM
from sarych.sft import ASSISTANT_MARKER, EOT_TOKEN, IGNORE_INDEX, build_sft_features, format_instruct_prompt, format_sft_text
from sarych.tokenizer_bpe import SarychBPETokenizer
from sarych.utils import choose_device


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _stats(values: list[int]) -> dict[str, float | int | None]:
    if not values:
        return {"min": None, "max": None, "mean": None, "median": None}
    return {"min": min(values), "max": max(values), "mean": round(mean(values), 3), "median": round(median(values), 3)}


def _token_label(tokenizer: SarychBPETokenizer, token_id: int) -> str:
    text = tokenizer.decode([token_id])
    if text == EOT_TOKEN:
        return EOT_TOKEN
    return repr(text)


def _rank_and_prob(logits: torch.Tensor, token_id: int) -> dict[str, float | int]:
    probs = F.softmax(logits, dim=-1)
    prob = float(probs[token_id].detach().cpu())
    rank = int((logits > logits[token_id]).sum().detach().cpu()) + 1
    return {"prob": prob, "rank": rank}


@torch.no_grad()
def _topk_after_prompt(
    *,
    model: SarychLM,
    tokenizer: SarychBPETokenizer,
    rows: list[dict[str, Any]],
    device: torch.device,
    top_k: int,
    max_prompts: int,
) -> list[dict[str, Any]]:
    model.eval()
    eos_id = tokenizer.token_to_id(EOT_TOKEN)
    diagnostics = []
    for row in rows[:max_prompts]:
        prompt = format_instruct_prompt(str(row["instruction"]), str(row.get("input", "")))
        prompt_ids = tokenizer.encode(prompt)[-model.config.block_size :]
        input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        logits = model(input_ids)[:, -1, :tokenizer.vocab_size][0]
        values, indices = torch.topk(logits, min(top_k, logits.numel()))
        output_ids = tokenizer.encode(str(row["output"]).strip() + EOT_TOKEN)
        first_expected = output_ids[0] if output_ids else None
        item: dict[str, Any] = {
            "id": row.get("id"),
            "task_type": row.get("task_type"),
            "prompt": prompt,
            "top_tokens": [
                {"token_id": int(index), "token": _token_label(tokenizer, int(index)), "logit": float(value.detach().cpu())}
                for value, index in zip(values, indices)
            ],
        }
        if eos_id is not None:
            item["eos_after_assistant"] = _rank_and_prob(logits, eos_id)
        if first_expected is not None:
            item["expected_first_token"] = {
                "token_id": first_expected,
                "token": _token_label(tokenizer, first_expected),
                **_rank_and_prob(logits, first_expected),
            }
        diagnostics.append(item)
    return diagnostics


def diagnose_rows(*, rows: list[dict[str, Any]], split: str, tokenizer: SarychBPETokenizer, max_seq_len: int) -> dict[str, Any]:
    eos_id = tokenizer.token_to_id(EOT_TOKEN)
    token_lengths: list[int] = []
    supervised_counts: list[int] = []
    zero_supervised: list[str] = []
    first_supervised_eos: list[str] = []
    empty_or_only_eos_outputs: list[str] = []
    first_supervised_tokens: Counter[int] = Counter()
    supervised_total = 0
    eos_supervised = 0
    decoded_examples = []
    samples = []
    for row in rows:
        features = build_sft_features(row, tokenizer, max_seq_len=max_seq_len)
        token_lengths.append(len(features.input_ids))
        supervised = [label for label in features.labels if label != IGNORE_INDEX]
        supervised_counts.append(len(supervised))
        supervised_total += len(supervised)
        eos_supervised += sum(1 for label in supervised if label == eos_id)
        if not supervised:
            zero_supervised.append(str(row.get("id")))
        else:
            first_supervised_tokens[supervised[0]] += 1
            if eos_id is not None and supervised[0] == eos_id:
                first_supervised_eos.append(str(row.get("id")))
        output_ids_without_eos = [token for token in tokenizer.encode(str(row.get("output", "")).strip()) if token != eos_id]
        if not output_ids_without_eos:
            empty_or_only_eos_outputs.append(str(row.get("id")))
        if len(decoded_examples) < 3:
            decoded_examples.append(
                {
                    "id": row.get("id"),
                    "text": tokenizer.decode(features.input_ids),
                    "supervised_text": tokenizer.decode(supervised, skip_special_tokens=False),
                }
            )
        if len(samples) < 5:
            samples.append(
                {
                    "id": row.get("id"),
                    "task_type": row.get("task_type"),
                    "prompt": format_sft_text(row, include_output=False),
                    "response": str(row.get("output", "")),
                }
            )
    return {
        "split": split,
        "num_examples": len(rows),
        "category_distribution": dict(sorted(Counter(str(row.get("task_type")) for row in rows).items())),
        "token_length_distribution": _stats(token_lengths),
        "supervised_labels_per_example": _stats(supervised_counts),
        "zero_supervised_label_examples": zero_supervised,
        "first_supervised_token_is_eos_examples": first_supervised_eos,
        "output_empty_or_only_eos_examples": empty_or_only_eos_outputs,
        "eos_label_ratio": (eos_supervised / supervised_total) if supervised_total else None,
        "most_common_first_supervised_tokens": [
            {"token_id": token_id, "token": _token_label(tokenizer, token_id), "count": count}
            for token_id, count in first_supervised_tokens.most_common(20)
        ],
        "sample_prompt_response_pairs": samples,
        "decoded_first_3_examples": decoded_examples,
    }


def run_diagnostics(args: argparse.Namespace) -> dict[str, Any]:
    tokenizer = SarychBPETokenizer.from_file(args.tokenizer)
    train_rows = _read_jsonl(Path(args.train))
    val_rows = _read_jsonl(Path(args.val))
    report: dict[str, Any] = {
        "train": diagnose_rows(rows=train_rows, split="train", tokenizer=tokenizer, max_seq_len=args.max_seq_len),
        "val": diagnose_rows(rows=val_rows, split="val", tokenizer=tokenizer, max_seq_len=args.max_seq_len),
        "tokenizer_path": args.tokenizer,
        "base_checkpoint": args.base_checkpoint,
        "sft_checkpoint": args.sft_checkpoint,
        "assistant_marker_token_ids": tokenizer.encode(f"{ASSISTANT_MARKER}\n"),
        "eos_token_id": tokenizer.token_to_id(EOT_TOKEN),
    }
    checkpoint_path = args.sft_checkpoint or args.base_checkpoint
    if checkpoint_path:
        device = choose_device(args.device)
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model = SarychLM(model_config_from_dict(checkpoint["config"])).to(device)
        load_checkpoint(checkpoint_path, model=model, optimizer=None, map_location=device, restore_rng=False)
        probe_rows = train_rows[: args.probe_prompts] + val_rows[: args.probe_prompts]
        report["checkpoint_next_token_diagnostics"] = _topk_after_prompt(
            model=model,
            tokenizer=tokenizer,
            rows=probe_rows,
            device=device,
            top_k=args.top_k,
            max_prompts=args.probe_prompts,
        )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose v0.4 SFT data alignment, EOS bias, and optional checkpoint prompt logits.")
    parser.add_argument("--train", required=True, help="Processed train JSONL.")
    parser.add_argument("--val", required=True, help="Processed val JSONL.")
    parser.add_argument("--tokenizer", default="data/tokenizers/sarych_bpe_8192_tinystories.json")
    parser.add_argument("--base-checkpoint", default=None)
    parser.add_argument("--sft-checkpoint", default=None)
    parser.add_argument("--max-seq-len", type=int, default=512)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--probe-prompts", type=int, default=5)
    parser.add_argument("--out", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run_diagnostics(args)
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
