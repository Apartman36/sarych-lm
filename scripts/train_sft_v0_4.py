from __future__ import annotations

import argparse
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sarych.checkpoint import latest_checkpoint, load_checkpoint, save_checkpoint
from sarych.config import apply_cli_overrides, load_yaml_config, model_config_from_dict
from sarych.model import SarychLM
from sarych.reporting import JsonlLogger, ThroughputMeter, cuda_memory_report
from sarych.sampling import generate_tokens
from sarych.sft import SFTJsonlDataset, format_instruct_prompt
from sarych.tokenizer_bpe import SarychBPETokenizer
from sarych.train import learning_rate_for_step, set_optimizer_lr
from sarych.utils import choose_device, choose_dtype, ensure_dir, set_seed


def _make_autocast_context(device: torch.device, use_bf16_autocast: bool):
    if use_bf16_autocast:
        return lambda: torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext


@torch.no_grad()
def estimate_sft_loss(
    model: SarychLM,
    dataset: SFTJsonlDataset,
    *,
    batch_size: int,
    eval_iters: int,
    device: torch.device,
    autocast_context,
) -> float:
    model.eval()
    losses = []
    for _ in range(eval_iters):
        x, y = dataset.get_batch(batch_size=batch_size, device=device)
        with autocast_context():
            _, loss = model(x, y)
        losses.append(float(loss.detach().cpu()))
    model.train()
    return sum(losses) / len(losses)


def train_sft_from_config(config: dict[str, Any]) -> dict[str, Any]:
    paths = config["paths"]
    train_config = config["train"]
    dataset_config = config["dataset"]
    base_config = config["base"]
    ensure_dir(paths["run_dir"])
    ensure_dir(paths["checkpoint_dir"])
    ensure_dir(paths["sample_dir"])

    set_seed(int(config["seed"]))
    device = choose_device(str(train_config["device"]))
    dtype, use_bf16_autocast = choose_dtype(device, str(train_config["dtype"]))
    autocast_context = _make_autocast_context(device, use_bf16_autocast)

    tokenizer = SarychBPETokenizer.from_file(base_config["tokenizer_path"])
    model = SarychLM(model_config_from_dict(config)).to(device)
    parameter_count = model.count_parameters()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_config["lr"]),
        betas=(float(train_config["beta1"]), float(train_config["beta2"])),
        weight_decay=float(train_config["weight_decay"]),
    )

    train_data = SFTJsonlDataset(
        dataset_config["train_jsonl"],
        tokenizer,
        max_seq_len=int(dataset_config["max_seq_len"]),
        seed=int(config["seed"]),
    )
    val_data = SFTJsonlDataset(
        dataset_config["val_jsonl"],
        tokenizer,
        max_seq_len=int(dataset_config["max_seq_len"]),
        seed=int(config["seed"]) + 1,
    )

    start_step = 0
    best_val_loss: float | None = None
    scheduler_state: dict[str, Any] = {}
    if bool(train_config.get("resume", True)):
        checkpoint = latest_checkpoint(paths["checkpoint_dir"])
        if checkpoint is not None:
            metadata = load_checkpoint(
                checkpoint,
                model=model,
                optimizer=optimizer,
                map_location=device,
                strict_rng_restore=bool(train_config.get("strict_rng_restore", True)),
            )
            start_step = int(metadata["step"])
            best_val_loss = metadata.get("best_val_loss")
            scheduler_state = metadata.get("scheduler_state_dict", {})
            extra_state = metadata.get("extra_state", {})
            if "train_data" in extra_state:
                train_data.load_state_dict(extra_state["train_data"])
            if "val_data" in extra_state:
                val_data.load_state_dict(extra_state["val_data"])
            print(f"Resuming v0.4 SFT from {checkpoint} at step {start_step}.")
        else:
            print("No v0.4 checkpoint found; loading base checkpoint.")
            load_checkpoint(base_config["checkpoint_path"], model=model, optimizer=None, map_location=device, restore_rng=False)
    else:
        print("Resume disabled; loading base checkpoint.")
        load_checkpoint(base_config["checkpoint_path"], model=model, optimizer=None, map_location=device, restore_rng=False)

    train_model = model
    if bool(train_config.get("compile", False)):
        if not hasattr(torch, "compile"):
            raise RuntimeError("train.compile is true, but torch.compile is unavailable.")
        train_model = torch.compile(model)  # type: ignore[assignment]
        setattr(train_model, "config", model.config)

    logger = JsonlLogger(paths["log_path"])
    meter = ThroughputMeter()
    max_steps = int(train_config["max_steps"])
    micro_batch_size = int(train_config["micro_batch_size"])
    grad_accumulation_steps = int(train_config["grad_accumulation_steps"])
    tokens_per_step = micro_batch_size * int(dataset_config["max_seq_len"]) * grad_accumulation_steps
    running_loss = 0.0
    steps_since_log = 0
    last_checkpoint_path: Path | None = None

    train_model.train()
    progress = tqdm(range(start_step, max_steps), initial=start_step, total=max_steps, desc="sft")
    for zero_based_step in progress:
        step = zero_based_step + 1
        lr = learning_rate_for_step(zero_based_step, train_config)
        set_optimizer_lr(optimizer, lr)
        optimizer.zero_grad(set_to_none=True)
        loss_accum = 0.0

        for _ in range(grad_accumulation_steps):
            x, y = train_data.get_batch(batch_size=micro_batch_size, device=device)
            with autocast_context():
                _, loss = train_model(x, y)
                loss = loss / grad_accumulation_steps
            loss.backward()
            loss_accum += float(loss.detach().cpu()) * grad_accumulation_steps

        if float(train_config["grad_clip"]) > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(train_config["grad_clip"]))
        optimizer.step()

        running_loss += loss_accum
        steps_since_log += 1
        scheduler_state = {"last_lr": lr, "last_step": step}
        tokens_processed = step * tokens_per_step
        val_loss = None

        if step % int(train_config["eval_every"]) == 0 or step == max_steps:
            val_loss = estimate_sft_loss(
                train_model,
                val_data,
                batch_size=int(train_config["eval_batch_size"]),
                eval_iters=int(train_config["eval_iters"]),
                device=device,
                autocast_context=autocast_context,
            )
            if best_val_loss is None or val_loss < best_val_loss:
                best_val_loss = val_loss

        if step % int(train_config["log_every"]) == 0 or step == max_steps:
            tokens_per_sec, elapsed = meter.update(tokens_processed)
            train_loss = running_loss / max(1, steps_since_log)
            logger.write(
                {
                    "step": step,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "lr": lr,
                    "tokens_processed": tokens_processed,
                    "tokens_per_sec": tokens_per_sec,
                    "elapsed_sec": elapsed,
                    "device": str(device),
                    "dtype": "bf16" if use_bf16_autocast else "fp32",
                    "parameter_count": parameter_count,
                    **cuda_memory_report(),
                }
            )
            progress.set_postfix(loss=f"{train_loss:.4f}", lr=f"{lr:.2e}")
            running_loss = 0.0
            steps_since_log = 0

        if step % int(train_config["sample_every"]) == 0 or step == max_steps:
            prompt = format_instruct_prompt(str(train_config.get("sample_instruction", "Write a short story about a kind fox.")), "")
            prompt_ids = tokenizer.encode(prompt)
            input_ids = torch.tensor([prompt_ids[-model.config.block_size :]], dtype=torch.long, device=device)
            with autocast_context():
                sample = generate_tokens(
                    train_model,
                    input_ids,
                    max_new_tokens=80,
                    temperature=0.8,
                    top_k=50,
                    vocab_size_limit=tokenizer.vocab_size,
                )
            sample_text = tokenizer.decode(sample[0].detach().cpu().tolist())
            sample_path = Path(paths["sample_dir"]) / f"sample_step_{step:07d}.txt"
            sample_path.write_text(sample_text + "\n", encoding="utf-8")
            train_model.train()

        if step % int(train_config["checkpoint_every"]) == 0 or step == max_steps:
            is_best = bool(train_config.get("save_best", True)) and val_loss is not None and val_loss == best_val_loss
            last_checkpoint_path = save_checkpoint(
                checkpoint_dir=paths["checkpoint_dir"],
                model=model,
                optimizer=optimizer,
                scheduler_state=scheduler_state,
                step=step,
                best_val_loss=best_val_loss,
                config=config,
                parameter_count=parameter_count,
                is_best=is_best,
                environment={"device": str(device), "dtype": str(dtype), "bf16_autocast": use_bf16_autocast},
                extra_state={"train_data": train_data.state_dict(), "val_data": val_data.state_dict()},
            )

    return {
        "final_step": max_steps,
        "best_val_loss": best_val_loss,
        "last_checkpoint_path": str(last_checkpoint_path) if last_checkpoint_path is not None else None,
        "log_path": paths["log_path"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune SARYCH-LM v0.4 on filtered Xiaomi SFT data.")
    parser.add_argument("--config", default="configs/v0_4_30m_instruct_xiaomi.yaml")
    resume_group = parser.add_mutually_exclusive_group()
    resume_group.add_argument("--resume", action="store_true", default=None)
    resume_group.add_argument("--no-resume", action="store_false", dest="resume")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--run-dir", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml_config(args.config)
    config = apply_cli_overrides(
        config,
        resume=args.resume,
        max_steps=args.max_steps,
        device=args.device,
        run_dir=args.run_dir,
    )
    result = train_sft_from_config(config)
    print(result)


if __name__ == "__main__":
    main()
