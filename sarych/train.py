from __future__ import annotations

import math
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

from sarych.checkpoint import latest_checkpoint, load_checkpoint, save_checkpoint
from sarych.config import model_config_from_dict
from sarych.data_synthetic import SyntheticTokenDataset
from sarych.data_text import MemmapTokenDataset
from sarych.env_report import write_env_report
from sarych.model import SarychLM
from sarych.reporting import JsonlLogger, ThroughputMeter, cuda_memory_report
from sarych.sampling import decode_token_ids, generate_tokens
from sarych.tokenizer_bpe import SarychBPETokenizer
from sarych.utils import choose_device, choose_dtype, ensure_dir, set_seed


def learning_rate_for_step(step: int, train_config: dict[str, Any]) -> float:
    max_lr = float(train_config["lr"])
    min_lr = float(train_config["min_lr"])
    warmup_steps = int(train_config["warmup_steps"])
    max_steps = int(train_config["max_steps"])
    if step < warmup_steps:
        return max_lr * float(step + 1) / float(max(1, warmup_steps))
    if step >= max_steps:
        return min_lr
    decay_ratio = (step - warmup_steps) / float(max(1, max_steps - warmup_steps))
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)


def set_optimizer_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr


@torch.no_grad()
def estimate_loss(
    model: SarychLM,
    dataset,
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


def _make_autocast_context(device: torch.device, use_bf16_autocast: bool):
    if use_bf16_autocast:
        return lambda: torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext


def _build_datasets(config: dict[str, Any]) -> tuple[Any, Any, int, SarychBPETokenizer | None]:
    dataset_config = config.get("dataset")
    if dataset_config and dataset_config.get("type") == "memmap_text":
        block_size = int(dataset_config["block_size"])
        vocab_size = int(config["model"]["vocab_size"])
        tokenizer = SarychBPETokenizer.from_file(dataset_config["tokenizer_path"])
        train_data = MemmapTokenDataset(
            dataset_config["train_bin"],
            block_size=block_size,
            seed=int(config["seed"]),
            vocab_size=vocab_size,
        )
        val_data = MemmapTokenDataset(
            dataset_config["val_bin"],
            block_size=block_size,
            seed=int(config["seed"]) + 1,
            vocab_size=vocab_size,
        )
        return train_data, val_data, block_size, tokenizer

    data_config = config["synthetic_data"]
    train_data = SyntheticTokenDataset(
        total_tokens=int(data_config["train_tokens"]),
        vocab_size=int(data_config["vocab_size"]),
        block_size=int(data_config["block_size"]),
        pattern_mode=str(data_config["pattern_mode"]),
        seed=int(config["seed"]),
    )
    val_data = SyntheticTokenDataset(
        total_tokens=int(data_config["val_tokens"]),
        vocab_size=int(data_config["vocab_size"]),
        block_size=int(data_config["block_size"]),
        pattern_mode=str(data_config["pattern_mode"]),
        seed=int(config["seed"]) + 1,
    )
    return train_data, val_data, int(data_config["block_size"]), None


def train_from_config(config: dict[str, Any]) -> dict[str, Any]:
    paths = config["paths"]
    train_config = config["train"]
    ensure_dir(paths["run_dir"])
    ensure_dir(paths["checkpoint_dir"])
    ensure_dir(paths["sample_dir"])

    set_seed(int(config["seed"]))
    device = choose_device(str(train_config["device"]))
    dtype, use_bf16_autocast = choose_dtype(device, str(train_config["dtype"]))
    autocast_context = _make_autocast_context(device, use_bf16_autocast)

    env_text = write_env_report(paths["env_report_path"])
    print(env_text)
    print(f"Device: {device}")
    print(f"DType mode: {'bf16 autocast' if use_bf16_autocast else 'fp32'}")

    raw_model = SarychLM(model_config_from_dict(config)).to(device)
    parameter_count = raw_model.count_parameters()
    print(f"Parameters: {parameter_count:,}")
    print(f"Estimated model size: {raw_model.estimate_model_size_mb():.2f} MB")

    train_data, val_data, dataset_block_size, tokenizer = _build_datasets(config)

    optimizer = torch.optim.AdamW(
        raw_model.parameters(),
        lr=float(train_config["lr"]),
        betas=(float(train_config["beta1"]), float(train_config["beta2"])),
        weight_decay=float(train_config["weight_decay"]),
    )
    scheduler_state: dict[str, Any] = {}
    start_step = 0
    best_val_loss: float | None = None

    if bool(train_config.get("resume", True)):
        checkpoint = latest_checkpoint(paths["checkpoint_dir"])
        if checkpoint is not None:
            print(f"Resuming from checkpoint: {checkpoint}")
            metadata = load_checkpoint(
                checkpoint,
                model=raw_model,
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
            print(f"Resuming from checkpoint: step {start_step}")
        else:
            print("No checkpoint found; starting fresh.")

    model = raw_model
    if bool(train_config.get("compile", False)):
        if not hasattr(torch, "compile"):
            raise RuntimeError("train.compile is true, but torch.compile is unavailable.")
        model = torch.compile(raw_model)  # type: ignore[assignment]
        setattr(model, "config", raw_model.config)

    logger = JsonlLogger(paths["log_path"])
    meter = ThroughputMeter()
    max_steps = int(train_config["max_steps"])
    micro_batch_size = int(train_config["micro_batch_size"])
    grad_accumulation_steps = int(train_config["grad_accumulation_steps"])
    tokens_per_step = micro_batch_size * dataset_block_size * grad_accumulation_steps
    running_loss = 0.0
    steps_since_log = 0
    last_checkpoint_path: Path | None = None

    model.train()
    try:
        progress = tqdm(range(start_step, max_steps), initial=start_step, total=max_steps, desc="train")
        for zero_based_step in progress:
            step = zero_based_step + 1
            lr = learning_rate_for_step(zero_based_step, train_config)
            set_optimizer_lr(optimizer, lr)
            optimizer.zero_grad(set_to_none=True)
            loss_accum = 0.0

            for _ in range(grad_accumulation_steps):
                x, y = train_data.get_batch(batch_size=micro_batch_size, device=device)
                with autocast_context():
                    _, loss = model(x, y)
                    loss = loss / grad_accumulation_steps
                loss.backward()
                loss_accum += float(loss.detach().cpu()) * grad_accumulation_steps

            if float(train_config["grad_clip"]) > 0:
                torch.nn.utils.clip_grad_norm_(raw_model.parameters(), float(train_config["grad_clip"]))
            optimizer.step()

            running_loss += loss_accum
            steps_since_log += 1
            tokens_processed = step * tokens_per_step
            scheduler_state = {"last_lr": lr, "last_step": step}

            val_loss = None
            if step % int(train_config["eval_every"]) == 0 or step == max_steps:
                val_loss = estimate_loss(
                    model,
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
                record = {
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
                logger.write(record)
                progress.set_postfix(loss=f"{train_loss:.4f}", lr=f"{lr:.2e}")
                running_loss = 0.0
                steps_since_log = 0

            if step % int(train_config["sample_every"]) == 0 or step == max_steps:
                if tokenizer is None:
                    prompt = torch.zeros((1, min(8, model.config.block_size)), dtype=torch.long, device=device)
                else:
                    prompt_ids = tokenizer.encode(str(train_config.get("sample_prompt", "Once upon a time")))
                    if not prompt_ids:
                        prompt_ids = [tokenizer.token_to_id("<|endoftext|>") or 0]
                    prompt = torch.tensor([prompt_ids[-model.config.block_size :]], dtype=torch.long, device=device)
                with autocast_context():
                    sample = generate_tokens(
                        model,
                        prompt,
                        max_new_tokens=40,
                        temperature=0.9,
                        top_k=50,
                        vocab_size_limit=tokenizer.vocab_size if tokenizer is not None else None,
                    )
                sample_text = tokenizer.decode(sample[0].detach().cpu().tolist()) if tokenizer is not None else decode_token_ids(sample[0])
                sample_path = Path(paths["sample_dir"]) / f"sample_step_{step:07d}.txt"
                sample_path.write_text(sample_text + "\n", encoding="utf-8")
                model.train()

            should_checkpoint = step % int(train_config["checkpoint_every"]) == 0 or step == max_steps
            if should_checkpoint:
                is_best = bool(train_config.get("save_best", True)) and val_loss is not None and val_loss == best_val_loss
                last_checkpoint_path = save_checkpoint(
                    checkpoint_dir=paths["checkpoint_dir"],
                    model=raw_model,
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
    except KeyboardInterrupt:
        print("Interrupted. Saving checkpoint before exit.")
        last_checkpoint_path = save_checkpoint(
            checkpoint_dir=paths["checkpoint_dir"],
            model=raw_model,
            optimizer=optimizer,
            scheduler_state=scheduler_state,
            step=max(start_step, scheduler_state.get("last_step", start_step)),
            best_val_loss=best_val_loss,
            config=config,
            parameter_count=parameter_count,
            is_best=False,
            environment={"device": str(device), "dtype": str(dtype), "bf16_autocast": use_bf16_autocast},
            extra_state={"train_data": train_data.state_dict(), "val_data": val_data.state_dict()},
        )

    return {
        "final_step": max_steps,
        "best_val_loss": best_val_loss,
        "last_checkpoint_path": str(last_checkpoint_path) if last_checkpoint_path is not None else None,
        "log_path": paths["log_path"],
    }
