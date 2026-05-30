from __future__ import annotations

from copy import deepcopy

import torch

from sarych.config import load_yaml_config, model_config_from_dict
from sarych.data_text import MemmapTokenDataset
from sarych.model import SarychLM
from sarych.tokenizer_bpe import train_byte_bpe_tokenizer
from sarych.train import train_from_config
from scripts.prepare_text_dataset_v0_2 import prepare_text_dataset


def test_v0_2_model_forward_backward_on_memmap_batch(tmp_path):
    sample = tmp_path / "sample.txt"
    sample.write_text(
        "Lina had a blue box. She put a soft ball in it.\n"
        "<|endoftext|>\n"
        "Tom saw the box and asked to play.\n",
        encoding="utf-8",
    )
    tokenizer_path = tmp_path / "tokenizer.json"
    tokenizer = train_byte_bpe_tokenizer(
        input_paths=[sample],
        output_path=tokenizer_path,
        vocab_size=512,
        special_tokens=["<|endoftext|>", "<|pad|>"],
    )
    processed = tmp_path / "processed"
    prepare_text_dataset(
        input_path=sample,
        tokenizer_path=tokenizer_path,
        output_dir=processed,
        block_size=8,
        val_fraction=0.2,
    )

    config = deepcopy(load_yaml_config("configs/v0_2_tinystories_smoke.yaml"))
    config["model"].update(
        {
            "vocab_size": tokenizer.vocab_size,
            "block_size": 8,
            "n_layer": 1,
            "n_head": 2,
            "n_embd": 16,
            "d_ff": 48,
        }
    )
    model = SarychLM(model_config_from_dict(config))
    dataset = MemmapTokenDataset(processed / "train.bin", block_size=8, seed=11, vocab_size=tokenizer.vocab_size)
    x, y = dataset.get_batch(batch_size=2, device="cpu")
    _, loss = model(x, y)
    loss.backward()

    assert torch.isfinite(loss)


def test_train_v0_2_runs_one_step_on_sample_memmap(tmp_path):
    sample = tmp_path / "sample.txt"
    sample.write_text(
        "A little cat sat by a sunny door.\n<|endoftext|>\nIt saw a bug and did not chase it.\n",
        encoding="utf-8",
    )
    tokenizer_path = tmp_path / "tokenizer.json"
    tokenizer = train_byte_bpe_tokenizer(
        input_paths=[sample],
        output_path=tokenizer_path,
        vocab_size=512,
        special_tokens=["<|endoftext|>", "<|pad|>"],
    )
    processed = tmp_path / "processed"
    prepare_text_dataset(
        input_path=sample,
        tokenizer_path=tokenizer_path,
        output_dir=processed,
        block_size=8,
        val_fraction=0.2,
    )

    config = deepcopy(load_yaml_config("configs/v0_2_tinystories_smoke.yaml"))
    run_dir = tmp_path / "run"
    config["model"].update(
        {
            "vocab_size": tokenizer.vocab_size,
            "block_size": 8,
            "n_layer": 1,
            "n_head": 2,
            "n_embd": 16,
            "d_ff": 48,
        }
    )
    config["train"].update(
        {
            "device": "cpu",
            "dtype": "fp32",
            "max_steps": 1,
            "micro_batch_size": 2,
            "grad_accumulation_steps": 1,
            "eval_batch_size": 2,
            "eval_iters": 1,
            "log_every": 1,
            "eval_every": 1,
            "sample_every": 1,
            "checkpoint_every": 1,
            "resume": False,
        }
    )
    config["dataset"].update(
        {
            "tokenizer_path": str(tokenizer_path),
            "train_bin": str(processed / "train.bin"),
            "val_bin": str(processed / "val.bin"),
            "block_size": 8,
        }
    )
    config["paths"] = {
        "run_dir": str(run_dir),
        "checkpoint_dir": str(run_dir / "checkpoints"),
        "log_path": str(run_dir / "train_log.jsonl"),
        "env_report_path": str(run_dir / "env_report.txt"),
        "sample_dir": str(run_dir / "samples"),
    }

    result = train_from_config(config)

    assert result["final_step"] == 1
    assert (run_dir / "checkpoints" / "checkpoint_latest.pt").exists()
