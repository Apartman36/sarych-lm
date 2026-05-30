from __future__ import annotations

from copy import deepcopy

from sarych.config import load_yaml_config
from sarych.train import train_from_config


def _tiny_resume_config(tmp_path):
    config = deepcopy(load_yaml_config("configs/v0_1_synthetic_sanity.yaml"))
    config["seed"] = 123
    config["model"].update(
        {
            "vocab_size": 64,
            "block_size": 16,
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
            "max_steps": 2,
            "micro_batch_size": 2,
            "grad_accumulation_steps": 1,
            "eval_batch_size": 2,
            "eval_iters": 1,
            "log_every": 1,
            "eval_every": 2,
            "sample_every": 2,
            "checkpoint_every": 2,
            "resume": False,
        }
    )
    config["synthetic_data"].update(
        {
            "train_tokens": 512,
            "val_tokens": 256,
            "vocab_size": 64,
            "block_size": 16,
        }
    )
    run_dir = tmp_path / "v0_1_resume"
    config["paths"] = {
        "run_dir": str(run_dir),
        "checkpoint_dir": str(run_dir / "checkpoints"),
        "log_path": str(run_dir / "train_log.jsonl"),
        "env_report_path": str(run_dir / "env_report.txt"),
        "sample_dir": str(run_dir / "samples"),
    }
    return config


def test_train_v0_1_resumes_from_latest_checkpoint(tmp_path):
    config = _tiny_resume_config(tmp_path)

    first = train_from_config(config)
    assert first["final_step"] == 2

    resumed_config = deepcopy(config)
    resumed_config["train"]["max_steps"] = 4
    resumed_config["train"]["resume"] = True
    resumed = train_from_config(resumed_config)

    assert resumed["final_step"] == 4
    assert (tmp_path / "v0_1_resume" / "checkpoints" / "checkpoint_latest.pt").exists()
