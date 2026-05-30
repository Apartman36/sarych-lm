from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import torch

from sarych.checkpoint import save_checkpoint
from sarych.config import load_yaml_config, model_config_from_dict
from sarych.model import SarychLM
from sarych.sft import (
    SFTJsonlDataset,
    build_sft_features,
    build_sft_splits,
    format_instruct_prompt,
    format_sft_text,
    validate_raw_sft_row,
)
from sarych.tokenizer_bpe import train_byte_bpe_tokenizer


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def _valid_row(row_id: str = "xm_sft_900001", task_type: str = "story_writing") -> dict:
    return {
        "id": row_id,
        "source": "xiaomi_mimo_v2_5_pro",
        "task_type": task_type,
        "instruction": "Write a short story about a gentle child who helps a friend.",
        "input": "",
        "output": (
            "Nina saw her friend Ben drop his crayons on the floor. She knelt beside him, "
            "picked them up one by one, and helped him draw a bright sun. Ben smiled and said thank you."
        ),
        "language": "en",
        "metadata": {
            "created_at": "2026-05-30T12:00:00Z",
            "generator": "manual",
            "model": "mimo-v2.5-pro",
            "temperature": 0.7,
            "max_tokens": 512,
            "prompt_template": "sft_v1",
        },
    }


def _tiny_tokenizer(tmp_path: Path):
    corpus = tmp_path / "corpus.txt"
    corpus.write_text(
        "\n".join(
            [
                "Write a short story about a gentle child who helps a friend.",
                "Nina saw her friend Ben drop his crayons on the floor.",
                "Plants need water because water helps them stay strong.",
                "<|user|>\n<|assistant|>\n<|endoftext|>",
            ]
        ),
        encoding="utf-8",
    )
    return train_byte_bpe_tokenizer(
        input_paths=[corpus],
        output_path=tmp_path / "tokenizer.json",
        vocab_size=512,
        special_tokens=["<|endoftext|>", "<|pad|>"],
    )


def test_raw_sft_schema_validation_accepts_expected_schema():
    result = validate_raw_sft_row(_valid_row())

    assert result.ok
    assert result.reason is None


def test_raw_sft_schema_validation_rejects_code_tasks():
    row = _valid_row(task_type="code_generation")

    result = validate_raw_sft_row(row)

    assert not result.ok
    assert result.reason == "invalid_task_type"


def test_build_sft_dataset_rejects_malformed_rows(tmp_path):
    _tiny_tokenizer(tmp_path)
    raw_path = tmp_path / "raw.jsonl"
    malformed = _valid_row()
    malformed.pop("output")
    _write_jsonl(raw_path, [malformed, _valid_row("xm_sft_900002", "simple_qa")])

    manifest = build_sft_splits(
        raw_path=raw_path,
        scored_path=None,
        tokenizer_path=tmp_path / "tokenizer.json",
        train_path=tmp_path / "processed" / "train.jsonl",
        val_path=tmp_path / "processed" / "val.jsonl",
        rejected_dir=tmp_path / "rejected",
        val_ratio=0.5,
        seed=123,
        max_seq_len=512,
    )

    assert manifest["total_raw_rows"] == 2
    assert manifest["accepted_rows"] == 1
    assert manifest["rejected_counts"]["malformed"] == 1
    assert (tmp_path / "rejected" / "malformed.jsonl").read_text(encoding="utf-8")


def test_build_sft_dataset_rejects_too_long_examples(tmp_path):
    _tiny_tokenizer(tmp_path)
    raw_path = tmp_path / "raw.jsonl"
    long_row = _valid_row()
    long_row["output"] = " ".join(["friendly"] * 80)
    _write_jsonl(raw_path, [long_row])

    manifest = build_sft_splits(
        raw_path=raw_path,
        scored_path=None,
        tokenizer_path=tmp_path / "tokenizer.json",
        train_path=tmp_path / "train.jsonl",
        val_path=tmp_path / "val.jsonl",
        rejected_dir=tmp_path / "rejected",
        val_ratio=0.2,
        seed=123,
        max_seq_len=24,
    )

    assert manifest["accepted_rows"] == 0
    assert manifest["rejected_counts"]["too_long"] == 1


def test_build_sft_dataset_deduplicates_normalized_instruction_and_output(tmp_path):
    _tiny_tokenizer(tmp_path)
    duplicate = _valid_row("xm_sft_900002")
    duplicate["instruction"] = "  Write a short story about a gentle child who helps a friend.  "
    duplicate["output"] = _valid_row()["output"].upper()
    raw_path = tmp_path / "raw.jsonl"
    _write_jsonl(raw_path, [_valid_row(), duplicate])

    manifest = build_sft_splits(
        raw_path=raw_path,
        scored_path=None,
        tokenizer_path=tmp_path / "tokenizer.json",
        train_path=tmp_path / "train.jsonl",
        val_path=tmp_path / "val.jsonl",
        rejected_dir=tmp_path / "rejected",
        val_ratio=0.5,
        seed=123,
        max_seq_len=512,
    )

    assert manifest["accepted_rows"] == 1
    assert manifest["rejected_counts"]["duplicates"] == 1


def test_train_val_split_is_deterministic_and_stratified(tmp_path):
    _tiny_tokenizer(tmp_path)
    rows = []
    names = ["one", "two", "three", "four", "five", "six", "seven", "eight"]
    for i in range(8):
        row = _valid_row(f"xm_sft_9{i:05d}", task_type=("story_writing" if i < 4 else "simple_qa"))
        row["instruction"] = f"Write a short story number {names[i]} about a gentle child helping a friend."
        row["output"] = (
            f"In story {i}, a child noticed a friend needed help with a small task. "
            "The child listened, shared a kind word, and worked slowly beside the friend. "
            "By the end, both children felt proud, calm, and ready to help again."
        )
        rows.append(row)
    raw_path = tmp_path / "raw.jsonl"
    _write_jsonl(raw_path, rows)

    kwargs = {
        "raw_path": raw_path,
        "scored_path": None,
        "tokenizer_path": tmp_path / "tokenizer.json",
        "rejected_dir": tmp_path / "rejected",
        "val_ratio": 0.25,
        "seed": 777,
        "max_seq_len": 512,
    }
    first = build_sft_splits(train_path=tmp_path / "a_train.jsonl", val_path=tmp_path / "a_val.jsonl", **kwargs)
    second = build_sft_splits(train_path=tmp_path / "b_train.jsonl", val_path=tmp_path / "b_val.jsonl", **kwargs)

    assert (tmp_path / "a_train.jsonl").read_text(encoding="utf-8") == (tmp_path / "b_train.jsonl").read_text(
        encoding="utf-8"
    )
    assert (tmp_path / "a_val.jsonl").read_text(encoding="utf-8") == (tmp_path / "b_val.jsonl").read_text(
        encoding="utf-8"
    )
    assert first["category_distribution"]["story_writing"]["val"] == 1
    assert first["category_distribution"]["simple_qa"]["val"] == 1
    assert first["accepted_rows"] == second["accepted_rows"]
    assert first["train_count"] == second["train_count"]
    assert first["val_count"] == second["val_count"]
    assert first["category_distribution"] == second["category_distribution"]


def test_sft_dataset_builds_output_only_labels(tmp_path):
    tokenizer = _tiny_tokenizer(tmp_path)
    row = _valid_row()
    features = build_sft_features(row, tokenizer, max_seq_len=128)

    prompt_ids = tokenizer.encode(format_sft_text(row, include_output=False))
    output_ids = tokenizer.encode(row["output"] + "<|endoftext|>")

    assert features.input_ids[: len(prompt_ids)] == prompt_ids
    assert features.labels[: len(prompt_ids)] == [-100] * len(prompt_ids)
    assert features.labels[len(prompt_ids) : len(prompt_ids) + len(output_ids)] == output_ids
    assert features.input_ids == prompt_ids + output_ids


def test_sft_jsonl_dataset_pads_labels_with_ignore_index(tmp_path):
    tokenizer = _tiny_tokenizer(tmp_path)
    data_path = tmp_path / "train.jsonl"
    _write_jsonl(data_path, [_valid_row()])
    dataset = SFTJsonlDataset(data_path, tokenizer, max_seq_len=128, seed=42)

    x, y = dataset.get_batch(batch_size=2, device=torch.device("cpu"))

    assert x.shape == (2, 128)
    assert y.shape == (2, 128)
    pad_id = tokenizer.token_to_id("<|pad|>")
    assert pad_id is not None
    assert torch.any(x == pad_id)
    assert torch.all(y[x == pad_id] == -100)
    assert torch.any(y != -100)


def test_model_forward_with_sft_ignore_labels_works():
    config = {
        "model": {
            "vocab_size": 128,
            "block_size": 16,
            "n_layer": 1,
            "n_head": 2,
            "n_embd": 16,
            "d_ff": 48,
            "dropout": 0.0,
            "bias": False,
            "norm": "rmsnorm",
            "activation": "swiglu",
            "position_encoding": "rope",
            "tie_embeddings": True,
        }
    }
    model = SarychLM(model_config_from_dict(config))
    input_ids = torch.randint(0, 128, (2, 16), dtype=torch.long)
    labels = input_ids.clone()
    labels[:, :8] = -100

    _, loss = model(input_ids, labels)
    loss.backward()

    assert torch.isfinite(loss)


def test_v0_4_config_loads():
    config = load_yaml_config("configs/v0_4_30m_instruct_xiaomi.yaml")

    assert config["run_name"] == "sarych-30m-instruct-xiaomi"
    assert config["model"]["vocab_size"] == 8192
    assert config["base"]["checkpoint_path"].endswith("checkpoint_latest.pt")
    assert config["dataset"]["max_seq_len"] == 512


def test_train_sft_v0_4_smoke_runs_two_cpu_steps(tmp_path):
    from scripts.train_sft_v0_4 import train_sft_from_config

    tokenizer = _tiny_tokenizer(tmp_path)
    config = deepcopy(load_yaml_config("configs/v0_4_30m_instruct_xiaomi.yaml"))
    config["model"].update({"vocab_size": tokenizer.vocab_size, "block_size": 128, "n_layer": 1, "n_head": 2, "n_embd": 16, "d_ff": 48})
    model = SarychLM(model_config_from_dict(config))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    base_checkpoint = save_checkpoint(
        checkpoint_dir=tmp_path / "base" / "checkpoints",
        model=model,
        optimizer=optimizer,
        scheduler_state={},
        step=1,
        best_val_loss=None,
        config=config,
        parameter_count=model.count_parameters(),
    )
    train_path = tmp_path / "train.jsonl"
    val_path = tmp_path / "val.jsonl"
    _write_jsonl(train_path, [_valid_row("xm_sft_900001"), _valid_row("xm_sft_900002", "simple_qa")])
    _write_jsonl(val_path, [_valid_row("xm_sft_900003", "dialogue")])
    config["base"]["checkpoint_path"] = str(base_checkpoint)
    config["base"]["tokenizer_path"] = str(tmp_path / "tokenizer.json")
    config["dataset"]["train_jsonl"] = str(train_path)
    config["dataset"]["val_jsonl"] = str(val_path)
    config["dataset"]["max_seq_len"] = 128
    config["train"].update(
        {
            "device": "cpu",
            "dtype": "fp32",
            "max_steps": 2,
            "micro_batch_size": 1,
            "grad_accumulation_steps": 1,
            "eval_batch_size": 1,
            "eval_iters": 1,
            "log_every": 1,
            "eval_every": 1,
            "sample_every": 100,
            "checkpoint_every": 2,
            "resume": False,
            "compile": False,
        }
    )
    config["paths"]["run_dir"] = str(tmp_path / "run")
    config["paths"]["checkpoint_dir"] = str(tmp_path / "run" / "checkpoints")
    config["paths"]["log_path"] = str(tmp_path / "run" / "train_log.jsonl")
    config["paths"]["sample_dir"] = str(tmp_path / "run" / "samples")

    result = train_sft_from_config(config)

    assert result["final_step"] == 2
    assert Path(result["last_checkpoint_path"]).exists()
    assert (tmp_path / "run" / "train_log.jsonl").exists()


def test_generate_instruct_prompt_format():
    prompt = format_instruct_prompt("Tell a tiny story.", "Use a fox.")

    assert prompt == "<|user|>\nTell a tiny story.\n\nUse a fox.\n\n<|assistant|>\n"
    assert format_instruct_prompt("Tell a tiny story.", "") == "<|user|>\nTell a tiny story.\n\n<|assistant|>\n"
