from __future__ import annotations

import json
from copy import deepcopy

import torch

from sarych.checkpoint import save_checkpoint
from sarych.config import load_yaml_config, model_config_from_dict
from sarych.model import SarychLM
from sarych.tokenizer_bpe import train_byte_bpe_tokenizer
from scripts.prepare_text_dataset_v0_2 import _read_text_with_eot, prepare_text_dataset


def test_v0_3_config_builds_30m_class_model():
    config = load_yaml_config("configs/v0_3_30m_tinystories_base.yaml")

    assert config["run_name"] == "sarych-30m-tinystories-base"
    assert config["model"]["vocab_size"] == 8192
    assert config["model"]["block_size"] == 512
    assert config["model"]["tie_embeddings"] is True

    model = SarychLM(model_config_from_dict(config))
    parameter_count = model.count_parameters()

    assert 25_000_000 <= parameter_count <= 35_000_000


def test_v0_3_model_forward_backward_on_tiny_random_batch():
    config = load_yaml_config("configs/v0_3_30m_tinystories_base.yaml")
    config["model"].update({"block_size": 16})
    model = SarychLM(model_config_from_dict(config))
    x = torch.randint(0, config["model"]["vocab_size"], (2, 16), dtype=torch.long)
    y = torch.randint(0, config["model"]["vocab_size"], (2, 16), dtype=torch.long)

    _, loss = model(x, y)
    loss.backward()

    assert torch.isfinite(loss)


def test_prepare_text_dataset_records_separate_validation_source(tmp_path):
    train_text = tmp_path / "train.txt"
    val_text = tmp_path / "valid.txt"
    train_text.write_text(("A small fox packed lunch for the road.\n\n" * 20), encoding="utf-8")
    val_text.write_text(("A kind bear waited near the red gate.\n\n" * 12), encoding="utf-8")
    tokenizer_path = tmp_path / "tokenizer.json"
    tokenizer = train_byte_bpe_tokenizer(
        input_paths=[train_text],
        output_path=tokenizer_path,
        vocab_size=512,
        special_tokens=["<|endoftext|>", "<|pad|>"],
    )

    metadata = prepare_text_dataset(
        input_path=train_text,
        val_input_path=val_text,
        tokenizer_path=tokenizer_path,
        output_dir=tmp_path / "processed",
        block_size=8,
        val_fraction=0.9,
    )

    metadata_path = tmp_path / "processed" / "metadata.json"
    on_disk = json.loads(metadata_path.read_text(encoding="utf-8"))

    assert metadata == on_disk
    assert metadata["train_source_path"] == str(train_text)
    assert metadata["val_source_path"] == str(val_text)
    assert metadata["validation_mode"] == "separate_file"
    assert metadata["val_fraction"] == 0.9
    assert metadata["val_fraction_used_for_split"] is False
    assert metadata["vocab_size"] == tokenizer.vocab_size
    assert metadata["train_token_count"] == len(tokenizer.encode(_read_text_with_eot(train_text)))
    assert metadata["val_token_count"] == len(tokenizer.encode(_read_text_with_eot(val_text)))


def test_prepare_text_dataset_streams_input_files(monkeypatch, tmp_path):
    train_text = tmp_path / "train.txt"
    val_text = tmp_path / "valid.txt"
    train_text.write_text(("A red hen made a cake for her friends.\n\n" * 20), encoding="utf-8")
    val_text.write_text(("A blue bird sang beside the gate.\n\n" * 12), encoding="utf-8")
    tokenizer_path = tmp_path / "tokenizer.json"
    train_byte_bpe_tokenizer(
        input_paths=[train_text],
        output_path=tokenizer_path,
        vocab_size=512,
        special_tokens=["<|endoftext|>", "<|pad|>"],
    )

    original_read_text = type(train_text).read_text

    def guarded_read_text(self, *args, **kwargs):
        if self in {train_text, val_text}:
            raise AssertionError("dataset preparation should stream raw text files")
        return original_read_text(self, *args, **kwargs)

    monkeypatch.setattr(type(train_text), "read_text", guarded_read_text)

    metadata = prepare_text_dataset(
        input_path=train_text,
        val_input_path=val_text,
        tokenizer_path=tokenizer_path,
        output_dir=tmp_path / "processed",
        block_size=8,
        val_fraction=0.1,
    )

    assert metadata["validation_mode"] == "separate_file"
    assert metadata["train_token_count"] > 8
    assert metadata["val_token_count"] > 8


def test_generate_v0_3_core_loads_checkpoint_and_tokenizer(tmp_path):
    from scripts.generate_v0_3 import generate_text

    sample = tmp_path / "sample.txt"
    sample.write_text(
        "Once upon a time there was a little kite.\n<|endoftext|>\nThe kite flew over a hill.\n",
        encoding="utf-8",
    )
    tokenizer_path = tmp_path / "tokenizer.json"
    tokenizer = train_byte_bpe_tokenizer(
        input_paths=[sample],
        output_path=tokenizer_path,
        vocab_size=512,
        special_tokens=["<|endoftext|>", "<|pad|>"],
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
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_path = save_checkpoint(
        checkpoint_dir=checkpoint_dir,
        model=model,
        optimizer=optimizer,
        scheduler_state={},
        step=1,
        best_val_loss=None,
        config=config,
        parameter_count=model.count_parameters(),
    )

    text = generate_text(
        checkpoint_path=checkpoint_path,
        tokenizer_path=tokenizer_path,
        prompt="Once",
        max_new_tokens=2,
        temperature=0.0,
        top_k=10,
        device="cpu",
    )

    assert isinstance(text, str)
    assert text
