from __future__ import annotations

import numpy as np
import torch

from sarych.data_text import MemmapTokenDataset
from sarych.tokenizer_bpe import train_byte_bpe_tokenizer
from scripts.prepare_text_dataset_v0_2 import prepare_text_dataset


def test_prepare_text_dataset_and_memmap_batches_shift_targets(tmp_path):
    sample = tmp_path / "sample.txt"
    sample.write_text(
        "Nia found a tiny hat. She put it on a toy bear.\n"
        "The bear sat on a red box and looked happy.\n"
        "<|endoftext|>\n"
        "The hat fell down. Nia picked it up and tried again.\n"
        "This time the bear kept the hat all day.\n",
        encoding="utf-8",
    )
    tokenizer_path = tmp_path / "tokenizer.json"
    tokenizer = train_byte_bpe_tokenizer(
        input_paths=[sample],
        output_path=tokenizer_path,
        vocab_size=512,
        special_tokens=["<|endoftext|>", "<|pad|>"],
    )
    output_dir = tmp_path / "processed"

    metadata = prepare_text_dataset(
        input_path=sample,
        tokenizer_path=tokenizer_path,
        output_dir=output_dir,
        block_size=8,
        val_fraction=0.2,
    )

    train_bin = output_dir / "train.bin"
    val_bin = output_dir / "val.bin"
    assert train_bin.exists()
    assert val_bin.exists()
    assert metadata["dtype"] == "uint16"
    assert metadata["vocab_size"] == tokenizer.vocab_size

    dataset = MemmapTokenDataset(train_bin, block_size=8, seed=7, vocab_size=tokenizer.vocab_size)
    x, y = dataset.get_batch(batch_size=4, device="cpu")

    assert x.shape == (4, 8)
    assert y.shape == (4, 8)
    assert x.dtype == torch.long
    assert y.dtype == torch.long
    assert torch.equal(x[:, 1:], y[:, :-1])
    assert int(x.max()) < tokenizer.vocab_size
    assert int(y.max()) < tokenizer.vocab_size

    raw = np.memmap(train_bin, dtype=np.uint16, mode="r")
    assert len(raw) == metadata["train_tokens"]
