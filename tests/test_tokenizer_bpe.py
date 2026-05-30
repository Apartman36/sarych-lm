from __future__ import annotations

from pathlib import Path

from sarych.tokenizer_bpe import SarychBPETokenizer, train_byte_bpe_tokenizer


def test_byte_bpe_tokenizer_trains_saves_loads_and_roundtrips(tmp_path):
    sample = tmp_path / "sample.txt"
    sample.write_text(
        "Once upon a time, Mira had a red cup.\n"
        "<|endoftext|>\n"
        "The cup was small, but it made her smile.\n",
        encoding="utf-8",
    )
    output = tmp_path / "tokenizer.json"

    tokenizer = train_byte_bpe_tokenizer(
        input_paths=[sample],
        output_path=output,
        vocab_size=512,
        special_tokens=["<|endoftext|>", "<|pad|>"],
    )
    loaded = SarychBPETokenizer.from_file(output)

    text = "Once upon a time, Mira smiled."
    token_ids = loaded.encode(text)
    decoded = tokenizer.decode(token_ids)

    assert output.exists()
    assert token_ids
    assert all(0 <= token_id < loaded.vocab_size for token_id in token_ids)
    assert "Mira" in decoded


def test_repository_sample_text_exists():
    sample = Path("data/samples/tiny_stories_sample.txt")
    assert sample.exists()
    assert "<|endoftext|>" in sample.read_text(encoding="utf-8")
