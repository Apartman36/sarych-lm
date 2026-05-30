from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from scripts.analyze_sft_jsonl import analyze_sft_jsonl
from scripts.diagnose_sft_v0_4 import diagnose_rows
from scripts.make_sft_seed_prompts import TARGET_DISTRIBUTION_1000, make_seed_rows, write_seed_output
from scripts.merge_sft_shards import merge_sft_shards
from sarych.tokenizer_bpe import train_byte_bpe_tokenizer


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def _raw_row(row_id: str, task_type: str, instruction: str, output: str) -> dict:
    return {
        "id": row_id,
        "source": "xiaomi_mimo_v2_5_pro",
        "task_type": task_type,
        "instruction": instruction,
        "input": "",
        "output": output,
        "language": "en",
        "metadata": {"generator": "fixture"},
    }


def _tokenizer(tmp_path: Path):
    corpus = tmp_path / "corpus.txt"
    corpus.write_text(
        "\n".join(
            [
                "<|user|>\nInstruction\n\n<|assistant|>\nThe fox asked for help.<|endoftext|>",
                "The red fox asked the owl for help.",
                "The blue bird sang beside the pond.",
            ]
        ),
        encoding="utf-8",
    )
    return train_byte_bpe_tokenizer(
        input_paths=[corpus],
        output_path=tmp_path / "tokenizer.json",
        vocab_size=384,
        special_tokens=["<|endoftext|>", "<|pad|>"],
    )


def test_make_sft_seed_prompts_is_deterministic_and_exact_for_1000():
    rows = make_seed_rows(count=1000, start_id=1, seed=1337)
    again = make_seed_rows(count=1000, start_id=1, seed=1337)

    assert rows == again
    assert len(rows) == 1000
    assert rows[0]["seed_id"] == "seed_000001"
    assert rows[-1]["seed_id"] == "seed_001000"
    assert rows[0]["target_sft_id"] == "xm_sft_000001"
    assert rows[-1]["target_sft_id"] == "xm_sft_001000"
    assert len({row["seed_id"] for row in rows}) == 1000
    assert len({row["target_sft_id"] for row in rows}) == 1000
    assert Counter(row["task_type"] for row in rows) == TARGET_DISTRIBUTION_1000
    assert len({row["instruction_blueprint"] for row in rows}) == 1000
    assert not any("Once there was" in row["instruction_blueprint"] for row in rows)
    assert not any("Once upon a time" in row["instruction_blueprint"] for row in rows)


def test_make_sft_seed_prompts_writes_shards(tmp_path):
    rows = make_seed_rows(count=25, start_id=7, seed=9)
    paths = write_seed_output(rows=rows, out=None, out_dir=tmp_path / "shards", shard_size=10)

    assert [path.name for path in paths] == ["sft_seeds_0001.jsonl", "sft_seeds_0002.jsonl", "sft_seeds_0003.jsonl"]
    assert sum(len(path.read_text(encoding="utf-8").strip().splitlines()) for path in paths) == 25


def test_merge_sft_shards_sorts_valid_rows_and_rejects_bad_lines(tmp_path):
    shard_dir = tmp_path / "raw" / "shards"
    _write_jsonl(
        shard_dir / "b.jsonl",
        [_raw_row("xm_sft_000002", "simple_qa", "Answer a simple question about rain.", "Rain falls from clouds and waters plants.")],
    )
    (shard_dir / "a.jsonl").write_text(
        json.dumps(
            _raw_row("xm_sft_000001", "story_writing", "Write a short story about a fox helper.", "The fox saw a lost owl and helped it find the gate.")
        )
        + "\n"
        + "{bad json\n",
        encoding="utf-8",
    )

    manifest = merge_sft_shards(
        input_dir=shard_dir,
        out=tmp_path / "raw" / "merged.jsonl",
        manifest=tmp_path / "manifest.json",
    )

    merged_lines = (tmp_path / "raw" / "merged.jsonl").read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["id"] for line in merged_lines] == ["xm_sft_000001", "xm_sft_000002"]
    assert manifest["accepted_rows"] == 2
    assert manifest["rejected_rows"] == 1
    assert manifest["category_counts"] == {"simple_qa": 1, "story_writing": 1}


def test_analyze_sft_jsonl_reports_duplicates_and_common_openings(tmp_path):
    rows = [
        _raw_row("xm_sft_000001", "story_writing", "Write a short story about a fox helper.", "The fox helped a bird with a map."),
        _raw_row("xm_sft_000002", "story_writing", "Write a short story about a fox helper.", "The fox helped a turtle with a bell."),
    ]
    path = tmp_path / "raw.jsonl"
    _write_jsonl(path, rows)

    report = analyze_sft_jsonl(path)

    assert report["total_rows"] == 2
    assert report["duplicate_instruction_count"] == 1
    assert report["most_common_first_3_output_words"][0][0] == "the fox helped"


def test_diagnose_sft_rows_reports_shifted_supervision(tmp_path):
    tokenizer = _tokenizer(tmp_path)
    rows = [_raw_row("xm_sft_000001", "story_writing", "Instruction", "The fox asked for help.")]

    report = diagnose_rows(rows=rows, split="train", tokenizer=tokenizer, max_seq_len=64)

    assert report["num_examples"] == 1
    assert report["zero_supervised_label_examples"] == []
    assert report["first_supervised_token_is_eos_examples"] == []
    assert report["output_empty_or_only_eos_examples"] == []
    assert report["most_common_first_supervised_tokens"][0]["token"] != "<|endoftext|>"
