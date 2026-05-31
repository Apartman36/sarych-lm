from __future__ import annotations

import json
from pathlib import Path

from scripts.convert_dolly_lite_to_sft import convert_dolly_lite
from scripts.make_tinystories_replay_sft import make_tinystories_replay
from scripts.mix_sft_sources import mix_sft_sources, parse_source_spec
from sarych.sft import build_sft_splits
from sarych.tokenizer_bpe import train_byte_bpe_tokenizer
from scripts.run_sft_experiment_grid import build_experiment_commands, run_sft_experiment_grid


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def _sft_row(row_id: str, source: str, task_type: str, instruction: str, output: str) -> dict:
    return {
        "id": row_id,
        "source": source,
        "task_type": task_type,
        "instruction": instruction,
        "input": "",
        "output": output,
        "language": "en",
        "metadata": {"fixture": True},
    }


def _tiny_tokenizer(tmp_path: Path):
    corpus = tmp_path / "corpus.txt"
    corpus.write_text(
        "\n".join(
            [
                "<|user|>\nWrite a short simple story.\n\n<|assistant|>\nThe kind fox helped a little bird find a warm nest before sunset.<|endoftext|>",
                "Mia held a bright kite and walked slowly up the grassy hill with her dad.",
                "The kitten found a blue button, gave it back to Ben, and purred softly.",
                "Rain helped the tiny flowers stand tall in the garden after a warm morning.",
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


def _story_output(index: int) -> str:
    return (
        f"Story {index} began when a kind child found a small lost toy near the garden gate. "
        f"The child asked two friends for help, looked under the bench, and listened carefully. "
        f"At last they found the toy beside a flower pot and returned it before supper."
    )


def test_dolly_lite_filters_and_maps_to_allowed_task_types(tmp_path):
    raw_path = tmp_path / "dolly.jsonl"
    out_path = tmp_path / "dolly_lite.jsonl"
    manifest_path = tmp_path / "manifest.json"
    simple_output = (
        "A tiny fox wanted to help her friend. She listened carefully, carried a small bag, "
        "and found a safe path home before sunset."
    )
    _write_jsonl(
        raw_path,
        [
            {
                "category": "creative_writing",
                "instruction": "Write a story about a helpful fox.",
                "context": "",
                "response": simple_output,
            },
            {
                "category": "closed_qa",
                "instruction": "Who was the president in 1999?",
                "context": "",
                "response": "This answer is long enough but it should be discarded because the category is not kept.",
            },
            {
                "category": "open_qa",
                "instruction": "Explain what a cloud is.",
                "context": "",
                "response": "A cloud is made of tiny drops of water floating high in the sky. It can look soft and white from the ground.",
            },
            {
                "category": "generation",
                "instruction": "Write a Python script for counting words.",
                "context": "",
                "response": "This response mentions Python and code many times, so it must never enter the small child friendly dataset.",
            },
        ],
    )

    manifest = convert_dolly_lite(raw_path, out_path, manifest_path)

    rows = [json.loads(line) for line in out_path.read_text(encoding="utf-8").splitlines()]
    assert [row["task_type"] for row in rows] == ["story_writing", "explanation_for_children"]
    assert all(row["source"] == "databricks_dolly_15k_lite" for row in rows)
    assert manifest["raw_rows"] == 4
    assert manifest["kept_rows"] == 2
    assert manifest["filter_reason_counts"]["discard_category"] == 1
    assert manifest["filter_reason_counts"]["contains_code_terms"] == 1
    assert json.loads(manifest_path.read_text(encoding="utf-8"))["kept_rows"] == 2


def test_dolly_lite_accepts_utf8_sig_jsonl_from_windows_tools(tmp_path):
    raw_path = tmp_path / "dolly_bom.jsonl"
    out_path = tmp_path / "dolly_lite.jsonl"
    row = {
        "category": "open_qa",
        "instruction": "Explain why rain helps flowers.",
        "context": "",
        "response": "Rain gives flowers water. The water moves into the roots and helps each flower stand tall, grow leaves, and make bright petals.",
    }
    raw_path.write_text(json.dumps(row) + "\n", encoding="utf-8-sig")

    manifest = convert_dolly_lite(raw_path, out_path)

    rows = [json.loads(line) for line in out_path.read_text(encoding="utf-8").splitlines()]
    assert manifest["kept_rows"] == 1
    assert rows[0]["task_type"] == "explanation_for_children"


def test_tinystories_replay_creates_story_and_continuation_rows(tmp_path):
    source = tmp_path / "TinyStories-valid.txt"
    output = tmp_path / "replay" / "tinystories_replay_sft_v0_4.jsonl"
    manifest = tmp_path / "replay" / "manifest.json"
    story = (
        "Mia had a red kite. She took it to the hill with her dad. "
        "The wind was gentle, and the kite went up slowly. "
        "Mia waited, held the string with care, and smiled when it danced. "
        "At sunset, she thanked her dad and carried the kite home."
    )
    source.write_text(story + "\n<|endoftext|>\n" + story.replace("Mia", "Ben"), encoding="utf-8")

    result = make_tinystories_replay(source, output, count=4, seed=7, manifest_path=manifest)

    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert result["written_rows"] == len(rows) == 4
    assert {row["task_type"] for row in rows} == {"story_writing", "story_continuation"}
    assert {row["source"] for row in rows} == {"tinystories_replay"}
    assert any(row["input"] for row in rows if row["task_type"] == "story_continuation")
    assert json.loads(manifest.read_text(encoding="utf-8"))["written_rows"] == 4


def test_replay_duplicate_policy_keeps_repeated_instruction_templates(tmp_path):
    _tiny_tokenizer(tmp_path)
    rows = [
        _sft_row(
            f"replay_{index:03d}",
            "tinystories_replay",
            "story_writing",
            "Write a short simple story.",
            _story_output(index),
        )
        for index in range(20)
    ]
    raw_path = tmp_path / "replay.jsonl"
    _write_jsonl(raw_path, rows)

    manifest = build_sft_splits(
        raw_path=raw_path,
        scored_path=None,
        tokenizer_path=tmp_path / "tokenizer.json",
        train_path=tmp_path / "train.jsonl",
        val_path=tmp_path / "val.jsonl",
        rejected_dir=tmp_path / "rejected",
        val_ratio=0.1,
        seed=123,
        max_seq_len=512,
        replay_source_prefix="tinystories_replay",
        keep_replay_duplicates=True,
        replay_dedup_mode="output_hash",
        disable_replay_low_diversity_filter=True,
    )

    assert manifest["accepted_rows"] == 20
    assert manifest["rejected_counts"]["duplicates"] == 0
    assert manifest["accepted_by_source"] == {"tinystories_replay": 20}


def test_non_replay_duplicate_policy_still_rejects_repeated_instruction_templates(tmp_path):
    _tiny_tokenizer(tmp_path)
    rows = [
        _sft_row(
            f"regular_{index:03d}",
            "databricks_dolly_15k_lite",
            "story_writing",
            "Write a short simple story.",
            _story_output(index),
        )
        for index in range(20)
    ]
    raw_path = tmp_path / "regular.jsonl"
    _write_jsonl(raw_path, rows)

    manifest = build_sft_splits(
        raw_path=raw_path,
        scored_path=None,
        tokenizer_path=tmp_path / "tokenizer.json",
        train_path=tmp_path / "train.jsonl",
        val_path=tmp_path / "val.jsonl",
        rejected_dir=tmp_path / "rejected",
        val_ratio=0.1,
        seed=123,
        max_seq_len=512,
        replay_source_prefix="tinystories_replay",
        keep_replay_duplicates=True,
        replay_dedup_mode="output_hash",
    )

    assert manifest["accepted_rows"] == 1
    assert manifest["rejected_counts"]["duplicates"] == 19
    assert manifest["rejected_by_source"] == {"databricks_dolly_15k_lite": 19}


def test_rejected_rows_include_original_metadata_and_record(tmp_path):
    _tiny_tokenizer(tmp_path)
    duplicate = _sft_row(
        "dup_2",
        "databricks_dolly_15k_lite",
        "story_writing",
        "Write a short simple story.",
        _story_output(1),
    )
    rows = [
        _sft_row("dup_1", "databricks_dolly_15k_lite", "story_writing", "Write a short simple story.", _story_output(1)),
        duplicate,
        _sft_row("bad_1", "tinystories_replay", "story_writing", "Write a short simple story.", ""),
    ]
    raw_path = tmp_path / "raw.jsonl"
    _write_jsonl(raw_path, rows)

    build_sft_splits(
        raw_path=raw_path,
        scored_path=None,
        tokenizer_path=tmp_path / "tokenizer.json",
        train_path=tmp_path / "train.jsonl",
        val_path=tmp_path / "val.jsonl",
        rejected_dir=tmp_path / "rejected",
        val_ratio=0.0,
        seed=123,
        max_seq_len=512,
    )

    duplicate_reject = json.loads((tmp_path / "rejected" / "duplicates.jsonl").read_text(encoding="utf-8").splitlines()[0])
    malformed_reject = json.loads((tmp_path / "rejected" / "malformed.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert duplicate_reject["reason"] == "duplicate"
    assert duplicate_reject["source"] == duplicate["source"]
    assert duplicate_reject["task_type"] == duplicate["task_type"]
    assert duplicate_reject["id"] == duplicate["id"]
    assert duplicate_reject["record"] == duplicate
    assert malformed_reject["source"] == "tinystories_replay"
    assert malformed_reject["task_type"] == "story_writing"
    assert malformed_reject["id"] == "bad_1"
    assert "record" in malformed_reject


def test_source_aware_manifest_reports_accepts_rejects_and_reasons(tmp_path):
    _tiny_tokenizer(tmp_path)
    replay_rows = [
        _sft_row(f"replay_{index}", "tinystories_replay", "story_writing", "Write a short simple story.", _story_output(index))
        for index in range(3)
    ]
    regular_rows = [
        _sft_row("regular_1", "databricks_dolly_15k_lite", "story_writing", "Write a short simple story.", _story_output(30)),
        _sft_row("regular_2", "databricks_dolly_15k_lite", "story_writing", "Write a short simple story.", _story_output(31)),
    ]
    raw_path = tmp_path / "mixed.jsonl"
    _write_jsonl(raw_path, replay_rows + regular_rows)

    manifest = build_sft_splits(
        raw_path=raw_path,
        scored_path=None,
        tokenizer_path=tmp_path / "tokenizer.json",
        train_path=tmp_path / "train.jsonl",
        val_path=tmp_path / "val.jsonl",
        rejected_dir=tmp_path / "rejected",
        val_ratio=0.0,
        seed=123,
        max_seq_len=512,
        replay_source_prefix="tinystories_replay",
        keep_replay_duplicates=True,
        replay_dedup_mode="output_hash",
    )

    assert manifest["accepted_by_source"] == {"databricks_dolly_15k_lite": 1, "tinystories_replay": 3}
    assert manifest["rejected_by_source"] == {"databricks_dolly_15k_lite": 1}
    assert manifest["accepted_by_task_type"] == {"story_writing": 4}
    assert manifest["rejected_by_task_type"] == {"story_writing": 1}
    assert manifest["rejected_reason_by_source"] == {"databricks_dolly_15k_lite": {"duplicates": 1}}


def test_trusted_source_policy_accepts_factory_validated_short_rows_but_keeps_hard_rejections(tmp_path):
    _tiny_tokenizer(tmp_path)
    trusted = "xiaomi_instruction_lite_v0_4"
    rows = [
        _sft_row("qa_1", trusted, "simple_qa", "What is rain for a child?", "Rain is water from clouds."),
        _sft_row("list_1", trusted, "structured_output", "List two gentle bedtime steps.", "1. Brush teeth.\n2. Read a story."),
        _sft_row("sum_1", trusted, "summarization", "Make this idea shorter for a child.", "A seed can grow into a plant."),
        _sft_row(
            "safe_1",
            trusted,
            "dialogue",
            "A child asks if they should take medicine from a shelf.",
            "Do not take it alone. Ask a trusted grown-up first.",
        ),
        _sft_row("qa_2", trusted, "simple_qa", "What is rain for a child?", "Rain is water from clouds."),
        _sft_row("long_1", trusted, "simple_qa", "Say friendly words about stars.", " ".join(["friendly"] * 300)),
    ]
    rows[1]["metadata"]["category"] = "simple_list"
    rows[2]["metadata"]["category"] = "summarization_rewrite"
    rows[3]["metadata"]["category"] = "safety_refusal"
    malformed = _sft_row("bad_1", trusted, "simple_qa", "Broken row.", "Missing output.")
    malformed.pop("output")
    raw_path = tmp_path / "trusted.jsonl"
    _write_jsonl(raw_path, rows + [malformed])

    manifest = build_sft_splits(
        raw_path=raw_path,
        scored_path=None,
        tokenizer_path=tmp_path / "tokenizer.json",
        train_path=tmp_path / "train.jsonl",
        val_path=tmp_path / "val.jsonl",
        rejected_dir=tmp_path / "rejected",
        val_ratio=0.0,
        seed=123,
        max_seq_len=160,
        trusted_source_prefixes=[trusted],
    )

    assert manifest["accepted_rows"] == 4
    assert manifest["rejected_counts"]["duplicates"] == 1
    assert manifest["rejected_counts"]["malformed"] == 1
    assert manifest["rejected_counts"]["too_long"] == 1
    assert manifest["accepted_by_category"] == {
        "safety_refusal": 1,
        "simple_list": 1,
        "summarization_rewrite": 1,
        "UNKNOWN": 1,
    }
    assert manifest["rejected_by_category"]["UNKNOWN"] == 3
    assert manifest["trusted_source_policy_active"] is True
    assert manifest["trusted_source_prefixes"] == [trusted]
    assert manifest["trusted_source_rows_seen"] == 7
    assert manifest["trusted_source_rows_accepted"] == 4
    assert manifest["trusted_source_rows_rejected"] == 3
    assert manifest["trusted_source_rejection_reasons"] == {"duplicates": 1, "malformed": 1, "too_long": 1}
    assert manifest["rejection_reasons_by_source"][trusted] == {"duplicates": 1, "malformed": 1, "too_long": 1}


def test_tinystories_replay_generator_manifest_and_modes(tmp_path):
    source = tmp_path / "TinyStories-valid.txt"
    output = tmp_path / "replay" / "tinystories_replay_sft_v0_4.jsonl"
    manifest_path = tmp_path / "replay" / "manifest.json"
    stories = []
    for index in range(6):
        stories.append(
            (
                f"Story {index} had a gentle child named Mia. She found a lost kitten near the blue gate. "
                "Mia called softly, gave the kitten water, and waited beside the path. "
                "Soon the kitten's owner came back, smiled, and thanked Mia for being kind."
            )
        )
    source.write_text("\n<|endoftext|>\n".join(stories), encoding="utf-8")

    manifest = make_tinystories_replay(
        source,
        output,
        count=8,
        seed=7,
        manifest_path=manifest_path,
        mode="mixed",
        min_words=20,
        max_words=80,
        unique_instructions=True,
    )

    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert rows
    assert all(row["source"].startswith("tinystories_replay") for row in rows)
    assert {row["task_type"] for row in rows} <= {"story_writing", "story_continuation"}
    assert manifest["stories_read"] == 6
    assert manifest["rows_written"] == len(rows)
    assert "story_writing_count" in manifest
    assert "story_continuation_count" in manifest
    assert "skipped_short" in manifest
    assert "skipped_long" in manifest
    assert "tokenizer_rejected_too_long" in manifest
    assert json.loads(manifest_path.read_text(encoding="utf-8"))["rows_written"] == len(rows)


def test_mix_sft_sources_respects_caps_dedups_and_seed(tmp_path):
    a = tmp_path / "a.jsonl"
    b = tmp_path / "b.jsonl"
    duplicate_output = "The fox helped the bird find a warm nest near the small green tree."
    _write_jsonl(
        a,
        [
            _sft_row("a1", "src_a", "simple_qa", "Tell me about rain.", "Rain falls from clouds and helps little plants grow in the ground."),
            _sft_row("a2", "src_a", "story_writing", "Write a fox helper story.", duplicate_output),
        ],
    )
    _write_jsonl(
        b,
        [
            _sft_row("b1", "src_b", "story_writing", "Write a fox helper story.", duplicate_output),
            _sft_row("b2", "src_b", "structured_output", "List kind acts.", "Help a friend, share a toy, and speak in a gentle voice."),
            _sft_row("b3", "src_b", "simple_reasoning", "Why did the cup fall?", "The cup fell because it was too close to the edge of the table."),
        ],
    )

    out1 = tmp_path / "mixed1.jsonl"
    out2 = tmp_path / "mixed2.jsonl"
    spec_a = parse_source_spec(f"alpha={a}:cap=2")
    spec_b = parse_source_spec(f"beta={b}:cap=2")
    result1 = mix_sft_sources([spec_a, spec_b], out1, seed=11)
    result2 = mix_sft_sources([spec_a, spec_b], out2, seed=11)

    rows = [json.loads(line) for line in out1.read_text(encoding="utf-8").splitlines()]
    assert out1.read_text(encoding="utf-8") == out2.read_text(encoding="utf-8")
    comparable1 = {key: value for key, value in result1.items() if key not in {"output_path", "timestamp"}}
    comparable2 = {key: value for key, value in result2.items() if key not in {"output_path", "timestamp"}}
    assert comparable1 == comparable2
    assert result1["selected_counts_by_source"] == {"alpha": 2, "beta": 1}
    assert result1["rejected_duplicate_rows"] == 1
    assert len(rows) == 3
    assert len({row["id"] for row in rows}) == 3
    assert all(row["id"].startswith(("alpha_", "beta_")) for row in rows)


def test_run_sft_experiment_grid_dry_run_writes_report_and_commands(tmp_path):
    commands = build_experiment_commands(
        configs=["configs/lr.yaml"],
        steps=[100, 200],
        run_root=tmp_path / "runs",
        device="cpu",
    )
    assert len(commands) == 2
    assert commands[0].train_cmd[:3] == ["python", "scripts/train_sft_v0_4.py", "--config"]
    assert "--no-resume" in commands[0].train_cmd
    assert "--run-dir" in commands[0].train_cmd

    result = run_sft_experiment_grid(
        configs=["configs/lr.yaml"],
        steps=[100],
        output_root=tmp_path / "artifacts",
        dry_run=True,
        timestamp="20260531_010203",
        device="cpu",
    )

    assert (Path(result["run_dir"]) / "report.md").exists()
    assert (Path(result["run_dir"]) / "results.jsonl").exists()
    report = (Path(result["run_dir"]) / "report.md").read_text(encoding="utf-8")
    assert "DRY RUN" in report
    assert "configs/lr.yaml" in report
