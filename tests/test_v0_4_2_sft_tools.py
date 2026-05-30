from __future__ import annotations

import json
from pathlib import Path

from scripts.convert_dolly_lite_to_sft import convert_dolly_lite
from scripts.make_tinystories_replay_sft import make_tinystories_replay
from scripts.mix_sft_sources import mix_sft_sources, parse_source_spec
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
