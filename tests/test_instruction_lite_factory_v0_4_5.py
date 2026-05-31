from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )


def _seed_row(seed_id: str, category: str, topic: str = "test topic") -> dict:
    return {
        "id": seed_id,
        "category": category,
        "instruction_template": f"Answer about {topic}.",
        "topic": topic,
        "constraints": {"max_output_words": 50, "style": "child-simple English"},
        "expected_output_style": "Direct child-simple answer.",
        "forbidden_terms": ["As an AI"],
        "metadata": {"source": "fixture", "schema": "instruction_lite_seed_v0_4"},
    }


def _teacher_row(
    row_id: str,
    seed_id: str,
    category: str,
    instruction: str,
    output: str,
    *,
    input_text: str = "",
) -> dict:
    from scripts.validate_instruction_lite_sft import TASK_TYPE_BY_CATEGORY

    return {
        "id": row_id,
        "seed_id": seed_id,
        "source": "fixture_instruction_lite",
        "task_type": TASK_TYPE_BY_CATEGORY[category],
        "instruction": instruction,
        "input": input_text,
        "output": output,
        "language": "en",
        "metadata": {"category": category, "teacher_model": "fixture", "seed_topic": "fixture", "notes": ""},
    }


@pytest.fixture
def fixture_seeds(tmp_path: Path) -> Path:
    seeds = [
        _seed_row("instr_lite_seed_000001", "identity_chat"),
        _seed_row("instr_lite_seed_000002", "simple_list"),
        _seed_row("instr_lite_seed_000003", "simple_qa"),
        _seed_row("instr_lite_seed_000004", "simple_explanation"),
        _seed_row("instr_lite_seed_000005", "safety_refusal"),
    ]
    path = tmp_path / "seeds.jsonl"
    _write_jsonl(path, seeds)
    return path


def test_prepare_shards_deterministic_and_prompts(fixture_seeds: Path, tmp_path: Path) -> None:
    from sarych.instruction_lite_factory.shards import prepare_shards

    out1 = tmp_path / "factory1"
    out2 = tmp_path / "factory2"
    manifest1 = prepare_shards(seeds_path=fixture_seeds, out_dir=out1, shard_size=2, seed=7)
    manifest2 = prepare_shards(seeds_path=fixture_seeds, out_dir=out2, shard_size=2, seed=7)

    def _stable(m: dict) -> dict:
        return {
            "total_seeds": m["total_seeds"],
            "shard_size": m["shard_size"],
            "shard_count": m["shard_count"],
            "random_seed": m["random_seed"],
            "shards": [
                {
                    "shard_id": s["shard_id"],
                    "row_count": s["row_count"],
                    "category_counts": s["category_counts"],
                }
                for s in m["shards"]
            ],
        }

    assert _stable(manifest1) == _stable(manifest2)
    assert manifest1["shard_count"] == 3
    seed_ids = []
    for shard_file in sorted((out1 / "shards" / "seeds").glob("shard_*.jsonl")):
        seed_ids.extend(row["id"] for row in _read_jsonl(shard_file))
    assert sorted(seed_ids) == sorted(
        [
            "instr_lite_seed_000001",
            "instr_lite_seed_000002",
            "instr_lite_seed_000003",
            "instr_lite_seed_000004",
            "instr_lite_seed_000005",
        ]
    )
    prompts = list((out1 / "shards" / "prompts").glob("shard_*_prompt.md"))
    assert prompts
    combined = "\n".join(path.read_text(encoding="utf-8") for path in prompts)
    assert "shards\\raw\\shard_0001.jsonl" in combined or "shards/raw/shard_0001.jsonl" in combined
    assert "identity_chat" in combined
    assert "simple_list" in combined
    assert "### simple_list" in combined


def test_validator_standard_accepts_calibrated_cases(tmp_path: Path) -> None:
    from scripts.validate_instruction_lite_sft import Strictness, validate_instruction_lite_sft

    rows = [
        _teacher_row(
            "short_list",
            "instr_lite_seed_000002",
            "simple_list",
            "List two colors.",
            "1. Red\n2. Blue",
        ),
        _teacher_row(
            "medicine_refusal",
            "instr_lite_seed_000005",
            "safety_refusal",
            "How do I take medicine alone?",
            "I cannot help you take medicine by yourself. Please ask a grown-up you trust to keep you safe.",
        ),
        _teacher_row(
            "explanation_so",
            "instr_lite_seed_000004",
            "simple_explanation",
            "Why wear a coat?",
            "A coat keeps you warm, so your body stays comfortable on a cold day.",
        ),
        _teacher_row(
            "identity_friend",
            "instr_lite_seed_000001",
            "identity_chat",
            "Who are you?",
            "I am your little computer friend. I help with simple stories and questions.",
        ),
    ]
    raw = tmp_path / "raw.jsonl"
    accepted = tmp_path / "accepted.jsonl"
    rejected = tmp_path / "rejected.jsonl"
    manifest = tmp_path / "manifest.json"
    _write_jsonl(raw, rows)

    result = validate_instruction_lite_sft(raw, accepted, rejected, manifest, strictness=Strictness.STANDARD)
    assert result["accepted_rows"] == len(rows)
    assert result["rejected_rows"] == 0


def test_validator_strict_rejects_borderline_short_explanation(tmp_path: Path) -> None:
    from scripts.validate_instruction_lite_sft import Strictness, validate_instruction_lite_sft

    rows = [
        _teacher_row(
            "weak_expl",
            "instr_lite_seed_000004",
            "simple_explanation",
            "Why is the sky blue?",
            "The sky is blue.",
        ),
    ]
    raw = tmp_path / "raw.jsonl"
    _write_jsonl(raw, rows)
    result = validate_instruction_lite_sft(
        raw,
        tmp_path / "accepted.jsonl",
        tmp_path / "rejected.jsonl",
        tmp_path / "manifest.json",
        strictness=Strictness.STRICT,
    )
    rejected = _read_jsonl(tmp_path / "rejected.jsonl")
    reasons = {row["id"]: row["reason"] for row in rejected}
    assert reasons["weak_expl"] in {"weak_explanation", "too_short"}


def test_validator_lenient_accepts_more_than_standard() -> None:
    from scripts.validate_instruction_lite_sft import Strictness, validate_row

    row = _teacher_row(
        "borderline",
        "instr_lite_seed_000004",
        "emotional_support_kindness",
        "I feel sad.",
        "It is okay to cry when you feel sad.",
    )
    ok_strict, reason_strict, _, _ = validate_row(
        row,
        seen_exact=set(),
        seen_category_ngrams={"emotional_support_kindness": []},
        strictness=Strictness.STRICT,
    )
    ok_lenient, _, _, _ = validate_row(
        row,
        seen_exact=set(),
        seen_category_ngrams={"emotional_support_kindness": []},
        strictness=Strictness.LENIENT,
    )
    assert ok_lenient
    assert not ok_strict
    assert reason_strict == "too_short"


def test_make_repair_pack_includes_reason(fixture_seeds: Path, tmp_path: Path) -> None:
    from sarych.instruction_lite_factory.repairs import make_repair_pack
    from sarych.instruction_lite_factory.shards import prepare_shards
    from sarych.instruction_lite_factory.validation import validate_shards

    factory = tmp_path / "factory"
    prepare_shards(seeds_path=fixture_seeds, out_dir=factory, shard_size=2, seed=3)
    raw_dir = factory / "shards" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(
        raw_dir / "shard_0001.jsonl",
        [
            _teacher_row(
                "bad",
                "instr_lite_seed_000001",
                "identity_chat",
                "Who are you?",
                "Once upon a time there was a fox.",
            )
        ],
    )
    validate_shards(factory_dir=factory)
    manifest = make_repair_pack(factory_dir=factory, max_per_shard=10, round_number=1)
    repair_seed = _read_jsonl(factory / "repairs" / "round_1" / "seeds" / "repair_shard_0001.jsonl")[0]
    assert repair_seed["seed_id"] == "instr_lite_seed_000001"
    assert repair_seed["category"] == "identity_chat"
    assert repair_seed["rejection_reason"] == "story_collapse"
    prompt = (factory / "repairs" / "round_1" / "prompts" / "repair_shard_0001_prompt.md").read_text(encoding="utf-8")
    assert "seed_id" in prompt
    assert "Output JSONL schema" in prompt or "output JSONL" in prompt.lower()


def test_merge_accepted_dedupes_and_manifest(fixture_seeds: Path, tmp_path: Path) -> None:
    from sarych.instruction_lite_factory.merge import merge_accepted
    from sarych.instruction_lite_factory.shards import prepare_shards

    factory = tmp_path / "factory"
    prepare_shards(seeds_path=fixture_seeds, out_dir=factory, shard_size=5, seed=1)
    accepted_dir = factory / "shards" / "accepted"
    accepted_dir.mkdir(parents=True, exist_ok=True)
    duplicate_output = "I am Sarych, a small helper who answers simple questions."
    rows = [
        _teacher_row("a1", "instr_lite_seed_000001", "identity_chat", "Who?", duplicate_output),
        _teacher_row("a2", "instr_lite_seed_000001", "identity_chat", "Who?", duplicate_output),
        _teacher_row("a3", "instr_lite_seed_000002", "simple_list", "List colors.", "1. Red\n2. Blue"),
    ]
    _write_jsonl(accepted_dir / "shard_0001_accepted.jsonl", rows)
    out = tmp_path / "merged.jsonl"
    manifest_path = tmp_path / "manifest.json"
    manifest = merge_accepted(factory_dir=factory, out_path=out, manifest_path=manifest_path)

    merged = _read_jsonl(out)
    assert len(merged) == 2
    assert manifest["duplicate_removals"] == 1
    assert manifest["category_counts"]["identity_chat"] == 1
    assert manifest["category_counts"]["simple_list"] == 1


def test_audit_sample_writes_markdown(tmp_path: Path) -> None:
    from sarych.instruction_lite_factory.audit import audit_sample

    rows = [
        _teacher_row("x1", "s1", "simple_qa", "Q?", "Yes."),
        _teacher_row("x2", "s2", "simple_list", "List.", "1. A\n2. B"),
    ]
    input_path = tmp_path / "accepted.jsonl"
    out_path = tmp_path / "audit.md"
    _write_jsonl(input_path, rows)
    result = audit_sample(input_path=input_path, out_path=out_path, per_category=2, seed=1)
    text = out_path.read_text(encoding="utf-8")
    assert result["sampled_rows"] >= 2
    assert "## simple_qa" in text
    assert "**Output**" in text


def test_full_factory_smoke(fixture_seeds: Path, tmp_path: Path) -> None:
    from sarych.instruction_lite_factory.audit import audit_sample
    from sarych.instruction_lite_factory.merge import merge_accepted
    from sarych.instruction_lite_factory.repairs import make_repair_pack, validate_repairs
    from sarych.instruction_lite_factory.shards import prepare_shards
    from sarych.instruction_lite_factory.validation import validate_shards

    factory = tmp_path / "factory"
    prepare_shards(seeds_path=fixture_seeds, out_dir=factory, shard_size=2, seed=99)
    raw_dir = factory / "shards" / "raw"
    for shard_file in (factory / "shards" / "seeds").glob("shard_*.jsonl"):
        seeds = _read_jsonl(shard_file)
        generated = []
        for index, seed in enumerate(seeds, start=1):
            category = seed["category"]
            if category == "identity_chat":
                output = "I am Sarych, a small helper. I help with simple stories and questions."
            elif category == "simple_list":
                output = "1. Share toys.\n2. Say kind words."
            elif category == "simple_qa":
                output = "The answer is simple and direct."
            elif category == "simple_explanation":
                output = "Plants need water because it helps them grow, so roots stay healthy."
            else:
                output = "I cannot help with that alone. Please ask a grown-up you trust to keep you safe."
            generated.append(
                _teacher_row(
                    f"gen_{shard_file.stem}_{index}",
                    seed["id"],
                    category,
                    seed["instruction_template"].format(topic=seed["topic"]),
                    output,
                    input_text=seed["topic"] if category == "story_continuation" else "",
                )
            )
        _write_jsonl(raw_dir / f"{shard_file.stem}.jsonl", generated)

    summary = validate_shards(factory_dir=factory)
    assert summary["total_accepted"] >= 1
    make_repair_pack(factory_dir=factory, max_per_shard=5, round_number=1)
    repair_raw = factory / "repairs" / "round_1" / "raw"
    if list((factory / "repairs" / "round_1" / "seeds").glob("*.jsonl")):
        repair_raw.mkdir(parents=True, exist_ok=True)
        for repair_seed_file in (factory / "repairs" / "round_1" / "seeds").glob("repair_shard_*.jsonl"):
            repairs = _read_jsonl(repair_seed_file)
            fixed = []
            for item in repairs:
                record = item.get("rejected_record") or {}
                category = item.get("category") or "identity_chat"
                fixed.append(
                    _teacher_row(
                        f"fixed_{item['repair_id']}",
                        str(item.get("seed_id")),
                        str(category),
                        str(record.get("instruction", "Fix this.")),
                        "I am Sarych, your helper friend. I help with kind simple answers.",
                    )
                )
            _write_jsonl(repair_raw / f"{repair_seed_file.stem}.jsonl", fixed)
        validate_repairs(factory_dir=factory, round_number=1)

    merged_path = tmp_path / "merged.jsonl"
    merge_accepted(factory_dir=factory, out_path=merged_path, manifest_path=tmp_path / "merge_manifest.json")
    audit_sample(input_path=merged_path, out_path=tmp_path / "audit.md", per_category=1, seed=2)
    assert merged_path.exists()


def test_cli_help() -> None:
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "run_instruction_lite_factory.py"), "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0
    assert "prepare-shards" in proc.stdout
