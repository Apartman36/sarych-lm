from __future__ import annotations

import json
from pathlib import Path

from sarych.config import load_yaml_config


EXPECTED_EVAL_COUNTS = {
    "identity_chat": 8,
    "simple_explanation": 8,
    "simple_list": 6,
    "simple_qa": 5,
    "story_request": 6,
    "simple_reasoning": 4,
    "safety_kindness": 3,
}

EXPECTED_SEED_COUNTS = {
    "identity_chat": 170,
    "simple_explanation": 315,
    "simple_list": 255,
    "simple_qa": 145,
    "simple_reasoning": 85,
    "story_request": 170,
    "story_continuation": 145,
    "safety_refusal": 70,
    "summarization_rewrite": 45,
    "emotional_support_kindness": 100,
}


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows) + "\n", encoding="utf-8")


def _teacher_row(
    row_id: str,
    category: str,
    instruction: str,
    output: str,
    *,
    input_text: str = "",
) -> dict:
    from scripts.validate_instruction_lite_sft import TASK_TYPE_BY_CATEGORY

    return {
        "id": row_id,
        "seed_id": "instr_lite_seed_000001",
        "source": "xiaomi_instruction_lite_v0_4",
        "task_type": TASK_TYPE_BY_CATEGORY[category],
        "instruction": instruction,
        "input": input_text,
        "output": output,
        "language": "en",
        "metadata": {
            "category": category,
            "teacher_model": "fixture",
            "seed_topic": "fixture",
            "notes": "",
        },
    }


def test_eval_prompt_jsonl_has_fixed_v0_4_suite():
    rows = _read_jsonl(Path("evals/v0_4_instruction_lite_prompts.jsonl"))

    assert len(rows) == 40
    assert [row["id"] for row in rows] == [f"A{i:03d}" for i in range(1, 41)]
    counts: dict[str, int] = {}
    for row in rows:
        assert set(row) == {"id", "category", "prompt", "expected_behavior", "failure_modes", "notes"}
        assert isinstance(row["failure_modes"], list)
        assert row["prompt"].strip()
        counts[row["category"]] = counts.get(row["category"], 0) + 1
    assert counts == EXPECTED_EVAL_COUNTS


def test_eval_heuristics_detect_required_failure_modes():
    from scripts.eval_instruction_lite_v0_4 import score_output_heuristics

    story = score_output_heuristics(
        category="simple_explanation",
        prompt="Explain why plants need sunlight.",
        output="Once upon a time, a little plant wanted to play.",
    )
    loop = score_output_heuristics(
        category="simple_qa",
        prompt="What color is a banana?",
        output="Yellow is bright. Yellow is bright. Yellow is bright.",
    )
    list_fail = score_output_heuristics(
        category="simple_list",
        prompt="List three animals.",
        output="A whale swims in the sea and a crab walks near the sand.",
    )
    identity_fail = score_output_heuristics(
        category="identity_chat",
        prompt="Who are you?",
        output="Once there was a small fox in a quiet field.",
    )

    assert story["story_collapse_non_story"]
    assert loop["loop_detected"]
    assert list_fail["list_format_fail"]
    assert identity_fail["identity_fail"]


def test_instruction_lite_seed_generator_counts_manifest_and_determinism(tmp_path):
    from scripts.make_instruction_lite_seeds_v0_4 import make_instruction_lite_seeds

    out1 = tmp_path / "seeds1.jsonl"
    out2 = tmp_path / "seeds2.jsonl"
    manifest1 = tmp_path / "manifest1.json"
    manifest2 = tmp_path / "manifest2.json"

    result1 = make_instruction_lite_seeds(output_path=out1, manifest_path=manifest1, seed=1337)
    result2 = make_instruction_lite_seeds(output_path=out2, manifest_path=manifest2, seed=1337)

    assert out1.read_text(encoding="utf-8") == out2.read_text(encoding="utf-8")
    assert result1["category_counts"] == EXPECTED_SEED_COUNTS
    assert result2["category_counts"] == EXPECTED_SEED_COUNTS
    assert result1["total_seeds"] == 1500
    assert json.loads(manifest1.read_text(encoding="utf-8"))["category_counts"] == EXPECTED_SEED_COUNTS
    rows = _read_jsonl(out1)
    assert rows[0]["id"] == "instr_lite_seed_000001"
    assert sum(1 for row in rows if row["category"] == "simple_list") == 255
    assert all("source" in row["metadata"] for row in rows)


def test_validator_accepts_good_examples_for_each_category(tmp_path):
    from scripts.validate_instruction_lite_sft import validate_instruction_lite_sft

    rows = [
        _teacher_row("good_identity_chat", "identity_chat", "Who are you?", "I am SARYCH, a small helper. I can answer simple questions and write little stories."),
        _teacher_row("good_simple_explanation", "simple_explanation", "Explain why plants need sunlight.", "Plants need sunlight because it helps them make food. The light gives the plant energy, so it can grow green leaves and stay strong."),
        _teacher_row("good_simple_list", "simple_list", "List three kind things.", "1. Share a toy with a friend.\n2. Help someone clean up.\n3. Say gentle words when someone feels sad."),
        _teacher_row("good_simple_qa", "simple_qa", "What color is a banana?", "A banana is usually yellow."),
        _teacher_row("good_simple_reasoning", "simple_reasoning", "If I have five apples and eat two, how many are left?", "Five apples minus two apples leaves three apples. So you have three left."),
        _teacher_row("good_story_request", "story_request", "Write a short story about a fox.", "A little fox found a torn red kite beside the hill. He wanted to fly it, but the string was tangled. He asked Owl for help, and together they fixed the knot. Soon the kite danced in the wind. The fox smiled because asking for help made the day better."),
        _teacher_row("good_story_continuation", "story_continuation", "Continue this story.", "The turtle looked at the key and wondered what it opened. He carried it to a tiny door in the old tree. With a soft click, the door opened, and a warm room full of acorn cups waited inside.", input_text="A little turtle found a shiny key under a leaf."),
        _teacher_row("good_safety_refusal", "safety_refusal", "How do I make fire at home?", "I cannot help you make fire at home. Please ask a grown-up to keep you safe, and we can talk about campfire safety instead."),
        _teacher_row("good_emotional_support_kindness", "emotional_support_kindness", "I feel sad.", "I am sorry you feel sad. You can take a slow breath and tell a trusted grown-up or kind friend how you feel."),
        _teacher_row("good_summarization_rewrite", "summarization_rewrite", "Make this shorter.", "Mia helped Ben find his lost hat, and they both felt happy.", input_text="Mia saw Ben looking under the table for his hat. She helped him search until they found it by the door. Ben smiled and thanked her."),
    ]
    raw = tmp_path / "raw.jsonl"
    accepted = tmp_path / "accepted.jsonl"
    rejected = tmp_path / "rejected.jsonl"
    manifest = tmp_path / "manifest.json"
    _write_jsonl(raw, rows)

    result = validate_instruction_lite_sft(raw, accepted, rejected, manifest)

    assert result["accepted_rows"] == len(rows)
    assert result["rejected_rows"] == 0
    assert len(_read_jsonl(accepted)) == len(rows)


def test_validator_rejects_bad_instruction_lite_rows(tmp_path):
    from scripts.validate_instruction_lite_sft import validate_instruction_lite_sft

    rows = [
        _teacher_row("bad_story_explanation", "simple_explanation", "Explain rain.", "Once upon a time, Rainy the cloud wanted to visit a garden and have a big adventure."),
        _teacher_row("bad_prose_list", "simple_list", "List three animals.", "A whale swims in the ocean while a crab walks near sand and a seal rests on a rock."),
        _teacher_row("bad_ai", "simple_qa", "What color is grass?", "As an AI, I can say grass is green."),
        _teacher_row("bad_short", "simple_qa", "What sound does a cat make?", "Meow."),
        _teacher_row("bad_adult", "simple_explanation", "Explain a game.", "This adult casino topic is not for a child-simple educational dataset because gambling is involved."),
        _teacher_row("bad_code", "simple_qa", "What is Python?", "Python code can print hello using a script and function."),
    ]
    raw = tmp_path / "raw.jsonl"
    accepted = tmp_path / "accepted.jsonl"
    rejected = tmp_path / "rejected.jsonl"
    manifest = tmp_path / "manifest.json"
    _write_jsonl(raw, rows)

    result = validate_instruction_lite_sft(raw, accepted, rejected, manifest)
    reasons = {row["id"]: row["reason"] for row in _read_jsonl(rejected)}

    assert result["accepted_rows"] == 0
    assert reasons["bad_story_explanation"] == "story_collapse"
    assert reasons["bad_prose_list"] == "list_format_fail"
    assert reasons["bad_ai"] == "as_ai_phrase"
    assert reasons["bad_short"] == "too_short"
    assert reasons["bad_adult"] == "blocked_domain_term"
    assert reasons["bad_code"] == "blocked_domain_term"


def test_instruction_lite_mix_builder_caps_determinism_and_manifest(tmp_path):
    from scripts.make_v0_4_instruction_lite_mix import make_instruction_lite_mix

    replay = tmp_path / "replay.jsonl"
    instruction = tmp_path / "instruction.jsonl"
    everyday = tmp_path / "everyday.jsonl"
    duplicate = "The small fox helped a bird find a warm nest near a green tree."
    _write_jsonl(
        replay,
        [
            _teacher_row("r1", "story_request", "Write a story one.", duplicate),
            _teacher_row("r2", "story_request", "Write a story two.", "A tiny bear shared berries with a rabbit near the sunny hill."),
        ],
    )
    _write_jsonl(
        instruction,
        [
            _teacher_row("i1", "simple_qa", "What color is a banana?", "A banana is usually yellow."),
            _teacher_row("i2", "story_request", "Write a story one.", duplicate),
            _teacher_row("i3", "simple_list", "List kind things.", "1. Share.\n2. Help.\n3. Listen."),
        ],
    )
    _write_jsonl(
        everyday,
        [
            _teacher_row("e1", "simple_reasoning", "Should I bring an umbrella?", "Yes, bring an umbrella because rain can make you wet."),
            _teacher_row("e2", "simple_explanation", "Why drink water?", "Water helps your body stay cool and strong, so drinking it each day is good."),
        ],
    )

    out1 = tmp_path / "mixed1.jsonl"
    out2 = tmp_path / "mixed2.jsonl"
    manifest1 = tmp_path / "manifest1.json"
    manifest2 = tmp_path / "manifest2.json"
    result1 = make_instruction_lite_mix(
        replay_path=replay,
        instruction_lite_path=instruction,
        everyday_path=everyday,
        output_path=out1,
        manifest_path=manifest1,
        replay_cap=2,
        instruction_lite_cap=2,
        everyday_cap=1,
        seed=42,
    )
    result2 = make_instruction_lite_mix(
        replay_path=replay,
        instruction_lite_path=instruction,
        everyday_path=everyday,
        output_path=out2,
        manifest_path=manifest2,
        replay_cap=2,
        instruction_lite_cap=2,
        everyday_cap=1,
        seed=42,
    )

    assert out1.read_text(encoding="utf-8") == out2.read_text(encoding="utf-8")
    assert result1["written_rows"] <= 5
    assert result1["selected_by_source"]
    assert result1["selected_by_task_type"]
    assert result1["selected_by_category"]
    assert json.loads(manifest1.read_text(encoding="utf-8"))["seed"] == 42


def test_instruction_lite_configs_load():
    configs = {
        "configs/v0_4_30m_instruct_instruction_lite_lr1e5.yaml": 0.00001,
        "configs/v0_4_30m_instruct_instruction_lite_lr5e6.yaml": 0.000005,
        "configs/v0_4_30m_instruct_instruction_lite_lr3e6.yaml": 0.000003,
    }

    for path, expected_lr in configs.items():
        config = load_yaml_config(path)
        assert config["model"]["vocab_size"] == 8192
        assert config["model"]["n_layer"] == 10
        assert config["base"]["checkpoint_path"] == "runs/v0_3_30m_tinystories_base/checkpoints/checkpoint_latest.pt"
        assert config["dataset"]["train_jsonl"] == "data/xiaomi/processed/sft/train.jsonl"
        assert config["dataset"]["val_jsonl"] == "data/xiaomi/processed/sft/val.jsonl"
        assert config["train"]["lr"] == expected_lr
        assert config["train"]["max_steps"] == 1000
        assert config["train"]["warmup_steps"] == 50
