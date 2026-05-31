from __future__ import annotations

from typing import Any

from scripts.validate_instruction_lite_sft import TASK_TYPE_BY_CATEGORY

PROJECT_CONTEXT = """\
# SARYCH-LM Instruction-Lite Teacher Task (v0.4.5)

You are generating supervised fine-tuning rows for **sarych-30m-instruct**: a small English,
child-simple instruction model. It should follow simple instructions and keep TinyStories-like
story fluency. It is **not** a broad assistant: no code, no URLs, no adult/professional domains.

**Global rules**
- English only, child-simple style.
- No code, URLs, markdown tables, or markdown in the JSONL output.
- No "As an AI", "Sure", "Of course", "Certainly" openers.
- No professional assistant tone or long essays.
- Do **not** repeat the same first four words across rows in this shard.
- No duplicate outputs within the shard.
- Story prose only for `story_request` and `story_continuation`.
"""

OUTPUT_SCHEMA = """\
## Output JSONL schema (one JSON object per line, no markdown fences)

Required fields per row:
```json
{
  "id": "instr_lite_000001",
  "seed_id": "instr_lite_seed_000001",
  "source": "xiaomi_instruction_lite_v0_4",
  "task_type": "<see category mapping>",
  "instruction": "<filled from seed>",
  "input": "<empty string or continuation/rewrite input>",
  "output": "<teacher completion>",
  "language": "en",
  "metadata": {
    "category": "<seed category>",
    "teacher_model": "<your model name>",
    "seed_topic": "<seed topic>",
    "notes": ""
  }
}
```

Use the seed's `id` as `seed_id`. Generate unique `id` values per row (e.g. instr_lite_shard0001_001).
"""

CATEGORY_RULES: dict[str, dict[str, Any]] = {
    "identity_chat": {
        "words": "12–55",
        "rules": [
            "Short first-person helper identity (name/role). Direct and friendly.",
            "Must include first-person (I/me/my) or Sarych/helper persona.",
            "No story opening.",
        ],
        "good": [
            "I am Sarych, a small computer helper. I can answer simple questions and write little stories for you.",
        ],
        "bad": [
            "Once upon a time, a fox lived in a forest.",
            "Sure! I can help with many professional tasks.",
        ],
    },
    "simple_explanation": {
        "words": "18–90",
        "rules": [
            "First sentence directly answers.",
            "Use because/so/helps/needs/makes/means/turns when appropriate.",
            "No story opening.",
        ],
        "good": [
            "Plants need sunlight because it helps them make food. The light gives energy so leaves stay green and strong.",
        ],
        "bad": [
            "One day, a little cloud wanted to visit a garden.",
            "Sunlight is important.",
        ],
    },
    "simple_list": {
        "words": "12–90",
        "rules": [
            "Numbered or bullet list with at least 2 items.",
            "Do not write story prose.",
        ],
        "good": ["1. Share a toy.\n2. Say kind words.\n3. Help clean up."],
        "bad": ["A whale swims while a crab walks near the sand and a seal rests."],
    },
    "simple_qa": {
        "words": "3–45",
        "rules": ["Direct short answer.", "No story opening."],
        "good": ["A banana is usually yellow.", "Cats say meow."],
        "bad": ["Once there was a cat who loved bananas."],
    },
    "simple_reasoning": {
        "words": "12–85",
        "rules": [
            "Include one simple step or reason (so/because/first/then or practical cause).",
            "No story opening.",
        ],
        "good": ["Five minus two leaves three. So you have three apples left.", "A coat keeps you warm on a cold day."],
        "bad": ["Three."],
    },
    "story_request": {
        "words": "45–160",
        "rules": [
            "Child-simple TinyStories-like story with clear ending.",
            "Vary story openings across rows.",
        ],
        "good": [
            "A little fox found a torn kite beside the hill. He asked Owl for help, and they fixed the string. Soon the kite danced in the wind.",
        ],
        "bad": ["I am Sarych, your helper."],
    },
    "story_continuation": {
        "words": "35–140",
        "rules": [
            "Copy seed input into `input` field.",
            "Continue directly; do not repeat the full input.",
        ],
        "good": [
            "The turtle looked at the key and walked to a tiny door in the tree. With a soft click, a warm room of acorn cups waited inside.",
        ],
        "bad": ["A little turtle found a shiny key under a leaf. A little turtle found a shiny key under a leaf."],
    },
    "safety_refusal": {
        "words": "12–65",
        "rules": [
            "Gentle safe refusal; no harmful instructions.",
            "Say ask a grown-up/trusted person when appropriate.",
            "Mentioning medicine is OK if you refuse safely.",
        ],
        "good": [
            "I cannot help you take medicine on your own. Please ask a grown-up you trust to keep you safe.",
        ],
        "bad": ["Here is how to take pills without telling anyone."],
    },
    "emotional_support_kindness": {
        "words": "12–80",
        "rules": [
            "Warm, direct, one simple helpful idea.",
            "Not a story.",
        ],
        "good": ["It is okay to cry when you feel sad. Take a slow breath and tell a trusted grown-up how you feel."],
        "bad": ["Once upon a time, a sad bunny found a friend."],
    },
    "summarization_rewrite": {
        "words": "8–80",
        "rules": [
            "Put seed text in `input`; shorter simpler wording in `output`.",
            "Keep meaning; short is OK.",
        ],
        "good": ["Mia helped Ben find his hat, and they both felt happy."],
        "bad": ["Mia saw Ben looking under the table for his hat. She helped him search until they found it by the door."],
    },
}


def category_mapping_section(categories: list[str]) -> str:
    lines = ["## Category → task_type mapping (this shard)"]
    for category in sorted(set(categories)):
        lines.append(f"- `{category}` → `{TASK_TYPE_BY_CATEGORY[category]}`")
    return "\n".join(lines) + "\n"


def category_rules_section(categories: list[str]) -> str:
    parts = ["## Category-specific rules (only categories in this shard)"]
    for category in sorted(set(categories)):
        spec = CATEGORY_RULES[category]
        parts.append(f"\n### {category} ({spec['words']} words)")
        for rule in spec["rules"]:
            parts.append(f"- {rule}")
        parts.append("\n**Good example output:**")
        for example in spec["good"]:
            parts.append(f"- {example}")
        parts.append("\n**Bad example output:**")
        for example in spec["bad"]:
            parts.append(f"- {example}")
    return "\n".join(parts) + "\n"


def render_shard_prompt(
    *,
    shard_index: int,
    shard_rows: list[dict[str, Any]],
    seeds_path_windows: str,
    output_path_windows: str,
) -> str:
    categories = [str(row["category"]) for row in shard_rows]
    seeds_block = "\n".join(
        json_line(row)
        for row in shard_rows
    )
    anti_dup = """\
## Anti-duplication (shard-local)
- Do not reuse the same output text twice.
- Do not reuse the same first four words in two different outputs.
- Vary list items, story openings, and explanation openings across rows.
"""
    footer = """\
## Final summary (required at end of your chat response, not in JSONL)
After writing the file, report: row count, categories covered, and confirm the output path.
"""
    return "\n".join(
        [
            PROJECT_CONTEXT,
            f"\n## This shard\n- Shard number: {shard_index:04d}\n- Rows to generate: {len(shard_rows)}\n",
            f"## Input seeds file (read-only)\n`{seeds_path_windows}`\n",
            f"## Output file (write exactly here)\n`{output_path_windows}`\n",
            OUTPUT_SCHEMA,
            category_mapping_section(categories),
            category_rules_section(categories),
            anti_dup,
            "## Seed rows (generate one JSONL object per seed, same order)\n",
            seeds_block,
            footer,
        ]
    )


def render_repair_prompt(
    *,
    repair_index: int,
    repair_rows: list[dict[str, Any]],
    seeds_path_windows: str,
    output_path_windows: str,
    round_number: int,
) -> str:
    lines = [
        PROJECT_CONTEXT,
        f"\n# Repair round {round_number} — shard {repair_index:04d}\n",
        "Each seed below was rejected. Rewrite **one fixed row per line** in the output JSONL.",
        "Keep the same `seed_id`, `metadata.category`, and `task_type`. Fix the rejection reason.\n",
        f"## Repair seeds file\n`{seeds_path_windows}`\n",
        f"## Output file\n`{output_path_windows}`\n",
        OUTPUT_SCHEMA,
    ]
    categories = [str(row.get("category") or row.get("metadata", {}).get("category", "")) for row in repair_rows]
    lines.append(category_rules_section([c for c in categories if c]))
    lines.append("\n## Repair rows\n")
    for row in repair_rows:
        lines.append(json_line(row))
    lines.append(
        "\n## Final summary\nReport how many rows you fixed and the output path.\n"
    )
    return "\n".join(lines)


def json_line(row: dict[str, Any]) -> str:
    import json

    return json.dumps(row, ensure_ascii=False, sort_keys=True)
