# Xiaomi SFT From Seeds v1

You are given one JSONL seed shard. Read every seed row and produce exactly one final SFT JSONL record per seed row.

Rules:
- Do not browse the web.
- Do not use citations.
- Do not use markdown.
- Do not add commentary before or after the JSONL.
- Preserve the seed row order.
- Preserve `target_sft_id` as the final record `id`.
- Preserve `task_type`.
- Write strict JSONL: one valid JSON object per line.
- Use English only.
- Make every instruction and output materially different from other rows.
- Follow all seed constraints, including word limits, avoided openings, avoided names, required content, and category-specific format requirements.
- Do not copy `instruction_blueprint` verbatim if it reads like a seed; turn it into a natural user instruction.
- Do not mention seed IDs in the final `instruction`, `input`, or `output`.
- Keep outputs simple, warm, child-friendly, and safe.

Input seed fields:
- `seed_id`: internal seed identifier.
- `target_sft_id`: final SFT record ID to preserve.
- `task_type`: final SFT task category to preserve.
- `instruction_blueprint`: source idea for the final user instruction.
- `input_blueprint`: optional source text for the final `input`.
- Additional fields such as `topic`, `setting`, `characters`, `lesson`, `tone`, and `constraints`.

Final output schema for each JSONL line:

```json
{
  "id": "xm_sft_000001",
  "source": "xiaomi_mimo_v2_5_pro",
  "task_type": "story_writing",
  "instruction": "Write a short story for young children about ...",
  "input": "",
  "output": "A complete child-friendly answer ...",
  "language": "en",
  "metadata": {
    "seed_id": "seed_000001",
    "generator": "opencode_manual_seeded",
    "model": "mimo-v2.5-pro",
    "web_used": false,
    "prompt_template": "xiaomi_sft_from_seeds_v1"
  }
}
```

Validation before finishing:
- The number of output lines equals the number of seed lines.
- Every output line parses as JSON.
- Every `id` is unique and equals its seed `target_sft_id`.
- Every `task_type` is one of:
  `story_writing`, `story_continuation`, `explanation_for_children`, `simple_qa`, `dialogue`, `summarization`, `simple_reasoning`, `structured_output`, `creative_generation`.
- No output starts with banned openings from its seed constraints.
- Banned names from the seed constraints are not used as character names.
- `web_used` is always false.
