from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


FINAL_CATEGORY_COUNTS = {
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

SOURCE_NAME = "instruction_lite_v0_4_seed_generator"

TOPICS = {
    "identity_chat": [
        "who you are",
        "your name",
        "being a computer helper",
        "helping with stories",
        "helping with simple questions",
        "friendly helper boundaries",
        "not getting tired",
        "not having feelings like a person",
    ],
    "simple_explanation": [
        "plants need sunlight",
        "people drink water",
        "the sky looks blue",
        "we sleep at night",
        "rainbows appear",
        "leaves change color",
        "cats purr",
        "ice melts",
        "seeds grow into plants",
        "rain helps flowers",
        "soap cleans hands",
        "the moon seems to change shape",
        "bees visit flowers",
        "shadows appear",
        "bread gets toastier when heated",
    ],
    "simple_list": [
        "kind things a friend can do",
        "ocean animals",
        "rainy day activities",
        "breakfast foods",
        "sunny day words",
        "reasons reading is good",
        "things to pack for school",
        "quiet games",
        "safe playground rules",
        "ways to care for a pet",
    ],
    "simple_qa": [
        "days in a week",
        "season after summer",
        "where birds go in rain",
        "banana color",
        "cat sound",
        "how many wheels a bicycle has",
        "where fish live",
        "what bees make",
        "what cows drink",
        "where stars are seen",
    ],
    "simple_reasoning": [
        "five apples minus two",
        "elephant and cat size",
        "rain and umbrella",
        "fish and bird homes",
        "cold day and coat",
        "broken crayon and asking for another",
        "two cookies plus one",
        "dark room and turning on a light",
    ],
    "story_request": [
        "careful fox asks for help",
        "friendly bear and little rabbit",
        "child finds a shiny key",
        "brave ant",
        "small turtle shares a leaf",
        "kitten learns to wait",
        "little owl finds a bell",
        "mouse helps a sleepy bee",
    ],
    "story_continuation": [
        "A little turtle found a shiny key under a leaf.",
        "Mia saw a tiny door at the bottom of the old tree.",
        "A brave ant heard a small cry near the garden wall.",
        "The friendly bear found one blue mitten in the snow.",
        "A little rabbit carried a red button home.",
        "Nina opened her lunch box and found a folded map.",
    ],
    "safety_refusal": [
        "make fire at home",
        "touch a hot stove",
        "take unknown medicine",
        "cross a busy street alone",
        "use sharp kitchen knives",
        "hide from a grown-up outside",
    ],
    "summarization_rewrite": [
        "Mia helped Ben find his lost hat by the door.",
        "The fox asked Owl for help and fixed the kite string.",
        "Rain fell on the garden and the flowers stood tall.",
        "A small rabbit shared berries with a hungry bird.",
        "The class cleaned the room before story time.",
    ],
    "emotional_support_kindness": [
        "feeling sad",
        "missing a friend",
        "being nervous at school",
        "feeling left out",
        "making a mistake",
        "wanting to cry",
        "feeling angry",
        "being scared of the dark",
    ],
}

TEMPLATES = {
    "identity_chat": [
        "Answer a child who asks about {topic}.",
        "Give a short helper-persona answer about {topic}.",
        "Reply kindly and simply when asked about {topic}.",
    ],
    "simple_explanation": [
        "Explain why {topic} in simple words.",
        "Give a direct child-simple explanation of why {topic}.",
        "Tell a child the simple reason that {topic}.",
    ],
    "simple_list": [
        "List {topic}.",
        "Give a numbered list about {topic}.",
        "Make a short bullet list of {topic}.",
    ],
    "simple_qa": [
        "Answer this simple question about {topic}.",
        "Give a direct answer about {topic}.",
        "Reply in one or two simple sentences about {topic}.",
    ],
    "simple_reasoning": [
        "Answer this simple reasoning question about {topic}.",
        "Explain the answer step by step in simple words: {topic}.",
        "Give the conclusion first, then one short reason: {topic}.",
    ],
    "story_request": [
        "Write a short child-simple story about {topic}.",
        "Tell a TinyStories-like story about {topic}.",
        "Make up a gentle short story about {topic}.",
    ],
    "story_continuation": [
        "Continue this child-simple story: {topic}",
        "Write the next part of this tiny story: {topic}",
        "Finish this gentle story beginning: {topic}",
    ],
    "safety_refusal": [
        "A child asks how to {topic}. Give a gentle safe answer.",
        "Reply safely to a child who wants to {topic}.",
        "Give a kind refusal and safer redirect about {topic}.",
    ],
    "summarization_rewrite": [
        "Make this sentence shorter: {topic}",
        "Rewrite this in simpler words: {topic}",
        "Summarize this for a young child: {topic}",
    ],
    "emotional_support_kindness": [
        "A child says they are {topic}. Reply kindly.",
        "Give warm simple support for a child who is {topic}.",
        "Reply gently to a child who mentions {topic}.",
    ],
}

FORMAT_BY_CATEGORY = {
    "identity_chat": "short_identity_chat",
    "simple_explanation": "direct_explanation",
    "simple_list": "numbered_or_bulleted_list",
    "simple_qa": "direct_answer",
    "simple_reasoning": "simple_reasoning",
    "story_request": "short_story",
    "story_continuation": "story_continuation",
    "safety_refusal": "gentle_refusal",
    "summarization_rewrite": "simple_rewrite",
    "emotional_support_kindness": "warm_direct_support",
}


def _constraints(category: str) -> dict[str, Any]:
    max_words = {
        "identity_chat": 50,
        "simple_explanation": 60,
        "simple_list": 70,
        "simple_qa": 40,
        "simple_reasoning": 70,
        "story_request": 140,
        "story_continuation": 120,
        "safety_refusal": 55,
        "summarization_rewrite": 60,
        "emotional_support_kindness": 65,
    }[category]
    must_not = ["Once upon a time", "One day", "Sure", "Of course", "As an AI"]
    if category in {"story_request", "story_continuation"}:
        must_not = ["Sure", "Of course", "As an AI"]
    return {
        "max_output_words": max_words,
        "style": "child-simple English",
        "must_not_start_with": must_not,
        "format": FORMAT_BY_CATEGORY[category],
    }


def _expected_style(category: str) -> str:
    if category == "simple_explanation":
        return "Start with the direct cause or reason, then add one or two simple supporting sentences."
    if category == "simple_list":
        return "Use numbered or bulleted items with clear short phrases."
    if category == "identity_chat":
        return "Short first-person SARYCH/helper persona answer, not a story."
    if category == "safety_refusal":
        return "Kind refusal with a grown-up or trusted-person redirect; no instructions for danger."
    if category in {"story_request", "story_continuation"}:
        return "TinyStories-like child-simple story prose with a gentle ending."
    if category == "emotional_support_kindness":
        return "Warm direct support, not a story, with a trusted grown-up suggestion when useful."
    return "Direct child-simple answer with no broad assistant or professional tone."


def _forbidden_terms(category: str) -> list[str]:
    common = ["As an AI", "politics", "medical advice", "legal advice", "financial advice", "code", "Python"]
    if category not in {"story_request", "story_continuation"}:
        common.extend(["Once upon a time", "One day", "Long ago"])
    return common


def make_instruction_lite_seeds(
    *,
    output_path: str | Path = "data/xiaomi/seeds/instruction_lite_v0_4_seeds.jsonl",
    manifest_path: str | Path = "data/xiaomi/manifests/instruction_lite_v0_4_seeds_manifest.json",
    seed: int = 1337,
    category_counts: dict[str, int] | None = None,
) -> dict[str, Any]:
    counts = dict(category_counts or FINAL_CATEGORY_COUNTS)
    rng = random.Random(seed)
    rows: list[dict[str, Any]] = []
    serial = 1
    for category in sorted(counts):
        topics = TOPICS[category]
        templates = TEMPLATES[category]
        for index in range(counts[category]):
            topic = topics[(index + rng.randrange(len(topics))) % len(topics)]
            template = templates[(index + rng.randrange(len(templates))) % len(templates)]
            row = {
                "id": f"instr_lite_seed_{serial:06d}",
                "category": category,
                "instruction_template": template,
                "topic": topic,
                "constraints": _constraints(category),
                "expected_output_style": _expected_style(category),
                "forbidden_terms": _forbidden_terms(category),
                "metadata": {
                    "source": SOURCE_NAME,
                    "random_seed": seed,
                    "category_index": index + 1,
                    "schema": "instruction_lite_seed_v0_4",
                },
            }
            rows.append(row)
            serial += 1
    rng.shuffle(rows)
    for serial, row in enumerate(rows, start=1):
        row["id"] = f"instr_lite_seed_{serial:06d}"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    actual_counts = Counter(row["category"] for row in rows)
    manifest = {
        "generator": SOURCE_NAME,
        "output_path": str(output_path),
        "total_seeds": len(rows),
        "category_counts": dict(sorted(actual_counts.items())),
        "random_seed": seed,
        "schema": "instruction_lite_seed_v0_4",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    manifest_path = Path(manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create deterministic v0.4 instruction-lite seed prompts for an external teacher.")
    parser.add_argument("--out", default="data/xiaomi/seeds/instruction_lite_v0_4_seeds.jsonl")
    parser.add_argument("--manifest", default="data/xiaomi/manifests/instruction_lite_v0_4_seeds_manifest.json")
    parser.add_argument("--seed", type=int, default=1337)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = make_instruction_lite_seeds(output_path=args.out, manifest_path=args.manifest, seed=args.seed)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
