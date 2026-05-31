#!/usr/bin/env python3
"""CLI for v0.4.5 instruction-lite data factory."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.validate_instruction_lite_sft import Strictness, parse_strictness
from sarych.instruction_lite_factory import (
    audit_sample,
    merge_accepted,
    make_repair_pack,
    prepare_shards,
    validate_repairs,
    validate_shards,
)


def _cmd_prepare_shards(args: argparse.Namespace) -> None:
    manifest = prepare_shards(
        seeds_path=args.seeds,
        out_dir=args.out_dir,
        shard_size=args.shard_size,
        seed=args.seed,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


def _cmd_validate_shards(args: argparse.Namespace) -> None:
    summary = validate_shards(factory_dir=args.factory_dir, strictness=args.strictness)
    print(json.dumps(summary, indent=2, sort_keys=True))


def _cmd_make_repair_pack(args: argparse.Namespace) -> None:
    manifest = make_repair_pack(
        factory_dir=args.factory_dir,
        max_per_shard=args.max_per_shard,
        round_number=args.round,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


def _cmd_validate_repairs(args: argparse.Namespace) -> None:
    summary = validate_repairs(
        factory_dir=args.factory_dir,
        round_number=args.round,
        strictness=args.strictness,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


def _cmd_merge_accepted(args: argparse.Namespace) -> None:
    manifest = merge_accepted(
        factory_dir=args.factory_dir,
        out_path=args.out,
        manifest_path=args.manifest,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


def _cmd_audit_sample(args: argparse.Namespace) -> None:
    result = audit_sample(
        input_path=args.input,
        out_path=args.out,
        per_category=args.per_category,
        seed=args.seed,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="v0.4.5 instruction-lite data factory")
    sub = parser.add_subparsers(dest="command", required=True)

    prepare = sub.add_parser("prepare-shards", help="Split seeds and write teacher shard prompts")
    prepare.add_argument("--seeds", required=True)
    prepare.add_argument("--out-dir", required=True)
    prepare.add_argument("--shard-size", type=int, default=100)
    prepare.add_argument("--seed", type=int, default=1337)
    prepare.set_defaults(func=_cmd_prepare_shards)

    validate = sub.add_parser("validate-shards", help="Validate teacher outputs in shards/raw")
    validate.add_argument("--factory-dir", required=True)
    validate.add_argument(
        "--strictness",
        type=parse_strictness,
        default=Strictness.STANDARD,
        help="strict|standard|lenient",
    )
    validate.set_defaults(func=_cmd_validate_shards)

    repair = sub.add_parser("make-repair-pack", help="Build repair prompts from rejected shards")
    repair.add_argument("--factory-dir", required=True)
    repair.add_argument("--max-per-shard", type=int, default=50)
    repair.add_argument("--round", type=int, default=1)
    repair.set_defaults(func=_cmd_make_repair_pack)

    validate_repair = sub.add_parser("validate-repairs", help="Validate repair round outputs")
    validate_repair.add_argument("--factory-dir", required=True)
    validate_repair.add_argument("--round", type=int, default=1)
    validate_repair.add_argument("--strictness", type=parse_strictness, default=Strictness.STANDARD)
    validate_repair.set_defaults(func=_cmd_validate_repairs)

    merge = sub.add_parser("merge-accepted", help="Merge accepted shard and repair outputs")
    merge.add_argument("--factory-dir", required=True)
    merge.add_argument("--out", required=True)
    merge.add_argument("--manifest", required=True)
    merge.set_defaults(func=_cmd_merge_accepted)

    audit = sub.add_parser("audit-sample", help="Write markdown audit sample by category")
    audit.add_argument("--input", required=True)
    audit.add_argument("--out", required=True)
    audit.add_argument("--per-category", type=int, default=5)
    audit.add_argument("--seed", type=int, default=1337)
    audit.set_defaults(func=_cmd_audit_sample)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
