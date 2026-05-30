from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sarych.config import apply_cli_overrides, load_yaml_config
from sarych.train import train_from_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SARYCH-LM v0.3 30M TinyStories base model.")
    parser.add_argument("--config", default="configs/v0_3_30m_tinystories_base.yaml")
    resume_group = parser.add_mutually_exclusive_group()
    resume_group.add_argument("--resume", action="store_true", default=None)
    resume_group.add_argument("--no-resume", action="store_false", dest="resume")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--run-dir", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml_config(args.config)
    config = apply_cli_overrides(
        config,
        resume=args.resume,
        max_steps=args.max_steps,
        device=args.device,
        run_dir=args.run_dir,
    )
    result = train_from_config(config)
    print(result)


if __name__ == "__main__":
    main()
