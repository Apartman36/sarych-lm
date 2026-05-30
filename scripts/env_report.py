from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sarych.env_report import collect_env_report, write_env_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Print and optionally save a SARYCH-LM environment report.")
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output:
        text = write_env_report(args.output)
    else:
        text = collect_env_report()
    print(text)


if __name__ == "__main__":
    main()
