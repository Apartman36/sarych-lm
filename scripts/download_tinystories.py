from __future__ import annotations

import argparse
import sys
import urllib.error
import urllib.request
from pathlib import Path


URLS = {
    "train": "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories-train.txt",
    "valid": "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories-valid.txt",
}


def _download(url: str, output_path: Path, *, force: bool = False) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and output_path.stat().st_size > 0 and not force:
        print(f"Exists, skipping: {output_path} ({output_path.stat().st_size} bytes)")
        return
    part_path = output_path.with_suffix(output_path.suffix + ".part")
    if force:
        output_path.unlink(missing_ok=True)
        part_path.unlink(missing_ok=True)
    mode = "wb"
    headers: dict[str, str] = {}
    if part_path.exists() and part_path.stat().st_size > 0:
        headers["Range"] = f"bytes={part_path.stat().st_size}-"
        mode = "ab"
    request = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            if mode == "ab" and response.status == 200:
                mode = "wb"
            with part_path.open(mode) as handle:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    handle.write(chunk)
    except (urllib.error.URLError, TimeoutError) as exc:
        raise RuntimeError(
            f"Download failed for {url}. Retry with the same command to resume, or use --force to restart: {exc}"
        ) from exc
    part_path.replace(output_path)
    print(f"Downloaded: {output_path} ({output_path.stat().st_size} bytes)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optionally download TinyStories raw text files into data/raw/.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--train", action="store_true")
    group.add_argument("--valid", action="store_true")
    group.add_argument("--all", action="store_true")
    parser.add_argument("--output-dir", default="data/raw")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected = ["train", "valid"] if args.all else ["train"] if args.train else ["valid"]
    output_dir = Path(args.output_dir)
    try:
        for split in selected:
            _download(URLS[split], output_dir / f"TinyStories-{split}.txt", force=args.force)
    except RuntimeError as exc:
        print(exc, file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
