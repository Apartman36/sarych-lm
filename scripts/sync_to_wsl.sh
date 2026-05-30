#!/usr/bin/env bash
set -euo pipefail

SOURCE="${1:-/mnt/c/Users/hustlePC/PycharmProjects/sarych-lm}"
TARGET="${2:-$HOME/projects/sarych-lm}"

mkdir -p "$TARGET"

EXCLUDES=(
  --exclude ".venv/"
  --exclude "runs/"
  --exclude "__pycache__/"
  --exclude ".pytest_cache/"
  --exclude ".mypy_cache/"
  --exclude ".ruff_cache/"
  --exclude ".git/"
  --exclude "*.pt"
  --exclude "*.bin"
  --exclude "*.npy"
  --exclude "*.npz"
)

echo "Source: $SOURCE"
echo "Target: $TARGET"

if command -v rsync >/dev/null 2>&1; then
  rsync -av "${EXCLUDES[@]}" "$SOURCE/" "$TARGET/"
else
  echo "rsync is not installed; using cp -a fallback without deleting target files."
  tmp_tar="$(mktemp)"
  tar -C "$SOURCE" \
    --exclude ".venv" \
    --exclude "runs" \
    --exclude "__pycache__" \
    --exclude ".pytest_cache" \
    --exclude ".mypy_cache" \
    --exclude ".ruff_cache" \
    --exclude ".git" \
    --exclude "*.pt" \
    --exclude "*.bin" \
    --exclude "*.npy" \
    --exclude "*.npz" \
    -cf "$tmp_tar" .
  tar -C "$TARGET" -xf "$tmp_tar"
  rm -f "$tmp_tar"
fi

cat <<'EOF'

Next commands inside WSL:
cd ~/projects/sarych-lm
source .venv/bin/activate
python scripts/env_report.py --output runs/v0_1_synthetic_sanity/env_report.txt
pytest -q
python scripts/train_v0_1.py --config configs/v0_1_synthetic_sanity.yaml
EOF
