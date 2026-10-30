#!/bin/bash
# Upload the code as a private Kaggle dataset, then start a private T4 GPU kernel.
# The Kaggle CLI reads its token from ~/.kaggle/access_token. This script never touches it.
#
# Usage: kaggle/push.sh
# DRY_RUN=1 kaggle/push.sh builds the upload folders, prints where, and uploads nothing.
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
USER_NAME=zahidhasancodeml
DATASET="$USER_NAME/llm-engine-code"
KERNEL="$USER_NAME/llm-engine-benchmark"
K="$ROOT/.venv/bin/kaggle"
BUILD=$(mktemp -d)

# Only these paths are uploaded. Anything else in the repository stays on this machine.
mkdir -p "$BUILD/data/code/benchmarks"
cp -R "$ROOT/engine" "$ROOT/tests" "$ROOT/pyproject.toml" "$BUILD/data/code/"
cp "$ROOT/benchmarks/real_model.py" "$BUILD/data/code/benchmarks/"
find "$BUILD/data/code" -name __pycache__ -prune -exec rm -rf {} +

cat > "$BUILD/data/dataset-metadata.json" <<EOF
{"title": "llm-engine-code", "id": "$DATASET", "licenses": [{"name": "other"}]}
EOF

mkdir -p "$BUILD/kernel"
cp "$ROOT/kaggle/run.py" "$BUILD/kernel/run.py"
cat > "$BUILD/kernel/kernel-metadata.json" <<EOF
{
  "id": "$KERNEL",
  "title": "llm-engine-benchmark",
  "code_file": "run.py",
  "language": "python",
  "kernel_type": "script",
  "is_private": true,
  "enable_gpu": true,
  "enable_internet": true,
  "machine_shape": "NvidiaTeslaT4",
  "dataset_sources": ["$DATASET"],
  "competition_sources": [],
  "kernel_sources": []
}
EOF

if [ "${DRY_RUN:-0}" = "1" ]; then
  echo "$BUILD"
  exit 0
fi

if "$K" datasets status "$DATASET" >/dev/null 2>&1; then
  "$K" datasets version -p "$BUILD/data" -m "update" --dir-mode zip
else
  "$K" datasets create -p "$BUILD/data" --dir-mode zip  # private by default
fi

# A kernel started before the dataset is ready would run the previous version.
for _ in $(seq 1 60); do
  if [ "$("$K" datasets status "$DATASET" 2>/dev/null | tail -1)" = "ready" ]; then
    break
  fi
  sleep 5
done

"$K" kernels push -p "$BUILD/kernel"
echo "status: $K kernels status $KERNEL"
