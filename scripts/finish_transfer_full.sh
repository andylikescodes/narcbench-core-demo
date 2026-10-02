#!/bin/bash
# After full Transfer transcripts land under upstream/scenarios/gemma2_9b/transfer/RUNPOD
set -euo pipefail
cd /workspace/narcbench-core-demo
ROOT=upstream/scenarios/gemma2_9b/transfer/RUNPOD
if [ ! -d "$ROOT" ]; then
  # maybe nested under upstream from tarball
  ROOT=$(find upstream -type d -path '*/gemma2_9b/transfer/RUNPOD' 2>/dev/null | head -1 || true)
fi
echo "[finish] transcripts root=$ROOT"
n=$(find "${ROOT:-/nonexistent}" -name 'transcript.json' 2>/dev/null | wc -l)
echo "[finish] transcript count=$n"
# Launch extract if needed
if [ "${1:-}" = "--launch-extract" ] && [ "$n" -ge 60 ]; then
  .venv-probe/bin/python -u scripts/runpod_launch_extract_acts.py \
    --runs-rel gemma2_9b/transfer/RUNPOD \
    --model google/gemma-2-9b-it \
    --layers 19-23 --gen-only --launch \
    2>&1 | tee results/transfer/extract_full_launch.log
fi
