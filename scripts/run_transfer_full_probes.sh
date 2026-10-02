#!/bin/bash
set -euo pipefail
cd /workspace/narcbench-core-demo
ACTS=data/activations/gemma2_9b/transfer/RUNPOD
CORE=data/activations/gemma2_9b/core/20261001T012639Z
OUT=results/transfer
PY=.venv-probe/bin/python

# Locate acts if nested
if [ ! -f "$ACTS/activations_gen.npz" ]; then
  ALT=$(find data/activations -path '*/transfer/RUNPOD/activations_gen.npz' 2>/dev/null | head -1)
  if [ -n "$ALT" ]; then ACTS=$(dirname "$ALT"); fi
fi
echo "[probes] acts=$ACTS"
ls -la "$ACTS" | head

# Full acts eval
$PY -u scripts/eval_transfer_from_core.py \
  --core-acts "$CORE" \
  --transfer-acts "$ACTS" \
  --out "$OUT/FULL_core_to_transfer_full" \
  --layers 19-23 --core-phase discussion --transfer-phase final \
  2>&1 | tee "$OUT/FULL_core_to_transfer_full.log"

# Project W_U k25 and jac k25 on Transfer FULL
$PY -u scripts/project_j_variants.py \
  --acts-dir "$ACTS" --variant wu_topk --k 25 --layers 19-23 --batch 96 --vocab-chunk 16384 \
  --out-root "$ACTS/variants/wu_topk_k25" \
  2>&1 | tee "$OUT/FULL_project_wu_k25.log"

$PY -u scripts/project_j_variants.py \
  --acts-dir "$ACTS" --variant jac_topk --k 25 --layers 19,21,23 --batch 96 --vocab-chunk 16384 \
  --jac-dir data/j_basis/gemma2_9b/jacobians \
  --out-root "$ACTS/variants/jac_topk_k25" \
  2>&1 | tee "$OUT/FULL_project_jac_k25.log"

for arm in j_only complement; do
  $PY -u scripts/eval_transfer_from_core.py \
    --core-acts "$CORE/variants/wu_topk_k25_L19-23/$arm" \
    --transfer-acts "$ACTS/variants/wu_topk_k25/$arm" \
    --out "$OUT/FULL_core_to_transfer_wu_k25_$arm" \
    --layers 19-23 --core-phase discussion --transfer-phase final \
    2>&1 | tee "$OUT/FULL_core_to_transfer_wu_k25_$arm.log"

  $PY -u scripts/eval_transfer_from_core.py \
    --core-acts "$CORE/variants/jac_topk_k25_L19-21-23/$arm" \
    --transfer-acts "$ACTS/variants/jac_topk_k25/$arm" \
    --out "$OUT/FULL_core_to_transfer_jac_k25_$arm" \
    --layers 19,21,23 --core-phase discussion --transfer-phase final \
    2>&1 | tee "$OUT/FULL_core_to_transfer_jac_k25_$arm.log"
done

echo DONE_PROBES
