#!/bin/bash
# Stage 2: passes of the probe sweep with shuffled labels on a laptop, from the tables a probe pod served. No GPU.
# The same call as the pod's s2_null_<n> stages (scripts/runpod_uninstructed_boot.sh), for the seeds FIRST..LAST and
# the named contrasts, into OUT/seed<n>; then the summary (OUT/NULLS.md, OUT/nulls.json). A seed whose probe.json
# exists is skipped, so the script can be stopped and started again. One pass of the vote contrast takes about five
# minutes on four cores.
#
#   bash scripts/uninstructed_nulls_local.sh <python> FIRST LAST JOBS CONTRASTS OUT
#   bash scripts/uninstructed_nulls_local.sh python3 11 100 4 vote results/uninstructed/stage2/nulls
#   bash scripts/uninstructed_nulls_local.sh python3 1 30 4 hidden,collusion_clean,collusion_rest results/uninstructed/stage2/nulls_other
set -uo pipefail
PY=$1; FIRST=$2; LAST=$3; JOBS=$4; CONTRASTS=$5; OUT=$6
cd "$(dirname "$0")/.." || exit 2
ACTS=results/uninstructed/acts/stage2
[ -f "$ACTS/tables/index.json" ] || { echo "no tables under $ACTS"; exit 3; }
mkdir -p "$OUT"
for sd in $(seq "$FIRST" "$LAST"); do
  if [ -f "$OUT/seed$sd/probe.json" ]; then echo "seed $sd: already there"; continue; fi
  t=$(date +%s)
  env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 "$PY" scripts/uninstructed_probe.py \
    --acts "$ACTS" --out "$OUT/seed$sd" --jobs "$JOBS" --contrasts "$CONTRASTS" --inputs full,bow --shuffle-labels "$sd" > "$OUT.seed$sd.log" 2>&1
  rc=$?
  echo "$(date +%H:%M:%S) $CONTRASTS seed $sd: exit $rc after $(( $(date +%s) - t )) s"
  [ $rc -eq 0 ] || { tail -5 "$OUT.seed$sd.log"; exit 4; }
  rm -f "$OUT.seed$sd.log"
done
"$PY" scripts/uninstructed_probe.py --summarize-nulls "$OUT" | tail -30
echo "$(date +%H:%M:%S) $CONTRASTS done: $(ls "$OUT"/seed*/probe.json | wc -l | tr -d ' ') passes in $OUT"
