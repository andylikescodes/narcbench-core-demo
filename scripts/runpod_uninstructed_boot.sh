#!/bin/bash
# Pod-side driver for the uninstructed-collusion experiment (started by scripts/runpod_launch_uninstructed.py after it
# clones this repository at a pushed SHA). Everything is logged under /tmp/run, which is served on port 8765
# (status.json, stages.json, job.log, logs/, results.tgz, manifest.json).
#
# JOB_MODE=pilot (docs/uninstructed_collusion/DESIGN.md, sections 10 and 11), stages in order:
#   setup, download (published lens + gated Gemma-2-9B-it),
#   gen_ref    version S0, 8 prompts per call
#   gen_stake  versions JOB_VERSIONS (default S1,S2,S3) with JOB_BATCH prompts per call (default 8: 16 ran out of
#              memory on the first pod). The second pilot runs T1,T2,T3,T4,T5 into JOB_PILOT_DIR=pilot2.
#   A stage named in JOB_SKIP is skipped. For gen_ref this is used when the S0 runs are already in the checkout.
#   code_rooms room codes by Gemma-2-9B-it, greedy decoding (coder A)
#   summarize  provisional rates from the votes and coder A
#   export_basis, timing   the timing test for the main run (durations only; its activations are deleted)
#   package
# Env: HF_TOKEN, RUNPOD_API_KEY (self-termination), RUNPOD_POD_ID, JOB_MAX_MINUTES, JOB_GRACE_MINUTES, JOB_PILOT_N,
#      JOB_PILOT_RUNS, JOB_BATCH, JOB_WORKERS, JOB_CODER_BATCH, JOB_TIMING_EPISODES, JOB_SKIP, JOB_VERSIONS,
#      JOB_PILOT_DIR, JOB_THREADS, JOB_SCENARIOS (pilot | all), JOB_STANCE_ROUNDS (comma list of earlier round folders),
#      JOB_REF_VERSIONS and JOB_REF_RUNS (the no-stake versions of gen_ref and their runs; default S0 and JOB_PILOT_RUNS).
set -uo pipefail
cd "$(dirname "$0")/.."
REPO=$(pwd)
RUN=/tmp/run; LOGS=$RUN/logs; mkdir -p "$LOGS"
OUT=$REPO/results/uninstructed
MODE=${JOB_MODE:-pilot}
MODEL=${JOB_MODEL:-google/gemma-2-9b-it}
MAX_MIN=${JOB_MAX_MINUTES:-150}
GRACE_MIN=${JOB_GRACE_MINUTES:-30}
SKIP=",${JOB_SKIP:-},"
PILOT_N=${JOB_PILOT_N:-12}
PILOT_RUNS=${JOB_PILOT_RUNS:-2}
BATCH=${JOB_BATCH:-8}
WORKERS=${JOB_WORKERS:-8}
CODER_BATCH=${JOB_CODER_BATCH:-8}
TIMING_EPISODES=${JOB_TIMING_EPISODES:-10}
VERSIONS=${JOB_VERSIONS:-S1,S2,S3}
PILOT_DIR=${JOB_PILOT_DIR:-pilot}
REF_VERSIONS=${JOB_REF_VERSIONS:-S0}               # the versions of the gen_ref stage (no stake), with their own number of runs
REF_RUNS=${JOB_REF_RUNS:-$PILOT_RUNS}
SCENARIOS=${JOB_SCENARIOS:-pilot}                  # pilot: the PILOT_N pilot scenarios; all: every scenario of the bank
STANCE_ROUNDS=${JOB_STANCE_ROUNDS:-}               # earlier round folders whose public positions are coded as well
if [ "$SCENARIOS" = all ]; then SCEN_ARGS=(--all); else SCEN_ARGS=(--pilot --pilot-n "$PILOT_N"); fi
P=$OUT/$PILOT_DIR
TIMING_VERSION=${VERSIONS%%,*}
# Threads of the numerical libraries. A pod shows every CPU of its host (96 on the second pod) to a container that may
# use a few of them; with one thread per visible CPU the non-negative refit of the timing test took 207 ms per
# activation. Use the container's quota when it can be read, else 8.
cpu_quota() {
  local q p
  if [ -r /sys/fs/cgroup/cpu.max ]; then read -r q p < /sys/fs/cgroup/cpu.max; [ "$q" != max ] && { echo $(( (q + p - 1) / p )); return; }; fi
  if [ -r /sys/fs/cgroup/cpu/cpu.cfs_quota_us ]; then
    q=$(cat /sys/fs/cgroup/cpu/cpu.cfs_quota_us); p=$(cat /sys/fs/cgroup/cpu/cpu.cfs_period_us)
    [ "$q" -gt 0 ] && { echo $(( (q + p - 1) / p )); return; }
  fi
  echo 0
}
QUOTA=$(cpu_quota)
THREADS=${JOB_THREADS:-${RUNPOD_CPU_COUNT:-$QUOTA}}
{ [ "$THREADS" -ge 1 ] 2>/dev/null && [ "$THREADS" -le 32 ]; } || THREADS=8
export OMP_NUM_THREADS=$THREADS OPENBLAS_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS NUMEXPR_NUM_THREADS=$THREADS VECLIB_MAXIMUM_THREADS=$THREADS
export HF_HOME=${HF_HOME:-/workspace/nb/hf}; mkdir -p "$HF_HOME"
export HF_TOKEN="${HF_TOKEN:-}"; export HF_HUB_ENABLE_HF_TRANSFER=0; export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LENS=$HF_HOME/lens/gemma-2-9b-it_jacobian_lens.pt
BASIS=$REPO/data/j_basis/gemma2_9b
T0=$(date +%s)

elapsed_min() { echo $(( ($(date +%s) - T0) / 60 )); }
remaining_min() { local r=$(( MAX_MIN - $(elapsed_min) )); echo $(( r > 1 ? r : 1 )); }
st() { python3 - "$@" <<'PY'
import json, sys, time, os
p = "/tmp/run/status.json"
try: cur = json.load(open(p))
except Exception: cur = {}
cur.update({"phase": sys.argv[1], "stage": sys.argv[2] if len(sys.argv) > 2 else cur.get("stage"),
            "detail": sys.argv[3] if len(sys.argv) > 3 else "", "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "elapsed_min": int(os.environ.get("ELAPSED_MIN", "0")), "pod": os.environ.get("RUNPOD_POD_ID", "")})
json.dump(cur, open(p, "w"))
print("[status]", json.dumps(cur), flush=True)
PY
}
log() { echo "$(date -u +%H:%M:%S) $*" | tee -a "$RUN/job.log"; }
record_stage() { python3 - "$1" "$2" "$3" <<'PY'
import json, sys
p = "/tmp/run/stages.json"
try: d = json.load(open(p))
except Exception: d = {}
d[sys.argv[1]] = {"rc": int(sys.argv[2]), "seconds": int(sys.argv[3])}
json.dump(d, open(p, "w"), indent=1)
PY
}
# stage NAME CMD... : runs under the remaining time budget, logs to logs/NAME.log, records rc + seconds
stage() {
  local name=$1; shift
  if [[ "$SKIP" == *",$name,"* ]]; then log "stage $name skipped (JOB_SKIP)"; record_stage "$name" 99 0; return 99; fi
  ELAPSED_MIN=$(elapsed_min) st running "$name"
  local t=$(date +%s)
  log "stage $name: $*"
  timeout "$(remaining_min)m" "$@" > "$LOGS/$name.log" 2>&1
  local rc=$?
  record_stage "$name" $rc $(( $(date +%s) - t ))
  log "stage $name: exit $rc after $(( ($(date +%s) - t) / 60 )) min"; tail -3 "$LOGS/$name.log" | cut -c1-300 >> "$RUN/job.log"
  return $rc
}
term() { python3 - <<'PY' || true
import os, json, urllib.request as u
p=os.environ.get('RUNPOD_POD_ID',''); k=os.environ.get('RUNPOD_API_KEY','')
if p and k:
    b=json.dumps({'query':'mutation { podTerminate(input: {podId: "%s"}) }'%p}).encode()
    r=u.Request('https://api.runpod.io/graphql',data=b,headers={'content-type':'application/json','Authorization':'Bearer '+k,'User-Agent':'nb-interp/1'},method='POST')
    print(u.urlopen(r,timeout=30).read()[:200])
PY
}
package() {
  ELAPSED_MIN=$(elapsed_min) st packaging package
  PL="$OUT/pod_logs/${RUNPOD_POD_ID:-pod}"
  mkdir -p "$PL"; cp -a "$LOGS"/. "$PL/" 2>/dev/null; cp "$RUN"/job.log "$RUN"/stages.json "$RUN"/status.json "$PL/" 2>/dev/null
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader > "$PL/gpu.txt" 2>/dev/null || true
  (cd "$REPO" && tar --exclude='results/uninstructed/acts' -czf "$RUN/results.tgz" results/uninstructed) 2>>"$RUN/job.log"
  python3 - <<'PY'
import hashlib, json, os
m = {}
p = "/tmp/run/results.tgz"
if os.path.exists(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""): h.update(b)
    m["results.tgz"] = {"size": os.path.getsize(p), "sha256": h.hexdigest()}
json.dump(m, open("/tmp/run/manifest.json", "w"), indent=1)
print(m)
PY
}
finish() {  # finish PHASE DETAIL
  package
  ELAPSED_MIN=$(elapsed_min) st "$1" done "$2"
  log "finished: $1 ($2); serving results for $GRACE_MIN min, then terminating"
  sleep "${GRACE_MIN}m"; term; exit 0
}

# ---- status server (Range-capable; replaces the bootstrap's plain server) ----
pkill -f "http.server 8765" 2>/dev/null || true
nohup python3 scripts/runpod_range_server.py "$RUN" 8765 > /tmp/http.log 2>&1 &
ELAPSED_MIN=0 st running boot "repo $(git rev-parse --short HEAD); gpu $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1); mode $MODE"
log "config: mode=$MODE model=$MODEL scenarios=$SCENARIOS pilot_n=$PILOT_N pilot_runs=$PILOT_RUNS ref=$REF_VERSIONS x$REF_RUNS versions=$VERSIONS stance_rounds=$STANCE_ROUNDS batch=$BATCH workers=$WORKERS max=$MAX_MIN min grace=$GRACE_MIN min skip=$SKIP"
log "config: versions=$VERSIONS results=$P threads=$THREADS (cpu quota $QUOTA, RUNPOD_CPU_COUNT=${RUNPOD_CPU_COUNT:-unset}, nproc $(nproc))"
if [ "$MODE" != pilot ]; then log "unknown JOB_MODE $MODE"; finish failed "unknown mode $MODE"; fi

# ---- 1. setup ----
stage setup bash -c "python3 -m pip install -q 'transformers>=4.56' accelerate 'scikit-learn>=1.5' scipy 'huggingface_hub>=0.26' sentencepiece protobuf && python3 -c 'import torch,transformers,sklearn; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0), transformers.__version__, sklearn.__version__)' && nvidia-smi && nproc && free -g && df -h /workspace /tmp" || finish failed setup

# ---- 2. downloads: public lens (for the timing test), gated Gemma ----
stage download python3 - <<PY || finish failed download
import os, shutil, time
from huggingface_hub import hf_hub_download, snapshot_download
t=time.time()
d = snapshot_download("$MODEL", token=os.environ.get("HF_TOKEN") or None, allow_patterns=["*.json", "*.safetensors", "tokenizer*", "*.jinja"])
print("model", d, sum(os.path.getsize(os.path.join(r,f)) for r,_,fs in os.walk(d) for f in fs)//2**20, "MiB", f"{time.time()-t:.0f}s", flush=True)
t=time.time()
try:
    p = hf_hub_download("neuronpedia/jacobian-lens", "gemma-2-9b-it/jlens/Salesforce-wikitext/gemma-2-9b-it_jacobian_lens.pt")
    os.makedirs(os.path.dirname("$LENS"), exist_ok=True); shutil.copyfile(p, "$LENS")
    print("lens", os.path.getsize("$LENS"), f"{time.time()-t:.0f}s", flush=True)
except Exception as e:  # the lens is only needed by the timing test
    print("lens download failed:", type(e).__name__, str(e)[:200], flush=True)
PY

# ---- 3. generation: S0 at the reference batch size, then the other versions ----
stage gen_ref python3 scripts/uninstructed_generate.py --model "$MODEL" --versions "$REF_VERSIONS" "${SCEN_ARGS[@]}" --runs "$REF_RUNS" \
  --workers 8 --batch 8 --tag "${REF_VERSIONS//,/-}" --out "$P/runs"
rc=$?; [ $rc -eq 0 ] || [ $rc -eq 99 ] || finish failed gen_ref
[ $rc -eq 99 ] && log "gen_ref skipped: $(ls "$P/runs/${REF_VERSIONS%%,*}" 2>/dev/null | wc -l) ${REF_VERSIONS%%,*} run folders in the checkout"
stage gen_stake python3 scripts/uninstructed_generate.py --model "$MODEL" --versions "$VERSIONS" "${SCEN_ARGS[@]}" --runs "$PILOT_RUNS" \
  --workers "$WORKERS" --batch "$BATCH" --tag "${VERSIONS//,/-}" --out "$P/runs"
rc=$?; [ $rc -eq 0 ] || [ $rc -eq 99 ] || finish failed gen_stake

# ---- 4. coder A and the provisional rates ----
if stage code_rooms python3 scripts/uninstructed_code_rooms.py --model "$MODEL" --runs "$P/runs" --batch "$CODER_BATCH" --out "$P/coding_gemma.json"; then
  stage summarize python3 scripts/uninstructed_summarize.py --runs "$P/runs" --coding "$P/coding_gemma.json" \
    --out "$P/provisional" || log "summarize failed; the rates are computed after the pull"
else
  log "code_rooms failed; the rooms are coded after the pull"
fi

# ---- 4b. public position of every member, coded by Gemma: this round, then the earlier rounds named in JOB_STANCE_ROUNDS ----
# (exploratory reading, DESIGN.md notes of 17 November 2026; a failure here loses nothing else)
if [[ "$SKIP" == *",stance,"* ]]; then
  log "stance skipped (JOB_SKIP)"; record_stage stance 99 0
else
  for rd in "$PILOT_DIR" ${STANCE_ROUNDS//,/ }; do
    [ -d "$OUT/$rd/runs" ] || { log "stance: no runs under $OUT/$rd"; continue; }
    [ -f "$OUT/$rd/stance_gemma.json" ] && { log "stance: $rd is coded already"; continue; }
    if [ "$(remaining_min)" -lt 8 ]; then log "stance of $rd skipped: under 8 min left"; record_stage "stance_$rd" 98 0; continue; fi
    stage "stance_$rd" python3 scripts/uninstructed_public_stance.py --model "$MODEL" --runs "$OUT/$rd/runs" --batch "$CODER_BATCH" \
      --key "$OUT/$rd/stance_key_all.json" --answers-out "$OUT/$rd/stance_gemma.json" || log "stance coding of $rd failed"
  done
fi

# ---- 5. timing test for the main run ----
if [ "$(remaining_min)" -lt 20 ]; then
  log "timing test skipped: under 20 min left"; record_stage timing 98 0
else
  if [ ! -f "$BASIS/unembed_fp16.npy" ]; then
    stage export_basis python3 scripts/jspace_interp_extract.py --model "$MODEL" --splits core --conditions original \
      --max-rows 4 --layers 19 --export-basis "$BASIS" --out /tmp/basis_probe
    rc=$?; [ $rc -eq 0 ] || [ $rc -eq 99 ] || log "export_basis failed; the decomposition is not timed"
  fi
  TARGS=()
  [ -f "$LENS" ] && [ -f "$BASIS/unembed_fp16.npy" ] && TARGS+=(--lens "$LENS" --basis-dir "$BASIS")
  stage timing python3 scripts/uninstructed_timing_test.py --model "$MODEL" --runs "$P/runs/$TIMING_VERSION" --episodes "$TIMING_EPISODES" \
    "${TARGS[@]}" --out "$P/timing.json"
  rc=$?; [ $rc -eq 0 ] || [ $rc -eq 99 ] || log "timing test failed"
fi

finish done "stages $(python3 -c "import json;d=json.load(open('/tmp/run/stages.json'));print(sum(1 for v in d.values() if v['rc']==0),'ok of',len(d))")"
