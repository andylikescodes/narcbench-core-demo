#!/bin/bash
# Pod-side driver for the jspace-interp rebuild (started by scripts/runpod_launch_jspace_interp.py after it clones this
# repository at a pushed SHA). Everything is logged under /tmp/run, which is served on port 8765 (status.json, stages.json,
# logs/, results.tgz, acts.tgz, manifest.json). Stages, in order:
#   setup, download (published lens + gated Gemma-2-9B-it), gen_warned (50 matched-vocabulary control episodes),
#   extract_warned, extract_original, extract_swap, extract_prompt_only, extract_nosys (each followed by queueing the
#   CPU analysis jobs for that condition), role_dirs, steer (per JOB_STEER_RUNS), wait for the analysis queue, package.
# Env: HF_TOKEN, RUNPOD_API_KEY (self-termination), RUNPOD_POD_ID, JOB_MAX_MINUTES, JOB_GRACE_MINUTES, JOB_N_RANDOM,
#      JOB_POSITIONS, JOB_LAYERS, JOB_STEER_RUNS, JOB_ANALYSIS_PROCS, JOB_WARNED_WORKERS, JOB_WARNED_BATCH, JOB_SKIP.
set -uo pipefail
cd "$(dirname "$0")/.."
REPO=$(pwd)
RUN=/tmp/run; LOGS=$RUN/logs; mkdir -p "$LOGS" "$RUN/jobs/todo" "$RUN/jobs/running" "$RUN/jobs/done"
OUT=$REPO/results/jspace_interp
MODEL=${JOB_MODEL:-google/gemma-2-9b-it}
LAYERS=${JOB_LAYERS:-19-23}
LAYERS_CSV=$(python3 -c "import sys; a=sys.argv[1]; print(','.join(str(x) for x in (range(int(a.split('-')[0]), int(a.split('-')[1])+1) if '-' in a else a.split(','))))" "$LAYERS")
POSITIONS=${JOB_POSITIONS:-last,prevote,eot,reply_mean,frac50,frac25,frac75}
N_RANDOM=${JOB_N_RANDOM:-200}
STEER_RUNS=${JOB_STEER_RUNS:-swap:22,original:22,swap:23,original:23}
MAX_MIN=${JOB_MAX_MINUTES:-480}
GRACE_MIN=${JOB_GRACE_MINUTES:-240}
SKIP=",${JOB_SKIP:-},"
NPROC=$(nproc)
# nproc and free report the host inside a RunPod container; the pod's own allocation is in its cgroup limits
cg_cpus() {
  local q p
  if [ -r /sys/fs/cgroup/cpu.max ]; then read -r q p < /sys/fs/cgroup/cpu.max; if [ "$q" != max ]; then echo $(( (q + p - 1) / p )); return; fi; fi
  if [ -r /sys/fs/cgroup/cpu/cpu.cfs_quota_us ]; then q=$(cat /sys/fs/cgroup/cpu/cpu.cfs_quota_us); p=$(cat /sys/fs/cgroup/cpu/cpu.cfs_period_us)
    if [ "$q" -gt 0 ] 2>/dev/null; then echo $(( (q + p - 1) / p )); return; fi; fi
  nproc
}
cg_mem_gb() {
  local m f
  for f in /sys/fs/cgroup/memory.max /sys/fs/cgroup/memory/memory.limit_in_bytes; do
    if [ -r "$f" ]; then m=$(cat "$f"); if [ "$m" != max ] && [ "$m" -lt 1000000000000000 ] 2>/dev/null; then echo $(( m / 1073741824 )); return; fi; fi
  done
  free -g | awk '/^Mem:/{print $2}'
}
CG_CPUS=$(cg_cpus); CG_MEM_GB=$(cg_mem_gb)
# one single-threaded analysis job per usable core, keeping one core for the GPU process, and at most one per JOB_GB_PER_PROC of RAM
GB_PER_PROC=${JOB_GB_PER_PROC:-6}
PROCS_CPU=$(( CG_CPUS > 2 ? CG_CPUS - 1 : 1 )); PROCS_MEM=$(( CG_MEM_GB / GB_PER_PROC )); [ "$PROCS_MEM" -lt 1 ] && PROCS_MEM=1
PROCS=${JOB_ANALYSIS_PROCS:-$(( PROCS_CPU < PROCS_MEM ? PROCS_CPU : PROCS_MEM ))}
export HF_HOME=${HF_HOME:-/workspace/nb/hf}; mkdir -p "$HF_HOME"
export HF_TOKEN="${HF_TOKEN:-}"; export HF_HUB_ENABLE_HF_TRANSFER=0; export TOKENIZERS_PARALLELISM=false
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
  mkdir -p "$OUT/pod_logs"; cp -a "$LOGS"/. "$OUT/pod_logs/" 2>/dev/null; cp "$RUN"/job.log "$RUN"/stages.json "$RUN"/status.json "$OUT/pod_logs/" 2>/dev/null
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader > "$OUT/pod_logs/gpu.txt" 2>/dev/null || true
  (cd "$REPO" && tar --exclude='results/jspace_interp/acts' -czf "$RUN/results.tgz" results/jspace_interp) 2>>"$RUN/job.log"
  (cd "$REPO" && [ -d results/jspace_interp/acts ] && tar -cf "$RUN/acts.tgz" results/jspace_interp/acts) 2>>"$RUN/job.log"
  python3 - <<'PY'
import hashlib, json, os
m = {}
for n in ("results.tgz", "acts.tgz"):
    p = os.path.join("/tmp/run", n)
    if os.path.exists(p):
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for b in iter(lambda: f.read(1 << 24), b""): h.update(b)
        m[n] = {"size": os.path.getsize(p), "sha256": h.hexdigest()}
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

# ---- analysis queue: jobs are shell files in jobs/todo, run PROCS at a time, each under the remaining budget ----
queue_job() {  # queue_job NAME CMD...
  local name=$1; shift
  printf '%s\n' "cd $REPO" "export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1" "exec timeout \$(python3 -c \"import time;print(max(1,int(($T0+${MAX_MIN}*60-time.time())/60)))\")m $*" > "$RUN/jobs/todo/$name.sh"
  log "queued $name"
}
run_queue() {
  while true; do
    local n_run=$(ls "$RUN/jobs/running" | grep -c '\.sh$')
    local next=$(ls "$RUN/jobs/todo" 2>/dev/null | sort | head -1)
    if [ -n "$next" ] && [ "$n_run" -lt "$PROCS" ]; then
      mv "$RUN/jobs/todo/$next" "$RUN/jobs/running/$next"
      ( bash "$RUN/jobs/running/$next" > "$LOGS/analysis_${next%.sh}.log" 2>&1; echo $? > "$RUN/jobs/running/$next.rc";
        mv "$RUN/jobs/running/$next" "$RUN/jobs/running/$next.rc" "$RUN/jobs/done/" 2>/dev/null ) &
      sleep 2; continue
    fi
    if [ -f "$RUN/jobs/STOP" ] && [ -z "$next" ] && [ "$n_run" -eq 0 ]; then return 0; fi
    if [ "$(elapsed_min)" -ge "$MAX_MIN" ]; then return 0; fi
    sleep 20
  done
}
queue_analysis_for() {  # queue_analysis_for CONDITION POSITIONS_CSV TAG_PREFIX
  local cond=$1 poss=$2 pre=$3 i=0
  IFS=',' read -ra PA <<< "$poss"
  for pos in "${PA[@]}"; do
    i=$((i+1))
    queue_job "$(printf '%s_%02d_%s_%s' "$pre" "$i" "$cond" "$pos")" python3 scripts/jspace_interp_analyze.py --root "$OUT/acts" --lens "$LENS" \
      --basis-dir "$BASIS" --tokenizer "$BASIS/tokenizer.json" --layers "$LAYERS" --ks 8,25,50,100 --positions "$pos" \
      --conditions "$cond" --n-random "$N_RANDOM" --n-random-lodo 20 --n-random-mv 20 --out "$OUT/gemma2_9b/${cond}_${pos}"
  done
}

# ---- status server (Range-capable; replaces the bootstrap's plain server) ----
pkill -f "http.server 8765" 2>/dev/null || true
nohup python3 scripts/runpod_range_server.py "$RUN" 8765 > /tmp/http.log 2>&1 &
ELAPSED_MIN=0 st running boot "repo $(git rev-parse --short HEAD); gpu $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1); nproc $NPROC; cgroup cpus $CG_CPUS mem ${CG_MEM_GB}G; analysis procs $PROCS"
log "config: model=$MODEL layers=$LAYERS positions=$POSITIONS n_random=$N_RANDOM steer=$STEER_RUNS max=$MAX_MIN min grace=$GRACE_MIN min procs=$PROCS (cgroup cpus $CG_CPUS, mem ${CG_MEM_GB}G, nproc $NPROC) skip=$SKIP"
run_queue &
QUEUE_PID=$!

# ---- 1. setup ----
stage setup bash -c "python3 -m pip install -q 'transformers>=4.56' accelerate 'scikit-learn>=1.5' scipy 'huggingface_hub>=0.26' sentencepiece protobuf && python3 -c 'import torch,transformers,sklearn; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0), transformers.__version__, sklearn.__version__)' && nvidia-smi && nproc && free -g && df -h /workspace /tmp" || finish failed setup

# ---- 2. downloads: public lens, gated Gemma ----
stage download python3 - <<PY || finish failed download
import os, shutil, time
from huggingface_hub import hf_hub_download, snapshot_download
t=time.time()
p = hf_hub_download("neuronpedia/jacobian-lens", "gemma-2-9b-it/jlens/Salesforce-wikitext/gemma-2-9b-it_jacobian_lens.pt")
os.makedirs(os.path.dirname("$LENS"), exist_ok=True); shutil.copyfile(p, "$LENS")
print("lens", os.path.getsize("$LENS"), f"{time.time()-t:.0f}s", flush=True)
t=time.time()
d = snapshot_download("$MODEL", token=os.environ.get("HF_TOKEN") or None, allow_patterns=["*.json", "*.safetensors", "tokenizer*", "*.jinja"])
print("model", d, sum(os.path.getsize(os.path.join(r,f)) for r,_,fs in os.walk(d) for f in fs)//2**20, "MiB", f"{time.time()-t:.0f}s", flush=True)
PY

# ---- 3. matched-vocabulary control episodes (GPU generation), then their activations ----
stage gen_warned python3 scripts/jspace_interp_gen_matched_vocab.py --model "$MODEL" --all --workers "${JOB_WARNED_WORKERS:-8}" \
  --batch "${JOB_WARNED_BATCH:-8}" --out "$OUT/matched_vocab_runs" || log "gen_warned failed; continuing without the control"
stage extract_warned python3 scripts/jspace_interp_extract.py --model "$MODEL" --splits warned --warned-runs "$OUT/matched_vocab_runs" \
  --conditions original prompt_only --layers "$LAYERS_CSV" --export-basis "$BASIS" --out "$OUT/acts" || log "extract_warned failed; continuing"
[ -f "$BASIS/unembed_fp16.npy" ] || stage export_basis python3 scripts/jspace_interp_extract.py --model "$MODEL" --splits core --conditions original \
  --max-rows 4 --layers 19 --export-basis "$BASIS" --out /tmp/basis_probe || finish failed export_basis

# atom norms per layer once, so the parallel analysis jobs read a finished cache instead of racing to write it
stage atom_norms python3 - <<PY || log "atom_norms failed; the analysis jobs compute them"
import numpy as np, sys
sys.path.insert(0, "$REPO")
from jspace_interp.nov5_jspace_decompose import atom_norms, load_lens, load_unembed, norm_scale, parse_layers
from pathlib import Path
layers = parse_layers("$LAYERS"); J, info = load_lens(Path("$LENS"), layers); W, _ = load_unembed(Path("$BASIS"))
s = norm_scale(np.load("$BASIS/final_norm_weight_fp32.npy"), "gemma")
for L in layers:
    an = atom_norms(W, s, J[L], Path("$BASIS") / f"derived_atom_norms_L{L}.npy"); print(L, float(an.mean()), float(an.max()), flush=True)
PY

# ---- 4. main extraction, one condition at a time; each queues its CPU analysis ----
# prompt_only first (short contexts, one job, the priority control), then original and swap; the queue runs lower names first,
# so when slots are scarce the controls and the earlier positions in JOB_POSITIONS go ahead of the frac positions and nosys
for cond in prompt_only original swap nosys; do
  if stage "extract_$cond" python3 scripts/jspace_interp_extract.py --model "$MODEL" --splits core transfer --conditions "$cond" \
       --layers "$LAYERS_CSV" --out "$OUT/acts"; then
    case $cond in
      prompt_only) queue_analysis_for prompt_only prompt_end 0 ;;
      original) queue_analysis_for original "$POSITIONS" 1 ;;
      swap) queue_analysis_for swap "$POSITIONS" 2 ;;
      nosys) queue_analysis_for nosys last 3 ;;
    esac
  else
    log "extract_$cond failed; its analysis is not queued"
  fi
done

# ---- 5. role directions (CPU, short) and steering (GPU) ----
for cond in swap original; do
  stage "role_dirs_$cond" python3 scripts/jspace_interp_role_dir.py --root "$OUT/acts" --condition "$cond" --position last \
    --layers "$(echo "$STEER_RUNS" | tr ',' '\n' | grep "^$cond:" | cut -d: -f2 | sort -u | paste -sd, -)" --lens "$LENS" --basis-dir "$BASIS" \
    --out "$OUT/role_dirs" || log "role_dirs_$cond failed"
done
IFS=',' read -ra SR <<< "$STEER_RUNS"
for run in "${SR[@]}"; do
  cond=${run%%:*}; L=${run##*:}
  rd="$OUT/role_dirs/role_dir_J25_${cond}_last_L${L}.npy"
  hj=$(python3 -c "import json,sys; r=[x for x in json.load(open('$OUT/role_dirs/role_dirs.json')) if x['condition']=='$cond' and x['layer']==$L]; print(r[0]['hj_norm_median'] if r else '')" 2>/dev/null)
  extra=(); [ -f "$rd" ] && extra+=(--role-direction "$rd"); [ -n "$hj" ] && extra+=(--hj-norm "$hj")
  if [ "$(remaining_min)" -lt 45 ]; then log "steer_${cond}_L${L}: skipped, under 45 min left"; record_stage "steer_${cond}_L${L}" 98 0; continue; fi
  stage "steer_${cond}_L${L}" python3 scripts/jspace_interp_steer.py --model "$MODEL" --lens "$LENS" --basis-dir "$BASIS" --layer "$L" \
    --condition "$cond" --n-random 20 "${extra[@]}" --out "$OUT/steer" || log "steer_${cond}_L${L} failed"
done

# ---- 6. wait for the analysis queue, then package and serve ----
touch "$RUN/jobs/STOP"
ELAPSED_MIN=$(elapsed_min) st running analysis_wait "todo $(ls $RUN/jobs/todo | wc -l) running $(ls $RUN/jobs/running | grep -c '\.sh$')"
while kill -0 $QUEUE_PID 2>/dev/null; do
  sleep 60
  ELAPSED_MIN=$(elapsed_min) st running analysis_wait "todo $(ls $RUN/jobs/todo | wc -l) running $(ls $RUN/jobs/running | grep -c '\.sh$') done $(ls $RUN/jobs/done | grep -c '\.sh$')"
  if [ "$(elapsed_min)" -ge "$MAX_MIN" ]; then log "time budget reached; stopping the analysis queue"; pkill -f jspace_interp_analyze.py; break; fi
done
sleep 5
finish done "stages $(python3 -c "import json;d=json.load(open('/tmp/run/stages.json'));print(sum(1 for v in d.values() if v['rc']==0),'ok of',len(d))")"
