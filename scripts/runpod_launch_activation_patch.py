#!/usr/bin/env python3
"""Launch self-contained activation-patch job on volume-cached smoke transcripts."""
from __future__ import annotations
import argparse, base64, json, os, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COLLUSION_SCRIPTS = Path("/workspace/collusion-exp/scripts")
SECRETS = Path("/home/box/.secrets/interp-explorer.env")
IMAGE = "runpod/pytorch:1.2.0-rc.162-cu1281-torch280-ubuntu2404"
VOLUME_ID = "cr1kjtzjs2"
POD_NAME = "job-narcbench-actpatch"
GPU_PREFS = [
    "NVIDIA A40", "NVIDIA GeForce RTX 4090", "NVIDIA RTX A6000",
    "NVIDIA GeForce RTX 3090", "NVIDIA RTX 4000 Ada Generation", "NVIDIA RTX A4000",
]

def read_env():
    if SECRETS.is_file():
        for line in SECRETS.read_text().splitlines():
            line=line.strip()
            if not line or line.startswith("#") or "=" not in line: continue
            k,v=line.split("=",1); os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-rel", default="gemma2_2b/core/20260930T214125Z")
    ap.add_argument("--model", default="google/gemma-2-2b-it")
    ap.add_argument("--launch", action="store_true")
    ap.add_argument("--max-minutes", type=int, default=90)
    ap.add_argument("--grace-minutes", type=int, default=8)
    args = ap.parse_args()
    read_env()
    script_b64 = base64.b64encode((ROOT/"scripts/activation_patch_core.py").read_bytes()).decode()
    boot = f"""#!/bin/bash
set -uo pipefail
RUN=/tmp/narcbench-job; mkdir -p "$RUN" /workspace/jobs/narcbench-results/activation_patch
python3 -m http.server 8765 --directory "$RUN" >/tmp/http.log 2>&1 &
echo '{{"phase":"boot"}}' > "$RUN/status.json"
phase() {{ echo "{{\\"phase\\":\\"$1\\"}}" > "$RUN/status.json"; echo "[phase] $1" | tee -a "$RUN/job.log"; }}
term() {{ python3 -c "import os,json,urllib.request as u;p=os.environ.get('RUNPOD_POD_ID','');k=os.environ.get('RUNPOD_API_KEY','');
b=json.dumps({{'query':'mutation {{ podTerminate(input: {{podId: \\"%s\\"}}) }}'%p}});
r=u.Request('https://api.runpod.io/graphql',data=b.encode(),headers={{'content-type':'application/json','Authorization':'Bearer '+k,'User-Agent':'nb/1'}},method='POST');
u.urlopen(r,timeout=30).read()" || true; }}
finish() {{ echo "{{\\"phase\\":\\"$1\\",\\"detail\\":\\"$2\\"}}" > "$RUN/status.json"; sleep {args.grace_minutes}m; term; exit 0; }}
phase setup
DEMO=/workspace/jobs/narcbench-actpatch; mkdir -p "$DEMO/scripts"
echo '{script_b64}' | base64 -d > "$DEMO/scripts/activation_patch_core.py"
PY=/workspace/interp-demo/.venv-explorer/bin/python; [ -x "$PY" ] || PY=python3
export HF_HOME=${{HF_HOME:-/workspace/interp-demo/models/hf}}
$PY -m pip install -q 'transformers>=4.40' accelerate sentencepiece protobuf 2>>"$RUN/job.log" || true
RUNS=""
for c in \\
  /workspace/jobs/narcbench-results/upstream/scenarios/{args.runs_rel} \\
  /workspace/jobs/narcbench-core-demo/upstream/scenarios/{args.runs_rel}; do
  [ -d "$c" ] && RUNS="$c" && break
done
[ -n "$RUNS" ] || finish failed missing_runs
OUT=/workspace/jobs/narcbench-results/activation_patch/smoke_$(date -u +%Y%m%dT%H%M%SZ)
phase running
set +e
timeout {args.max_minutes}m $PY -u "$DEMO/scripts/activation_patch_core.py" --runs-dir "$RUNS" --model {args.model} --out "$OUT" --layers 12,16,20 --max-new-tokens 64 >"$RUN/cmd.log" 2>&1
EC=$?; set -e
echo $EC > "$RUN/exit_code"; cat "$RUN/cmd.log" | tee -a "$RUN/job.log"
tar -czf "$RUN/results.tgz" -C /workspace/jobs/narcbench-results activation_patch || true
if [ $EC -eq 0 ]; then finish done ok; else finish failed exit_$EC; fi
"""
    boot_b64 = base64.b64encode(boot.encode()).decode()
    print(json.dumps({"boot_b64": len(boot_b64), "script_b64": len(script_b64), "will_launch": args.launch}, indent=2))
    if not args.launch:
        return 0
    if len(boot_b64) > 90000:
        sys.exit("boot too large")
    sys.path.insert(0, str(COLLUSION_SCRIPTS))
    from runpod_graphql import graphql
    env = {"JOB_BOOT_B64": boot_b64, "RUNPOD_API_KEY": os.environ["RUNPOD_API_KEY"]}
    if os.environ.get("HF_TOKEN"): env["HF_TOKEN"] = os.environ["HF_TOKEN"]
    docker_args = "bash -c 'echo $JOB_BOOT_B64 | base64 -d > /tmp/boot.sh; exec bash /tmp/boot.sh'"
    last=None
    for cloud in ("SECURE","COMMUNITY"):
      for gpu in GPU_PREFS:
        # Prefer no conflict: allow concurrent with full regen
        inp = {"name": POD_NAME, "imageName": IMAGE, "gpuTypeId": gpu, "cloudType": cloud,
               "gpuCount":1,"volumeInGb":0,"containerDiskInGb":40,"minVcpuCount":4,"minMemoryInGb":16,
               "ports":"8765/http","env":[{"key":k,"value":str(v)} for k,v in env.items()],
               "dockerArgs": docker_args, "supportPublicIp": False, "startSsh": False,
               "networkVolumeId": VOLUME_ID, "volumeMountPath": "/workspace"}
        q="""mutation ($input: PodFindAndDeployOnDemandInput!) { podFindAndDeployOnDemand(input: $input) { id desiredStatus costPerHr machine { gpuDisplayName dataCenterId } } }"""
        try:
            data=graphql(q,{"input":inp},timeout=120)
            if data.get("errors"): raise RuntimeError(str(data["errors"])[:300])
            pod=(data.get("data") or {}).get("podFindAndDeployOnDemand")
            if not pod or not pod.get("id"): raise RuntimeError("empty")
            print(json.dumps({"created":pod,"gpu":gpu,"cloud":cloud},indent=2))
            print(f"Status: https://{pod['id']}-8765.proxy.runpod.net/status.json")
            Path("/tmp/narcbench_patch_pod.json").write_text(json.dumps({"id":pod["id"],"gpu":gpu},indent=2))
            return 0
        except Exception as e:
            last=e; print(f"[launch] {cloud} {gpu}: {e}")
    sys.exit(f"all failed: {last}")

if __name__ == "__main__":
    raise SystemExit(main())
