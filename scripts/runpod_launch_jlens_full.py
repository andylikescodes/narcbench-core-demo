#!/usr/bin/env python3
"""Launch, watch and pull the full J-lens estimation (scripts/estimate_jlens_full.py) on a RunPod GPU.

Self-contained: keys come from the environment only (RUNPOD_API_KEY required to spend, watch or pull;
HF_TOKEN optional, used if the model is not already cached on the network volume). Default is estimate-only.

    python3 scripts/runpod_launch_jlens_full.py                       # print the plan and the cost estimate
    RUNPOD_API_KEY=... python3 scripts/runpod_launch_jlens_full.py --launch
    python3 scripts/runpod_launch_jlens_full.py --status <pod id>     # progress (status.json on the pod)
    python3 scripts/runpod_launch_jlens_full.py --pull <pod id>       # results.tgz -> data/j_basis/gemma2_9b/jlens_full/
    python3 scripts/runpod_launch_jlens_full.py --terminate <pod id>

The pod runs a timing smoke (8 prompts, rank 32, one layer) first, writes the projected hours to status.json,
shrinks rank 256 -> 128 and then prompts 1000 -> 600 -> 400 until the projection fits 90% of --max-minutes,
then runs the estimate with checkpoints, copies the outputs to the network
volume, serves results.tgz for --grace-minutes, and terminates itself.
"""
from __future__ import annotations

import argparse
import base64
import json
import math
import os
import sys
import tarfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IMAGE = "runpod/pytorch:1.2.0-rc.162-cu1281-torch280-ubuntu2404"
VOLUME_ID = "cr1kjtzjs2"  # network volume with the HF cache under /workspace/interp-demo/models/hf
POD_NAME = "job-narcbench-jlens-full"
OUT_REL = "gemma2_9b/jlens_full"
GPU_PREFS = [
    "NVIDIA RTX 6000 Ada Generation", "NVIDIA L40S", "NVIDIA A40", "NVIDIA RTX A6000",
    "NVIDIA A100 80GB PCIe", "NVIDIA A100-SXM4-80GB", "NVIDIA H100 PCIe",
    "NVIDIA RTX PRO 4500", "NVIDIA GeForce RTX 4090", "NVIDIA L4",
]
RATES = {
    "NVIDIA RTX 6000 Ada Generation": 0.89, "NVIDIA L40S": 0.89, "NVIDIA A40": 0.49, "NVIDIA RTX A6000": 0.79,
    "NVIDIA A100 80GB PCIe": 1.39, "NVIDIA A100-SXM4-80GB": 1.64, "NVIDIA H100 PCIe": 2.49,
    "NVIDIA RTX PRO 4500": 0.72, "NVIDIA GeForce RTX 4090": 0.74, "NVIDIA L4": 0.49,
}


def graphql(query: str, variables: dict | None = None, timeout: int = 120) -> dict:
    key = os.environ.get("RUNPOD_API_KEY")
    if not key:
        raise SystemExit("RUNPOD_API_KEY is not set in the environment")
    body = json.dumps({"query": query, "variables": variables or {}}).encode()
    req = urllib.request.Request(
        "https://api.runpod.io/graphql", data=body, method="POST",
        headers={"content-type": "application/json", "Authorization": f"Bearer {key}", "User-Agent": "nb-jlens/1"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def boot_script(a) -> str:
    est = (ROOT / "scripts/estimate_jlens_full.py").read_bytes()
    est_b64 = base64.b64encode(est).decode()
    mid_layer = sorted(parse_layers(a.layers))[len(parse_layers(a.layers)) // 2]
    n_layers = len(parse_layers(a.layers))
    hf = 'export HF_TOKEN="${HF_TOKEN:-}"'
    return f"""#!/bin/bash
set -uo pipefail
RUN=/tmp/narcbench-jlens; mkdir -p "$RUN"
python3 -m http.server 8765 --directory "$RUN" >/tmp/http.log 2>&1 &
st() {{ echo "$1" > "$RUN/status.json"; echo "[status] $1" | tee -a "$RUN/job.log"; }}
st '{{"phase":"boot"}}'
term() {{ python3 - <<'PY' || true
import os, json, urllib.request as u
p=os.environ.get('RUNPOD_POD_ID',''); k=os.environ.get('RUNPOD_API_KEY','')
b=json.dumps({{'query':'mutation {{ podTerminate(input: {{podId: "%s"}}) }}'%p}}).encode()
r=u.Request('https://api.runpod.io/graphql',data=b,headers={{'content-type':'application/json','Authorization':'Bearer '+k,'User-Agent':'nb-jlens/1'}},method='POST')
u.urlopen(r,timeout=30).read()
PY
}}
finish() {{ st "{{\\"phase\\":\\"$1\\",\\"detail\\":\\"$2\\"}}"; sleep {a.grace_minutes}m; term; exit 0; }}
DEMO=/workspace/jobs/narcbench-jlens; mkdir -p "$DEMO/scripts"
echo '{est_b64}' | base64 -d > "$DEMO/scripts/estimate_jlens_full.py"
PY=/workspace/interp-demo/.venv-explorer/bin/python; [ -x "$PY" ] || PY=python3
export HF_HOME=${{HF_HOME:-/workspace/interp-demo/models/hf}}; {hf}
st '{{"phase":"setup"}}'
$PY -m pip install -q 'transformers>=4.45' accelerate sentencepiece protobuf datasets 2>>"$RUN/job.log" || true
OUT=/workspace/jobs/narcbench-results/j_basis/{OUT_REL}; mkdir -p "$OUT"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader > "$RUN/gpu.txt" 2>/dev/null || true
set +e
if [ "{int(a.smoke)}" = "1" ]; then
  st '{{"phase":"smoke"}}'
  timeout 60m $PY -u "$DEMO/scripts/estimate_jlens_full.py" --model {a.model} --layers {mid_layer} \\
    --n-prompts 8 --holdout 2 --rank 32 --chunk {a.chunk} --max-len 64 --dtype {a.dtype} \\
    --prompt-mix wikitext:8,alpaca:8 --out "$OUT/smoke" --status "$RUN/status_smoke.json" >"$RUN/smoke.log" 2>&1
  SEC=$?; tail -5 "$RUN/smoke.log" >> "$RUN/job.log"
  if [ $SEC -ne 0 ]; then cp "$RUN/smoke.log" "$RUN/cmd.log"; finish failed smoke_exit_$SEC; fi
  $PY - <<'PY' > "$RUN/projection.json"
import json, math
m=json.load(open("/workspace/jobs/narcbench-results/j_basis/{OUT_REL}/smoke/jlens_meta.json"))
s=m["pass1_s"]+m["pass2_s"]; per_unit=s/(8*32*1)/({a.max_len}/64)   # seconds per (prompt x rank x layer) at max_len
cap_h={a.max_minutes}/60*0.9; n={a.n_prompts}; r={a.rank}
def hours(n,r): return per_unit*n*r*{n_layers}/3600
tried=[]
for n_,r_ in ((n,r),(n,r//2),(600,r//2),(400,r//2)):
    tried.append([n_,r_,round(hours(n_,r_),2)])
    if hours(n_,r_)<=cap_h: n,r=n_,r_; break
else:
    n,r=400,r//2
print(json.dumps({{"smoke_seconds":round(s,1),"jvp_mode":m["jvp_mode"],"batched_vjp":m["batched_vjp"],"projected_hours_requested":round(hours({a.n_prompts},{a.rank}),2),"chosen_n_prompts":n,"chosen_rank":r,"chosen_hours":round(hours(n,r),2),"tried":tried}}))
PY
  st "$(python3 -c "import json;p=json.load(open('$RUN/projection.json'));p['phase']='running';print(json.dumps(p))")"
  N_RUN=$(python3 -c "import json;print(json.load(open('$RUN/projection.json'))['chosen_n_prompts'])")
  R_RUN=$(python3 -c "import json;print(json.load(open('$RUN/projection.json'))['chosen_rank'])")
else
  st '{{"phase":"running"}}'; N_RUN={a.n_prompts}; R_RUN={a.rank}
fi
timeout {a.max_minutes}m $PY -u "$DEMO/scripts/estimate_jlens_full.py" --model {a.model} --layers {a.layers} \\
  --n-prompts $N_RUN --holdout {a.holdout} --rank $R_RUN --chunk {a.chunk} --max-len {a.max_len} \\
  --dtype {a.dtype} --prompt-mix {a.prompt_mix} --out "$OUT" --status "$RUN/status_full.json" --resume >"$RUN/cmd.log" 2>&1
EC=$?; set -e
echo $EC > "$RUN/exit_code"; tail -40 "$RUN/cmd.log" >> "$RUN/job.log"
mkdir -p /workspace/jobs/narcbench-core-demo/data/j_basis/{OUT_REL}
cp -a "$OUT"/. /workspace/jobs/narcbench-core-demo/data/j_basis/{OUT_REL}/ 2>/dev/null || true
tar -czf "$RUN/results.tgz" -C /workspace/jobs/narcbench-results/j_basis {OUT_REL} || true
if [ $EC -eq 0 ]; then finish done ok; else finish failed exit_$EC; fi
"""


def parse_layers(spec: str) -> list[int]:
    if "-" in spec and "," not in spec:
        a, b = spec.split("-", 1)
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in spec.split(",") if x.strip()]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="google/gemma-2-9b-it")
    ap.add_argument("--layers", default="19-23")
    ap.add_argument("--n-prompts", type=int, default=1000)
    ap.add_argument("--holdout", type=int, default=20)
    ap.add_argument("--rank", type=int, default=256)
    ap.add_argument("--chunk", type=int, default=32, help="tangents per batched pass; the estimator halves it on OOM")
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--prompt-mix", default="wikitext:510,alpaca:510")
    ap.add_argument("--smoke", type=int, default=1, help="1: run the timing smoke first")
    ap.add_argument("--max-minutes", type=int, default=720)
    ap.add_argument("--grace-minutes", type=int, default=45)
    ap.add_argument("--s-per-pass", type=float, default=0.6, help="assumed seconds per batched JVP or VJP pass")
    ap.add_argument("--gpu", default=None, help="force one GPU type id")
    ap.add_argument("--launch", action="store_true", help="create the pod (spends money)")
    ap.add_argument("--status", metavar="POD_ID")
    ap.add_argument("--pull", metavar="POD_ID")
    ap.add_argument("--terminate", metavar="POD_ID")
    a = ap.parse_args()

    if a.status or a.pull:
        pod = a.status or a.pull
        base = f"https://{pod}-8765.proxy.runpod.net"
        for name in ("status.json", "projection.json", "status_full.json"):
            try:
                with urllib.request.urlopen(f"{base}/{name}", timeout=30) as r:
                    print(name, r.read().decode().strip())
            except Exception as e:  # noqa: BLE001
                print(name, "unavailable:", type(e).__name__)
        if a.pull:
            dest = ROOT / "data/j_basis"
            dest.mkdir(parents=True, exist_ok=True)
            tgz = dest / "jlens_full_results.tgz"
            urllib.request.urlretrieve(f"{base}/results.tgz", tgz)
            with tarfile.open(tgz) as t:
                t.extractall(dest)
            print("extracted to", dest / OUT_REL)
        return 0
    if a.terminate:
        print(graphql('mutation { podTerminate(input: {podId: "%s"}) }' % a.terminate))
        return 0

    layers = parse_layers(a.layers)
    passes = a.n_prompts * 2 * math.ceil(a.rank / a.chunk) * len(layers)
    hours = passes * a.s_per_pass / 3600 * 1.15 + 0.5
    gpu = a.gpu or GPU_PREFS[0]
    est = {"batched_passes": passes, "assumed_s_per_pass": a.s_per_pass, "hours_est": round(hours, 1),
           "preferred_gpu": gpu, "rate_usd_hr": RATES.get(gpu), "cost_est_usd": round(hours * RATES.get(gpu, 1.0), 2),
           "settings": {k: getattr(a, k) for k in ("model", "layers", "n_prompts", "holdout", "rank", "chunk", "max_len", "dtype", "prompt_mix", "max_minutes")}}
    print(json.dumps(est, indent=2))
    boot = boot_script(a)
    boot_b64 = base64.b64encode(boot.encode()).decode()
    (ROOT / "results/jspace").mkdir(parents=True, exist_ok=True)
    (ROOT / "results/jspace/jlens_full_launch_estimate.json").write_text(json.dumps(est, indent=2) + "\n")
    if not a.launch:
        print("\n[estimate-only] re-run with --launch to spend; the smoke on the pod measures the real rate", file=sys.stderr)
        return 0
    if len(boot_b64) > 120000:
        print("boot script too large", file=sys.stderr)
        return 1
    env = {"JOB_BOOT_B64": boot_b64, "RUNPOD_API_KEY": os.environ["RUNPOD_API_KEY"]}
    if os.environ.get("HF_TOKEN"):
        env["HF_TOKEN"] = os.environ["HF_TOKEN"]
    docker_args = "bash -c 'echo $JOB_BOOT_B64 | base64 -d > /tmp/boot.sh; exec bash /tmp/boot.sh'"
    gpus = [a.gpu] if a.gpu else GPU_PREFS
    last = None
    for cloud in ("SECURE", "COMMUNITY"):
        for g in gpus:
            inp = {"name": POD_NAME, "imageName": IMAGE, "gpuTypeId": g, "cloudType": cloud, "gpuCount": 1,
                   "volumeInGb": 0, "containerDiskInGb": 40, "minVcpuCount": 4, "minMemoryInGb": 32,
                   "ports": "8765/http", "env": [{"key": k, "value": str(v)} for k, v in env.items()],
                   "dockerArgs": docker_args, "supportPublicIp": False, "startSsh": False,
                   "networkVolumeId": VOLUME_ID, "volumeMountPath": "/workspace"}
            q = """mutation ($input: PodFindAndDeployOnDemandInput!) {
                  podFindAndDeployOnDemand(input: $input) { id desiredStatus costPerHr machine { gpuDisplayName dataCenterId } } }"""
            try:
                data = graphql(q, {"input": inp})
                if data.get("errors"):
                    raise RuntimeError(str(data["errors"])[:300])
                pod = (data.get("data") or {}).get("podFindAndDeployOnDemand")
                if not pod or not pod.get("id"):
                    raise RuntimeError("empty pod")
                out = {"created": pod, "gpu": g, "cloud": cloud, "estimate": est}
                print(json.dumps(out, indent=2))
                print(f"status: https://{pod['id']}-8765.proxy.runpod.net/status.json")
                (ROOT / "results/jspace/jlens_full_pod.json").write_text(json.dumps(out, indent=2) + "\n")
                return 0
            except Exception as e:  # noqa: BLE001
                last = e
                print(f"[launch] {cloud} {g}: {e}")
    print(f"all GPU types failed: {last}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
