#!/usr/bin/env python3
"""Launch SEPARATE RunPod job to extract Gemma activations from Core transcripts.

Does NOT touch full Core pod lnqm4yawhv13on. Default is estimate-only;
pass --launch only with explicit spend approval.

  python3 scripts/runpod_launch_extract_acts.py \
    --runs-rel gemma2_9b/core/20261001T011552Z \
    --model google/gemma-2-9b-it --layers 19-23 --gen-only

  python3 scripts/runpod_launch_extract_acts.py ... --launch
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COLLUSION_SCRIPTS = Path("/workspace/collusion-exp/scripts")
SECRETS = (
    Path("/home/box/.secrets/interp-explorer.env"),
    Path("/workspace/runpod-study/interp-explorer.env"),
    Path.home() / ".secrets" / "interp-explorer.env",
)
IMAGE = "runpod/pytorch:1.2.0-rc.162-cu1281-torch280-ubuntu2404"
VOLUME_ID = "cr1kjtzjs2"
POD_NAME = "job-narcbench-extract-acts"
# Prefer GPUs that fit gemma-2-9b-it (~20GB+); avoid conflicting with A6000 Core if scarce
GPU_PREFS_9B = [
    "NVIDIA L4",
    "NVIDIA RTX PRO 4500 Blackwell Server Edition",
    "NVIDIA RTX PRO 4500 Blackwell",
    "NVIDIA A100 80GB PCIe",
    "NVIDIA GeForce RTX 4090",
    "NVIDIA A40",
    "NVIDIA RTX A5000",
    "NVIDIA GeForce RTX 3090",
    "NVIDIA RTX A6000",
    "NVIDIA L40S",
    "NVIDIA RTX 6000 Ada Generation",
    "NVIDIA RTX PRO 5000 Blackwell",
]
RATES = {
    "NVIDIA L4": 0.49,
    "NVIDIA RTX PRO 4500 Blackwell Server Edition": 0.72,
    "NVIDIA RTX PRO 4500 Blackwell": 0.72,
    "NVIDIA A100 80GB PCIe": 1.39,
    "NVIDIA GeForce RTX 4090": 0.74,
    "NVIDIA A40": 0.49,
    "NVIDIA RTX A6000": 0.53,
    "NVIDIA L40S": 0.89,
    "NVIDIA RTX A5000": 0.49,
    "NVIDIA GeForce RTX 3090": 0.50,
    "NVIDIA RTX 6000 Ada Generation": 0.89,
    "NVIDIA RTX PRO 5000 Blackwell": 0.82,
}
DO_NOT_TOUCH = "lnqm4yawhv13on"


def read_env() -> None:
    for path in SECRETS:
        if not path.is_file():
            continue
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def estimate(kind: str) -> dict:
    # smoke 6 runs × ~18 turns ≈ 108 forwards; full 100 × 18 ≈ 1800
    if kind == "smoke":
        hours, n = 0.6, 108
    else:
        hours, n = 4.0, 1800
    gpu = GPU_PREFS_9B[0]
    rate = RATES.get(gpu, 0.6)
    return {
        "kind": kind,
        "n_forwards_est": n,
        "hours_est": hours,
        "preferred_gpu": gpu,
        "rate_usd_hr": rate,
        "cost_est_usd": round(hours * rate, 2),
        "do_not_touch_pod": DO_NOT_TOUCH,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-rel", default="gemma2_9b/core/20261001T011552Z")
    ap.add_argument("--model", default="google/gemma-2-9b-it")
    ap.add_argument("--layers", default="19-23")
    ap.add_argument("--gen-only", action="store_true", default=True)
    ap.add_argument("--launch", action="store_true")
    ap.add_argument("--max-minutes", type=int, default=120)
    ap.add_argument("--grace-minutes", type=int, default=8)
    ap.add_argument("--kind", choices=["smoke", "full"], default="smoke")
    args = ap.parse_args()
    read_env()

    est = estimate(args.kind)
    script = (ROOT / "scripts/extract_activations_from_transcripts.py").read_bytes()
    script_b64 = base64.b64encode(script).decode()
    out_rel = args.runs_rel  # mirror transcript run_id under activations/
    gen_flag = "--gen-only" if args.gen_only else ""

    boot = f"""#!/bin/bash
set -uo pipefail
# HARD RULE: never interact with Core pod {DO_NOT_TOUCH}
RUN=/tmp/narcbench-extract; mkdir -p "$RUN" /workspace/jobs/narcbench-results/activations
python3 -m http.server 8765 --directory "$RUN" >/tmp/http.log 2>&1 &
echo '{{"phase":"boot","do_not_touch":"{DO_NOT_TOUCH}"}}' > "$RUN/status.json"
phase() {{ echo "{{\\"phase\\":\\"$1\\"}}" > "$RUN/status.json"; echo "[phase] $1" | tee -a "$RUN/job.log"; }}
term() {{ python3 -c "import os,json,urllib.request as u;p=os.environ.get('RUNPOD_POD_ID','');k=os.environ.get('RUNPOD_API_KEY','');
b=json.dumps({{'query':'mutation {{ podTerminate(input: {{podId: \\"%s\\"}}) }}'%p}});
r=u.Request('https://api.runpod.io/graphql',data=b.encode(),headers={{'content-type':'application/json','Authorization':'Bearer '+k,'User-Agent':'nb-extract/1'}},method='POST');
u.urlopen(r,timeout=30).read()" || true; }}
finish() {{ echo "{{\\"phase\\":\\"$1\\",\\"detail\\":\\"$2\\"}}" > "$RUN/status.json"; sleep {args.grace_minutes}m; term; exit 0; }}
phase setup
DEMO=/workspace/jobs/narcbench-extract-acts; mkdir -p "$DEMO/scripts"
echo '{script_b64}' | base64 -d > "$DEMO/scripts/extract_activations_from_transcripts.py"
PY=/workspace/interp-demo/.venv-explorer/bin/python; [ -x "$PY" ] || PY=python3
export HF_HOME=${{HF_HOME:-/workspace/interp-demo/models/hf}}
$PY -m pip install -q 'transformers>=4.40' accelerate sentencepiece protobuf 2>>"$RUN/job.log" || true
RUNS=""
for c in \\
  /workspace/jobs/narcbench-results/upstream/scenarios/{args.runs_rel} \\
  /workspace/jobs/narcbench-core-demo/upstream/scenarios/{args.runs_rel} \\
  /workspace/narcbench-core-demo/upstream/scenarios/{args.runs_rel}; do
  [ -d "$c" ] && RUNS="$c" && break
done
[ -n "$RUNS" ] || finish failed missing_runs
OUT=/workspace/jobs/narcbench-results/activations/{out_rel}
mkdir -p "$OUT"
phase running
set +e
timeout {args.max_minutes}m $PY -u "$DEMO/scripts/extract_activations_from_transcripts.py" \\
  --runs-dir "$RUNS" --out "$OUT" --model {args.model} --layers {args.layers} {gen_flag} \\
  >"$RUN/cmd.log" 2>&1
EC=$?; set -e
echo $EC > "$RUN/exit_code"; cat "$RUN/cmd.log" | tee -a "$RUN/job.log"
# Also copy into volume mirror path for box pull
mkdir -p /workspace/jobs/narcbench-core-demo/data/activations/{out_rel}
cp -a "$OUT"/. /workspace/jobs/narcbench-core-demo/data/activations/{out_rel}/ 2>/dev/null || true
tar -czf "$RUN/results.tgz" -C /workspace/jobs/narcbench-results activations || true
if [ $EC -eq 0 ]; then finish done ok; else finish failed exit_$EC; fi
"""
    boot_b64 = base64.b64encode(boot.encode()).decode()
    report = {
        "estimate": est,
        "runs_rel": args.runs_rel,
        "model": args.model,
        "layers": args.layers,
        "boot_b64_len": len(boot_b64),
        "script_b64_len": len(script_b64),
        "will_launch": args.launch,
        "do_not_touch": DO_NOT_TOUCH,
        "next_after_pull": (
            f"python3 scripts/train_collusion_probe.py "
            f"--acts-dir data/activations/{out_rel} "
            f"--out results/stage1/{args.kind} --layers {args.layers}"
        ),
    }
    print(json.dumps(report, indent=2))
    (ROOT / "results/stage1/extract_launch_estimate.json").write_text(json.dumps(report, indent=2) + "\n")

    if not args.launch:
        print("\n[estimate-only] re-run with --launch after explicit spend approval", file=sys.stderr)
        return 0
    if len(boot_b64) > 90000:
        print("boot too large", file=sys.stderr)
        return 1
    if not os.environ.get("RUNPOD_API_KEY"):
        print("RUNPOD_API_KEY missing", file=sys.stderr)
        return 1

    sys.path.insert(0, str(COLLUSION_SCRIPTS))
    from runpod_graphql import graphql  # type: ignore

    env = {"JOB_BOOT_B64": boot_b64, "RUNPOD_API_KEY": os.environ["RUNPOD_API_KEY"]}
    if os.environ.get("HF_TOKEN"):
        env["HF_TOKEN"] = os.environ["HF_TOKEN"]
    docker_args = "bash -c 'echo $JOB_BOOT_B64 | base64 -d > /tmp/boot.sh; exec bash /tmp/boot.sh'"
    last = None
    for cloud in ("SECURE", "COMMUNITY"):
        for gpu in GPU_PREFS_9B:
            inp = {
                "name": POD_NAME,
                "imageName": IMAGE,
                "gpuTypeId": gpu,
                "cloudType": cloud,
                "gpuCount": 1,
                "volumeInGb": 0,
                "containerDiskInGb": 40,
                "minVcpuCount": 4,
                "minMemoryInGb": 32,
                "ports": "8765/http",
                "env": [{"key": k, "value": str(v)} for k, v in env.items()],
                "dockerArgs": docker_args,
                "supportPublicIp": False,
                "startSsh": False,
                "networkVolumeId": VOLUME_ID,
                "volumeMountPath": "/workspace",
            }
            q = """mutation ($input: PodFindAndDeployOnDemandInput!) {
              podFindAndDeployOnDemand(input: $input) {
                id desiredStatus costPerHr machine { gpuDisplayName dataCenterId }
              }
            }"""
            try:
                data = graphql(q, {"input": inp}, timeout=120)
                if data.get("errors"):
                    raise RuntimeError(str(data["errors"])[:300])
                pod = (data.get("data") or {}).get("podFindAndDeployOnDemand")
                if not pod or not pod.get("id"):
                    raise RuntimeError("empty pod")
                if pod["id"] == DO_NOT_TOUCH:
                    raise RuntimeError("refusing to reuse Core pod id")
                out = {"created": pod, "gpu": gpu, "cloud": cloud, "do_not_touch": DO_NOT_TOUCH}
                print(json.dumps(out, indent=2))
                print(f"Status: https://{pod['id']}-8765.proxy.runpod.net/status.json")
                Path("/tmp/narcbench_extract_pod.json").write_text(json.dumps(out, indent=2))
                (ROOT / "results/stage1/extract_pod.json").write_text(json.dumps(out, indent=2) + "\n")
                return 0
            except Exception as e:
                last = e
                print(f"[launch] {cloud} {gpu}: {e}")
    print(f"all failed: {last}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
