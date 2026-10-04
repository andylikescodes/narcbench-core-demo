#!/usr/bin/env python3
"""Estimate + optionally launch RunPod job for true per-layer Jacobian J-lens.

Default estimate-only. Pass --launch to spend (Andy authorized keep-going).

Thrifty defaults: layers 19,21,23 · n_prompts=16 · rank=128
  → ~16*(128 jvp + 128 vjp)*3 ≈ 12k half-model passes; est 1.5–3 h on 4090/A40.
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
)
IMAGE = "runpod/pytorch:1.2.0-rc.162-cu1281-torch280-ubuntu2404"
VOLUME_ID = "cr1kjtzjs2"
POD_NAME = "job-narcbench-jacobian"
GPU_PREFS = [
    "NVIDIA GeForce RTX 4090",
    "NVIDIA A40",
    "NVIDIA RTX A6000",
    "NVIDIA L40S",
    "NVIDIA RTX A5000",
    "NVIDIA GeForce RTX 3090",
    "NVIDIA RTX 4000 Ada Generation",
    "NVIDIA RTX A4000",
    "NVIDIA L4",
    "NVIDIA RTX 2000 Ada Generation",
    "NVIDIA RTX 6000 Ada Generation",
    "NVIDIA A100 80GB PCIe",
    "NVIDIA A100-SXM4-80GB",
    "NVIDIA H100 PCIe",
]
RATES = {
    "NVIDIA GeForce RTX 4090": 0.74,
    "NVIDIA A40": 0.49,
    "NVIDIA RTX A6000": 0.79,
    "NVIDIA L40S": 0.89,
    "NVIDIA RTX A5000": 0.49,
    "NVIDIA GeForce RTX 3090": 0.50,
    "NVIDIA RTX 4000 Ada Generation": 0.28,
    "NVIDIA RTX A4000": 0.25,
    "NVIDIA L4": 0.49,
    "NVIDIA RTX 2000 Ada Generation": 0.24,
    "NVIDIA RTX 6000 Ada Generation": 0.89,
    "NVIDIA A100 80GB PCIe": 1.39,
    "NVIDIA A100-SXM4-80GB": 1.64,
    "NVIDIA H100 PCIe": 2.49,
}


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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/gemma-2-9b-it")
    ap.add_argument("--layers", default="19,21,23")
    ap.add_argument("--n-prompts", type=int, default=16)
    ap.add_argument("--rank", type=int, default=128)
    ap.add_argument("--max-len", type=int, default=64)
    ap.add_argument("--max-minutes", type=int, default=180)
    ap.add_argument("--grace-minutes", type=int, default=6)
    ap.add_argument("--launch", action="store_true")
    args = ap.parse_args()
    read_env()

    n_layers = len([x for x in args.layers.split(",") if x.strip()])
    # rough: each jvp/vjp ~0.05–0.15s on 9B half-depth; use 0.08s
    n_passes = args.n_prompts * args.rank * 2 * n_layers
    hours = max(0.75, n_passes * 0.08 / 3600.0) * 1.4  # +overhead
    gpu = GPU_PREFS[0]
    rate = RATES[gpu]
    est = {
        "n_passes_est": n_passes,
        "hours_est": round(hours, 2),
        "preferred_gpu": gpu,
        "rate_usd_hr": rate,
        "cost_est_usd": round(hours * rate, 2),
        "layers": args.layers,
        "n_prompts": args.n_prompts,
        "rank": args.rank,
        "note": "Terminate-on-finish baked into boot; pull J_*.npy from volume/results.tgz",
    }
    script = (ROOT / "scripts/estimate_jacobian_jlens.py").read_bytes()
    script_b64 = base64.b64encode(script).decode()
    out_rel = "gemma2_9b/jacobians"

    boot = f"""#!/bin/bash
set -uo pipefail
RUN=/tmp/narcbench-jac; mkdir -p "$RUN" /workspace/jobs/narcbench-results/j_basis
python3 -m http.server 8765 --directory "$RUN" >/tmp/http.log 2>&1 &
echo '{{"phase":"boot"}}' > "$RUN/status.json"
phase() {{ echo "{{\\"phase\\":\\"$1\\"}}" > "$RUN/status.json"; echo "[phase] $1" | tee -a "$RUN/job.log"; }}
term() {{ python3 -c "import os,json,urllib.request as u;p=os.environ.get('RUNPOD_POD_ID','');k=os.environ.get('RUNPOD_API_KEY','');
b=json.dumps({{'query':'mutation {{ podTerminate(input: {{podId: \\"%s\\"}}) }}'%p}});
r=u.Request('https://api.runpod.io/graphql',data=b.encode(),headers={{'content-type':'application/json','Authorization':'Bearer '+k,'User-Agent':'nb-jac/1'}},method='POST');
u.urlopen(r,timeout=30).read()" || true; }}
finish() {{ echo "{{\\"phase\\":\\"$1\\",\\"detail\\":\\"$2\\"}}" > "$RUN/status.json"; sleep {args.grace_minutes}m; term; exit 0; }}
phase setup
DEMO=/workspace/jobs/narcbench-jacobian; mkdir -p "$DEMO/scripts"
echo '{script_b64}' | base64 -d > "$DEMO/scripts/estimate_jacobian_jlens.py"
PY=/workspace/interp-demo/.venv-explorer/bin/python; [ -x "$PY" ] || PY=python3
export HF_HOME=${{HF_HOME:-/workspace/interp-demo/models/hf}}
$PY -m pip install -q 'transformers>=4.40' accelerate sentencepiece protobuf 2>>"$RUN/job.log" || true
OUT=/workspace/jobs/narcbench-results/j_basis/{out_rel}
mkdir -p "$OUT"
phase running
set +e
timeout {args.max_minutes}m $PY -u "$DEMO/scripts/estimate_jacobian_jlens.py" \\
  --model {args.model} --layers {args.layers} --n-prompts {args.n_prompts} \\
  --rank {args.rank} --max-len {args.max_len} --out "$OUT" \\
  >"$RUN/cmd.log" 2>&1
EC=$?; set -e
echo $EC > "$RUN/exit_code"; cat "$RUN/cmd.log" | tee -a "$RUN/job.log"
mkdir -p /workspace/jobs/narcbench-core-demo/data/j_basis/{out_rel}
cp -a "$OUT"/. /workspace/jobs/narcbench-core-demo/data/j_basis/{out_rel}/ 2>/dev/null || true
tar -czf "$RUN/results.tgz" -C /workspace/jobs/narcbench-results j_basis || true
if [ $EC -eq 0 ]; then finish done ok; else finish failed exit_$EC; fi
"""
    boot_b64 = base64.b64encode(boot.encode()).decode()
    report = {"estimate": est, "boot_b64_len": len(boot_b64), "will_launch": args.launch}
    print(json.dumps(report, indent=2))
    (ROOT / "results/stage2_jacobian/launch_estimate.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    if not args.launch:
        print("\n[estimate-only] re-run with --launch to spend", file=sys.stderr)
        return 0
    if len(boot_b64) > 90000:
        print("boot too large", file=sys.stderr)
        return 1

    sys.path.insert(0, str(COLLUSION_SCRIPTS))
    from runpod_graphql import graphql  # type: ignore

    env = {"JOB_BOOT_B64": boot_b64, "RUNPOD_API_KEY": os.environ["RUNPOD_API_KEY"]}
    if os.environ.get("HF_TOKEN"):
        env["HF_TOKEN"] = os.environ["HF_TOKEN"]
    docker_args = "bash -c 'echo $JOB_BOOT_B64 | base64 -d > /tmp/boot.sh; exec bash /tmp/boot.sh'"
    last = None
    for cloud in ("SECURE", "COMMUNITY"):
        for gpu in GPU_PREFS:
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
                out = {"created": pod, "gpu": gpu, "cloud": cloud, "estimate": est}
                print(json.dumps(out, indent=2))
                print(f"Status: https://{pod['id']}-8765.proxy.runpod.net/status.json")
                (ROOT / "results/stage2_jacobian/jac_pod.json").write_text(
                    json.dumps(out, indent=2) + "\n"
                )
                return 0
            except Exception as e:
                last = e
                print(f"[launch] {cloud} {gpu}: {e}")
    print(f"all failed: {last}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
