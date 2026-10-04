#!/usr/bin/env python3
"""Launch RunPod job to regen NARCBench-Transfer on Gemma-2-9B.

Default estimate-only. --launch spends (Andy keep-going authorized).
Smoke (12 runs) first; --full for 72.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NARC = Path("/workspace/narcbench-upstream/repo")
COLLUSION_SCRIPTS = Path("/workspace/collusion-exp/scripts")
SECRETS = (
    Path("/home/box/.secrets/interp-explorer.env"),
    Path("/workspace/runpod-study/interp-explorer.env"),
)
IMAGE = "runpod/pytorch:1.2.0-rc.162-cu1281-torch280-ubuntu2404"
VOLUME_ID = "cr1kjtzjs2"
POD_NAME = "job-narcbench-gemma-transfer"
GPU_PREFS = [
    "NVIDIA RTX PRO 4500 Blackwell Server Edition",
    "NVIDIA RTX PRO 4500 Blackwell",
    "NVIDIA L4",
    "NVIDIA RTX 4000 Ada Generation",
    "NVIDIA GeForce RTX 3090",
    "NVIDIA GeForce RTX 4090",
    "NVIDIA RTX 5000 Ada Generation",
    "NVIDIA RTX PRO 4000 Blackwell",
    "NVIDIA A100 80GB PCIe",
    "NVIDIA A100-SXM4-80GB",
    "NVIDIA L40S",
    "NVIDIA RTX 6000 Ada Generation",
    "NVIDIA RTX A4500",
    "NVIDIA H100 80GB HBM3",
]
RATES = {
    "NVIDIA H100 80GB HBM3": 2.69,
    "NVIDIA RTX A4500": 0.19,
    "NVIDIA RTX PRO 4000 Blackwell": 0.50,
    "NVIDIA RTX 5000 Ada Generation": 0.49,
    "NVIDIA RTX PRO 4500 Blackwell Server Edition": 0.72,
    "NVIDIA RTX PRO 4500 Blackwell": 0.72,
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


def make_tar() -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for rel in [
            "generation/hf_backend.py",
            "generation/__init__.py",
            "scripts/regen_gemma_transfer.py",
        ]:
            fp = ROOT / rel
            tar.add(fp, arcname=rel)
        # upstream transfer + config
        tar.add(NARC / "generation" / "transfer.py", arcname="narcbench_upstream/generation/transfer.py")
        tar.add(NARC / "config.py", arcname="narcbench_upstream/config.py")
        # empty init
    return buf.getvalue()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/gemma-2-9b-it")
    ap.add_argument("--smoke", action="store_true", default=True)
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--launch", action="store_true")
    ap.add_argument("--max-minutes", type=int, default=240)
    ap.add_argument("--grace-minutes", type=int, default=8)
    args = ap.parse_args()
    if args.full:
        args.smoke = False
    read_env()

    n_runs = 72 if args.full else 12
    hours = 14.0 if args.full else 2.5
    gpu = GPU_PREFS[0]
    rate = RATES[gpu]
    est = {
        "kind": "full" if args.full else "smoke",
        "n_runs": n_runs,
        "n_calls_upper": n_runs * 18,
        "hours_est": hours,
        "preferred_gpu": gpu,
        "rate_usd_hr": rate,
        "cost_est_usd": round(hours * rate, 2),
    }
    raw = make_tar()
    b64 = base64.b64encode(raw).decode()
    # chunk into env vars if needed
    chunks = [b64[i : i + 12000] for i in range(0, len(b64), 12000)]
    mode_flag = "--full" if args.full else "--smoke"
    chunk_exports = "\n".join(
        f"export JOB_TAR_B64_{i:02d}='{c}'" for i, c in enumerate(chunks)
    )
    # Actually env size limits — use boot that reads from JOB_TAR_* set by API
    boot = f"""#!/bin/bash
set -uo pipefail
RUN=/tmp/narcbench-xfer; mkdir -p "$RUN" /workspace/jobs/narcbench-results/upstream/scenarios
python3 -m http.server 8765 --directory "$RUN" >/tmp/http.log 2>&1 &
echo '{{"phase":"boot"}}' > "$RUN/status.json"
phase() {{ echo "{{\\"phase\\":\\"$1\\"}}" > "$RUN/status.json"; echo "[phase] $1" | tee -a "$RUN/job.log"; }}
term() {{ python3 -c "import os,json,urllib.request as u;p=os.environ.get('RUNPOD_POD_ID','');k=os.environ.get('RUNPOD_API_KEY','');
b=json.dumps({{'query':'mutation {{ podTerminate(input: {{podId: \\"%s\\"}}) }}'%p}});
r=u.Request('https://api.runpod.io/graphql',data=b.encode(),headers={{'content-type':'application/json','Authorization':'Bearer '+k,'User-Agent':'nb-xfer/1'}},method='POST');
u.urlopen(r,timeout=30).read()" || true; }}
finish() {{ echo "{{\\"phase\\":\\"$1\\",\\"detail\\":\\"$2\\"}}" > "$RUN/status.json"; sleep {args.grace_minutes}m; term; exit 0; }}
phase assemble
python3 - <<'PY' >>"$RUN/job.log" 2>&1 || finish failed assemble
import os, base64, pathlib, sys
n = int(os.environ.get("JOB_TAR_N", "0"))
parts = []
for i in range(n):
    v = os.environ.get(f"JOB_TAR_B64_{{i:02d}}")
    if not v:
        print("MISSING", i); sys.exit(2)
    parts.append(v)
raw = base64.b64decode("".join(parts))
path = pathlib.Path("/tmp/xfer.tgz"); path.write_bytes(raw)
print("wrote", path, len(raw))
PY
phase unpack
DEMO=/workspace/jobs/narcbench-transfer; mkdir -p "$DEMO"
tar --no-same-owner -xzf /tmp/xfer.tgz -C "$DEMO" >>"$RUN/job.log" 2>&1 || finish failed untar
# layout expected by regen script
mkdir -p /workspace/narcbench-upstream/repo/generation
cp -f "$DEMO/narcbench_upstream/generation/transfer.py" /workspace/narcbench-upstream/repo/generation/transfer.py
cp -f "$DEMO/narcbench_upstream/config.py" /workspace/narcbench-upstream/repo/config.py
mkdir -p /workspace/narcbench-core-demo/generation /workspace/narcbench-core-demo/scripts
cp -f "$DEMO/generation/"*.py /workspace/narcbench-core-demo/generation/
cp -f "$DEMO/scripts/regen_gemma_transfer.py" /workspace/narcbench-core-demo/scripts/
# patch NARC path in regen to /workspace/narcbench-upstream/repo
sed -i 's|/workspace/narcbench-upstream/repo|/workspace/narcbench-upstream/repo|' /workspace/narcbench-core-demo/scripts/regen_gemma_transfer.py
phase deps
PY=/workspace/interp-demo/.venv-explorer/bin/python; [ -x "$PY" ] || PY=python3
export HF_HOME=${{HF_HOME:-/workspace/interp-demo/models/hf}}
$PY -m pip install -q 'transformers>=4.40' accelerate sentencepiece protobuf 2>>"$RUN/job.log" || true
OUT=/workspace/jobs/narcbench-results/upstream/scenarios/gemma2_9b/transfer
mkdir -p "$OUT"
phase running
set +e
timeout {args.max_minutes}m $PY -u /workspace/narcbench-core-demo/scripts/regen_gemma_transfer.py \\
  --model {args.model} {mode_flag} --out-root "$OUT/RUNPOD" \\
  >"$RUN/cmd.log" 2>&1
EC=$?; set -e
echo $EC > "$RUN/exit_code"; cat "$RUN/cmd.log" | tee -a "$RUN/job.log"
tar -czf "$RUN/results.tgz" -C /workspace/jobs/narcbench-results upstream || true
if [ $EC -eq 0 ]; then finish done ok; else finish failed exit_$EC; fi
"""
    boot_b64 = base64.b64encode(boot.encode()).decode()
    report = {
        "estimate": est,
        "n_tar_chunks": len(chunks),
        "tar_bytes": len(raw),
        "boot_b64_len": len(boot_b64),
        "will_launch": args.launch,
    }
    print(json.dumps(report, indent=2))
    (ROOT / "results/transfer/launch_estimate.json").write_text(json.dumps(report, indent=2) + "\n")
    if not args.launch:
        print("\n[estimate-only]", file=sys.stderr)
        return 0
    if len(boot_b64) > 90000:
        print("boot too large", len(boot_b64), file=sys.stderr)
        return 1

    sys.path.insert(0, str(COLLUSION_SCRIPTS))
    from runpod_graphql import graphql  # type: ignore

    env = {
        "JOB_BOOT_B64": boot_b64,
        "JOB_TAR_N": str(len(chunks)),
        "RUNPOD_API_KEY": os.environ["RUNPOD_API_KEY"],
    }
    for i, c in enumerate(chunks):
        env[f"JOB_TAR_B64_{i:02d}"] = c
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
                    raise RuntimeError(str(data["errors"])[:400])
                pod = (data.get("data") or {}).get("podFindAndDeployOnDemand")
                if not pod or not pod.get("id"):
                    raise RuntimeError("empty pod")
                out = {"created": pod, "gpu": gpu, "cloud": cloud, "estimate": est}
                print(json.dumps(out, indent=2))
                print(f"Status: https://{pod['id']}-8765.proxy.runpod.net/status.json")
                (ROOT / "results/transfer/xfer_pod.json").write_text(json.dumps(out, indent=2) + "\n")
                return 0
            except Exception as e:
                last = e
                print(f"[launch] {cloud} {gpu}: {e}")
    print(f"all failed: {last}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
