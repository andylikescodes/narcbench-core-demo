#!/usr/bin/env python3
"""Estimate + optionally launch disposable RunPod job for Gemma Core regen.

Default: print cost estimate and balance; do NOT spend.
Pass --launch to create a pod (requires explicit approval from Andy).

Smoke (3×2): ~108 generative turns on gemma-2-2b-it → fits RTX 2000 Ada (~$0.24/h).
Full 50×2: ~1800 turns → budget hours, not minutes.
9B needs ≥24–32 GB VRAM (--gpu big).
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
import secrets
import sys
import tarfile
import time
import urllib.request
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
POD_NAME = "job-narcbench-gemma-core"
GPU_PREFS_2B = [
    "NVIDIA A40",  # Medium stock in EU-RO-1 (volume DC)
    "NVIDIA RTX A4000",
    "NVIDIA RTX 4000 Ada Generation",
    "NVIDIA GeForce RTX 3090",
    "NVIDIA GeForce RTX 4090",
    "NVIDIA RTX 2000 Ada Generation",
    "NVIDIA L4",
    "NVIDIA RTX A6000",
]
GPU_PREFS_9B = [
    "NVIDIA RTX A6000",
    "NVIDIA A40",
    "NVIDIA L40S",
    "NVIDIA GeForce RTX 4090",
    "NVIDIA RTX A5000",
]
RATES = {
    "NVIDIA RTX 2000 Ada Generation": 0.24,
    "NVIDIA RTX 4000 Ada Generation": 0.28,
    "NVIDIA RTX A4000": 0.25,
    "NVIDIA L4": 0.49,
    "NVIDIA GeForce RTX 3090": 0.50,
    "NVIDIA A40": 0.49,
    "NVIDIA GeForce RTX 4090": 0.74,
    "NVIDIA RTX A5000": 0.49,
    "NVIDIA RTX A6000": 0.79,
    "NVIDIA L40S": 0.89,
}
UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"


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


def graphql(query: str, variables: dict | None = None) -> dict:
    body: dict = {"query": query}
    if variables is not None:
        body["variables"] = variables
    req = urllib.request.Request(
        "https://api.runpod.io/graphql",
        data=json.dumps(body).encode(),
        headers={
            "content-type": "application/json",
            "Authorization": "Bearer " + os.environ["RUNPOD_API_KEY"],
            "User-Agent": UA,
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode())


def balance() -> dict:
    d = graphql("{ myself { clientBalance spendLimit currentSpendPerHr } }")
    return (d.get("data") or {}).get("myself") or {}


def estimate(kind: str, model: str) -> dict:
    # ~18 calls/mode (control skips 2 private → 16); use 18 as upper bound
    if kind == "smoke":
        n_scen, modes = 3, 2
        hours_2b, hours_9b = 0.35, 1.2
    else:
        n_scen, modes = 50, 2
        hours_2b, hours_9b = 4.0, 14.0
    n_calls = n_scen * modes * 18
    is_9b = "9b" in model
    hours = hours_9b if is_9b else hours_2b
    gpu = GPU_PREFS_9B[0] if is_9b else GPU_PREFS_2B[0]
    rate = RATES.get(gpu, 0.40)
    return {
        "kind": kind,
        "model": model,
        "n_scenarios": n_scen,
        "n_modes": modes,
        "n_calls_upper": n_calls,
        "preferred_gpu": gpu,
        "approx_rate_usd_per_hr": rate,
        "approx_hours": hours,
        "approx_cost_usd": round(hours * rate, 2),
        "note": "Wall-clock depends on cache hits + generation length; estimates are conservative.",
    }


def make_tar_bytes() -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for rel in [
            "generation",
            "scripts/regen_gemma_core.py",
            "scripts/build_index.py",
            "README.md",
        ]:
            path = ROOT / rel
            if path.is_dir():
                for fp in path.rglob("*"):
                    if fp.is_file() and "__pycache__" not in fp.parts:
                        tar.add(fp, arcname=str(fp.relative_to(ROOT)))
            elif path.is_file():
                tar.add(path, arcname=rel)
    return buf.getvalue()


def make_tar_b64() -> str:
    return base64.b64encode(make_tar_bytes()).decode("ascii")


def chunk_b64(data: bytes, size: int = 12000) -> list[str]:
    b64 = base64.b64encode(data).decode("ascii")
    return [b64[i : i + size] for i in range(0, len(b64), size)]


def build_boot(secret: str, max_min: int, grace: int, n_chunks: int, model: str, smoke: bool) -> str:
    mode_flag = "--smoke" if smoke else "--all"
    return f"""#!/bin/bash
set -uo pipefail
RUN=/tmp/narcbench-job
mkdir -p "$RUN" /workspace/jobs/narcbench-core-demo /workspace/jobs/narcbench-results
python3 -m http.server 8765 --directory "$RUN" >/tmp/http.log 2>&1 &
echo '{{"phase":"boot"}}' > "$RUN/status.json"
phase() {{ echo "{{\\"phase\\":\\"$1\\"}}" > "$RUN/status.json"; echo "[phase] $1" | tee -a "$RUN/job.log"; }}
term() {{ python3 -c "import os,json,urllib.request as u;p=os.environ.get('RUNPOD_POD_ID','');k=os.environ.get('RUNPOD_API_KEY','');
b=json.dumps({{'query':'mutation {{ podTerminate(input: {{podId: \\"%s\\"}}) }}'%p}});
r=u.Request('https://api.runpod.io/graphql',data=b.encode(),headers={{'content-type':'application/json','Authorization':'Bearer '+k,'User-Agent':'nb/1'}},method='POST');
u.urlopen(r,timeout=30).read()" || true; }}
finish() {{ echo "{{\\"phase\\":\\"$1\\",\\"detail\\":\\"$2\\"}}" > "$RUN/status.json"; sleep {grace}m; term; exit 0; }}
phase assemble
python3 - <<'PY' >>"$RUN/job.log" 2>&1 || finish failed assemble
import os, base64, pathlib, sys
keys = sorted(k for k in os.environ if k.startswith("JOB_TAR_"))
print("tar_keys", keys)
n = int(os.environ.get("JOB_TAR_N", "0"))
parts = []
for i in range(n):
    k = f"JOB_TAR_B64_{{i:02d}}"
    v = os.environ.get(k)
    if not v:
        print("MISSING", k)
        sys.exit(2)
    parts.append(v)
raw = base64.b64decode("".join(parts))
path = pathlib.Path("/tmp/demo.tgz")
path.write_bytes(raw)
print("wrote", path, "bytes", len(raw), "magic", raw[:2])
if raw[:2] != b"\\x1f\\x8b":
    print("NOT_GZIP")
    sys.exit(3)
PY
phase unpack
DEMO=/workspace/jobs/narcbench-core-demo
tar --no-same-owner -xzf /tmp/demo.tgz -C "$DEMO" >>"$RUN/job.log" 2>&1 || finish failed untar
ls -la "$DEMO" >>"$RUN/job.log" 2>&1 || true
phase deps
PYBIN=python3
if [ -x /workspace/interp-demo/.venv-explorer/bin/python ]; then PYBIN=/workspace/interp-demo/.venv-explorer/bin/python; fi
export HF_HOME=${{HF_HOME:-/workspace/interp-demo/models/hf}}
mkdir -p "$HF_HOME/hub"
$PYBIN -m pip install -q 'transformers>=4.40' accelerate sentencepiece protobuf 2>>"$RUN/job.log" || true
phase running
cd "$DEMO" || finish failed cd
export PYTHONPATH="$DEMO:${{PYTHONPATH:-}}"
set +e
timeout {max_min}m $PYBIN -u scripts/regen_gemma_core.py {mode_flag} --model {model} --temperature 0.7 --max-new-tokens 256 >"$RUN/cmd.log" 2>&1
EC=$?
set -e
echo $EC > "$RUN/exit_code"
cat "$RUN/cmd.log" | tee -a "$RUN/job.log"
tar -czf "$RUN/results.tgz" -C "$DEMO" upstream/scenarios || true
cp -f "$RUN/results.tgz" /workspace/jobs/narcbench-results/last_results.tgz 2>/dev/null || true
cp -a "$DEMO/upstream" /workspace/jobs/narcbench-results/upstream 2>/dev/null || true
if [ $EC -eq 0 ]; then finish done ok; else finish failed exit_$EC; fi
"""



def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", action="store_true", default=True)
    ap.add_argument("--full", action="store_true", help="50×2 instead of smoke")
    ap.add_argument("--model", default="google/gemma-2-2b-it")
    ap.add_argument("--launch", action="store_true", help="Actually create a pod (spends $)")
    ap.add_argument("--max-minutes", type=int, default=60)
    ap.add_argument("--grace-minutes", type=int, default=5)
    ap.add_argument("--estimate-only", action="store_true", default=False)
    args = ap.parse_args()
    kind = "full" if args.full else "smoke"

    read_env()
    est = estimate(kind, args.model)
    bal = {}
    if os.environ.get("RUNPOD_API_KEY"):
        try:
            bal = balance()
        except Exception as e:
            bal = {"error": str(e)}
    report = {"estimate": est, "balance": bal, "will_launch": bool(args.launch)}
    print(json.dumps(report, indent=2))

    if not args.launch:
        print(
            "\nNo pod created. Re-run with --launch after Andy approves spend "
            f"(~${est['approx_cost_usd']} for {kind} on {args.model}).",
            file=sys.stderr,
        )
        return 0

    if not os.environ.get("RUNPOD_API_KEY"):
        sys.exit("RUNPOD_API_KEY missing")

    # Prefer browser-UA GraphQL (collusion-exp); SDK often hits CF 403.
    sys.path.insert(0, str(COLLUSION_SCRIPTS))
    from runpod_graphql import graphql  # type: ignore

    def create_pod_full(
        *,
        name,
        image_name,
        gpu_type_id,
        cloud_type,
        container_disk_in_gb,
        ports,
        env,
        docker_args,
        network_volume_id=None,
        volume_mount_path="/workspace",
    ):
        inp = {
            "name": name,
            "imageName": image_name,
            "gpuTypeId": gpu_type_id,
            "cloudType": cloud_type,
            "gpuCount": 1,
            "volumeInGb": 0,
            "containerDiskInGb": container_disk_in_gb,
            "minVcpuCount": 4,
            "minMemoryInGb": 16,
            "ports": ports,
            "env": [{"key": k, "value": str(v)} for k, v in env.items()],
            "dockerArgs": docker_args,
            "supportPublicIp": False,
            "startSsh": False,
        }
        if network_volume_id:
            inp["networkVolumeId"] = network_volume_id
            inp["volumeMountPath"] = volume_mount_path
        q = """mutation ($input: PodFindAndDeployOnDemandInput!) {
          podFindAndDeployOnDemand(input: $input) {
            id desiredStatus costPerHr machine { gpuDisplayName dataCenterId }
          }
        }"""
        data = graphql(q, {"input": inp}, timeout=120)
        if data.get("errors"):
            raise RuntimeError(str(data["errors"])[:400])
        pod = (data.get("data") or {}).get("podFindAndDeployOnDemand")
        if not pod or not pod.get("id"):
            raise RuntimeError("empty create: " + json.dumps(data)[:400])
        return pod

    secret = secrets.token_hex(12)
    tar_bytes = make_tar_bytes()
    chunks = chunk_b64(tar_bytes, size=12000)
    boot = build_boot(
        secret,
        args.max_minutes,
        args.grace_minutes,
        len(chunks),
        args.model,
        smoke=(kind == "smoke"),
    )
    boot_b64 = base64.b64encode(boot.encode()).decode("ascii")
    print(
        f"[launch] boot b64 chars={len(boot_b64)} tar_bytes={len(tar_bytes)} chunks={len(chunks)}"
    )
    if len(boot_b64) > 40000:
        sys.exit(f"boot still too large: {len(boot_b64)}")

    gpu_list = GPU_PREFS_9B if "9b" in args.model else GPU_PREFS_2B
    env = {
        "JOB_BOOT_B64": boot_b64,
        "RUNPOD_API_KEY": os.environ["RUNPOD_API_KEY"],
        "JOB_TAR_N": str(len(chunks)),
    }
    for i, ch in enumerate(chunks):
        env[f"JOB_TAR_B64_{i:02d}"] = ch
    if os.environ.get("HF_TOKEN"):
        env["HF_TOKEN"] = os.environ["HF_TOKEN"]

    docker_args = "bash -c 'echo $JOB_BOOT_B64 | base64 -d > /tmp/boot.sh; exec bash /tmp/boot.sh'"

    last_err = None
    attempts = []
    for cloud in ("SECURE", "COMMUNITY"):
        for gpu in gpu_list:
            attempts.append((cloud, gpu, VOLUME_ID))
    for cloud in ("SECURE", "COMMUNITY"):
        for gpu in gpu_list:
            attempts.append((cloud, gpu, None))

    for cloud, gpu, vol in attempts:
        try:
            pod = create_pod_full(
                name=POD_NAME,
                image_name=IMAGE,
                gpu_type_id=gpu,
                cloud_type=cloud,
                container_disk_in_gb=50,
                ports="8765/http",
                network_volume_id=vol,
                volume_mount_path="/workspace",
                env=env,
                docker_args=docker_args,
            )
            print(
                json.dumps(
                    {
                        "created": pod,
                        "gpu": gpu,
                        "cloud": cloud,
                        "volume": vol,
                        "secret": secret,
                    },
                    indent=2,
                )[:2000]
            )
            pid = pod.get("id")
            if pid:
                print(f"Status: https://{pid}-8765.proxy.runpod.net/status.json")
                print(f"Log: https://{pid}-8765.proxy.runpod.net/job.log")
                print(f"Results: https://{pid}-8765.proxy.runpod.net/results.tgz")
                Path("/tmp/narcbench_last_pod.json").write_text(
                    json.dumps(
                        {
                            "id": pid,
                            "gpu": gpu,
                            "cloud": cloud,
                            "volume": vol,
                            "secret": secret,
                            "model": args.model,
                            "kind": kind,
                            "costPerHr": pod.get("costPerHr"),
                        },
                        indent=2,
                    )
                )
            return 0
        except Exception as e:
            last_err = e
            print(f"[launch] {cloud} {gpu} vol={bool(vol)} failed: {e}")
    sys.exit(f"All GPUs failed: {last_err}")

if __name__ == "__main__":
    raise SystemExit(main())
