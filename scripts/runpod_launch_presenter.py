#!/usr/bin/env python3
"""Host NARCBench presenter on RunPod :8765 (safety-capped hours)."""
from __future__ import annotations
import argparse, base64, json, os, sys
from pathlib import Path

COLLUSION_SCRIPTS = Path("/workspace/collusion-exp/scripts")
SECRETS = Path("/home/box/.secrets/interp-explorer.env")
IMAGE = "runpod/pytorch:1.2.0-rc.162-cu1281-torch280-ubuntu2404"
VOLUME_ID = "cr1kjtzjs2"
POD_NAME = "job-narcbench-presenter"
GPU_PREFS = [
    "NVIDIA RTX 4000 Ada Generation",
    "NVIDIA RTX A4000",
    "NVIDIA RTX 2000 Ada Generation",
    "NVIDIA A40",
    "NVIDIA GeForce RTX 3090",
    "NVIDIA GeForce RTX 4090",
]

def read_env() -> None:
    if SECRETS.is_file():
        for line in SECRETS.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

def chunk_b64(data: bytes, size: int = 12000) -> list[str]:
    b64 = base64.b64encode(data).decode()
    return [b64[i : i + size] for i in range(0, len(b64), size)]

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--launch", action="store_true")
    ap.add_argument("--max-hours", type=float, default=3.0)
    ap.add_argument("--tar", default="/tmp/nb_presenter.tgz")
    args = ap.parse_args()
    read_env()
    raw = Path(args.tar).read_bytes()
    chunks = chunk_b64(raw)
    max_sec = int(args.max_hours * 3600)
    boot = f"""#!/bin/bash
set -uo pipefail
RUN=/tmp/nbpres; mkdir -p "$RUN" /workspace/jobs/narcbench-presenter
python3 -m http.server 8765 --directory "$RUN" >/tmp/http.log 2>&1 &
echo '{{"phase":"boot"}}' > "$RUN/status.json"
echo boot > "$RUN/index.html"
phase() {{ echo "{{\\"phase\\":\\"$1\\"}}" > "$RUN/status.json"; echo "[phase] $1" | tee -a "$RUN/job.log"; }}
term() {{ python3 -c "import os,json,urllib.request as u;p=os.environ.get('RUNPOD_POD_ID','');k=os.environ.get('RUNPOD_API_KEY','');b=json.dumps({{'query':'mutation {{ podTerminate(input: {{podId: \\"%s\\"}}) }}'%p}});r=u.Request('https://api.runpod.io/graphql',data=b.encode(),headers={{'content-type':'application/json','Authorization':'Bearer '+k,'User-Agent':'nb/1'}},method='POST');u.urlopen(r,timeout=30).read()" || true; }}
phase assemble
python3 - <<'PY' >>"$RUN/job.log" 2>&1 || {{ echo fail_assemble > "$RUN/index.html"; sleep 600; term; exit 1; }}
import os, base64, pathlib
n = int(os.environ["JOB_TAR_N"])
parts = [os.environ[f"JOB_TAR_B64_{{i:02d}}"] for i in range(n)]
pathlib.Path("/tmp/demo.tgz").write_bytes(base64.b64decode("".join(parts)))
print("tar", pathlib.Path("/tmp/demo.tgz").stat().st_size)
PY
DEMO=/workspace/jobs/narcbench-presenter
tar --no-same-owner -xzf /tmp/demo.tgz -C "$DEMO" >>"$RUN/job.log" 2>&1
ls -la "$DEMO" >>"$RUN/job.log" 2>&1
phase serve
# stop placeholder server, start real app on 8765
kill %1 2>/dev/null || pkill -f "http.server 8765" || true
sleep 1
cd "$DEMO"
export PORT=8765
( sleep {max_sec}; term ) &
python3 app/server.py >"$RUN/app.log" 2>&1 &
APP=$!
sleep 3
if kill -0 $APP 2>/dev/null; then
  echo '{{"phase":"serving"}}' > "$RUN/status.json"
  # also copy status into a place - app owns 8765 now; write heartbeat file for volume
  echo serving > /workspace/jobs/narcbench-presenter/HEARTBEAT
  wait $APP
  EC=$?
else
  echo '{{"phase":"failed","detail":"app_start"}}' > "$RUN/status.json"
  # restart simple server so we can read logs
  python3 -m http.server 8765 --directory "$RUN" >/tmp/http.log 2>&1 &
  sleep {max_sec}
  term
  exit 1
fi
term
exit 0
"""
    boot_b64 = base64.b64encode(boot.encode()).decode()
    env = {
        "JOB_BOOT_B64": boot_b64,
        "RUNPOD_API_KEY": os.environ["RUNPOD_API_KEY"],
        "JOB_TAR_N": str(len(chunks)),
    }
    for i, ch in enumerate(chunks):
        env[f"JOB_TAR_B64_{i:02d}"] = ch
    env_chars = sum(len(str(v)) for v in env.values())
    print(json.dumps({"boot_b64": len(boot_b64), "chunks": len(chunks), "env_chars": env_chars, "will_launch": args.launch}, indent=2))
    if env_chars > 100000:
        sys.exit(f"env too large: {env_chars}")
    if not args.launch:
        return 0
    sys.path.insert(0, str(COLLUSION_SCRIPTS))
    from runpod_graphql import graphql
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
                "containerDiskInGb": 30,
                "minVcpuCount": 2,
                "minMemoryInGb": 8,
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
                    raise RuntimeError("empty")
                print(json.dumps({"created": pod, "gpu": gpu, "cloud": cloud}, indent=2))
                pid = pod["id"]
                url = f"https://{pid}-8765.proxy.runpod.net/"
                print(f"Presenter URL: {url}")
                print(f"Status JSON: https://{pid}-8765.proxy.runpod.net/status.json")
                print(f"Job log: https://{pid}-8765.proxy.runpod.net/job.log")
                Path("/tmp/narcbench_presenter_pod.json").write_text(
                    json.dumps({"id": pid, "gpu": gpu, "url": url, "max_hours": args.max_hours}, indent=2)
                )
                return 0
            except Exception as e:
                last = e
                print(f"[launch] {cloud} {gpu}: {e}")
    sys.exit(f"all failed: {last}")

if __name__ == "__main__":
    raise SystemExit(main())
