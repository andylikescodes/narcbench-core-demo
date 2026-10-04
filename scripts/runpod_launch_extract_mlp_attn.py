#!/usr/bin/env python3
"""Launch ONE RunPod job: extract residual+attn+mlp acts for Core then Transfer.

Model loads once. Auto-terminates. Status on :8765.
Default estimate-only; pass --launch with standing greenlight / Paige steer.

  python3 scripts/runpod_launch_extract_mlp_attn.py
  python3 scripts/runpod_launch_extract_mlp_attn.py --launch

Outputs on volume:
  /workspace/jobs/narcbench-results/activations/gemma2_9b/core/<id>_mlp_attn/
  /workspace/jobs/narcbench-results/activations/gemma2_9b/transfer/RUNPOD_mlp_attn/
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
COLLUSION_SCRIPTS = Path("/workspace/collusion-exp/scripts")
SECRETS = (
    Path("/home/box/.secrets/interp-explorer.env"),
    Path("/workspace/runpod-study/interp-explorer.env"),
    Path.home() / ".secrets" / "interp-explorer.env",
)
IMAGE = "runpod/pytorch:1.2.0-rc.162-cu1281-torch280-ubuntu2404"
VOLUME_ID = "cr1kjtzjs2"
POD_NAME = "job-narcbench-extract-mlp-attn"
GPU_PREFS_9B = [
    "NVIDIA L4",
    "NVIDIA RTX PRO 4500 Blackwell Server Edition",
    "NVIDIA RTX PRO 4500 Blackwell",
    "NVIDIA A40",
    "NVIDIA RTX A5000",
    "NVIDIA GeForce RTX 4090",
    "NVIDIA GeForce RTX 3090",
    "NVIDIA RTX A6000",
    "NVIDIA L40S",
    "NVIDIA A100 80GB PCIe",
]
RATES = {
    "NVIDIA L4": 0.49,
    "NVIDIA RTX PRO 4500 Blackwell Server Edition": 0.72,
    "NVIDIA RTX PRO 4500 Blackwell": 0.72,
    "NVIDIA A40": 0.49,
    "NVIDIA RTX A5000": 0.49,
    "NVIDIA GeForce RTX 4090": 0.74,
    "NVIDIA GeForce RTX 3090": 0.50,
    "NVIDIA RTX A6000": 0.53,
    "NVIDIA L40S": 0.89,
    "NVIDIA A100 80GB PCIe": 1.39,
}
DO_NOT_TOUCH = "lnqm4yawhv13on"
PT = ZoneInfo("America/Los_Angeles")
CORE_REL_DEFAULT = "gemma2_9b/core/20261001T012639Z"
XFER_REL_DEFAULT = "gemma2_9b/transfer/RUNPOD"


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


def estimate(*, n_core: int = 1700, n_xfer: int = 936) -> dict:
    """Same-forward residual+attn+mlp hooks ≈ residual wall time (not 3×).

    Historical Transfer residual (~936) ≈ 11 min on RTX PRO 4500.
    Core+Transfer 2636 ≈ 2.8× → ~31 min on RTX PRO; L4 slower → ~1.0–1.5 h.
    """
    n = n_core + n_xfer
    # conservative: 1.5s/forward average on L4 (includes long prompts); model load 10 min
    hours_l4 = round((n * 1.5 + 600) / 3600.0, 2)
    hours_l4 = max(hours_l4, 0.75)
    hours_4500 = round((n * 0.7 + 600) / 3600.0, 2)
    hours_4500 = max(hours_4500, 0.5)
    gpu = GPU_PREFS_9B[0]
    rate = RATES[gpu]
    return {
        "kind": "full_core_then_transfer_mlp_attn",
        "n_core_samples": n_core,
        "n_transfer_samples": n_xfer,
        "n_forwards_est": n,
        "sites": ["residual", "attn", "mlp"],
        "layers": "19-23",
        "hook_note": (
            "Same forward pass hooks residual (DecoderLayer), attn (self_attn out = "
            "residual-write), mlp (mlp out = residual-write) at last token. Wall time "
            "≈ residual-only extract, not 3×."
        ),
        "hours_est_l4": hours_l4,
        "hours_est_rtx_pro_4500": hours_4500,
        "hours_est": hours_l4,
        "preferred_gpu": gpu,
        "gpu_prefs": GPU_PREFS_9B,
        "rate_usd_hr": rate,
        "cost_est_usd": round(hours_l4 * rate, 2),
        "cost_est_range_usd": [
            round(hours_4500 * 0.72, 2),
            round(hours_l4 * 0.74, 2),
        ],
        "balance_note": "clientBalance ≈ $27.27; combined extract target ≪ $5",
        "volume_id": VOLUME_ID,
        "core_runs_rel": CORE_REL_DEFAULT,
        "transfer_runs_rel": XFER_REL_DEFAULT,
        "do_not_touch_pod": DO_NOT_TOUCH,
        "historical_residual_xfer_rtx_pro": "~11 min / ~$0.15",
    }


def resolve_runs(rel: str, var: str) -> str:
    return f"""
{var}=""
for c in \\
  /workspace/jobs/narcbench-results/upstream/scenarios/{rel} \\
  /workspace/jobs/narcbench-core-demo/upstream/scenarios/{rel} \\
  /workspace/narcbench-core-demo/upstream/scenarios/{rel}; do
  [ -d "$c" ] && {var}="$c" && break
done
"""


def build_boot(args: argparse.Namespace, script_b64: str) -> str:
    core_rel = args.core_runs_rel
    xfer_rel = args.xfer_runs_rel
    core_out = f"{core_rel}_mlp_attn"
    xfer_out = f"{xfer_rel}_mlp_attn"
    sites = args.sites
    return f"""#!/bin/bash
set -uo pipefail
# HARD RULE: never interact with Core pod {DO_NOT_TOUCH}
RUN=/tmp/narcbench-extract-mlp-attn; mkdir -p "$RUN" /workspace/jobs/narcbench-results/activations
python3 -m http.server 8765 --directory "$RUN" >/tmp/http.log 2>&1 &
echo '{{"phase":"boot","do_not_touch":"{DO_NOT_TOUCH}","sites":"{sites}"}}' > "$RUN/status.json"
phase() {{ echo "{{\\"phase\\":\\"$1\\"}}" > "$RUN/status.json"; echo "[phase] $1" | tee -a "$RUN/job.log"; }}
term() {{ python3 -c "import os,json,urllib.request as u;p=os.environ.get('RUNPOD_POD_ID','');k=os.environ.get('RUNPOD_API_KEY','');
b=json.dumps({{'query':'mutation {{ podTerminate(input: {{podId: \\"%s\\"}}) }}'%p}});
r=u.Request('https://api.runpod.io/graphql',data=b.encode(),headers={{'content-type':'application/json','Authorization':'Bearer '+k,'User-Agent':'nb-extract-mlp/1'}},method='POST');
u.urlopen(r,timeout=30).read()" || true; }}
finish() {{ echo "{{\\"phase\\":\\"$1\\",\\"detail\\":\\"$2\\"}}" > "$RUN/status.json"; sleep {args.grace_minutes}m; term; exit 0; }}
phase setup
DEMO=/workspace/jobs/narcbench-extract-mlp-attn; mkdir -p "$DEMO/scripts"
echo '{script_b64}' | base64 -d > "$DEMO/scripts/extract_activations_from_transcripts.py"
PY=/workspace/interp-demo/.venv-explorer/bin/python; [ -x "$PY" ] || PY=python3
export HF_HOME=${{HF_HOME:-/workspace/interp-demo/models/hf}}
$PY -m pip install -q 'transformers>=4.40' accelerate sentencepiece protobuf 2>>"$RUN/job.log" || true
{resolve_runs(core_rel, "RUNS_CORE")}
{resolve_runs(xfer_rel, "RUNS_XFER")}
[ -n "$RUNS_CORE" ] || finish failed missing_core_runs
[ -n "$RUNS_XFER" ] || finish failed missing_transfer_runs
echo "[setup] CORE=$RUNS_CORE XFER=$RUNS_XFER" | tee -a "$RUN/job.log"
OUT_CORE=/workspace/jobs/narcbench-results/activations/{core_out}
OUT_XFER=/workspace/jobs/narcbench-results/activations/{xfer_out}
mkdir -p "$OUT_CORE" "$OUT_XFER"
# ---- Core ----
phase core_extract
set +e
timeout {args.max_minutes}m $PY -u "$DEMO/scripts/extract_activations_from_transcripts.py" \\
  --runs-dir "$RUNS_CORE" --out "$OUT_CORE" --model {args.model} --layers {args.layers} \\
  --gen-only --sites {sites} \\
  >"$RUN/cmd_core.log" 2>&1
EC_CORE=$?; set -e
echo $EC_CORE > "$RUN/exit_code_core"
cat "$RUN/cmd_core.log" | tee -a "$RUN/job.log"
[ $EC_CORE -eq 0 ] || finish failed core_exit_$EC_CORE
# ---- Transfer (model already warm if same process — but script reloads; acceptable) ----
phase transfer_extract
set +e
timeout {args.max_minutes}m $PY -u "$DEMO/scripts/extract_activations_from_transcripts.py" \\
  --runs-dir "$RUNS_XFER" --out "$OUT_XFER" --model {args.model} --layers {args.layers} \\
  --gen-only --sites {sites} \\
  >"$RUN/cmd_xfer.log" 2>&1
EC_XFER=$?; set -e
echo $EC_XFER > "$RUN/exit_code_xfer"
cat "$RUN/cmd_xfer.log" | tee -a "$RUN/job.log"
[ $EC_XFER -eq 0 ] || finish failed xfer_exit_$EC_XFER
# mirror for box pull convenience
mkdir -p /workspace/jobs/narcbench-core-demo/data/activations/{core_out}
mkdir -p /workspace/jobs/narcbench-core-demo/data/activations/{xfer_out}
cp -a "$OUT_CORE"/. /workspace/jobs/narcbench-core-demo/data/activations/{core_out}/ 2>/dev/null || true
cp -a "$OUT_XFER"/. /workspace/jobs/narcbench-core-demo/data/activations/{xfer_out}/ 2>/dev/null || true
tar -czf "$RUN/results.tgz" -C /workspace/jobs/narcbench-results activations/{core_out} activations/{xfer_out} || true
ls -la "$OUT_CORE" "$OUT_XFER" | tee -a "$RUN/job.log"
finish done ok
"""


def client_balance() -> float | None:
    try:
        sys.path.insert(0, str(COLLUSION_SCRIPTS))
        from runpod_graphql import graphql  # type: ignore

        d = graphql("{ myself { clientBalance currentSpendPerHr } }")
        me = ((d.get("data") or {}).get("myself") or {})
        return float(me.get("clientBalance") or 0), float(me.get("currentSpendPerHr") or 0)  # type: ignore
    except Exception as e:
        print(f"[warn] balance fetch failed: {e}", file=sys.stderr)
        return None  # type: ignore


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--core-runs-rel", default=CORE_REL_DEFAULT)
    ap.add_argument("--xfer-runs-rel", default=XFER_REL_DEFAULT)
    ap.add_argument("--model", default="google/gemma-2-9b-it")
    ap.add_argument("--layers", default="19-23")
    ap.add_argument("--sites", default="residual,attn,mlp")
    ap.add_argument("--launch", action="store_true")
    ap.add_argument("--max-minutes", type=int, default=90, help="per-world timeout")
    ap.add_argument("--grace-minutes", type=int, default=5)
    args = ap.parse_args()
    read_env()

    est = estimate()
    bal_info = client_balance()
    balance = bal_info[0] if isinstance(bal_info, tuple) else None
    spend = bal_info[1] if isinstance(bal_info, tuple) else None

    script = (ROOT / "scripts/extract_activations_from_transcripts.py").read_bytes()
    script_b64 = base64.b64encode(script).decode()
    boot = build_boot(args, script_b64)
    boot_b64 = base64.b64encode(boot.encode()).decode()

    out_dir = ROOT / "results/transfer_stable"
    out_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "estimate": est,
        "hours_est": est["hours_est"],
        "cost_est_usd": est["cost_est_usd"],
        "rate_usd_hr": est["rate_usd_hr"],
        "preferred_gpu": est["preferred_gpu"],
        "gpu_prefs": est["gpu_prefs"],
        "core_runs_rel": args.core_runs_rel,
        "xfer_runs_rel": args.xfer_runs_rel,
        "core_out_rel": f"{args.core_runs_rel}_mlp_attn",
        "xfer_out_rel": f"{args.xfer_runs_rel}_mlp_attn",
        "model": args.model,
        "layers": args.layers,
        "sites": args.sites,
        "boot_b64_len": len(boot_b64),
        "script_b64_len": len(script_b64),
        "will_launch": args.launch,
        "do_not_touch": DO_NOT_TOUCH,
        "volume_id": VOLUME_ID,
        "balance_usd": balance,
        "spend_per_hr": spend,
        "soft_stop": "Track I / Paper 2 / interp-demo / family-house NOT touched",
        "note": "Sequential extract Core→Transfer one pod; residual α sweeps NOT re-run",
        "output_on_volume": "/workspace/jobs/narcbench-results/activations/..._mlp_attn/",
        "local_pull_targets": [
            f"data/activations/{args.core_runs_rel}_mlp_attn",
            f"data/activations/{args.xfer_runs_rel}_mlp_attn",
        ],
        "written_pt": datetime.now(PT).strftime("%Y-%m-%d %H:%M:%S %Z"),
    }
    est_path = out_dir / "MLP_ATTN_EXTRACT_ESTIMATE.json"
    est_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print(f"\n[wrote] {est_path}", file=sys.stderr)

    if not args.launch:
        print("\n[estimate-only] re-run with --launch after greenlight", file=sys.stderr)
        return 0
    if len(boot_b64) > 100000:
        print(f"boot too large: {len(boot_b64)}", file=sys.stderr)
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
    started = datetime.now(PT)
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
                rate_actual = float(pod.get("costPerHr") or RATES.get(gpu, 0.6))
                hours = est["hours_est_rtx_pro_4500"] if "4500" in gpu else est["hours_est"]
                launch = {
                    "pod_id": pod["id"],
                    "pod_name": POD_NAME,
                    "gpu": gpu,
                    "cloud": cloud,
                    "cost_per_hr": rate_actual,
                    "started_pt": started.strftime("%Y-%m-%d %H:%M:%S %Z"),
                    "kind": "mlp_attn_extract_core_then_transfer",
                    "sites": args.sites,
                    "layers": args.layers,
                    "model": args.model,
                    "core_runs_rel": args.core_runs_rel,
                    "xfer_runs_rel": args.xfer_runs_rel,
                    "core_out_rel": f"{args.core_runs_rel}_mlp_attn",
                    "xfer_out_rel": f"{args.xfer_runs_rel}_mlp_attn",
                    "status_url": f"https://{pod['id']}-8765.proxy.runpod.net/status.json",
                    "hours_est": hours,
                    "estimate_usd": round(hours * rate_actual, 2),
                    "balance_usd": balance,
                    "spend_per_hr_at_launch": spend,
                    "volume_id": VOLUME_ID,
                    "output_on_volume": "/workspace/jobs/narcbench-results/activations/",
                    "soft_stop": "Track I / Paper 2 / interp-demo / family-house NOT touched",
                    "machine": pod.get("machine"),
                    "do_not_touch": DO_NOT_TOUCH,
                    "success": True,
                    "created": pod,
                }
                print(json.dumps(launch, indent=2))
                launch_path = out_dir / "MLP_ATTN_EXTRACT_LAUNCH.json"
                launch_path.write_text(json.dumps(launch, indent=2) + "\n")
                Path("/tmp/narcbench_extract_mlp_attn_pod.json").write_text(json.dumps(launch, indent=2))
                print(f"Status: {launch['status_url']}", file=sys.stderr)
                print(f"[wrote] {launch_path}", file=sys.stderr)
                return 0
            except Exception as e:
                last = e
                print(f"[launch] {cloud} {gpu}: {e}")
    print(f"all failed: {last}", file=sys.stderr)
    fail = {"success": False, "error": str(last), "estimate": est, "balance_usd": balance}
    (out_dir / "MLP_ATTN_EXTRACT_LAUNCH.json").write_text(json.dumps(fail, indent=2) + "\n")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
