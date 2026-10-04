#!/usr/bin/env python3
"""Estimate or launch the Core within-trajectory mid-private RunPod smoke.

Estimate-only by default. Pass --launch only after the cost gate is approved:

  python3 scripts/runpod_launch_within_traj.py
  python3 scripts/runpod_launch_within_traj.py --launch

Boot clones this public repo and checks out GIT_SHA, then runs
scripts/within_traj_mid_private.py from that checkout. RUNPOD_API_KEY and
HF_TOKEN are read from the environment. Direction .npy files stay on the
network volume. The GraphQL create body is the tiny boot plus a few env vars.
Soft-stop: no Track I / Paper 2 / interp-demo / family-house.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
REPO_URL = "https://github.com/andylikescodes/narcbench-core-demo.git"
HARNESS_REL = "scripts/within_traj_mid_private.py"
IMAGE = "runpod/pytorch:1.2.0-rc.162-cu1281-torch280-ubuntu2404"
VOLUME_ID = "cr1kjtzjs2"
POD_NAME = "job-narcbench-within-traj-mid-private"
GPU_TRIES = [
    ("SECURE", "NVIDIA L4"),
    ("SECURE", "NVIDIA GeForce RTX 4090"),
]
RATES = {
    "NVIDIA L4": 0.49,
    "NVIDIA GeForce RTX 4090": 0.74,
}
GPU = "NVIDIA L4"
RATE_USD_HR = 0.49
PT = ZoneInfo("America/Los_Angeles")
# Refuse anything that is not well under the 100KB GraphQL/Cloudflare ceiling.
MAX_CREATE_BODY_BYTES = 64 * 1024
UA = "nb-within-traj/3"

ESTIMATE_OUT = ROOT / "results/transfer_stable/WITHIN_TRAJ_MID_PRIVATE_ESTIMATE.json"
LAUNCH_OUT = ROOT / "results/transfer_stable/WITHIN_TRAJ_MID_PRIVATE_LAUNCH.json"
# Data on the network volume, not code. Not shipped in the create body.
DIRECTIONS_ON_VOLUME = "/workspace/jobs/narcbench-data/transfer_stable/directions"
HF_CACHE_ON_VOLUME = "/workspace/jobs/hf-cache"
RESULTS_REL = "within_traj_mid_private"

ARMS = [
    "baseline_tf",
    "ablate_role_attn_L22",
    "ablate_role_resid_L21",
    "ablate_role_perp_resid_L21",
    "ablate_random_attn_L22",
    "steer_role_attn_L22",
]

CREATE_MUTATION = """mutation ($input: PodFindAndDeployOnDemandInput!) {
  podFindAndDeployOnDemand(input: $input) {
    id desiredStatus costPerHr machine { gpuDisplayName dataCenterId }
  }
}"""

SHA_RE = re.compile(r"^[0-9a-f]{40}$")
REF_RE = re.compile(r"^[A-Za-z0-9._/-]+$")
MODEL_RE = re.compile(r"^[A-Za-z0-9_./:-]+$")


def estimate(max_scenarios: int, max_new_tokens: int) -> dict:
    """Conservative L4 estimate: model load plus six short arms per scenario."""
    n_arms = len(ARMS)
    n_generations = max_scenarios * n_arms
    model_load_sec = 600
    sec_per_arm = 35.0 if max_new_tokens <= 64 else 50.0
    hours = round((model_load_sec + n_generations * sec_per_arm) / 3600.0, 3)
    return {
        "kind": "within_traj_mid_private_core_smoke",
        "world": "Core",
        "source_run": "gemma2_9b/core/20261001T012639Z",
        "n_scenarios": max_scenarios,
        "n_arms": n_arms,
        "n_generations_est": n_generations,
        "k_frac": 0.5,
        "max_new_tokens": max_new_tokens,
        "hours_est": hours,
        "preferred_gpu": GPU,
        "rate_usd_hr": RATE_USD_HR,
        "cost_est_usd": round(hours * RATE_USD_HR, 2),
        "cost_est_range_usd": [round(hours * 0.49, 2), round(hours * 0.74, 2)],
        "model_load_sec": model_load_sec,
        "sec_per_arm_assumed": sec_per_arm,
        "volume_id": VOLUME_ID,
        "sites": ["attn_L22", "residual_L21"],
        "arms": ARMS,
        "protocol": "teacher_forced_recorded_private_prefix_then_intervene_from_k",
        "payload_delivery": "git_checkout_sha",
    }


def git_identity() -> tuple[str, str]:
    sha = subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
    ).strip()
    ref = subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-parse", "--abbrev-ref", "HEAD"], text=True
    ).strip()
    if ref == "HEAD":
        ref = "main"
    return sha, ref


def worktree_dirty() -> bool:
    out = subprocess.check_output(
        ["git", "-C", str(ROOT), "status", "--porcelain"], text=True
    )
    return bool(out.strip())


def blob_in_commit(sha: str, rel: str) -> bool:
    proc = subprocess.run(
        ["git", "-C", str(ROOT), "cat-file", "-e", f"{sha}:{rel}"],
        capture_output=True,
    )
    return proc.returncode == 0


def assert_option_labels_fix(path: Path) -> None:
    """Refuse a harness that calls .keys() on a list (or on an unguarded value)."""
    text = path.read_text()
    start = text.find("def option_label_items")
    if start < 0:
        raise SystemExit("REFUSE: option_label_items missing from harness")
    end = text.find("\ndef ", start + 1)
    body = text[start:end if end > 0 else None]
    if "isinstance(raw, dict)" not in body or "isinstance(raw, (list, tuple))" not in body:
        raise SystemExit("REFUSE: option_labels list-or-dict guard missing")
    dict_branch = False
    for line in body.splitlines():
        stripped = line.strip()
        if ".keys()" in stripped and not dict_branch:
            raise SystemExit("REFUSE: .keys() used outside the dict branch of option_label_items")
        if stripped.startswith("if isinstance(raw, dict)"):
            dict_branch = True
        elif stripped.startswith(("if ", "elif ", "else", "def ")):
            dict_branch = False


def build_boot(args: argparse.Namespace) -> str:
    """Tiny boot: clone this repo, check out GIT_SHA, run the harness, self-term."""
    return f"""#!/bin/bash
set -uo pipefail
export GIT_TERMINAL_PROMPT=0
RUN=/tmp/narcbench-within-traj
SRC=/tmp/narcbench-src
RESULTS=/workspace/jobs/narcbench-results/{RESULTS_REL}
mkdir -p "$RUN" "$RESULTS"
phase() {{ echo "{{\\"phase\\":\\"$1\\"}}" > "$RUN/status.json"; echo "[phase] $1" | tee -a "$RUN/job.log"; }}
term() {{ python3 -c "import os,json,urllib.request as u;p=os.environ.get('RUNPOD_POD_ID','');k=os.environ.get('RUNPOD_API_KEY',''); b=json.dumps({{'query':'mutation {{ podTerminate(input: {{podId: \\\"%s\\\"}}) }}'%p}}); r=u.Request('https://api.runpod.io/graphql',data=b.encode(),headers={{'content-type':'application/json','Authorization':'Bearer '+k,'User-Agent':'nb-within-traj/3'}},method='POST'); u.urlopen(r,timeout=30).read()" || true; }}
finish() {{ cp -f "$RUN/job.log" "$RESULTS/last_boot.log" 2>/dev/null || true; echo "{{\\"phase\\":\\"$1\\",\\"detail\\":\\"$2\\"}}" > "$RUN/status.json"; sleep {args.grace_minutes}m; term; exit 0; }}
python3 -m http.server 8765 --directory "$RUN" >/tmp/http.log 2>&1 &
phase checkout
if ! command -v git >/dev/null 2>&1; then
  apt-get update -qq && apt-get install -y -qq git >>"$RUN/job.log" 2>&1 || finish failed git_install
fi
rm -rf "$SRC"
GIT_REPO_URL="{REPO_URL}"
SHA="${{GIT_SHA:?missing GIT_SHA}}"
REF="${{GIT_REF:?missing GIT_REF}}"
git clone --branch "$REF" --single-branch "$GIT_REPO_URL" "$SRC" >>"$RUN/job.log" 2>&1 || finish failed git_clone
git -C "$SRC" checkout --detach "$SHA" >>"$RUN/job.log" 2>&1 || finish failed git_checkout
GOT=$(git -C "$SRC" rev-parse HEAD)
echo "[setup] git_sha=$GOT" | tee -a "$RUN/job.log"
printf '%s\\n' "$GOT" > "$RUN/git_sha"
[ "$GOT" = "$SHA" ] || finish failed git_sha_mismatch
[ -f "$SRC/{HARNESS_REL}" ] || finish failed missing_harness
phase directions
DIRS="${{DIRECTIONS_DIR:-{DIRECTIONS_ON_VOLUME}}}"
[ -f "$DIRS/lr_role_attn_L22.npy" ] && [ -f "$DIRS/lr_role_L21.npy" ] || finish failed missing_directions
echo "[setup] directions=$DIRS" | tee -a "$RUN/job.log"
export HF_HOME="${{HF_HOME:-{HF_CACHE_ON_VOLUME}}}"
mkdir -p "$HF_HOME"
python3 -m pip install -q 'transformers>=4.40' accelerate sentencepiece protobuf numpy >>"$RUN/job.log" 2>&1 || true
OUT="$RESULTS/smoke_$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$OUT"
phase running
set +e
timeout {args.max_minutes}m python3 -u "$SRC/{HARNESS_REL}" \\
  --run-dir "$SRC/upstream/scenarios/gemma2_9b/core/20261001T012639Z" \\
  --directions "$DIRS" \\
  --out "$OUT" \\
  --model {args.model} \\
  --max-scenarios {args.max_scenarios} \\
  --k-frac {args.k_frac} \\
  --max-new-tokens {args.max_new_tokens} \\
  --seed {args.seed} \\
  >"$RUN/cmd.log" 2>&1
EC=$?; set -e
printf '%s\\n' "$EC" > "$RUN/exit_code"
cat "$RUN/cmd.log" | tee -a "$RUN/job.log"
cp -a "$OUT" "$RESULTS/latest_smoke" 2>/dev/null || true
if [ "$EC" -eq 0 ]; then finish done ok; else finish failed exit_$EC; fi
"""


def build_create_input(boot_b64: str, gpu: str, cloud: str, git_sha: str, git_ref: str) -> dict:
    env = {
        "JOB_BOOT_B64": boot_b64,
        "GIT_SHA": git_sha,
        "GIT_REF": git_ref,
        "RUNPOD_API_KEY": os.environ["RUNPOD_API_KEY"],
    }
    if os.environ.get("HF_TOKEN"):
        env["HF_TOKEN"] = os.environ["HF_TOKEN"]
    docker_args = "bash -c 'echo $JOB_BOOT_B64 | base64 -d > /tmp/boot.sh; exec bash /tmp/boot.sh'"
    return {
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


def body_bytes(query: str, variables: dict) -> int:
    return len(json.dumps({"query": query, "variables": variables}).encode())


class GraphqlHttpError(Exception):
    def __init__(self, status: int, body: str):
        self.status = status
        self.body = body
        low = body.lower()
        self.is_cloudflare_body_reject = status == 413 or (
            status in (403, 400) and "cloudflare" in low
        )
        super().__init__(f"HTTP {status}: {body[:300]}")


def graphql(query: str, variables: dict | None = None, timeout: int = 60) -> dict:
    payload: dict = {"query": query}
    if variables is not None:
        payload["variables"] = variables
    req = urllib.request.Request(
        "https://api.runpod.io/graphql",
        data=json.dumps(payload).encode(),
        headers={
            "content-type": "application/json",
            "Authorization": "Bearer " + os.environ["RUNPOD_API_KEY"],
            "User-Agent": UA,
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        err_body = exc.read().decode("utf-8", "replace")
        raise GraphqlHttpError(exc.code, err_body) from None


def list_pods() -> list:
    data = graphql(
        "query { myself { pods { id name desiredStatus costPerHr machine { gpuDisplayName } } } }",
        {},
        timeout=30,
    )
    return ((data.get("data") or {}).get("myself") or {}).get("pods") or []


def billable_gpu_pods(pods: list) -> list:
    busy = []
    for pod in pods:
        gpu = ((pod.get("machine") or {}).get("gpuDisplayName") or "")
        status = (pod.get("desiredStatus") or "").upper()
        if gpu and status not in ("EXITED", "TERMINATED"):
            busy.append(
                {
                    "id": pod.get("id"),
                    "name": pod.get("name"),
                    "gpu": gpu,
                    "desiredStatus": pod.get("desiredStatus"),
                }
            )
    return busy


def terminate_pod(pod_id: str) -> None:
    graphql(
        "mutation ($input: PodTerminateInput!) { podTerminate(input: $input) { id } }",
        {"input": {"podId": pod_id}},
        timeout=30,
    )


def measure_post_bytes(boot_b64: str, git_sha: str, git_ref: str) -> int:
    saved_key = os.environ.get("RUNPOD_API_KEY")
    saved_hf = os.environ.get("HF_TOKEN")
    if not saved_key:
        os.environ["RUNPOD_API_KEY"] = "0" * 40
    if not saved_hf:
        os.environ["HF_TOKEN"] = "0" * 40
    try:
        sample = build_create_input(boot_b64, GPU, "SECURE", git_sha, git_ref)
        return body_bytes(CREATE_MUTATION, {"input": sample})
    finally:
        if not saved_key:
            os.environ.pop("RUNPOD_API_KEY", None)
        if not saved_hf:
            os.environ.pop("HF_TOKEN", None)


def refuse_payload(boot: str, boot_b64: str) -> str | None:
    blob = boot + "\n" + boot_b64
    if "JOB_PAYLOAD_B64" in blob or "JOB_TGZ" in blob or "JOB_TAR_" in blob:
        return "job payload env present"
    if "/upload" in boot or "do_PUT" in boot or "tarfile" in boot:
        return "volume upload present in boot"
    for banned in ("/home/box", "collusion-exp", "interp-demo", "runpod-study"):
        if banned in boot:
            return f"box-only path in boot: {banned}"
    return None


def remote_has_sha(sha: str, ref: str) -> bool:
    """True when the public branch tip is this SHA or the SHA is an ancestor of it.

    stderr is discarded: a failed fetch can echo a credentialed remote URL.
    """
    fetch = subprocess.run(
        ["git", "-C", str(ROOT), "fetch", "--quiet", "origin", f"refs/heads/{ref}:refs/remotes/origin/{ref}"],
        capture_output=True,
    )
    if fetch.returncode != 0:
        return False
    ancestor = subprocess.run(
        ["git", "-C", str(ROOT), "merge-base", "--is-ancestor", sha, f"origin/{ref}"],
        capture_output=True,
    )
    return ancestor.returncode == 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="google/gemma-2-9b-it")
    ap.add_argument("--max-scenarios", type=int, default=6)
    ap.add_argument("--k-frac", type=float, default=0.5)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--launch", action="store_true", help="Actually create one L4 pod")
    ap.add_argument("--max-minutes", type=int, default=90)
    ap.add_argument("--grace-minutes", type=int, default=6)
    ap.add_argument("--estimate-out", type=Path, default=ESTIMATE_OUT)
    ap.add_argument("--launch-card", type=Path, default=LAUNCH_OUT)
    ap.add_argument("--cost-max", type=float, default=1.50)
    ap.add_argument("--balance-min", type=float, default=20.0)
    ap.add_argument("--git-sha", default="", help="Override checkout SHA (default: HEAD)")
    ap.add_argument("--git-ref", default="", help="Branch to clone before checkout")
    args = ap.parse_args()
    if not MODEL_RE.fullmatch(args.model):
        print("ERROR: --model must be a single token (no spaces)", file=sys.stderr)
        return 1
    if args.max_scenarios < 1 or args.max_scenarios > 6:
        print("ERROR: --max-scenarios must be between 1 and 6", file=sys.stderr)
        return 1

    head_sha, head_ref = git_identity()
    git_sha = (args.git_sha or head_sha).strip()
    git_ref = (args.git_ref or head_ref).strip()
    if not SHA_RE.fullmatch(git_sha):
        print("ERROR: git SHA must be 40 hex chars", file=sys.stderr)
        return 1
    if not REF_RE.fullmatch(git_ref):
        print("ERROR: git ref has unexpected characters", file=sys.stderr)
        return 1

    harness = ROOT / HARNESS_REL
    assert_option_labels_fix(harness)
    harness_in_sha = blob_in_commit(git_sha, HARNESS_REL)
    dirty = worktree_dirty()

    est = estimate(args.max_scenarios, args.max_new_tokens)
    boot = build_boot(args)
    boot_b64 = base64.b64encode(boot.encode()).decode("ascii")
    post_bytes = measure_post_bytes(boot_b64, git_sha, git_ref)
    payload_problem = refuse_payload(boot, boot_b64)

    report = {
        "estimate": est,
        "hours_est": est["hours_est"],
        "cost_est_usd": est["cost_est_usd"],
        "rate_usd_hr": RATE_USD_HR,
        "preferred_gpu": GPU,
        "n_scenarios": args.max_scenarios,
        "arms": ARMS,
        "model": args.model,
        "will_launch": bool(args.launch),
        "volume_id": VOLUME_ID,
        "git_repo_url": REPO_URL,
        "git_sha": git_sha,
        "git_ref": git_ref,
        "git_pull_blocked": None,
        "harness_in_sha": harness_in_sha,
        "worktree_dirty": dirty,
        "labels_fix": "option_label_items_list_or_dict",
        "payload_delivery": "git_checkout_sha",
        "payload_in_graphql_env": False,
        "job_tarball_in_create": False,
        "directions_on_volume": DIRECTIONS_ON_VOLUME,
        "graphql_post_body_bytes": post_bytes,
        "graphql_post_body_limit": MAX_CREATE_BODY_BYTES,
        "boot_b64_len": len(boot_b64),
        "output_on_volume": f"/workspace/jobs/narcbench-results/{RESULTS_REL}/smoke_*",
        "status_port": 8765,
        "status_url_template": "https://<pod-id>-8765.proxy.runpod.net/status.json",
        "cost_gate_usd": args.cost_max,
        "balance_min_usd": args.balance_min,
        "gpu_tries": [{"cloud": c, "gpu": g} for c, g in GPU_TRIES],
        "launch_command": "python3 scripts/runpod_launch_within_traj.py --launch",
        "dry_run_command": "python3 scripts/runpod_launch_within_traj.py",
        "soft_stop": "Track I / Paper 2 / interp-demo / family-house NOT touched; no GPU unless --launch",
    }
    print(json.dumps(report, indent=2))
    args.estimate_out.parent.mkdir(parents=True, exist_ok=True)
    args.estimate_out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"\n[wrote] {args.estimate_out}", file=sys.stderr)
    print(f"[pack] boot_b64={len(boot_b64)} post_body={post_bytes}", file=sys.stderr)

    if payload_problem:
        print(f"REFUSE: {payload_problem}", file=sys.stderr)
        return 4
    if post_bytes >= MAX_CREATE_BODY_BYTES:
        print(
            f"REFUSE: GraphQL POST body {post_bytes} >= {MAX_CREATE_BODY_BYTES}",
            file=sys.stderr,
        )
        return 4
    if not args.launch:
        print("\n[estimate-only] NOT launching. Re-run with --launch when ready.", file=sys.stderr)
        return 0

    if not harness_in_sha:
        print(
            f"REFUSE launch: {HARNESS_REL} is not in git SHA {git_sha}",
            file=sys.stderr,
        )
        return 1
    if dirty:
        print("REFUSE launch: worktree is dirty; commit and push the SHA first", file=sys.stderr)
        return 1
    if not remote_has_sha(git_sha, git_ref):
        print(
            "REFUSE launch: SHA is not on the public remote branch yet",
            file=sys.stderr,
        )
        return 1
    if est["cost_est_usd"] > args.cost_max:
        print(f"REFUSE launch: cost_est ${est['cost_est_usd']} > ${args.cost_max}", file=sys.stderr)
        return 2
    if not os.environ.get("RUNPOD_API_KEY"):
        print("RUNPOD_API_KEY missing", file=sys.stderr)
        return 1

    balance = None
    spend = None
    try:
        bal = graphql("query { myself { clientBalance currentSpendPerHr } }", {}, timeout=30)
        me = ((bal.get("data") or {}).get("myself") or {})
        balance = me.get("clientBalance")
        spend = me.get("currentSpendPerHr")
    except Exception as exc:
        print(f"[warn] balance fetch failed ({type(exc).__name__})", file=sys.stderr)
    print(f"[balance] ${balance} spendPerHr={spend}", file=sys.stderr)

    if balance is not None and float(balance) < args.balance_min:
        print(f"REFUSE launch: balance ${balance} < ${args.balance_min}", file=sys.stderr)
        card = {
            "launched": False,
            "reason": "balance_below_min",
            "balance_usd": balance,
            "balance_min_usd": args.balance_min,
            "graphql_post_body_bytes": post_bytes,
            "git_sha": git_sha,
            "estimate": est,
        }
        args.launch_card.write_text(json.dumps(card, indent=2) + "\n")
        return 3

    pods_now = list_pods()
    busy = billable_gpu_pods(pods_now)
    if busy:
        print(f"REFUSE launch: billable GPU already up: {busy}", file=sys.stderr)
        card = {
            "launched": False,
            "reason": "billable_gpu_already_running",
            "pods": busy,
            "balance_usd": balance,
            "graphql_post_body_bytes": post_bytes,
            "git_sha": git_sha,
            "estimate": est,
        }
        args.launch_card.write_text(json.dumps(card, indent=2) + "\n")
        return 5

    last: Exception | None = None
    for cloud, gpu in GPU_TRIES:
        rate = RATES.get(gpu, RATE_USD_HR)
        inp = build_create_input(boot_b64, gpu, cloud, git_sha, git_ref)
        nbytes = body_bytes(CREATE_MUTATION, {"input": inp})
        if nbytes >= MAX_CREATE_BODY_BYTES:
            print(f"REFUSE POST: {cloud} {gpu} body {nbytes} >= {MAX_CREATE_BODY_BYTES}", file=sys.stderr)
            card = {
                "launched": False,
                "reason": "graphql_body_too_large",
                "graphql_post_body_bytes": nbytes,
                "balance_usd": balance,
                "git_sha": git_sha,
                "estimate": est,
            }
            args.launch_card.write_text(json.dumps(card, indent=2) + "\n")
            return 4
        print(f"[launch] trying {cloud} {gpu}; post_body_bytes={nbytes}", file=sys.stderr)
        try:
            data = graphql(CREATE_MUTATION, {"input": inp}, timeout=120)
            if data.get("errors"):
                raise RuntimeError(str(data["errors"])[:400])
            pod = (data.get("data") or {}).get("podFindAndDeployOnDemand")
            if not pod or not pod.get("id"):
                raise RuntimeError("empty pod")
            status_url = f"https://{pod['id']}-8765.proxy.runpod.net/status.json"
            started = datetime.now(PT).strftime("%Y-%m-%d %H:%M:%S %Z")
            card = {
                "launched": True,
                "pod_id": pod["id"],
                "pod_name": POD_NAME,
                "gpu": gpu,
                "cloud": cloud,
                "cost_per_hr": pod.get("costPerHr") or rate,
                "balance_usd": balance,
                "balance_at_launch_usd": balance,
                "spend_per_hr_at_launch": spend,
                "started_pt": started,
                "status_url": status_url,
                "estimate": est,
                "gpu_tries": [{"cloud": c, "gpu": g} for c, g in GPU_TRIES],
                "volume_id": VOLUME_ID,
                "git_repo_url": REPO_URL,
                "git_sha": git_sha,
                "git_ref": git_ref,
                "git_pull_blocked": None,
                "labels_fix": "option_label_items_list_or_dict",
                "payload_delivery": "git_checkout_sha",
                "payload_in_graphql_env": False,
                "job_tarball_in_create": False,
                "graphql_post_body_bytes": nbytes,
                "cloudflare_walk_stop": True,
                "output_on_volume": f"/workspace/jobs/narcbench-results/{RESULTS_REL}/smoke_*",
                "output_latest_tgz": f"/workspace/jobs/narcbench-results/{RESULTS_REL}/latest.tgz",
                "soft_stop": "Track I / Paper 2 / interp-demo / family-house NOT touched",
                "machine": pod.get("machine") or {},
            }
            args.launch_card.parent.mkdir(parents=True, exist_ok=True)
            args.launch_card.write_text(json.dumps(card, indent=2) + "\n")
            print(json.dumps(card, indent=2))
            print(f"Pod ID: {pod['id']}")
            print(f"$/hr: {card['cost_per_hr']}")
            print(f"Balance: ${balance}")
            print(f"POST body bytes: {nbytes}")
            print(f"Status URL: {status_url}")
            print(f"[wrote] {args.launch_card}", file=sys.stderr)
            return 0
        except GraphqlHttpError as exc:
            last = exc
            print(f"[launch] {cloud} {gpu}: HTTP {exc.status}", file=sys.stderr)
            if exc.is_cloudflare_body_reject:
                print("[launch] Cloudflare body-size reject — STOPPING GPU walk", file=sys.stderr)
                fail_card = {
                    "launched": False,
                    "reason": "cloudflare_body_size_reject",
                    "http_status": exc.status,
                    "graphql_post_body_bytes": nbytes,
                    "cloudflare_walk_stop": True,
                    "gpu_stopped_at": {"cloud": cloud, "gpu": gpu},
                    "balance_usd": balance,
                    "git_sha": git_sha,
                    "estimate": est,
                }
                args.launch_card.parent.mkdir(parents=True, exist_ok=True)
                args.launch_card.write_text(json.dumps(fail_card, indent=2) + "\n")
                return 1
            time.sleep(1.0)
        except Exception as exc:
            last = exc
            msg = str(exc).lower()
            print(f"[launch] {cloud} {gpu}: {type(exc).__name__}", file=sys.stderr)
            if "no instances" in msg or "capacity" in msg or "not available" in msg:
                time.sleep(1.0)
                continue
            time.sleep(1.0)

    fail_card = {
        "launched": False,
        "reason": "all_gpu_prefs_failed",
        "last_error": str(last)[:400] if last else None,
        "gpu_tries": [{"cloud": c, "gpu": g} for c, g in GPU_TRIES],
        "balance_usd": balance,
        "graphql_post_body_bytes": post_bytes,
        "cloudflare_walk_stop": True,
        "estimate": est,
        "git_sha": git_sha,
        "ready_to_launch": True,
        "launch_command": "python3 scripts/runpod_launch_within_traj.py --launch",
    }
    args.launch_card.parent.mkdir(parents=True, exist_ok=True)
    args.launch_card.write_text(json.dumps(fail_card, indent=2) + "\n")
    print(f"all failed: {type(last).__name__ if last else 'unknown'}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
