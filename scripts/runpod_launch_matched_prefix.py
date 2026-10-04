#!/usr/bin/env python3
"""Estimate or launch matched-prefix activation interchange on RunPod.

Estimate-only by default. Pass --launch only after the cost gate is approved:

  python3 scripts/runpod_launch_matched_prefix.py
  python3 scripts/runpod_launch_matched_prefix.py --launch --max-pairs 10
  python3 scripts/runpod_launch_matched_prefix.py --suite transfer --max-pairs 12

Boot clones this public repo and checks out GIT_SHA, then runs
scripts/matched_prefix_interchange.py from that checkout. --extras-only runs
the multi-site, PCA, and role-perpendicular kitchen sink. --role-perp-confirm
runs the residual-L21 role-perp card (not that kitchen sink) on every row of
the volume pairs file. Direction .npy files and pairs stay on the network
volume. RUNPOD_API_KEY and HF_TOKEN come from the environment. The GraphQL
create body is the tiny boot plus a few env vars. --launch refuses unless the
harness is in the checked-out SHA.
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
HARNESS_REL = "scripts/matched_prefix_interchange.py"
IMAGE = "runpod/pytorch:1.2.0-rc.162-cu1281-torch280-ubuntu2404"
VOLUME_ID = "cr1kjtzjs2"
GPU_TRIES = [
    ("SECURE", "NVIDIA L4"),
    ("SECURE", "NVIDIA GeForce RTX 4090"),
]
RATES = {
    "NVIDIA L4": 0.49,
    "NVIDIA GeForce RTX 4090": 0.74,
}
GPU = "NVIDIA L4"
PT = ZoneInfo("America/Los_Angeles")
MAX_CREATE_BODY_BYTES = 64 * 1024
UA = "nb-matched-prefix/3"
DIRECTIONS_ON_VOLUME = "/workspace/jobs/narcbench-data/transfer_stable/directions"
HF_CACHE_ON_VOLUME = "/workspace/jobs/hf-cache"
# Read from the volume at boot. Never copied into the GraphQL body.
CORE_DIRECTION_FILES = [
    "lr_role_attn_L22.npy",
    "lr_role_L21.npy",
    "lr_mode_L21.npy",
]
EXTRAS_DIRECTION_FILES = [
    "lr_role_perp_mode_L21.npy",
    "pca_contrast_k8_L23.npy",
    "pca_contrast_k8_L23_lr_ambient.npy",
]

CORE_ARMS = [
    "baseline_colluder",
    "baseline_honest",
    "patch_c2h_attn_L22",
    "patch_h2c_attn_L22",
    "patch_c2h_resid_L21",
    "patch_h2c_resid_L21",
    "ablate_role_attn_L22_colluder",
    "ablate_role_resid_L21_colluder",
    "ablate_mode_resid_L21_colluder",
    "ablate_random_attn_L22_colluder",
    "ablate_role_perp_resid_L21_colluder",
]
EXTRAS_ONLY_ARMS = [
    "baseline_colluder",
    "baseline_honest",
    "patch_c2h_attn_L22+resid_L21",
    "patch_h2c_attn_L22+resid_L21",
    "ablate_role_attn_L22+resid_L21_colluder",
    "project_out_pca_k8_resid_L23_colluder",
    "ablate_pca_lr_ambient_resid_L23_colluder",
    "patch_c2h_role_perp_resid_L21",
    "patch_h2c_role_perp_resid_L21",
]
# Residual L21 confirm card. Not the extras-only multi-site kitchen sink.
ROLE_PERP_CONFIRM_ARMS = [
    "baseline_colluder",
    "baseline_honest",
    "patch_h2c_role_perp_resid_L21",
    "patch_c2h_role_perp_resid_L21",
    "ablate_role_perp_resid_L21_colluder",
    "ablate_role_resid_L21_colluder",
    "patch_h2c_role_resid_L21",
    "patch_c2h_role_resid_L21",
    "patch_h2c_random_resid_L21",
    "patch_c2h_random_resid_L21",
    "project_out_pca_k8_resid_L23_colluder",
    "ablate_pca_lr_ambient_resid_L23_colluder",
]
SUITE_META = {
    "core": {
        "kind": "matched_prefix_interchange_core_smoke",
        "world": "Core",
        "source_run": "gemma2_9b/core/20261001T012639Z",
        "pod_name": "job-narcbench-matched-prefix",
        "vol_subdir": "matched_prefix",
        "pairs_on_volume": "/workspace/jobs/narcbench-data/matched_prefix/pairs.jsonl",
        "estimate_out": ROOT / "results/transfer_stable/MATCHED_PREFIX_INTERCHANGE_ESTIMATE.json",
        "launch_out": ROOT / "results/transfer_stable/MATCHED_PREFIX_INTERCHANGE_LAUNCH.json",
        "target_band_usd": [0.30, 0.80],
    },
    "transfer": {
        "kind": "matched_prefix_interchange_transfer_smoke",
        "world": "Transfer",
        "source_run": "gemma2_9b/transfer/RUNPOD",
        "pod_name": "job-narcbench-matched-prefix-xfer",
        "vol_subdir": "matched_prefix_transfer",
        "pairs_on_volume": "/workspace/jobs/narcbench-data/matched_prefix/pairs_transfer.jsonl",
        "estimate_out": ROOT / "results/transfer_stable/MATCHED_PREFIX_TRANSFER_ESTIMATE.json",
        "launch_out": ROOT / "results/transfer_stable/MATCHED_PREFIX_TRANSFER_LAUNCH.json",
        "target_band_usd": [0.30, 1.00],
    },
}

CREATE_MUTATION = """mutation ($input: PodFindAndDeployOnDemandInput!) {
  podFindAndDeployOnDemand(input: $input) {
    id desiredStatus costPerHr machine { gpuDisplayName dataCenterId }
  }
}"""

SHA_RE = re.compile(r"^[0-9a-f]{40}$")
REF_RE = re.compile(r"^[A-Za-z0-9._/-]+$")
MODEL_RE = re.compile(r"^[A-Za-z0-9_./:-]+$")


def estimate(n_pairs: int | None, max_new_tokens: int, n_arms: int, *, suite: str) -> dict:
    meta = SUITE_META[suite]
    if n_pairs is None:
        return {
            "kind": meta["kind"],
            "suite": suite,
            "n_pairs": None,
            "n_arms": n_arms,
            "n_generations_est": None,
            "n_captures_est": None,
            "max_new_tokens": max_new_tokens,
            "hours_est": None,
            "preferred_gpu": GPU,
            "gpu_tries": [{"cloud": c, "gpu": g} for c, g in GPU_TRIES],
            "rate_usd_hr": RATES[GPU],
            "cost_est_usd": None,
            "cost_est_range_usd": None,
            "target_band_usd": list(meta["target_band_usd"]),
            "volume_id": VOLUME_ID,
            "sec_per_gen_assumed": 10.0 if max_new_tokens <= 16 else 18.0,
            "model_load_sec": 600,
            "world": meta["world"],
            "source_run": meta["source_run"],
            "protocol": "teacher_forced_matched_prefix_activation_interchange",
            "payload_delivery": "git_checkout_sha",
            "n_pairs_note": "every non-empty row of the volume pairs file; this machine has no local copy to count",
        }
    n_gens = n_pairs * n_arms
    n_captures = n_pairs * 2
    sec_per_gen = 10.0 if max_new_tokens <= 16 else 18.0
    sec_per_cap = 4.0
    load_sec = 600
    hours = round((n_gens * sec_per_gen + n_captures * sec_per_cap + load_sec) / 3600.0, 3)
    hours = max(hours, 0.25)
    rate = RATES[GPU]
    return {
        "kind": meta["kind"],
        "suite": suite,
        "n_pairs": n_pairs,
        "n_arms": n_arms,
        "n_generations_est": n_gens,
        "n_captures_est": n_captures,
        "max_new_tokens": max_new_tokens,
        "hours_est": hours,
        "preferred_gpu": GPU,
        "gpu_tries": [{"cloud": c, "gpu": g} for c, g in GPU_TRIES],
        "rate_usd_hr": rate,
        "cost_est_usd": round(hours * rate, 2),
        "cost_est_range_usd": [round(hours * 0.49, 2), round(hours * 0.74, 2)],
        "target_band_usd": list(meta["target_band_usd"]),
        "volume_id": VOLUME_ID,
        "sec_per_gen_assumed": sec_per_gen,
        "model_load_sec": load_sec,
        "world": meta["world"],
        "source_run": meta["source_run"],
        "protocol": "teacher_forced_matched_prefix_activation_interchange",
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


def arms_for(args: argparse.Namespace) -> list[str]:
    if args.role_perp_confirm:
        return list(ROLE_PERP_CONFIRM_ARMS)
    if args.extras_only:
        return list(EXTRAS_ONLY_ARMS)
    if args.extras:
        extra = [a for a in EXTRAS_ONLY_ARMS if a not in ("baseline_colluder", "baseline_honest")]
        return list(CORE_ARMS) + extra
    return list(CORE_ARMS)


def direction_files_for(args: argparse.Namespace) -> list[str]:
    needed = list(CORE_DIRECTION_FILES)
    if args.extras or args.extras_only or args.role_perp_confirm:
        needed.extend(EXTRAS_DIRECTION_FILES)
    return needed


def build_boot(args: argparse.Namespace, *, vol_subdir: str, pairs_on_volume: str) -> str:
    if args.role_perp_confirm:
        extras_argv = " --role-perp-confirm"
    elif args.extras_only:
        extras_argv = " --extras-only"
    elif args.extras:
        extras_argv = " --extras"
    else:
        extras_argv = ""
    dir_checks = " && ".join(
        f'[ -f "$DIRS/{name}" ]' for name in direction_files_for(args)
    )
    return f"""#!/bin/bash
set -uo pipefail
export GIT_TERMINAL_PROMPT=0
RUN=/tmp/narcbench-matched-prefix
SRC=/tmp/narcbench-src
RESULTS=/workspace/jobs/narcbench-results/{vol_subdir}
mkdir -p "$RUN" "$RESULTS"
phase() {{ echo "{{\\"phase\\":\\"$1\\"}}" > "$RUN/status.json"; echo "[phase] $1" | tee -a "$RUN/job.log"; }}
term() {{ python3 -c "import os,json,urllib.request as u;p=os.environ.get('RUNPOD_POD_ID','');k=os.environ.get('RUNPOD_API_KEY',''); b=json.dumps({{'query':'mutation {{ podTerminate(input: {{podId: \\\"%s\\\"}}) }}'%p}}); r=u.Request('https://api.runpod.io/graphql',data=b.encode(),headers={{'content-type':'application/json','Authorization':'Bearer '+k,'User-Agent':'nb-matched-prefix/3'}},method='POST'); u.urlopen(r,timeout=30).read()" || true; }}
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
phase data
DIRS="${{DIRECTIONS_DIR:-{DIRECTIONS_ON_VOLUME}}}"
PAIRS="${{PAIRS_FILE:-{pairs_on_volume}}}"
{dir_checks} || finish failed missing_directions
[ -f "$PAIRS" ] || finish failed missing_pairs
NPAIRS=$(grep -cve '^[[:space:]]*$' "$PAIRS" || true)
echo "[setup] directions=$DIRS pairs=$PAIRS pairs_rows=$NPAIRS" | tee -a "$RUN/job.log"
export HF_HOME="${{HF_HOME:-{HF_CACHE_ON_VOLUME}}}"
mkdir -p "$HF_HOME"
python3 -m pip install -q 'transformers>=4.40' accelerate sentencepiece protobuf numpy >>"$RUN/job.log" 2>&1 || true
OUT="$RESULTS/smoke_$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$OUT"
phase running
set +e
timeout {args.max_minutes}m python3 -u "$SRC/{HARNESS_REL}" \\
  --pairs "$PAIRS" \\
  --directions "$DIRS" \\
  --out "$OUT" \\
  --model {args.model} \\
  --max-pairs {args.max_pairs} \\
  --max-new-tokens {args.max_new_tokens} \\
  --seed {args.seed}{extras_argv} \\
  >"$RUN/cmd.log" 2>&1
EC=$?; set -e
printf '%s\\n' "$EC" > "$RUN/exit_code"
cat "$RUN/cmd.log" | tee -a "$RUN/job.log"
cp -a "$OUT" "$RESULTS/latest_smoke" 2>/dev/null || true
if [ "$EC" -eq 0 ]; then finish done ok; else finish failed exit_$EC; fi
"""


def build_create_input(
    boot_b64: str,
    *,
    pod_name: str,
    gpu: str,
    cloud: str,
    git_sha: str,
    git_ref: str,
) -> dict:
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
        "name": pod_name,
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


def measure_post_bytes(boot_b64: str, pod_name: str, git_sha: str, git_ref: str) -> int:
    saved_key = os.environ.get("RUNPOD_API_KEY")
    saved_hf = os.environ.get("HF_TOKEN")
    if not saved_key:
        os.environ["RUNPOD_API_KEY"] = "0" * 40
    if not saved_hf:
        os.environ["HF_TOKEN"] = "0" * 40
    try:
        sample = build_create_input(
            boot_b64,
            pod_name=pod_name,
            gpu=GPU,
            cloud="SECURE",
            git_sha=git_sha,
            git_ref=git_ref,
        )
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
    ap.add_argument("--suite", choices=("core", "transfer"), default="core")
    ap.add_argument(
        "--pairs",
        type=Path,
        default=None,
        help="Optional local pairs file, used only to count rows for the estimate",
    )
    ap.add_argument("--model", default="google/gemma-2-9b-it")
    ap.add_argument(
        "--max-pairs",
        type=int,
        default=None,
        help="leading rows to score; default 10, or 0 (every row) with --role-perp-confirm",
    )
    ap.add_argument("--max-new-tokens", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--extras", action="store_true")
    ap.add_argument("--extras-only", action="store_true")
    ap.add_argument(
        "--role-perp-confirm",
        action="store_true",
        help="boot the residual-L21 role-perp confirm card on every volume pairs row",
    )
    ap.add_argument("--launch", action="store_true")
    ap.add_argument("--max-minutes", type=int, default=90)
    ap.add_argument("--grace-minutes", type=int, default=6)
    ap.add_argument("--estimate-out", type=Path, default=None)
    ap.add_argument("--launch-card", type=Path, default=None)
    ap.add_argument("--cost-max", type=float, default=1.5)
    ap.add_argument("--balance-min", type=float, default=20.0)
    ap.add_argument("--git-sha", default="")
    ap.add_argument("--git-ref", default="")
    args = ap.parse_args()

    if not MODEL_RE.fullmatch(args.model):
        print("ERROR: --model must be a single token (no spaces)", file=sys.stderr)
        return 1
    mode_flags = int(bool(args.extras)) + int(bool(args.extras_only)) + int(bool(args.role_perp_confirm))
    if mode_flags > 1:
        print("ERROR: pass only one of --extras, --extras-only, --role-perp-confirm", file=sys.stderr)
        return 1
    if args.role_perp_confirm:
        args.max_pairs = 0 if args.max_pairs is None else args.max_pairs
    elif args.max_pairs is None:
        args.max_pairs = 10
    if args.max_pairs < 0 or (args.max_pairs == 0 and not args.role_perp_confirm):
        print("ERROR: --max-pairs 0 is only valid with --role-perp-confirm", file=sys.stderr)
        return 1

    meta = SUITE_META[args.suite]
    if args.estimate_out is None:
        if args.role_perp_confirm:
            args.estimate_out = ROOT / "results/transfer_stable/MATCHED_PREFIX_ROLE_PERP_CONFIRM_ESTIMATE.json"
        elif args.extras or args.extras_only:
            args.estimate_out = ROOT / "results/transfer_stable/MATCHED_PREFIX_CHEAP_EXTRAS_ESTIMATE.json"
        else:
            args.estimate_out = meta["estimate_out"]
    if args.launch_card is None:
        if args.role_perp_confirm:
            args.launch_card = ROOT / "results/transfer_stable/MATCHED_PREFIX_ROLE_PERP_CONFIRM_LAUNCH.json"
        elif args.extras or args.extras_only:
            args.launch_card = ROOT / "results/transfer_stable/MATCHED_PREFIX_CHEAP_EXTRAS_LAUNCH.json"
        else:
            args.launch_card = meta["launch_out"]

    vol_subdir = meta["vol_subdir"]
    pod_name = meta["pod_name"]
    pairs_on_volume = meta["pairs_on_volume"]
    if args.role_perp_confirm:
        vol_subdir = f"{vol_subdir}_role_perp_confirm"
        pod_name = f"{pod_name}-role-perp-confirm"
    elif args.extras or args.extras_only:
        vol_subdir = f"{vol_subdir}_cheap_extras"
        pod_name = f"{pod_name}-cheap-extras"

    n_file = None
    if args.pairs is not None:
        pairs_path = args.pairs if args.pairs.is_absolute() else ROOT / args.pairs
        if pairs_path.is_file():
            n_file = sum(1 for line in pairs_path.read_text().splitlines() if line.strip())
    if args.max_pairs == 0:
        n_pairs = n_file
    else:
        n_pairs = min(args.max_pairs, n_file) if n_file is not None else args.max_pairs
    arms_list = arms_for(args)
    est = estimate(n_pairs, args.max_new_tokens, len(arms_list), suite=args.suite)
    if args.role_perp_confirm:
        est["kind"] = est["kind"].replace("_smoke", "_role_perp_confirm")
        est["role_perp_confirm"] = True
        est["scores_every_pairs_row"] = args.max_pairs == 0
    elif args.extras or args.extras_only:
        est["kind"] = est["kind"].replace("_smoke", "_cheap_extras_smoke")
        est["cheap_extras"] = True
        est["extras_only"] = bool(args.extras_only)

    head_sha, head_ref = git_identity()
    git_sha = (args.git_sha or head_sha).strip()
    git_ref = (args.git_ref or head_ref).strip()
    if not SHA_RE.fullmatch(git_sha) or not REF_RE.fullmatch(git_ref):
        print("ERROR: bad git sha or ref", file=sys.stderr)
        return 1

    harness_in_sha = blob_in_commit(git_sha, HARNESS_REL)
    dirty = worktree_dirty()
    boot = build_boot(args, vol_subdir=vol_subdir, pairs_on_volume=pairs_on_volume)
    boot_b64 = base64.b64encode(boot.encode()).decode("ascii")
    post_bytes = measure_post_bytes(boot_b64, pod_name, git_sha, git_ref)
    payload_problem = refuse_payload(boot, boot_b64)

    report = {
        "estimate": est,
        "suite": args.suite,
        "hours_est": est["hours_est"],
        "cost_est_usd": est["cost_est_usd"],
        "rate_usd_hr": est["rate_usd_hr"],
        "preferred_gpu": GPU,
        "gpu_tries": [{"cloud": c, "gpu": g} for c, g in GPU_TRIES],
        "n_pairs": n_pairs,
        "n_pairs_local_file": n_file,
        "arms": arms_list,
        "n_arms": len(arms_list),
        "model": args.model,
        "max_new_tokens": args.max_new_tokens,
        "will_launch": bool(args.launch),
        "volume_id": VOLUME_ID,
        "git_repo_url": REPO_URL,
        "git_sha": git_sha,
        "git_ref": git_ref,
        "harness": HARNESS_REL,
        "harness_in_sha": harness_in_sha,
        "worktree_dirty": dirty,
        "payload_delivery": "git_checkout_sha",
        "payload_in_graphql_env": False,
        "job_tarball_in_create": False,
        "directions_on_volume": DIRECTIONS_ON_VOLUME,
        "direction_files_on_volume": direction_files_for(args),
        "pairs_on_volume": pairs_on_volume,
        "graphql_post_body_bytes": post_bytes,
        "graphql_post_body_limit": MAX_CREATE_BODY_BYTES,
        "boot_b64_len": len(boot_b64),
        "output_on_volume": f"/workspace/jobs/narcbench-results/{vol_subdir}/smoke_*",
        "status_port": 8765,
        "cost_gate_usd": args.cost_max,
        "balance_min_usd": args.balance_min,
        "launch_command": (
            f"python3 scripts/runpod_launch_matched_prefix.py --suite {args.suite} --launch "
            f"--max-pairs {args.max_pairs}"
            + (
                " --role-perp-confirm"
                if args.role_perp_confirm
                else (" --extras-only" if args.extras_only else (" --extras" if args.extras else ""))
            )
        ),
        "dry_run_command": (
            f"python3 scripts/runpod_launch_matched_prefix.py --suite {args.suite}"
            + (
                " --role-perp-confirm"
                if args.role_perp_confirm
                else (" --extras-only" if args.extras_only else (" --extras" if args.extras else ""))
            )
        ),
        "harness_boot_argv": (
            f"python3 scripts/matched_prefix_interchange.py --pairs {pairs_on_volume} "
            f"--directions {DIRECTIONS_ON_VOLUME} --max-pairs {args.max_pairs} "
            f"--model {args.model} --max-new-tokens {args.max_new_tokens} --seed {args.seed}"
            + (
                " --role-perp-confirm"
                if args.role_perp_confirm
                else (" --extras-only" if args.extras_only else (" --extras" if args.extras else ""))
            )
        ),
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
        print(f"REFUSE: GraphQL POST body {post_bytes} >= {MAX_CREATE_BODY_BYTES}", file=sys.stderr)
        return 4
    if not args.launch:
        print("\n[estimate-only] NOT launching. Re-run with --launch when ready.", file=sys.stderr)
        return 0

    if not harness_in_sha:
        print(
            f"REFUSE launch: {HARNESS_REL} is not in git SHA {git_sha}. "
            "Experiment code ships only through this repo.",
            file=sys.stderr,
        )
        return 1
    if dirty:
        print("REFUSE launch: worktree is dirty; commit and push the SHA first", file=sys.stderr)
        return 1
    if not remote_has_sha(git_sha, git_ref):
        print("REFUSE launch: SHA is not on the public remote branch yet", file=sys.stderr)
        return 1
    if est["cost_est_usd"] is None:
        print(
            "[warn] cost gate not applied: this machine has no local pairs file to count. "
            "The pod scores every row of the volume pairs file.",
            file=sys.stderr,
        )
    elif est["cost_est_usd"] > args.cost_max:
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

    if balance is not None and float(balance) < args.balance_min:
        print(f"REFUSE launch: balance ${balance} < ${args.balance_min}", file=sys.stderr)
        card = {
            "launched": False,
            "reason": "balance_below_min",
            "balance_usd": balance,
            "git_sha": git_sha,
            "graphql_post_body_bytes": post_bytes,
            "estimate": est,
        }
        args.launch_card.write_text(json.dumps(card, indent=2) + "\n")
        return 3

    busy = billable_gpu_pods(list_pods())
    if busy:
        print(f"REFUSE launch: billable GPU already up: {busy}", file=sys.stderr)
        card = {
            "launched": False,
            "reason": "billable_gpu_already_running",
            "pods": busy,
            "balance_usd": balance,
            "git_sha": git_sha,
            "graphql_post_body_bytes": post_bytes,
            "estimate": est,
        }
        args.launch_card.write_text(json.dumps(card, indent=2) + "\n")
        return 5

    last: Exception | None = None
    for cloud, gpu in GPU_TRIES:
        inp = build_create_input(
            boot_b64,
            pod_name=pod_name,
            gpu=gpu,
            cloud=cloud,
            git_sha=git_sha,
            git_ref=git_ref,
        )
        nbytes = body_bytes(CREATE_MUTATION, {"input": inp})
        if nbytes >= MAX_CREATE_BODY_BYTES:
            print(f"REFUSE POST: body {nbytes}", file=sys.stderr)
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
                "pod_name": pod_name,
                "gpu": gpu,
                "cloud": cloud,
                "cost_per_hr": pod.get("costPerHr") or RATES.get(gpu, est["rate_usd_hr"]),
                "started_pt": started,
                "kind": est["kind"],
                "n_pairs": n_pairs,
                "arms": arms_list,
                "status_url": status_url,
                "hours_est": est["hours_est"],
                "estimate_usd": est["cost_est_usd"],
                "balance_usd": balance,
                "spend_per_hr_at_launch": spend,
                "volume_id": VOLUME_ID,
                "git_repo_url": REPO_URL,
                "git_sha": git_sha,
                "git_ref": git_ref,
                "payload_delivery": "git_checkout_sha",
                "payload_in_graphql_env": False,
                "job_tarball_in_create": False,
                "graphql_post_body_bytes": nbytes,
                "output_on_volume": f"/workspace/jobs/narcbench-results/{vol_subdir}/smoke_*",
                "soft_stop": "Track I / Paper 2 / interp-demo / family-house NOT touched",
                "machine": pod.get("machine") or {},
            }
            args.launch_card.write_text(json.dumps(card, indent=2) + "\n")
            print(json.dumps(card, indent=2))
            print(f"Status: {status_url}")
            return 0
        except GraphqlHttpError as exc:
            last = exc
            print(f"[launch] {cloud} {gpu}: HTTP {exc.status}", file=sys.stderr)
            if exc.is_cloudflare_body_reject:
                print("[launch] Cloudflare body-size reject — STOPPING GPU walk", file=sys.stderr)
                fail = {
                    "launched": False,
                    "reason": "cloudflare_body_size_reject",
                    "http_status": exc.status,
                    "graphql_post_body_bytes": nbytes,
                    "git_sha": git_sha,
                    "estimate": est,
                }
                args.launch_card.write_text(json.dumps(fail, indent=2) + "\n")
                return 1
            time.sleep(1.0)
        except Exception as exc:
            last = exc
            print(f"[launch] {cloud} {gpu}: {type(exc).__name__}", file=sys.stderr)
            time.sleep(1.0)

    print(f"all failed: {type(last).__name__ if last else 'unknown'}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
