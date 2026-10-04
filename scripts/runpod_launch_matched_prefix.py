#!/usr/bin/env python3
"""Estimate or launch matched-prefix activation interchange on RunPod.

Estimate-only by default. Pass --launch only after the cost gate is approved:

  python3 scripts/runpod_launch_matched_prefix.py
  python3 scripts/runpod_launch_matched_prefix.py --launch --max-pairs 10
  python3 scripts/runpod_launch_matched_prefix.py --suite transfer --max-pairs 12

Boot clones this public repo and checks out a public SHA, then runs
scripts/runpod_matched_prefix_boot.sh from that checkout. The GraphQL create
env carries the SHA and volume paths only. It does not carry a boot script,
JOB_BOOT_B64, JOB_PAYLOAD_B64, or a job tarball. --extras-only runs the
multi-site, PCA, and role-perpendicular kitchen sink. --role-perp-confirm
runs the residual-L21 role-perp card on every row of the volume pairs file.
--final-resid-controls runs the pre-logit residual copies on the first 12
core pairs and does not read direction files. Direction .npy files and pairs
stay on the network volume. RUNPOD_API_KEY and HF_TOKEN come from the
environment. --launch refuses unless the harness and boot script are in the
checked-out SHA.
Soft-stop: no Track I / Paper 2 / interp-demo / family-house.
"""
from __future__ import annotations

import argparse
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
BOOT_REL = "scripts/runpod_matched_prefix_boot.sh"
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
    "patch_h2c_full_resid_L21",
    "patch_c2h_full_resid_L21",
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
# Pre-logit residual copies. Not the role-perp confirm list.
FINAL_RESID_ARMS = [
    "baseline_colluder",
    "baseline_honest",
    "patch_h2c_full_final_last",
    "patch_c2h_full_final_last",
]
FINAL_RESID_CONDITIONAL_ARMS = [
    "patch_h2c_full_final_private",
    "patch_c2h_full_final_private",
]
FINAL_RESID_CORE_PAIRS = 12
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


def mode_cli(args: argparse.Namespace) -> str:
    if getattr(args, "final_resid_controls", False):
        return " --final-resid-controls"
    if args.role_perp_confirm:
        return " --role-perp-confirm"
    if args.extras_only:
        return " --extras-only"
    if args.extras:
        return " --extras"
    return ""


def arms_for(args: argparse.Namespace) -> list[str]:
    if args.final_resid_controls:
        return list(FINAL_RESID_ARMS)
    if args.role_perp_confirm:
        return list(ROLE_PERP_CONFIRM_ARMS)
    if args.extras_only:
        return list(EXTRAS_ONLY_ARMS)
    if args.extras:
        extra = [a for a in EXTRAS_ONLY_ARMS if a not in ("baseline_colluder", "baseline_honest")]
        return list(CORE_ARMS) + extra
    return list(CORE_ARMS)


def direction_files_for(args: argparse.Namespace) -> list[str]:
    if args.final_resid_controls:
        return []
    needed = list(CORE_DIRECTION_FILES)
    if args.extras or args.extras_only or args.role_perp_confirm:
        needed.extend(EXTRAS_DIRECTION_FILES)
    return needed


def mode_name(args: argparse.Namespace) -> str:
    if getattr(args, "final_resid_controls", False):
        return "final-resid-controls"
    if args.role_perp_confirm:
        return "role-perp-confirm"
    if args.extras_only:
        return "extras-only"
    if args.extras:
        return "extras"
    return "core"


# Short bootstrap only. The smoke itself lives in scripts/runpod_matched_prefix_boot.sh
# at the pinned SHA, so script source is not placed in the GraphQL env.
DOCKER_ARGS = (
    "bash -c '"
    "set -euo pipefail; "
    "export DEBIAN_FRONTEND=noninteractive; "
    "if ! command -v git >/dev/null 2>&1; then "
    "apt-get update -qq && apt-get install -y -qq git ca-certificates; "
    "fi; "
    ': "${JOB_GIT_SHA:?}"; '
    ': "${JOB_REPO_URL:?}"; '
    'if [ -n "${JOB_IMAGE_REPO:-}" ]; then '
    'REPO="$JOB_IMAGE_REPO"; '
    'GOT=$(git -C "$REPO" rev-parse HEAD); '
    'if [ "$GOT" != "$JOB_GIT_SHA" ]; then echo "image pin $GOT != $JOB_GIT_SHA" >&2; exit 2; fi; '
    "else "
    "REPO=/tmp/narcbench-core-demo; "
    'rm -rf "$REPO"; '
    'git init "$REPO"; '
    'git -C "$REPO" remote add origin "$JOB_REPO_URL"; '
    'if ! git -C "$REPO" fetch --depth 1 origin "$JOB_GIT_SHA"; then '
    'rm -rf "$REPO"; '
    'git clone "$JOB_REPO_URL" "$REPO"; '
    'git -C "$REPO" checkout --detach "$JOB_GIT_SHA"; '
    "else "
    'git -C "$REPO" checkout --detach FETCH_HEAD; '
    "fi; "
    'GOT=$(git -C "$REPO" rev-parse HEAD); '
    'if [ "$GOT" != "$JOB_GIT_SHA" ]; then echo "clone pin $GOT != $JOB_GIT_SHA" >&2; exit 2; fi; '
    "fi; "
    'exec bash "$REPO/scripts/runpod_matched_prefix_boot.sh"'
    "'"
)

MAX_ENV_VALUE_CHARS = 4096
MAX_DOCKER_ARGS_CHARS = 8192
ALLOWED_ENV_KEYS = frozenset(
    {
        "JOB_GIT_SHA",
        "JOB_REPO_URL",
        "JOB_IMAGE_REPO",
        "JOB_PAIRS",
        "JOB_DIRECTIONS",
        "JOB_OUT_ROOT",
        "JOB_MODE",
        "JOB_MAX_PAIRS",
        "JOB_MODEL",
        "JOB_MAX_NEW_TOKENS",
        "JOB_SEED",
        "JOB_MAX_MINUTES",
        "JOB_GRACE_MINUTES",
        "HF_TOKEN",
        "HF_HOME",
        "RUNPOD_API_KEY",
    }
)
FORBIDDEN_MARKERS = (
    "JOB_PAYLOAD_B64",
    "JOB_BOOT_B64",
    "JOB_TGZ_B64",
    "JOB_TGZ_NAME",
    "JOB_TGZ_SHA256",
)
REPO_URL_RE = re.compile(
    r"^https://github.com/andylikescodes/narcbench-core-demo(\.git)?$"
)
MODES = frozenset({"core", "extras", "extras-only", "role-perp-confirm", "final-resid-controls"})


class LaunchRefused(Exception):
    def __init__(self, reason: str, code: int = 4):
        super().__init__(reason)
        self.reason = reason
        self.code = code


def assert_volume_path(path: str, label: str) -> str:
    if not path.startswith("/workspace/") or ".." in path.split("/"):
        raise LaunchRefused(f"{label} must be an absolute path on the network volume (/workspace/...)")
    if path == "/workspace/interp-demo" or path.startswith("/workspace/interp-demo/"):
        raise LaunchRefused(f"{label} must not use /workspace/interp-demo")
    return path


def _looks_like_tarball(value: str) -> bool:
    if value.startswith("\x1f\x8b") or "ustar" in value[:600]:
        return True
    stripped = value.strip()
    if stripped.startswith("H4sI") and len(stripped) > 64:
        return True
    return False


def validate_create_input(create_input: dict) -> int:
    """Refuse a create body that carries a boot script, payload tarball, or script source."""
    env_pairs = create_input.get("env") or []
    env = {}
    for item in env_pairs:
        key = str(item.get("key") or "")
        value = "" if item.get("value") is None else str(item.get("value"))
        if key in env:
            raise LaunchRefused(f"duplicate create env key {key}")
        env[key] = value
        if key not in ALLOWED_ENV_KEYS:
            raise LaunchRefused(f"create env key not allowed: {key}")
        upper = key.upper()
        if "PAYLOAD" in upper or "TARBALL" in upper or upper.endswith("TGZ") or upper.endswith("_B64"):
            raise LaunchRefused(f"refusing payload-like create env key {key}")
        if len(value) > MAX_ENV_VALUE_CHARS:
            raise LaunchRefused(f"create env {key} is {len(value)} chars; script source stays out of GraphQL env")
        if _looks_like_tarball(value):
            raise LaunchRefused(f"create env {key} looks like a payload tarball")
        if value.startswith("#!") or "def run_scenario" in value or "make_dir_patch_hook" in value:
            raise LaunchRefused(f"create env {key} contains script source")
    if not SHA_RE.fullmatch(env.get("JOB_GIT_SHA", "")):
        raise LaunchRefused("JOB_GIT_SHA must be a 40-hex pin")
    if not REPO_URL_RE.fullmatch(env.get("JOB_REPO_URL", "")):
        raise LaunchRefused("JOB_REPO_URL must be https://github.com/andylikescodes/narcbench-core-demo")
    if env.get("JOB_MODE") not in MODES:
        raise LaunchRefused("JOB_MODE is not a known matched-prefix suite")
    for label in ("JOB_PAIRS", "JOB_DIRECTIONS", "JOB_OUT_ROOT", "HF_HOME"):
        if label not in env:
            raise LaunchRefused(f"create env missing {label}")
        assert_volume_path(env[label], label)
    docker_args = str(create_input.get("dockerArgs") or "")
    if len(docker_args) > MAX_DOCKER_ARGS_CHARS:
        raise LaunchRefused("dockerArgs embeds the job; boot the pinned checkout instead")
    if "base64 -d" in docker_args or "JOB_BOOT_B64" in docker_args:
        raise LaunchRefused("dockerArgs must not decode an embedded boot script")
    blob = json.dumps({"env": env_pairs, "dockerArgs": docker_args})
    for marker in FORBIDDEN_MARKERS:
        if marker in blob:
            raise LaunchRefused(f"refusing create body that contains {marker}")
    if "JOB_GIT_SHA" not in docker_args or "runpod_matched_prefix_boot.sh" not in docker_args:
        raise LaunchRefused("dockerArgs must pin JOB_GIT_SHA and exec the in-repo boot script")
    nbytes = body_bytes(CREATE_MUTATION, {"input": create_input})
    if nbytes >= MAX_CREATE_BODY_BYTES:
        raise LaunchRefused(f"GraphQL POST body {nbytes} >= {MAX_CREATE_BODY_BYTES}")
    return nbytes


def build_create_input(args: argparse.Namespace, api_key: str) -> dict:
    env = {
        "JOB_GIT_SHA": args.git_sha,
        "JOB_REPO_URL": REPO_URL,
        "JOB_PAIRS": args.pairs_on_volume,
        "JOB_DIRECTIONS": DIRECTIONS_ON_VOLUME,
        "JOB_OUT_ROOT": args.volume_out,
        "JOB_MODE": args.mode_name,
        "JOB_MAX_PAIRS": str(args.max_pairs),
        "JOB_MODEL": args.model,
        "JOB_MAX_NEW_TOKENS": str(args.max_new_tokens),
        "JOB_SEED": str(args.seed),
        "JOB_MAX_MINUTES": str(args.max_minutes),
        "JOB_GRACE_MINUTES": str(args.grace_minutes),
        "HF_HOME": HF_CACHE_ON_VOLUME,
        "RUNPOD_API_KEY": api_key,
    }
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        env["HF_TOKEN"] = hf_token
    create_input = {
        "name": args.pod_name,
        "imageName": IMAGE,
        "gpuTypeId": args.gpu,
        "cloudType": args.cloud,
        "gpuCount": 1,
        "volumeInGb": 0,
        "containerDiskInGb": 40,
        "minVcpuCount": 4,
        "minMemoryInGb": 32,
        "ports": "8765/http",
        "env": [{"key": key, "value": str(value)} for key, value in env.items()],
        "dockerArgs": DOCKER_ARGS,
        "supportPublicIp": False,
        "startSsh": False,
        "networkVolumeId": VOLUME_ID,
        "volumeMountPath": "/workspace",
    }
    validate_create_input(create_input)
    return create_input


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
    ap.add_argument(
        "--final-resid-controls",
        action="store_true",
        help="boot the pre-logit residual controls on the first 12 core pairs",
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
    mode_flags = (
        int(bool(args.extras))
        + int(bool(args.extras_only))
        + int(bool(args.role_perp_confirm))
        + int(bool(args.final_resid_controls))
    )
    if mode_flags > 1:
        print(
            "ERROR: pass only one of --extras, --extras-only, --role-perp-confirm, --final-resid-controls",
            file=sys.stderr,
        )
        return 1
    if args.final_resid_controls and args.suite != "core":
        print("ERROR: --final-resid-controls scores the 12-pair core set", file=sys.stderr)
        return 1
    if args.final_resid_controls:
        args.max_pairs = FINAL_RESID_CORE_PAIRS if args.max_pairs is None else args.max_pairs
        if args.max_pairs != FINAL_RESID_CORE_PAIRS:
            print(
                f"ERROR: --final-resid-controls scores {FINAL_RESID_CORE_PAIRS} core pairs",
                file=sys.stderr,
            )
            return 1
    elif args.role_perp_confirm:
        args.max_pairs = 0 if args.max_pairs is None else args.max_pairs
    elif args.max_pairs is None:
        args.max_pairs = 10
    if args.max_pairs < 0 or (args.max_pairs == 0 and not args.role_perp_confirm):
        print("ERROR: --max-pairs 0 is only valid with --role-perp-confirm", file=sys.stderr)
        return 1

    meta = SUITE_META[args.suite]
    if args.estimate_out is None:
        if args.final_resid_controls:
            args.estimate_out = ROOT / "results/transfer_stable/MATCHED_PREFIX_FINAL_RESID_ESTIMATE.json"
        elif args.role_perp_confirm:
            args.estimate_out = ROOT / "results/transfer_stable/MATCHED_PREFIX_ROLE_PERP_CONFIRM_ESTIMATE.json"
        elif args.extras or args.extras_only:
            args.estimate_out = ROOT / "results/transfer_stable/MATCHED_PREFIX_CHEAP_EXTRAS_ESTIMATE.json"
        else:
            args.estimate_out = meta["estimate_out"]
    if args.launch_card is None:
        if args.final_resid_controls:
            args.launch_card = ROOT / "results/transfer_stable/MATCHED_PREFIX_FINAL_RESID_LAUNCH.json"
        elif args.role_perp_confirm:
            args.launch_card = ROOT / "results/transfer_stable/MATCHED_PREFIX_ROLE_PERP_CONFIRM_LAUNCH.json"
        elif args.extras or args.extras_only:
            args.launch_card = ROOT / "results/transfer_stable/MATCHED_PREFIX_CHEAP_EXTRAS_LAUNCH.json"
        else:
            args.launch_card = meta["launch_out"]

    vol_subdir = meta["vol_subdir"]
    pod_name = meta["pod_name"]
    pairs_on_volume = meta["pairs_on_volume"]
    if args.final_resid_controls:
        vol_subdir = f"{vol_subdir}_final_resid"
        pod_name = f"{pod_name}-final-resid"
    elif args.role_perp_confirm:
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
    if args.final_resid_controls:
        est["kind"] = est["kind"].replace("_smoke", "_final_resid_controls")
        est["final_resid_controls"] = True
        est["conditional_arms"] = list(FINAL_RESID_CONDITIONAL_ARMS)
        est["n_pairs_cap"] = FINAL_RESID_CORE_PAIRS
    elif args.role_perp_confirm:
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
    boot_in_sha = blob_in_commit(git_sha, BOOT_REL)
    dirty = worktree_dirty()
    args.git_sha = git_sha
    args.pod_name = pod_name
    args.pairs_on_volume = pairs_on_volume
    args.volume_out = f"/workspace/jobs/narcbench-results/{vol_subdir}"
    args.mode_name = mode_name(args)
    args.gpu = GPU
    args.cloud = "SECURE"
    try:
        assert_volume_path(args.pairs_on_volume, "pairs")
        assert_volume_path(args.volume_out, "volume_out")
        assert_volume_path(DIRECTIONS_ON_VOLUME, "directions")
        assert_volume_path(HF_CACHE_ON_VOLUME, "hf_home")
    except LaunchRefused as exc:
        print(f"REFUSE: {exc.reason}", file=sys.stderr)
        return exc.code
    real_key = os.environ.get("RUNPOD_API_KEY") or ""
    measure_key = real_key or ("0" * 64)
    try:
        sample = build_create_input(args, measure_key)
        post_bytes = validate_create_input(sample)
    except LaunchRefused as exc:
        print(f"REFUSE: {exc.reason}", file=sys.stderr)
        return exc.code

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
        "boot_script": BOOT_REL,
        "harness_in_sha": harness_in_sha,
        "boot_script_in_sha": boot_in_sha,
        "worktree_dirty": dirty,
        "code_source": "git_clone_sha",
        "payload_delivery": "git_checkout_sha",
        "payload_in_graphql_env": False,
        "job_boot_b64_in_create_env": False,
        "job_payload_b64_in_create_env": False,
        "script_source_in_create_env": False,
        "job_tarball_in_create": False,
        "job_mode": args.mode_name,
        "directions_on_volume": DIRECTIONS_ON_VOLUME,
        "direction_files_on_volume": direction_files_for(args),
        "pairs_on_volume": pairs_on_volume,
        "graphql_post_body_bytes": post_bytes,
        "graphql_post_body_limit": MAX_CREATE_BODY_BYTES,
        "output_on_volume": f"/workspace/jobs/narcbench-results/{vol_subdir}/smoke_*",
        "status_port": 8765,
        "cost_gate_usd": args.cost_max,
        "balance_min_usd": args.balance_min,
        "conditional_arms": list(FINAL_RESID_CONDITIONAL_ARMS) if args.final_resid_controls else [],
        "launch_command": (
            f"python3 scripts/runpod_launch_matched_prefix.py --suite {args.suite} --launch "
            f"--max-pairs {args.max_pairs}{mode_cli(args)}"
        ),
        "dry_run_command": (
            f"python3 scripts/runpod_launch_matched_prefix.py --suite {args.suite}{mode_cli(args)}"
        ),
        "harness_boot_argv": (
            f"python3 scripts/matched_prefix_interchange.py --pairs {pairs_on_volume} "
            + ("" if args.final_resid_controls else f"--directions {DIRECTIONS_ON_VOLUME} ")
            + f"--max-pairs {args.max_pairs} "
            f"--model {args.model} --max-new-tokens {args.max_new_tokens} --seed {args.seed}"
            + mode_cli(args)
        ),
        "soft_stop": "Track I / Paper 2 / interp-demo / family-house NOT touched; no GPU unless --launch",
    }
    print(json.dumps(report, indent=2))
    args.estimate_out.parent.mkdir(parents=True, exist_ok=True)
    args.estimate_out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"\n[wrote] {args.estimate_out}", file=sys.stderr)
    print(f"[pin] sha={git_sha} mode={args.mode_name} post_body={post_bytes}", file=sys.stderr)

    if post_bytes >= MAX_CREATE_BODY_BYTES:
        print(f"REFUSE: GraphQL POST body {post_bytes} >= {MAX_CREATE_BODY_BYTES}", file=sys.stderr)
        return 4
    if not args.launch:
        print("\n[estimate-only] NOT launching. Re-run with --launch when ready.", file=sys.stderr)
        return 0

    if not harness_in_sha or not boot_in_sha:
        print(
            f"REFUSE launch: {HARNESS_REL} or {BOOT_REL} is not in git SHA {git_sha}. "
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
        args.gpu = gpu
        args.cloud = cloud
        try:
            inp = build_create_input(args, os.environ["RUNPOD_API_KEY"])
            nbytes = validate_create_input(inp)
        except LaunchRefused as exc:
            print(f"REFUSE: {exc.reason}", file=sys.stderr)
            return exc.code
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
