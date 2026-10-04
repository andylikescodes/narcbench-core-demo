#!/usr/bin/env python3
"""Estimate or launch the Core within-trajectory mid-private RunPod smoke.

Estimate-only by default. Pass --launch only after the cost gate is approved.
The pod boots by cloning this repo and checking out --git-sha, or by using an
image that already contains that checkout. Scenario transcripts and direction
.npy files stay on the network volume. The GraphQL create env carries the pin
and those paths only — never a boot script, JOB_BOOT_B64, JOB_PAYLOAD_B64, or a tarball.
RUNPOD_API_KEY is read from the environment. This file does not load secret files.

  python3 scripts/runpod_launch_within_traj.py --git-sha <40-hex>
  python3 scripts/runpod_launch_within_traj.py --launch --git-sha <40-hex>

Soft-stop: no Track I / Paper 2 / interp-demo edits. This script does not
launch a pod unless --launch is passed.
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
HARNESS_REL = "scripts/within_traj_mid_private.py"
BOOT_REL = "scripts/runpod_within_traj_boot.sh"

REPO_URL = "https://github.com/andylikescodes/narcbench-core-demo.git"
IMAGE = "runpod/pytorch:1.2.0-rc.162-cu1281-torch280-ubuntu2404"
VOLUME_ID = "cr1kjtzjs2"
POD_NAME = "job-narcbench-within-traj-mid-private"
# EU-RO-1 (volume pin): L4 SECURE has stock; 4090 has stock.
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

# Cloudflare rejects GraphQL bodies around 140KB. ~100KB has succeeded.
# Stay at or under 100KB and keep this job far smaller than that.
MAX_CREATE_BODY_BYTES = 100_000
MAX_ENV_VALUE_CHARS = 4096
MAX_DOCKER_ARGS_CHARS = 8192

DEFAULT_VOLUME_RUN = "/workspace/jobs/narcbench-data/gemma2_9b/core/20261001T012639Z"
DEFAULT_VOLUME_DIRECTIONS = "/workspace/jobs/narcbench-data/transfer_stable/directions"
DEFAULT_VOLUME_OUT = "/workspace/jobs/narcbench-results/within_traj_mid_private"
DEFAULT_HF_HOME = "/workspace/jobs/narcbench-hf"
ESTIMATE_OUT = ROOT / "results/transfer_stable/WITHIN_TRAJ_MID_PRIVATE_ESTIMATE.json"
LAUNCH_OUT = ROOT / "results/transfer_stable/WITHIN_TRAJ_MID_PRIVATE_LAUNCH.json"

ARMS = [
    "baseline_tf",
    "ablate_role_attn_L22",
    "ablate_role_resid_L21",
    "ablate_role_perp_resid_L21",
    "ablate_random_attn_L22",
    "steer_role_attn_L22",
]

# Short bootstrap only. The smoke itself lives in scripts/runpod_within_traj_boot.sh
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
    'exec bash "$REPO/scripts/runpod_within_traj_boot.sh"'
    "'"
)

CREATE_MUTATION = """mutation ($input: PodFindAndDeployOnDemandInput!) {
  podFindAndDeployOnDemand(input: $input) {
    id desiredStatus costPerHr machine { gpuDisplayName dataCenterId }
  }
}"""

ALLOWED_ENV_KEYS = frozenset(
    {
        "JOB_GIT_SHA",
        "JOB_REPO_URL",
        "JOB_IMAGE_REPO",
        "JOB_RUN_DIR",
        "JOB_DIRECTIONS",
        "JOB_OUT_ROOT",
        "JOB_MODEL",
        "JOB_MAX_SCENARIOS",
        "JOB_K_FRAC",
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
SHA_RE = re.compile(r"^[0-9a-f]{40}$")
REPO_URL_RE = re.compile(
    r"^https://github.com/andylikescodes/narcbench-core-demo(\.git)?$"
)


class LaunchRefused(Exception):
    def __init__(self, reason: str, code: int = 4):
        super().__init__(reason)
        self.reason = reason
        self.code = code


class GraphqlHttpError(Exception):
    def __init__(self, status: int, body: str):
        self.status = status
        self.body = body
        super().__init__(f"HTTP {status}: {(body or '')[:400]}")

    @property
    def is_cloudflare_body_reject(self) -> bool:
        text = (self.body or "").lower()
        if self.status == 413:
            return True
        return "cloudflare" in text and (
            "too large" in text or "payload" in text or "413" in text
        )


def git_identity() -> tuple[str, str]:
    sha = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    ref = subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-parse", "--abbrev-ref", "HEAD"], text=True
    ).strip()
    if ref == "HEAD":
        ref = "main"
    return sha, ref


def worktree_dirty() -> bool:
    out = subprocess.check_output(["git", "-C", str(ROOT), "status", "--porcelain"], text=True)
    return bool(out.strip())


def blob_in_commit(sha: str, rel: str) -> bool:
    proc = subprocess.run(
        ["git", "-C", str(ROOT), "cat-file", "-e", f"{sha}:{rel}"],
        capture_output=True,
    )
    return proc.returncode == 0


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


def assert_option_labels_fix(path: Path) -> None:
    """Refuse a harness that calls .keys() on a list (or on an unguarded value)."""
    text = path.read_text()
    start = text.find("def option_label_items")
    if start < 0:
        raise LaunchRefused("option_label_items missing from harness", code=1)
    end = text.find("\ndef ", start + 1)
    body = text[start:end if end > 0 else None]
    if "isinstance(raw, dict)" not in body or "isinstance(raw, (list, tuple))" not in body:
        raise LaunchRefused("option_labels list-or-dict guard missing", code=1)
    dict_branch = False
    for line in body.splitlines():
        stripped = line.strip()
        if ".keys()" in stripped and not dict_branch:
            raise LaunchRefused(".keys() used outside the dict branch of option_label_items", code=1)
        if stripped.startswith("if isinstance(raw, dict)"):
            dict_branch = True
        elif stripped.startswith(("if ", "elif ", "else", "def ")):
            dict_branch = False


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
        "code_source": "git_clone_sha",
    }


def _reject_shell_chars(text: str, label: str) -> None:
    if any(ch in text for ch in ("\n", "\r", "'", '"', "$", "`")):
        raise LaunchRefused(f"{label} has characters that cannot go in the pod env")
    if len(text) > 512:
        raise LaunchRefused(f"{label} is too long")


def assert_volume_path(path: str, label: str) -> str:
    text = str(path or "").strip()
    if not text.startswith("/workspace/"):
        raise LaunchRefused(f"{label} must be an absolute path on the network volume (/workspace/...)")
    if any(part == ".." for part in text.split("/")):
        raise LaunchRefused(f"{label} must not contain ..")
    if text == "/workspace/interp-demo" or text.startswith("/workspace/interp-demo/"):
        raise LaunchRefused(f"{label} must not use /workspace/interp-demo")
    _reject_shell_chars(text, label)
    return text


def assert_checkout_path(path: str, label: str) -> str:
    """Path of a repo already baked into an image. Not required to sit on the volume."""
    text = str(path or "").strip()
    if not text.startswith("/") or text.startswith("//"):
        raise LaunchRefused(f"{label} must be an absolute path inside the image")
    if any(part == ".." for part in text.split("/")):
        raise LaunchRefused(f"{label} must not contain ..")
    if text == "/workspace/interp-demo" or text.startswith("/workspace/interp-demo/"):
        raise LaunchRefused(f"{label} must not use /workspace/interp-demo")
    _reject_shell_chars(text, label)
    return text


def body_bytes(query: str, variables: dict) -> int:
    payload = {"query": query, "variables": variables}
    return len(json.dumps(payload, separators=(",", ":")).encode())


def graphql(query: str, variables: dict, timeout: int = 60) -> dict:
    key = os.environ.get("RUNPOD_API_KEY") or ""
    if not key:
        raise LaunchRefused("RUNPOD_API_KEY missing", code=1)
    body = json.dumps({"query": query, "variables": variables}).encode()
    req = urllib.request.Request(
        "https://api.runpod.io/graphql",
        data=body,
        headers={
            "content-type": "application/json",
            "Authorization": "Bearer " + key,
            "User-Agent": "nb-within-traj-git/1",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return json.loads(response.read().decode())
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode(errors="replace")
        raise GraphqlHttpError(exc.code, raw) from exc


def list_pods() -> list:
    data = graphql(
        "query { myself { pods { id name desiredStatus costPerHr gpuCount machine { gpuDisplayName } } } }",
        {},
        timeout=30,
    )
    if data.get("errors"):
        raise RuntimeError(str(data["errors"])[:400])
    return ((data.get("data") or {}).get("myself") or {}).get("pods") or []


def billable_gpu_pods(pods: list) -> list:
    busy = []
    for pod in pods or []:
        status = str(pod.get("desiredStatus") or "").upper()
        if status in {"EXITED", "TERMINATED", "DEAD"}:
            continue
        machine = pod.get("machine") or {}
        gpu = machine.get("gpuDisplayName") or ""
        gpu_count = pod.get("gpuCount") or 0
        cost = pod.get("costPerHr") or 0
        if not gpu and not gpu_count:
            continue
        try:
            cost_val = float(cost or 0)
        except (TypeError, ValueError):
            cost_val = 0.0
        if cost_val > 0 or status == "RUNNING":
            busy.append(
                {
                    "id": pod.get("id"),
                    "name": pod.get("name"),
                    "desiredStatus": status,
                    "gpu": gpu,
                    "costPerHr": cost,
                }
            )
    return busy


def terminate_pod(pod_id: str) -> dict:
    query = 'mutation { podTerminate(input: {podId: "%s"}) { id desiredStatus } }' % pod_id
    return graphql(query, {}, timeout=30)


def _looks_like_tarball(value: str) -> bool:
    if value.startswith("\x1f\x8b") or "ustar" in value[:600]:
        return True
    stripped = value.strip()
    if stripped.startswith("H4sI") and len(stripped) > 64:
        # gzip payload, base64 (typical tarball prefix)
        return True
    return False


def validate_create_input(create_input: dict) -> int:
    """Refuse a create body that would carry a payload tarball or script source.

    Returns the GraphQL POST body size in bytes.
    """
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
            raise LaunchRefused(
                f"create env {key} is {len(value)} chars; script source and tarballs stay out of GraphQL env"
            )
        if _looks_like_tarball(value):
            raise LaunchRefused(f"create env {key} looks like a payload tarball")
        if value.startswith("#!") or "def run_scenario" in value or "make_persist_ablate_hook" in value:
            raise LaunchRefused(f"create env {key} contains script source")

    if not SHA_RE.fullmatch(env.get("JOB_GIT_SHA", "")):
        raise LaunchRefused("JOB_GIT_SHA must be a 40-hex pin")
    repo_url = env.get("JOB_REPO_URL", "")
    if not REPO_URL_RE.fullmatch(repo_url):
        raise LaunchRefused("JOB_REPO_URL must be https://github.com/andylikescodes/narcbench-core-demo")
    for label in ("JOB_RUN_DIR", "JOB_DIRECTIONS", "JOB_OUT_ROOT", "HF_HOME"):
        if label not in env:
            raise LaunchRefused(f"create env missing {label}")
        assert_volume_path(env[label], label)
    if env.get("JOB_IMAGE_REPO"):
        assert_checkout_path(env["JOB_IMAGE_REPO"], "JOB_IMAGE_REPO")

    docker_args = str(create_input.get("dockerArgs") or "")
    if len(docker_args) > MAX_DOCKER_ARGS_CHARS:
        raise LaunchRefused(
            f"dockerArgs is {len(docker_args)} chars; boot the pinned checkout instead of embedding the job"
        )
    if "base64 -d" in docker_args or "JOB_BOOT_B64" in docker_args:
        raise LaunchRefused("dockerArgs must not decode an embedded boot script")

    blob = json.dumps({"env": env_pairs, "dockerArgs": docker_args})
    for marker in FORBIDDEN_MARKERS:
        if marker in blob or marker in env:
            raise LaunchRefused(f"refusing create body that contains {marker}")

    if "git" not in docker_args or "JOB_GIT_SHA" not in docker_args:
        raise LaunchRefused("dockerArgs must pin the checkout with JOB_GIT_SHA")
    if "runpod_within_traj_boot.sh" not in docker_args:
        raise LaunchRefused("dockerArgs must exec the in-repo boot script")
    if "narcbench-core-demo" not in docker_args and "JOB_REPO_URL" not in docker_args:
        raise LaunchRefused("dockerArgs must clone andylikescodes/narcbench-core-demo")

    nbytes = body_bytes(CREATE_MUTATION, {"input": create_input})
    if nbytes >= MAX_CREATE_BODY_BYTES:
        raise LaunchRefused(
            f"GraphQL POST body {nbytes} >= {MAX_CREATE_BODY_BYTES}"
        )
    return nbytes


def build_create_input(args: argparse.Namespace, api_key: str) -> dict:
    env = {
        "JOB_GIT_SHA": args.git_sha,
        "JOB_REPO_URL": args.repo_url,
        "JOB_RUN_DIR": args.volume_run_dir,
        "JOB_DIRECTIONS": args.volume_directions,
        "JOB_OUT_ROOT": args.volume_out,
        "JOB_MODEL": args.model,
        "JOB_MAX_SCENARIOS": str(args.max_scenarios),
        "JOB_K_FRAC": str(args.k_frac),
        "JOB_MAX_NEW_TOKENS": str(args.max_new_tokens),
        "JOB_SEED": str(args.seed),
        "JOB_MAX_MINUTES": str(args.max_minutes),
        "JOB_GRACE_MINUTES": str(args.grace_minutes),
        "HF_HOME": args.hf_home,
        "RUNPOD_API_KEY": api_key,
    }
    if args.image_repo:
        env["JOB_IMAGE_REPO"] = args.image_repo
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        env["HF_TOKEN"] = hf_token
    create_input = {
        "name": POD_NAME,
        "imageName": args.image,
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
        "networkVolumeId": args.volume_id,
        "volumeMountPath": "/workspace",
    }
    validate_create_input(create_input)
    return create_input


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--git-sha", default="", help="40-hex commit already on GitHub. Default: HEAD")
    ap.add_argument("--repo-url", default=REPO_URL)
    ap.add_argument("--image", default=IMAGE, help="Container image. Default boots via git clone.")
    ap.add_argument(
        "--image-repo",
        default="",
        help="If set, the image already contains this repo checkout; boot verifies HEAD equals --git-sha",
    )
    ap.add_argument("--volume-id", default=VOLUME_ID)
    ap.add_argument("--volume-run-dir", default=DEFAULT_VOLUME_RUN)
    ap.add_argument("--volume-directions", default=DEFAULT_VOLUME_DIRECTIONS)
    ap.add_argument("--volume-out", default=DEFAULT_VOLUME_OUT)
    ap.add_argument("--hf-home", default=DEFAULT_HF_HOME)
    ap.add_argument("--model", default="google/gemma-2-9b-it")
    ap.add_argument("--max-scenarios", type=int, default=6)
    ap.add_argument("--k-frac", type=float, default=0.5)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--launch", action="store_true", help="Create one GPU pod")
    ap.add_argument("--max-minutes", type=int, default=90)
    ap.add_argument("--grace-minutes", type=int, default=6)
    ap.add_argument("--estimate-out", type=Path, default=ESTIMATE_OUT)
    ap.add_argument("--launch-card", type=Path, default=LAUNCH_OUT)
    ap.add_argument("--cost-max", type=float, default=1.50)
    ap.add_argument("--balance-min", type=float, default=20.0)
    ap.add_argument("--gpu", default=GPU)
    ap.add_argument("--cloud", default="SECURE")
    args = ap.parse_args(argv)

    if not args.git_sha:
        args.git_sha = git_identity()[0]
    sha = args.git_sha.strip().lower()
    if not SHA_RE.fullmatch(sha):
        print("ERROR: --git-sha must be a 40-hex commit", file=sys.stderr)
        return 2
    args.git_sha = sha
    if not REPO_URL_RE.fullmatch(args.repo_url.strip()):
        print("ERROR: --repo-url must be https://github.com/andylikescodes/narcbench-core-demo", file=sys.stderr)
        return 2
    args.repo_url = args.repo_url.strip()
    if not args.repo_url.endswith(".git"):
        args.repo_url += ".git"
    if args.max_scenarios < 1 or args.max_scenarios > 6:
        print("ERROR: --max-scenarios must be between 1 and 6", file=sys.stderr)
        return 1
    if args.image_repo and args.image == IMAGE:
        print(
            "ERROR: --image-repo is for an image that already contains the checkout; pass --image",
            file=sys.stderr,
        )
        return 2

    try:
        args.volume_run_dir = assert_volume_path(args.volume_run_dir, "--volume-run-dir")
        args.volume_directions = assert_volume_path(args.volume_directions, "--volume-directions")
        args.volume_out = assert_volume_path(args.volume_out, "--volume-out")
        args.hf_home = assert_volume_path(args.hf_home, "--hf-home")
        if args.image_repo:
            args.image_repo = assert_checkout_path(args.image_repo, "--image-repo")
    except LaunchRefused as exc:
        print(f"REFUSE: {exc.reason}", file=sys.stderr)
        return exc.code

    est = estimate(args.max_scenarios, args.max_new_tokens)
    est["k_frac"] = args.k_frac
    est["code_source"] = "image_pinned_checkout" if args.image_repo else "git_clone_sha"
    real_key = os.environ.get("RUNPOD_API_KEY") or ""
    # Measure the create body with the real key when it is present. Otherwise use a
    # same-width stand-in so estimate mode does not need credentials and does not
    # invent a key. Launch always remeasures with the real key.
    measure_key = real_key or ("0" * 64)
    try:
        sample = build_create_input(args, measure_key)
        post_bytes = validate_create_input(sample)
    except LaunchRefused as exc:
        print(f"REFUSE: {exc.reason}", file=sys.stderr)
        return exc.code

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
        "volume_id": args.volume_id,
        "volume_run_dir": args.volume_run_dir,
        "volume_directions": args.volume_directions,
        "volume_out": args.volume_out,
        "hf_home": args.hf_home,
        "code_source": est["code_source"],
        "repo_url": args.repo_url,
        "git_sha": args.git_sha,
        "image": args.image,
        "image_repo": args.image_repo or None,
        "graphql_post_body_bytes": post_bytes,
        "graphql_post_body_limit": MAX_CREATE_BODY_BYTES,
        "graphql_post_body_uses_placeholder_key": not bool(real_key),
        "payload_tarball_in_create_env": False,
        "job_payload_b64_in_create_env": False,
        "script_source_in_create_env": False,
        "output_on_volume": args.volume_out + "/smoke_*",
        "status_port": 8765,
        "status_url_template": "https://<pod-id>-8765.proxy.runpod.net/status.json",
        "cost_gate_usd": args.cost_max,
        "balance_min_usd": args.balance_min,
        "gpu_tries": [{"cloud": cloud, "gpu": gpu} for cloud, gpu in GPU_TRIES],
        "launch_command": "python3 scripts/runpod_launch_within_traj.py --launch --git-sha " + args.git_sha,
        "dry_run_command": "python3 scripts/runpod_launch_within_traj.py --git-sha " + args.git_sha,
        "soft_stop": "Track I / Paper 2 / interp-demo NOT touched; no GPU unless --launch",
    }
    print(json.dumps(report, indent=2))
    _write_json(args.estimate_out, report)
    print(f"\n[wrote] {args.estimate_out}", file=sys.stderr)
    print(f"[pin] sha={args.git_sha} post_body={post_bytes}", file=sys.stderr)

    if not args.launch:
        print("\n[estimate-only] NOT launching. Re-run with --launch when ready.", file=sys.stderr)
        return 0

    if worktree_dirty():
        print("REFUSE launch: worktree is dirty; commit and push the SHA first", file=sys.stderr)
        return 1
    if not blob_in_commit(args.git_sha, HARNESS_REL) or not blob_in_commit(args.git_sha, BOOT_REL):
        print(
            f"REFUSE launch: {HARNESS_REL} or {BOOT_REL} is not in git SHA {args.git_sha}",
            file=sys.stderr,
        )
        return 1
    try:
        assert_option_labels_fix(ROOT / HARNESS_REL)
    except LaunchRefused as exc:
        print(f"REFUSE launch: {exc.reason}", file=sys.stderr)
        return exc.code
    _sha_now, ref = git_identity()
    if not remote_has_sha(args.git_sha, ref):
        print("REFUSE launch: SHA is not on the public remote branch yet", file=sys.stderr)
        return 1

    if est["cost_est_usd"] > args.cost_max:
        print(f"REFUSE launch: cost_est ${est['cost_est_usd']} > ${args.cost_max}", file=sys.stderr)
        return 2
    if not real_key:
        print("RUNPOD_API_KEY missing", file=sys.stderr)
        return 1

    try:
        sample = build_create_input(args, real_key)
        post_bytes = validate_create_input(sample)
    except LaunchRefused as exc:
        print(f"REFUSE: {exc.reason}", file=sys.stderr)
        return exc.code
    report["graphql_post_body_bytes"] = post_bytes
    report["graphql_post_body_uses_placeholder_key"] = False

    balance = None
    spend = None
    try:
        bal = graphql("query { myself { clientBalance currentSpendPerHr } }", {}, timeout=30)
        me = ((bal.get("data") or {}).get("myself") or {})
        balance = me.get("clientBalance")
        spend = me.get("currentSpendPerHr")
    except Exception as exc:
        print(f"[warn] balance fetch: {exc}", file=sys.stderr)
    print(f"[balance] ${balance} spendPerHr={spend}", file=sys.stderr)

    if balance is not None and float(balance) < args.balance_min:
        print(f"REFUSE launch: balance ${balance} < ${args.balance_min}", file=sys.stderr)
        _write_json(
            args.launch_card,
            {
                "launched": False,
                "reason": "balance_below_min",
                "balance_usd": balance,
                "balance_min_usd": args.balance_min,
                "graphql_post_body_bytes": post_bytes,
                "git_sha": args.git_sha,
                "estimate": est,
            },
        )
        return 3

    try:
        pods_now = list_pods()
    except Exception as exc:
        print(f"REFUSE launch: could not list pods: {exc}", file=sys.stderr)
        return 1
    busy = billable_gpu_pods(pods_now)
    if busy:
        print(f"REFUSE launch: billable GPU already up: {busy}", file=sys.stderr)
        _write_json(
            args.launch_card,
            {
                "launched": False,
                "reason": "billable_gpu_already_running",
                "pods": busy,
                "balance_usd": balance,
                "graphql_post_body_bytes": post_bytes,
                "git_sha": args.git_sha,
                "estimate": est,
            },
        )
        return 5

    last: Exception | None = None
    for cloud, gpu in GPU_TRIES:
        rate = RATES.get(gpu, RATE_USD_HR)
        args.cloud = cloud
        args.gpu = gpu
        try:
            create_input = build_create_input(args, real_key)
            nbytes = validate_create_input(create_input)
        except LaunchRefused as exc:
            print(f"REFUSE: {exc.reason}", file=sys.stderr)
            _write_json(
                args.launch_card,
                {
                    "launched": False,
                    "reason": exc.reason,
                    "graphql_post_body_bytes": post_bytes,
                    "git_sha": args.git_sha,
                    "estimate": est,
                },
            )
            return exc.code
        print(f"[launch] trying {cloud} {gpu}; post_body_bytes={nbytes}", file=sys.stderr)
        try:
            data = graphql(CREATE_MUTATION, {"input": create_input}, timeout=120)
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
                "volume_id": args.volume_id,
                "volume_run_dir": args.volume_run_dir,
                "volume_directions": args.volume_directions,
                "volume_out": args.volume_out,
                "code_source": est["code_source"],
                "repo_url": args.repo_url,
                "git_sha": args.git_sha,
                "image": args.image,
                "image_repo": args.image_repo or None,
                "graphql_post_body_bytes": nbytes,
                "payload_tarball_in_create_env": False,
                "job_payload_b64_in_create_env": False,
                "script_source_in_create_env": False,
                "output_on_volume": args.volume_out + "/smoke_*",
                "output_latest_tgz": args.volume_out + "/latest.tgz",
                "soft_stop": "Track I / Paper 2 / interp-demo NOT touched",
                "machine": pod.get("machine") or {},
            }
            _write_json(args.launch_card, card)
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
            print(f"[launch] {cloud} {gpu}: {exc}", file=sys.stderr)
            if exc.is_cloudflare_body_reject:
                print("[launch] Cloudflare body-size reject — STOPPING GPU walk", file=sys.stderr)
                _write_json(
                    args.launch_card,
                    {
                        "launched": False,
                        "reason": "cloudflare_body_size_reject",
                        "http_status": exc.status,
                        "http_body_snippet": str(exc)[:400],
                        "graphql_post_body_bytes": nbytes,
                        "cloudflare_walk_stop": True,
                        "gpu_stopped_at": {"cloud": cloud, "gpu": gpu},
                        "balance_usd": balance,
                        "git_sha": args.git_sha,
                        "estimate": est,
                    },
                )
                return 1
            time.sleep(1.0)
        except Exception as exc:
            last = exc
            msg = str(exc).lower()
            print(f"[launch] {cloud} {gpu}: {exc}", file=sys.stderr)
            if "no instances" in msg or "capacity" in msg or "not available" in msg:
                time.sleep(1.0)
                continue
            time.sleep(1.0)

    _write_json(
        args.launch_card,
        {
            "launched": False,
            "reason": "all_gpu_prefs_failed",
            "last_error": str(last),
            "gpu_tries": [{"cloud": c, "gpu": g} for c, g in GPU_TRIES],
            "balance_usd": balance,
            "graphql_post_body_bytes": post_bytes,
            "cloudflare_walk_stop": True,
            "estimate": est,
            "git_sha": args.git_sha,
            "ready_to_launch": True,
            "launch_command": report["launch_command"],
        },
    )
    print(f"all failed: {last}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
