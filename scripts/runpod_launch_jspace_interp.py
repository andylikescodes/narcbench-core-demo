#!/usr/bin/env python3
"""Launch, watch, pull and terminate the jspace-interp rebuild on one fresh RunPod GPU pod.

The pod clones this repository at a pushed commit (--sha, default the local HEAD), runs
scripts/runpod_jspace_interp_boot.sh (downloads the published Jacobian lens and the gated Gemma-2-9B-it, generates the
50 matched-vocabulary control episodes, extracts activations for the four context conditions at seven read positions,
runs the k x position sweep with 200-draw random nulls as CPU jobs, computes role directions and runs the steering
experiment), then serves results.tgz and acts.tgz on port 8765 until they are pulled.

Keys come from the environment only: RUNPOD_API_KEY (to spend, watch, pull, terminate) and HF_TOKEN (gated Gemma).
Default is estimate-only; nothing is created without --launch.

    python3 scripts/runpod_launch_jspace_interp.py                     # plan and cost estimate
    python3 scripts/runpod_launch_jspace_interp.py --preflight         # gates: balance, running pods, GPU prices/stock, pushed SHA
    python3 scripts/runpod_launch_jspace_interp.py --launch [--gpu "NVIDIA A40"] [--watch]
    python3 scripts/runpod_launch_jspace_interp.py --status <pod>      # status.json, stages, job.log tail, pod state
    python3 scripts/runpod_launch_jspace_interp.py --watch-pod <pod>   # poll until done, pull, verify, terminate
    python3 scripts/runpod_launch_jspace_interp.py --pull <pod>        # results.tgz + acts.tgz -> results/jspace_interp/, verified
    python3 scripts/runpod_launch_jspace_interp.py --terminate <pod>
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import subprocess
import sys
import tarfile
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IMAGE = "runpod/pytorch:1.2.0-rc.162-cu1281-torch280-ubuntu2404"  # torch 2.8 / CUDA 12.8; used for the Nov 5 lens pods
POD_NAME = "job-narcbench-jspace-interp"
GIT_REPO = "https://github.com/andylikescodes/narcbench-core-demo.git"
GIT_BRANCH = "jspace-interp"
# gpuTypeId values as RunPod lists them (2026-11-05). 48 GB cards first (the user's choice), 80 GB A100s as the fast option.
GPU_PREFS = ["NVIDIA A40", "NVIDIA RTX A6000", "NVIDIA L40", "NVIDIA L40S", "NVIDIA RTX 6000 Ada Generation",
             "NVIDIA A100 80GB PCIe", "NVIDIA A100-SXM4-80GB"]
GPU_FAST = ["NVIDIA A100 80GB PCIe", "NVIDIA A100-SXM4-80GB", "NVIDIA H100 PCIe", "NVIDIA L40S", "NVIDIA A40", "NVIDIA RTX A6000"]
RATES = {"NVIDIA A40": 0.59, "NVIDIA RTX A6000": 0.59, "NVIDIA L40": 0.82, "NVIDIA L40S": 1.09,
         "NVIDIA RTX 6000 Ada Generation": 0.99, "NVIDIA A100 80GB PCIe": 1.79, "NVIDIA A100-SXM4-80GB": 1.79, "NVIDIA H100 PCIe": 2.89}
# relative GPU speed for the estimate (A40 = 1)
SPEED = {"NVIDIA A40": 1.0, "NVIDIA RTX A6000": 1.0, "NVIDIA L40": 1.3, "NVIDIA L40S": 1.6, "NVIDIA RTX 6000 Ada Generation": 1.5,
         "NVIDIA A100 80GB PCIe": 1.7, "NVIDIA A100-SXM4-80GB": 1.9, "NVIDIA H100 PCIe": 2.6}
# GPU stage minutes on an A40 (from the Nov 5 pods' throughput and the row counts: 2,636 turns per condition at ~0.5 s)
GPU_STAGES = {"setup+download": 15, "gen_warned (50 episodes, batched)": 30, "extract_warned": 10,
              "extract original/swap/prompt_only/nosys": 70, "role_dirs": 10, "steer (4 runs, cached prefix)": 80}
ANALYSIS_CELLS = 80            # (original + swap) x 7 positions x 5 layers + prompt_only 5 + nosys 5
ANALYSIS_MIN_PER_CELL = 25     # measured on the Nov 5 caches: 12 min with 10 draws + bootstrap each -> ~25 with 200 draws, no draw bootstrap


def graphql(query: str, variables: dict | None = None, timeout: int = 120) -> dict:
    key = os.environ.get("RUNPOD_API_KEY")
    if not key:
        raise SystemExit("RUNPOD_API_KEY is not set in the environment")
    body = json.dumps({"query": query, "variables": variables or {}}).encode()
    req = urllib.request.Request("https://api.runpod.io/graphql", data=body, method="POST",
                                 headers={"content-type": "application/json", "Authorization": f"Bearer {key}", "User-Agent": "nb-interp/1"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def fetch(url: str, timeout: int = 30) -> bytes | None:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "nb-interp/1"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.read()
    except Exception:  # noqa: BLE001  (proxy 404/502 before the container is up, timeouts)
        return None


def fetch_json(url: str, timeout: int = 30) -> dict | None:
    b = fetch(url, timeout)
    if b is None:
        return None
    try:
        return json.loads(b.decode())
    except Exception:  # noqa: BLE001
        return None


def bootstrap_script(a) -> str:
    """Small boot kept in the pod env: status server, shallow fetch of the pushed SHA, hand-off to the repo's boot script."""
    return f"""#!/bin/bash
set -u
RUN=/tmp/run; mkdir -p $RUN/logs
echo '{{"phase":"boot","stage":"clone"}}' > $RUN/status.json
(cd $RUN && nohup python3 -m http.server 8765 >/tmp/http0.log 2>&1 &)
term() {{ python3 - <<'PY' || true
import os, json, urllib.request as u
p=os.environ.get('RUNPOD_POD_ID',''); k=os.environ.get('RUNPOD_API_KEY','')
b=json.dumps({{'query':'mutation {{ podTerminate(input: {{podId: "%s"}}) }}'%p}}).encode()
r=u.Request('https://api.runpod.io/graphql',data=b,headers={{'content-type':'application/json','Authorization':'Bearer '+k,'User-Agent':'nb-interp/1'}},method='POST')
u.urlopen(r,timeout=30).read()
PY
}}
fail() {{ echo "{{\\"phase\\":\\"failed\\",\\"stage\\":\\"$1\\"}}" > $RUN/status.json; tail -20 $RUN/logs/clone.log > $RUN/job.log 2>/dev/null; sleep {a.grace_minutes}m; term; exit 1; }}
mkdir -p /workspace/nb && cd /workspace/nb && rm -rf repo && mkdir repo && cd repo
git init -q && git remote add origin "{GIT_REPO}" && git fetch -q --depth 1 origin {a.sha} >> $RUN/logs/clone.log 2>&1 || fail fetch
git checkout -q FETCH_HEAD >> $RUN/logs/clone.log 2>&1 || fail checkout
echo "checked out $(git rev-parse HEAD)" >> $RUN/logs/clone.log
exec bash scripts/runpod_jspace_interp_boot.sh
"""


def local_sha() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()


def remote_sha(branch: str) -> str | None:
    r = subprocess.run(["git", "ls-remote", "origin", f"refs/heads/{branch}"], cwd=ROOT, capture_output=True, text=True)
    return r.stdout.split()[0] if r.returncode == 0 and r.stdout.strip() else None


def sha_is_on_remote(sha: str) -> bool:
    """True if origin has the commit (ls-remote finds the branch at it, or it is an ancestor of the remote branch)."""
    rs = remote_sha(GIT_BRANCH)
    if rs == sha:
        return True
    subprocess.run(["git", "fetch", "-q", "origin", GIT_BRANCH], cwd=ROOT, capture_output=True)
    r = subprocess.run(["git", "merge-base", "--is-ancestor", sha, f"origin/{GIT_BRANCH}"], cwd=ROOT, capture_output=True)
    return r.returncode == 0


def estimate(a) -> dict:
    gpus = [a.gpu] if a.gpu else (GPU_FAST if a.fast else GPU_PREFS)
    rows = {}
    for g in gpus:
        sp = SPEED.get(g, 1.0)
        gpu_min = sum(GPU_STAGES.values()) / sp
        extract_done = (GPU_STAGES["setup+download"] + GPU_STAGES["gen_warned (50 episodes, batched)"] + GPU_STAGES["extract_warned"]
                        + GPU_STAGES["extract original/swap/prompt_only/nosys"] * 0.35) / sp  # original's analysis starts here
        cpu_min = ANALYSIS_CELLS * ANALYSIS_MIN_PER_CELL / max(a.analysis_procs, 1)
        total_min = max(gpu_min, extract_done + cpu_min) + 10
        total_h = min(total_min, a.max_minutes) / 60
        rate = RATES.get(g)
        rows[g] = {"rate_usd_hr": rate, "gpu_chain_hours": round(gpu_min / 60, 1), "analysis_hours_after_extraction": round(cpu_min / 60, 1),
                   "expected_hours": round(total_h, 1), "expected_usd": round(total_h * rate, 2) if rate else None,
                   "worst_case_usd": round((a.max_minutes + a.grace_minutes) / 60 * rate, 2) if rate else None}
    return {"stages_minutes_on_A40": GPU_STAGES, "analysis_cells": ANALYSIS_CELLS, "analysis_min_per_cell": ANALYSIS_MIN_PER_CELL,
            "analysis_procs_assumed": a.analysis_procs, "time_cap_minutes": a.max_minutes, "grace_minutes": a.grace_minutes,
            "settings": {"n_random": a.n_random, "positions": a.positions, "layers": a.layers, "steer_runs": a.steer_runs},
            "by_gpu": rows}


def account() -> dict:
    me = graphql("query { myself { clientBalance currentSpendPerHr pods { id name desiredStatus costPerHr machine { gpuDisplayName dataCenterId } } } }")
    if me.get("errors"):
        raise SystemExit(f"cannot read the account: {str(me['errors'])[:300]}")
    return me["data"]["myself"]


def gpu_market(names: list[str]) -> dict:
    q = ("query { gpuTypes { id displayName memoryInGb secureCloud communityCloud securePrice communityPrice "
         "lowestPrice(input:{gpuCount:1}) { minimumBidPrice uninterruptablePrice stockStatus } } }")
    d = graphql(q)
    if d.get("errors"):
        return {"error": str(d["errors"])[:300]}
    out = {}
    for g in d["data"]["gpuTypes"]:
        if g["id"] in names:
            lp = g.get("lowestPrice") or {}
            out[g["id"]] = {"memory_gb": g.get("memoryInGb"), "secure_usd_hr": g.get("securePrice"), "community_usd_hr": g.get("communityPrice"),
                            "on_demand_lowest": lp.get("uninterruptablePrice"), "stock": lp.get("stockStatus")}
    return out


def gates(a, est: dict) -> tuple[bool, dict]:
    me = account()
    balance = float(me.get("clientBalance") or 0.0)
    running = [p for p in (me.get("pods") or []) if p.get("desiredStatus") == "RUNNING" and (p.get("costPerHr") or 0) > 0.05]
    sha = a.sha
    pushed = sha_is_on_remote(sha)
    dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    gpus = [a.gpu] if a.gpu else (GPU_FAST if a.fast else GPU_PREFS)
    market = gpu_market(gpus)
    rep = {"balance_usd": round(balance, 2), "spend_per_hr": me.get("currentSpendPerHr"), "billable_pods_running": running,
           "sha": sha, "sha_on_origin": pushed, "working_tree_dirty": bool(dirty), "hf_token_present": bool(os.environ.get("HF_TOKEN")),
           "gpu_market": market, "estimate_by_gpu": est["by_gpu"]}
    ok = balance >= a.min_balance and not running and pushed and bool(os.environ.get("HF_TOKEN")) and not dirty
    rep["gates_ok"] = ok
    rep["why_not"] = [m for cond, m in ((balance < a.min_balance, f"balance below {a.min_balance}"), (bool(running), "a billable pod is running"),
                                         (not pushed, "the SHA is not on origin (push first)"), (not os.environ.get("HF_TOKEN"), "HF_TOKEN missing"),
                                         (bool(dirty), "tracked files are modified (commit first)")) if cond]
    return ok, rep


def create_pod(a, est: dict) -> dict:
    boot = bootstrap_script(a)
    env = {"JOB_BOOT_B64": base64.b64encode(boot.encode()).decode(), "RUNPOD_API_KEY": os.environ["RUNPOD_API_KEY"],
           "HF_TOKEN": os.environ["HF_TOKEN"], "GIT_SHA": a.sha, "JOB_MAX_MINUTES": str(a.max_minutes), "JOB_GRACE_MINUTES": str(a.grace_minutes),
           "JOB_N_RANDOM": str(a.n_random), "JOB_POSITIONS": a.positions, "JOB_LAYERS": a.layers, "JOB_STEER_RUNS": a.steer_runs,
           "JOB_WARNED_WORKERS": str(a.warned_workers), "JOB_WARNED_BATCH": str(a.warned_batch)}
    if a.analysis_procs_env:
        env["JOB_ANALYSIS_PROCS"] = str(a.analysis_procs_env)
    if a.skip:
        env["JOB_SKIP"] = a.skip
    docker_args = "bash -c 'echo $JOB_BOOT_B64 | base64 -d > /tmp/boot.sh; exec bash /tmp/boot.sh'"
    gpus = [a.gpu] if a.gpu else (GPU_FAST if a.fast else GPU_PREFS)
    attempts = []
    for vcpu, mem in ((a.min_vcpu, a.min_memory_gb), (4, 24)):
        for cloud in ("SECURE", "COMMUNITY"):
            for g in gpus:
                inp = {"name": POD_NAME, "imageName": IMAGE, "gpuTypeId": g, "cloudType": cloud, "gpuCount": 1,
                       "volumeInGb": 0, "containerDiskInGb": a.disk_gb, "minVcpuCount": vcpu, "minMemoryInGb": mem,
                       "ports": "8765/http", "env": [{"key": k, "value": v} for k, v in env.items()],
                       "dockerArgs": docker_args, "supportPublicIp": False, "startSsh": False}
                if a.min_download > 0:
                    inp["minDownload"] = a.min_download
                q = """mutation ($input: PodFindAndDeployOnDemandInput!) {
                      podFindAndDeployOnDemand(input: $input) { id desiredStatus costPerHr machine { podHostId gpuDisplayName dataCenterId cpuCount } } }"""
                try:
                    r = graphql(q, {"input": inp})
                    if r.get("errors"):
                        raise RuntimeError(r["errors"][0].get("message", "")[:200])
                    pod = (r.get("data") or {}).get("podFindAndDeployOnDemand")
                    if not pod or not pod.get("id"):
                        raise RuntimeError("empty pod")
                    pod.update({"gpu": g, "cloud": cloud, "min_vcpu": vcpu, "min_memory_gb": mem})
                    return {"created": pod, "attempts": attempts}
                except (RuntimeError, urllib.error.HTTPError) as e:
                    msg = str(e)[:200]
                    attempts.append({"gpu": g, "cloud": cloud, "vcpu": vcpu, "error": msg})
                    print(f"[launch] {cloud} {g} (vcpu>={vcpu}): {msg}", file=sys.stderr, flush=True)
    return {"created": None, "attempts": attempts}


def pod_state(pod: str) -> dict | None:
    r = graphql('query { pod(input: {podId: "%s"}) { id desiredStatus costPerHr runtime { uptimeInSeconds } machine { podHostId gpuDisplayName dataCenterId cpuCount } } }' % pod)
    return (r.get("data") or {}).get("pod")


def status(pod: str, verbose: bool = True) -> dict:
    base = f"https://{pod}-8765.proxy.runpod.net"
    st = fetch_json(f"{base}/status.json")
    stages = fetch_json(f"{base}/stages.json")
    tail = fetch(f"{base}/job.log", 30)
    out = {"status": st, "stages": stages, "job_log_tail": tail.decode("utf-8", "replace").splitlines()[-12:] if tail else None}
    if os.environ.get("RUNPOD_API_KEY"):
        try:
            out["pod"] = pod_state(pod)
        except Exception as e:  # noqa: BLE001
            out["pod"] = f"state unavailable: {type(e).__name__}"
    if verbose:
        print(json.dumps(out, indent=1))
    return out


def sha256_of(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def download(url: str, dest: Path, size: int, sha: str) -> dict:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size == size and sha256_of(dest) == sha:
        print(f"  {dest.name}: already present and verified", flush=True)
        return {"status": "present"}
    part = dest.with_name(dest.name + ".part")
    have = part.stat().st_size if part.exists() else 0
    if have > size:
        part.unlink(); have = 0
    t0, start_have, tries = time.time(), have, 0
    while have < size:
        hdr = {"User-Agent": "nb-interp/1"}
        if have:
            hdr["Range"] = f"bytes={have}-"
        req = urllib.request.Request(url, headers=hdr)
        try:
            with urllib.request.urlopen(req, timeout=180) as r, open(part, "ab" if have else "wb") as f:
                if have and r.status != 206:
                    f.seek(0); f.truncate(); have = 0
                last = time.time()
                while True:
                    b = r.read(1 << 22)
                    if not b:
                        break
                    f.write(b); have += len(b)
                    if time.time() - last > 15:
                        rate = (have - start_have) / max(time.time() - t0, 1e-6) / 1e6
                        print(f"  {dest.name}: {have/1e6:,.0f}/{size/1e6:,.0f} MB  {rate:.1f} MB/s", flush=True)
                        last = time.time()
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
            tries += 1
            if tries > 40:
                raise SystemExit(f"{dest.name}: giving up after {tries} attempts: {e}")
            print(f"  {dest.name}: {type(e).__name__} after {have/1e6:,.0f} MB; retry {tries} in 10 s", flush=True)
            time.sleep(10)
            have = part.stat().st_size if part.exists() else 0
    got = sha256_of(part)
    if part.stat().st_size != size or got != sha:
        raise SystemExit(f"{dest.name}: verification failed (size {part.stat().st_size} vs {size}, sha {got[:16]} vs {sha[:16]}); kept {part}")
    part.replace(dest)
    el = time.time() - t0
    print(f"  {dest.name}: {size/1e6:,.0f} MB in {el:.0f}s, sha256 ok", flush=True)
    return {"status": "downloaded", "seconds": round(el, 1)}


def safe_extract(tgz: Path, prefix: str):
    with tarfile.open(tgz) as t:
        members = [m for m in t.getmembers() if m.name.startswith(prefix) and ".." not in m.name.split("/")]
        t.extractall(ROOT, members=members)
    return len(members)


def pull(pod: str, dest_dir: Path) -> dict:
    base = f"https://{pod}-8765.proxy.runpod.net"
    man = fetch_json(f"{base}/manifest.json", 60)
    if not man:
        raise SystemExit("no manifest.json on the pod yet (status.json says what phase it is in)")
    dest_dir.mkdir(parents=True, exist_ok=True)
    rec = {"pod": pod, "files": {}}
    for name, info in man.items():
        r = download(f"{base}/{name}", dest_dir / name, int(info["size"]), info["sha256"])
        rec["files"][name] = {**info, **r, "local": str((dest_dir / name).relative_to(ROOT))}
    if (dest_dir / "results.tgz").exists():
        rec["extracted_results"] = safe_extract(dest_dir / "results.tgz", "results/jspace_interp/")
    if (dest_dir / "acts.tgz").exists():
        rec["extracted_acts"] = safe_extract(dest_dir / "acts.tgz", "results/jspace_interp/acts")
    for name in ("status.json", "stages.json", "job.log"):
        b = fetch(f"{base}/{name}", 60)
        if b:
            (dest_dir / name).write_bytes(b)
    (dest_dir / "pull.json").write_text(json.dumps(rec, indent=1) + "\n")
    print(json.dumps({k: v for k, v in rec.items() if k != "files"} | {"files": {k: v.get("status") for k, v in rec["files"].items()}}, indent=1))
    return rec


def terminate(pod: str) -> dict:
    return graphql('mutation { podTerminate(input: {podId: "%s"}) }' % pod)


def watch(pod: str, dest_dir: Path, hours: float, do_terminate: bool, poll: int = 60) -> int:
    t0 = time.time()
    last = None
    while time.time() - t0 < hours * 3600:
        st = fetch_json(f"https://{pod}-8765.proxy.runpod.net/status.json", 20)
        key = (st or {}).get("phase"), (st or {}).get("stage"), (st or {}).get("detail")
        if key != last:
            print(f"[{(time.time()-t0)/60:5.0f} min] {json.dumps(st) if st else 'no status yet'}", flush=True)
            last = key
        if st and st.get("phase") in ("done", "failed"):
            print(f"pod reports {st.get('phase')}; pulling", flush=True)
            try:
                pull(pod, dest_dir)
            except SystemExit as e:
                print(f"pull failed: {e}", file=sys.stderr)
                return 5
            if do_terminate:
                print("terminating", json.dumps(terminate(pod)), flush=True)
                time.sleep(5)
                try:
                    print("pod state after terminate:", json.dumps(pod_state(pod)), flush=True)
                except Exception:  # noqa: BLE001
                    pass
            return 0 if st.get("phase") == "done" else 4
        if os.environ.get("RUNPOD_API_KEY") and int(time.time() - t0) % 600 < poll:
            try:
                ps = pod_state(pod)
                if ps is None:
                    print("pod no longer exists", flush=True)
                    return 6
            except Exception:  # noqa: BLE001
                pass
        time.sleep(poll)
    print("watch window over; the pod keeps running", file=sys.stderr)
    return 7


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sha", default=None, help="commit the pod checks out (default: local HEAD; must be on origin)")
    ap.add_argument("--gpu", default=None, help="force one gpuTypeId")
    ap.add_argument("--fast", action="store_true", help="prefer the 80 GB A100s over the 48 GB cards")
    ap.add_argument("--max-minutes", type=int, default=480, help="time cap on the pod; what is unfinished is packaged as is")
    ap.add_argument("--grace-minutes", type=int, default=180, help="how long the pod serves results before terminating itself")
    ap.add_argument("--n-random", type=int, default=200)
    ap.add_argument("--positions", default="last,prevote,eot,reply_mean,frac50,frac25,frac75")
    ap.add_argument("--layers", default="19-23")
    ap.add_argument("--steer-runs", default="swap:22,original:22,swap:23,original:23")
    ap.add_argument("--warned-workers", type=int, default=8)
    ap.add_argument("--warned-batch", type=int, default=8)
    ap.add_argument("--analysis-procs", type=int, default=7, help="assumed CPU analysis processes for the estimate")
    ap.add_argument("--analysis-procs-env", type=int, default=0, help="force JOB_ANALYSIS_PROCS on the pod (default nproc-2)")
    ap.add_argument("--skip", default="", help="comma list of pod stages to skip (JOB_SKIP)")
    ap.add_argument("--min-vcpu", type=int, default=8)
    ap.add_argument("--min-memory-gb", type=int, default=40)
    ap.add_argument("--disk-gb", type=int, default=60)
    ap.add_argument("--min-download", type=int, default=800, help="hosts with at least this download speed (Mbps); 0 disables")
    ap.add_argument("--min-balance", type=float, default=20.0)
    ap.add_argument("--preflight", action="store_true")
    ap.add_argument("--launch", action="store_true", help="create the pod (spends money)")
    ap.add_argument("--watch", action="store_true", help="after --launch: poll, pull when done, terminate")
    ap.add_argument("--watch-hours", type=float, default=11.0)
    ap.add_argument("--no-terminate", action="store_true", help="watch/pull without terminating the pod")
    ap.add_argument("--status", metavar="POD_ID")
    ap.add_argument("--watch-pod", metavar="POD_ID")
    ap.add_argument("--pull", metavar="POD_ID")
    ap.add_argument("--terminate", metavar="POD_ID")
    a = ap.parse_args()
    a.sha = a.sha or local_sha()
    runs_dir = ROOT / "results/jspace_interp/pod_runs"

    if a.status:
        status(a.status)
        return 0
    if a.terminate:
        print(json.dumps(terminate(a.terminate)))
        return 0
    if a.pull:
        pull(a.pull, runs_dir / a.pull)
        if not a.no_terminate:
            print("terminating", json.dumps(terminate(a.pull)))
        return 0
    if a.watch_pod:
        return watch(a.watch_pod, runs_dir / a.watch_pod, a.watch_hours, not a.no_terminate)

    est = estimate(a)
    print(json.dumps(est, indent=1))
    if not (a.launch or a.preflight):
        print("\n[estimate-only] --preflight checks the gates read-only; --launch creates the pod", file=sys.stderr)
        return 0
    ok, rep = gates(a, est)
    print(json.dumps(rep, indent=1))
    if not ok:
        print("gates failed: " + "; ".join(rep["why_not"]), file=sys.stderr)
        return 2
    if a.preflight:
        return 0
    res = create_pod(a, est)
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    if not res["created"]:
        print("no GPU type could be deployed", file=sys.stderr)
        (runs_dir / f"failed_launch_{stamp}.json").parent.mkdir(parents=True, exist_ok=True)
        (runs_dir / f"failed_launch_{stamp}.json").write_text(json.dumps(res, indent=1) + "\n")
        return 3
    pod = res["created"]["id"]
    rec = {"created_utc": stamp, "pod": res["created"], "attempts_before": res["attempts"], "sha": a.sha, "image": IMAGE,
           "settings": {k: getattr(a, k) for k in ("max_minutes", "grace_minutes", "n_random", "positions", "layers", "steer_runs",
                                                     "warned_workers", "warned_batch", "skip", "min_vcpu", "min_memory_gb", "disk_gb")},
           "estimate": est["by_gpu"].get(res["created"]["gpu"]), "status_url": f"https://{pod}-8765.proxy.runpod.net/status.json"}
    (runs_dir / pod).mkdir(parents=True, exist_ok=True)
    (runs_dir / pod / "pod.json").write_text(json.dumps(rec, indent=1) + "\n")
    print(json.dumps(rec, indent=1))
    if a.watch:
        return watch(pod, runs_dir / pod, a.watch_hours, not a.no_terminate)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
