#!/usr/bin/env python3
"""Launch, watch, pull and terminate a pod of the uninstructed-collusion experiment (one GPU at a time).

The pod clones this repository at a pushed commit of branch `uninstructed-collusion` and runs
scripts/runpod_uninstructed_boot.sh. Mode `pilot` (docs/uninstructed_collusion/DESIGN.md, sections 10 and 11): the
versions given by --versions on the pilot's twelve scenarios, room codes by Gemma, a timing test for the main run. The
first pilot was S0 plus S1,S2,S3 into results/uninstructed/pilot; the second is
`--skip gen_ref --versions T1,T2,T3,T4,T5 --pilot-dir pilot2`. Results are served on port 8765 until they are pulled;
the watcher pulls them, verifies the checksum and terminates the pod.

Keys come from the environment only: RUNPOD_API_KEY (to spend, watch, pull, terminate) and HF_TOKEN (gated Gemma).
Default is estimate-only; nothing is created without --launch.

    python3 scripts/runpod_launch_uninstructed.py                      # plan and cost estimate
    python3 scripts/runpod_launch_uninstructed.py --preflight          # gates: balance, running pods, price, pushed SHA, clean code
    python3 scripts/runpod_launch_uninstructed.py --launch --watch     # create the pod, then poll, pull, terminate
    python3 scripts/runpod_launch_uninstructed.py --status <pod>
    python3 scripts/runpod_launch_uninstructed.py --watch-pod <pod>
    python3 scripts/runpod_launch_uninstructed.py --pull <pod>         # pulls and terminates (--no-terminate keeps it)
    python3 scripts/runpod_launch_uninstructed.py --terminate <pod>
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import subprocess
import sys
import tarfile
import time
import urllib.error
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import runpod_launch_jspace_interp as base  # noqa: E402  (account, market, download, status and terminate helpers)

GIT_BRANCH = "uninstructed-collusion"
POD_NAME = "job-narcbench-uninstructed"
BOOT = "scripts/runpod_uninstructed_boot.sh"
OUT_PREFIX = "results/uninstructed/"
RUNS_DIR = ROOT / "results/uninstructed/pod_runs"
GPU = "NVIDIA A40"
# USD per hour, read from the RunPod price list on 16 and 17 November 2026. The RTX A6000 is the workstation version of
# the A40 (same chip, 48 GB): it is the stand-in when no A40 can be deployed, as on the morning of 17 November.
GPU_RATES = {"NVIDIA A40": {"SECURE": 0.59, "COMMUNITY": 0.35}, "NVIDIA RTX A6000": {"SECURE": 0.69, "COMMUNITY": 0.33}}
GEN_S_PER_PROMPT = 1.58                            # measured on an A40 on 17 November 2026 UTC: versions S1 to S3, 1,728 prompts in 2,723 s,
                                                   # batches of 8 (version S0, whose prompts are shorter: 1.44 s)
STANCE_S_PER_MEMBER = 0.6                          # assumed: the room codes took 0.45 s per room on the A40 (336 rooms in 177 s with the model load)
PROMPTS_PER_EPISODE = 24                           # 8 room messages, 12 public turns, 4 votes; 2 fewer with a scripted first message
CODE_PATHS = ["uninstructed", "generation", "jspace_interp", "scripts"]


def estimate(a) -> dict:
    from uninstructed import protocol as P

    skip = {x for x in a.skip.split(",") if x}
    ref = [] if "gen_ref" in skip else [v for v in a.ref_versions.split(",") if v]
    versions = ref + ([] if "gen_stake" in skip else [v for v in a.versions.split(",") if v])
    unknown = [v for v in versions if v not in P.VERSIONS]
    if unknown:
        raise SystemExit(f"unknown versions: {unknown}")
    ids = [x for x in a.scenario_ids.split(",") if x]
    known = P.load_scenarios()[2]
    stakes = P.load_stakes()
    for sid in ids:
        if sid not in known:
            raise SystemExit(f"unknown scenario {sid}")
        for v in versions:                           # a version with a kind of stake sentences runs only where they exist
            if P.spec_of(v)["kind"] and sid not in stakes["kinds"][P.spec_of(v)["kind"]]:
                raise SystemExit(f"version {v} has no stake sentences for scenario {sid}")
    if not ids and any(P.spec_of(v)["kind"] for v in versions):
        raise SystemExit("a version with a kind of stake sentences needs --scenario-ids")
    n_scenarios = len(ids) if ids else len(P.load_scenarios()[1]) if a.scenarios == "all" else a.pilot_n
    per_version = {v: n_scenarios * ((a.ref_runs or a.pilot_runs) if v in ref else a.pilot_runs) for v in versions}
    episodes = sum(per_version.values())
    # public positions, coded by Gemma for this round and for the earlier rounds named in --stance-rounds
    stance_members = 0 if "stance" in skip else 4 * episodes + sum(
        4 * len(list((ROOT / OUT_PREFIX / rd / "runs").glob("*/*/run_config.json"))) for rd in a.stance_rounds.split(",") if rd)
    prompts = sum(per_version[v] * (PROMPTS_PER_EPISODE - (2 if P.spec_of(v)["opener"] else 0)) for v in versions)
    gen_min = prompts * GEN_S_PER_PROMPT / 60
    stages = {"boot, setup, downloads": 6, f"generation ({episodes} episodes, {prompts} prompts at {GEN_S_PER_PROMPT} s)": round(gen_min),
              "model loads": 3, f"room codes by Gemma ({2 * episodes} new rooms)": 3 + round(2 * episodes / 80),
              "timing test (basis export, replay, decomposition, probes)": 0 if "timing" in skip else 9,
              f"public positions by Gemma ({stance_members} members at {STANCE_S_PER_MEMBER} s)": (
                  0 if not stance_members else 1 + len([rd for rd in a.stance_rounds.split(",") if rd]) + round(stance_members * STANCE_S_PER_MEMBER / 60)),
              "package": 2}
    exp_h = sum(stages.values()) / 60
    cap_h = (a.max_minutes + a.grace_minutes + a.extra_minutes) / 60     # the extra minutes run only if the results are not fetched
    rate = GPU_RATES[a.gpu]
    return {"mode": a.mode, "gpu": a.gpu, "versions": versions, "scenarios": n_scenarios, "results_dir": f"results/uninstructed/{a.pilot_dir}",
            "episodes": episodes,
            "prompts": prompts, "stages_minutes": stages, "expected_hours": round(exp_h, 2),
            "expected_usd": {c: round(exp_h * r, 2) for c, r in rate.items()}, "time_cap_minutes": a.max_minutes, "grace_minutes": a.grace_minutes,
            "extra_minutes_if_not_fetched": a.extra_minutes, "worst_case_usd": round(cap_h * rate["SECURE"], 2)}


def sha_on_branch(sha: str) -> bool:
    rs = base.remote_sha(GIT_BRANCH)
    if rs == sha:
        return True
    subprocess.run(["git", "fetch", "-q", "origin", GIT_BRANCH], cwd=ROOT, capture_output=True)
    return subprocess.run(["git", "merge-base", "--is-ancestor", sha, f"origin/{GIT_BRANCH}"], cwd=ROOT, capture_output=True).returncode == 0


def code_dirty() -> list[str]:
    """Modified or untracked files in the folders the pod runs code from: they would not be on the pod."""
    r = subprocess.run(["git", "status", "--porcelain", "--"] + CODE_PATHS, cwd=ROOT, capture_output=True, text=True)
    return [ln for ln in r.stdout.splitlines() if ln.strip() and "__pycache__" not in ln]


def gates(a, est: dict) -> tuple[bool, dict]:
    me = base.account()
    balance = float(me.get("clientBalance") or 0.0)
    running = [p for p in (me.get("pods") or []) if p.get("desiredStatus") == "RUNNING" and (p.get("costPerHr") or 0) > 0.05]
    pushed = sha_on_branch(a.sha)
    dirty = code_dirty()
    worst = est["worst_case_usd"]
    rep = {"balance_usd": round(balance, 2), "min_balance": a.min_balance, "worst_case_usd": worst, "balance_after_worst_case": round(balance - worst, 2),
           "reserve": a.reserve, "spend_per_hr_idle": me.get("currentSpendPerHr"), "billable_pods_running": running, "sha": a.sha,
           "sha_on_origin_branch": pushed, "code_dirty": dirty, "hf_token_present": bool(os.environ.get("HF_TOKEN")),
           "gpu_market": base.gpu_market([a.gpu])}
    checks = ((balance < a.min_balance, f"balance below {a.min_balance}"), (balance - worst < a.reserve, f"worst case would leave less than {a.reserve}"),
              (bool(running), "a billable pod is running"), (not pushed, f"the SHA is not on origin/{GIT_BRANCH} (push first)"),
              (not os.environ.get("HF_TOKEN"), "HF_TOKEN missing"), (bool(dirty), "code folders have uncommitted or untracked files (commit first)"))
    rep["why_not"] = [m for cond, m in checks if cond]
    rep["gates_ok"] = not rep["why_not"]
    return rep["gates_ok"], rep


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
git init -q && git remote add origin "{base.GIT_REPO}" && git fetch -q --depth 1 origin {a.sha} >> $RUN/logs/clone.log 2>&1 || fail fetch
git checkout -q FETCH_HEAD >> $RUN/logs/clone.log 2>&1 || fail checkout
echo "checked out $(git rev-parse HEAD)" >> $RUN/logs/clone.log
exec bash {BOOT}
"""


def create_pod(a) -> dict:
    env = {"JOB_BOOT_B64": base64.b64encode(bootstrap_script(a).encode()).decode(), "RUNPOD_API_KEY": os.environ["RUNPOD_API_KEY"],
           "HF_TOKEN": os.environ["HF_TOKEN"], "GIT_SHA": a.sha, "JOB_MODE": a.mode, "JOB_MAX_MINUTES": str(a.max_minutes),
           "JOB_GRACE_MINUTES": str(a.grace_minutes), "JOB_EXTRA_MINUTES": str(a.extra_minutes), "JOB_PILOT_N": str(a.pilot_n), "JOB_PILOT_RUNS": str(a.pilot_runs),
           "JOB_BATCH": str(a.batch), "JOB_WORKERS": str(a.workers), "JOB_TIMING_EPISODES": str(a.timing_episodes),
           "JOB_VERSIONS": a.versions, "JOB_PILOT_DIR": a.pilot_dir, "JOB_SCENARIOS": a.scenarios, "JOB_STANCE_ROUNDS": a.stance_rounds,
           "JOB_REF_VERSIONS": a.ref_versions, "JOB_REF_RUNS": str(a.ref_runs or a.pilot_runs)}
    if a.scenario_ids:
        env["JOB_SCENARIO_IDS"] = a.scenario_ids
    if a.skip:
        env["JOB_SKIP"] = a.skip
    docker_args = "bash -c 'echo $JOB_BOOT_B64 | base64 -d > /tmp/boot.sh; exec bash /tmp/boot.sh'"
    attempts = []
    clouds = ["SECURE"] + ([] if a.secure_only else ["COMMUNITY"])
    for vcpu, mem in ((a.min_vcpu, a.min_memory_gb), (4, 24)):
        for cloud in clouds:
            inp = {"name": f"{POD_NAME}-{a.mode}", "imageName": base.IMAGE, "gpuTypeId": a.gpu, "cloudType": cloud, "gpuCount": 1,
                   "volumeInGb": 0, "containerDiskInGb": a.disk_gb, "minVcpuCount": vcpu, "minMemoryInGb": mem, "ports": "8765/http",
                   "env": [{"key": k, "value": v} for k, v in env.items()], "dockerArgs": docker_args, "supportPublicIp": False, "startSsh": False}
            if a.min_download > 0:
                inp["minDownload"] = a.min_download
            q = """mutation ($input: PodFindAndDeployOnDemandInput!) {
                  podFindAndDeployOnDemand(input: $input) { id desiredStatus costPerHr machine { podHostId gpuDisplayName dataCenterId cpuCount } } }"""
            try:
                r = base.graphql(q, {"input": inp})
                if r.get("errors"):
                    raise RuntimeError(r["errors"][0].get("message", "")[:200])
                pod = (r.get("data") or {}).get("podFindAndDeployOnDemand")
                if not pod or not pod.get("id"):
                    raise RuntimeError("empty pod")
                pod.update({"gpu": a.gpu, "cloud": cloud, "min_vcpu": vcpu, "min_memory_gb": mem})
                return {"created": pod, "attempts": attempts}
            except (RuntimeError, urllib.error.HTTPError) as e:
                attempts.append({"cloud": cloud, "vcpu": vcpu, "error": str(e)[:200]})
                print(f"[launch] {cloud} {a.gpu} (vcpu>={vcpu}): {str(e)[:200]}", file=sys.stderr, flush=True)
    return {"created": None, "attempts": attempts}


def safe_extract(tgz: Path) -> int:
    with tarfile.open(tgz) as t:
        members = [m for m in t.getmembers() if m.name.startswith(OUT_PREFIX) and ".." not in m.name.split("/") and not m.name.startswith("/")]
        t.extractall(ROOT, members=members)
    return len(members)


def pull(pod: str, full: bool = False, url: str | None = None) -> dict:
    """Fetch a pod's results. The small files come first (status, stages, log, provisional rates), then round.tgz, the
    archive of this round's folder and this pod's logs, which is extracted at once. results.tgz, the whole results
    folder, is fetched only with `full` or when the pod has no round.tgz (pods before 18 November 2026)."""
    url = url or f"https://{pod}-8765.proxy.runpod.net"
    dest = RUNS_DIR / pod
    dest.mkdir(parents=True, exist_ok=True)
    for name in ("status.json", "stages.json", "job.log", "provisional_summary.json"):
        b = base.fetch(f"{url}/{name}", 60)
        if b:
            (dest / name).write_bytes(b)
    man = base.fetch_json(f"{url}/manifest.json", 60)
    if man is None:
        raise SystemExit("no manifest.json on the pod yet (status.json says what phase it is in)")
    rec = {"pod": pod, "files": {}}
    names = [n for n in ("round.tgz",) if n in man]
    names += [n for n in man if n != "round.tgz" and (full or "round.tgz" not in man)]
    for name in names:
        info = man[name]
        r = base.download(f"{url}/{name}", dest / name, int(info["size"]), info["sha256"], max_tries=90 if name == "round.tgz" else 40)
        rec["files"][name] = {**info, **r}
        if name in ("round.tgz", "results.tgz"):
            rec[f"extracted_members_{name}"] = safe_extract(dest / name)
    (dest / "pull.json").write_text(json.dumps(rec, indent=1) + "\n")
    print(json.dumps(rec, indent=1))
    return rec


def terminate_checked(pod: str) -> None:
    print("terminating", json.dumps(base.terminate(pod)), flush=True)
    time.sleep(6)
    try:
        print("pod state after terminate:", json.dumps(base.pod_state(pod)), flush=True)
    except Exception as e:  # noqa: BLE001
        print("pod state unavailable:", type(e).__name__, flush=True)


def watch(pod: str, hours: float, do_terminate: bool, poll: int = 60, full: bool = False) -> int:
    t0, last, missing = time.time(), None, 0
    while time.time() - t0 < hours * 3600:
        st = base.fetch_json(f"https://{pod}-8765.proxy.runpod.net/status.json", 20)
        key = (st or {}).get("phase"), (st or {}).get("stage"), (st or {}).get("detail")
        if key != last:
            print(f"[{(time.time() - t0) / 60:5.0f} min] {json.dumps(st) if st else 'no status yet'}", flush=True)
            last = key
        if st and st.get("phase") in ("done", "failed"):
            print(f"pod reports {st.get('phase')}; pulling", flush=True)
            rc = 0 if st.get("phase") == "done" else 4
            try:
                pull(pod, full)
            except SystemExit as e:
                # the results are still on the pod and nowhere else: it is left to its own timer, not terminated here
                print(f"pull failed: {e}\nThe pod is left running until its own timer ends it; try again with --pull {pod}", file=sys.stderr, flush=True)
                return 5
            if do_terminate:
                terminate_checked(pod)
            return rc
        if int(time.time() - t0) % 600 < poll:      # every ten minutes: does the pod still exist?
            try:
                if base.pod_state(pod) is None:
                    missing += 1
                    if missing >= 2:
                        print("pod no longer exists", flush=True)
                        return 6
                else:
                    missing = 0
            except Exception:  # noqa: BLE001
                pass
        time.sleep(poll)
    print("watch window over", file=sys.stderr, flush=True)
    if do_terminate:                                 # never leave a billed pod behind
        terminate_checked(pod)
    return 7


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["pilot"], default="pilot")
    ap.add_argument("--gpu", choices=sorted(GPU_RATES), default=GPU, help="GPU type (default: the A40 of the approved plan)")
    ap.add_argument("--sha", default=None, help="commit the pod checks out (default: local HEAD; must be on origin)")
    ap.add_argument("--max-minutes", type=int, default=150, help="time cap on the pod; what is unfinished is packaged as is")
    ap.add_argument("--grace-minutes", type=int, default=30, help="how long the pod serves results before terminating itself")
    ap.add_argument("--extra-minutes", type=int, default=20, help="how much longer it serves them when round.tgz has not been fetched")
    ap.add_argument("--full-archive", action="store_true", help="also fetch results.tgz, the whole results folder (needed with --stance-rounds)")
    ap.add_argument("--pilot-n", type=int, default=12)
    ap.add_argument("--pilot-runs", type=int, default=2)
    ap.add_argument("--versions", default="S1,S2,S3", help="versions of the gen_stake stage (S0 is the gen_ref stage)")
    ap.add_argument("--pilot-dir", default="pilot", help="folder under results/uninstructed/ that receives the runs and codings")
    ap.add_argument("--scenarios", choices=["pilot", "all"], default="pilot", help="the --pilot-n pilot scenarios, or all scenarios of the bank")
    ap.add_argument("--scenario-ids", default="", help="comma list of scenario ids; replaces --scenarios (the fifth round's twelve)")
    ap.add_argument("--ref-versions", default="S0", help="the no-stake versions of the gen_ref stage")
    ap.add_argument("--ref-runs", type=int, default=0, help="runs per scenario of the gen_ref versions (default: --pilot-runs)")
    ap.add_argument("--stance-rounds", default="", help="comma list of earlier round folders whose public positions Gemma codes as well")
    ap.add_argument("--batch", type=int, default=8, help="prompts per generate call for the stake versions (16 ran out of GPU memory)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--timing-episodes", type=int, default=10)
    ap.add_argument("--skip", default="", help="comma list of pod stages to skip (JOB_SKIP)")
    ap.add_argument("--secure-only", action="store_true", help="do not fall back to the community cloud")
    ap.add_argument("--min-vcpu", type=int, default=8)
    ap.add_argument("--min-memory-gb", type=int, default=40)
    ap.add_argument("--disk-gb", type=int, default=60)
    ap.add_argument("--min-download", type=int, default=800)
    ap.add_argument("--min-balance", type=float, default=20.0, help="launch floor; the user approves any lower value explicitly")
    ap.add_argument("--reserve", type=float, default=5.0, help="what the worst case must leave on the account")
    ap.add_argument("--preflight", action="store_true")
    ap.add_argument("--launch", action="store_true", help="create the pod (spends money)")
    ap.add_argument("--watch", action="store_true", help="after --launch: poll, pull when done, terminate")
    ap.add_argument("--watch-hours", type=float, default=3.4)
    ap.add_argument("--no-terminate", action="store_true")
    ap.add_argument("--status", metavar="POD_ID")
    ap.add_argument("--watch-pod", metavar="POD_ID")
    ap.add_argument("--pull", metavar="POD_ID")
    ap.add_argument("--terminate", metavar="POD_ID")
    a = ap.parse_args()
    a.sha = a.sha or base.local_sha()

    if a.status:
        base.status(a.status)
        return 0
    if a.terminate:
        terminate_checked(a.terminate)
        return 0
    if a.pull:
        pull(a.pull, a.full_archive)
        if not a.no_terminate:
            terminate_checked(a.pull)
        return 0
    if a.watch_pod:
        return watch(a.watch_pod, a.watch_hours, not a.no_terminate, full=a.full_archive)

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
    res = create_pod(a)
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    if not res["created"]:
        print(f"no {a.gpu} could be deployed", file=sys.stderr)
        (RUNS_DIR / f"failed_launch_{stamp}.json").write_text(json.dumps(res, indent=1) + "\n")
        return 3
    pod = res["created"]["id"]
    rec = {"created_utc": stamp, "pod": res["created"], "attempts_before": res["attempts"], "sha": a.sha, "branch": GIT_BRANCH, "image": base.IMAGE,
           "settings": {k: getattr(a, k) for k in ("mode", "gpu", "max_minutes", "grace_minutes", "extra_minutes", "pilot_n", "pilot_runs", "versions", "pilot_dir", "scenarios", "scenario_ids", "stance_rounds", "ref_versions", "ref_runs",
                                                     "batch", "workers", "timing_episodes", "skip", "min_balance", "reserve")},
           "estimate": est, "balance_before": rep["balance_usd"], "status_url": f"https://{pod}-8765.proxy.runpod.net/status.json"}
    (RUNS_DIR / pod).mkdir(parents=True, exist_ok=True)
    (RUNS_DIR / pod / "launch.json").write_text(json.dumps(rec, indent=1) + "\n")
    print(json.dumps(rec, indent=1), flush=True)
    if a.watch:
        return watch(pod, a.watch_hours, not a.no_terminate, full=a.full_archive or bool(a.stance_rounds))
    return 0


if __name__ == "__main__":
    sys.exit(main())
