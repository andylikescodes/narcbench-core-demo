#!/usr/bin/env python3
"""Poll a matched-prefix RunPod job, fetch its results when it ends, optionally terminate it.

The boot script serves, on the pod's port 8765 (RunPod proxy
https://<pod>-8765.proxy.runpod.net): /status.json, /job.log, /cmd.log and
/out/{meta.json,summary.json,per_pair.json,RESULTS.md} for the current run.
At the end the boot also appends every result file to job.log between
"=====BEGIN <name>=====" / "=====END <name>=====" markers, so a single
/job.log fetch is enough even if /out/ cannot be reached.

  python3 scripts/watch_matched_prefix_pod.py --card results/transfer_stable/MATCHED_PREFIX_LAYER_SWEEP_LAUNCH.json \\
      --dest docs/handoff/2026-10-05-layer-sweep
  python3 scripts/watch_matched_prefix_pod.py --pod abc123 --dest /tmp/run --follow --poll 60 --max-minutes 120
  python3 scripts/watch_matched_prefix_pod.py --pod abc123 --dest /tmp/run --terminate-after-fetch

Exit codes: 0 finished and results fetched · 2 job failed (log fetched) ·
10 still running (or booting) · 30 status URL unreachable · 1 bad arguments.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

RESULT_FILES = ("meta.json", "summary.json", "per_pair.json", "RESULTS.md")
BLOCK_RE = re.compile(r"=====BEGIN (?P<name>[A-Za-z_.]+)=====\n(?P<body>.*?)\n=====END (?P=name)=====", re.S)
TERMINAL = {"done", "failed"}
UA = "nb-matched-prefix-watch/1"


def status_url_for(pod_id: str) -> str:
    return f"https://{pod_id}-8765.proxy.runpod.net"


def fetch(url: str, timeout: int = 30) -> bytes | None:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read()
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError):
        return None


def blocks_from_job_log(text: str) -> dict[str, str]:
    return {m.group("name"): m.group("body") for m in BLOCK_RE.finditer(text)}


def read_status(base: str) -> dict | None:
    raw = fetch(f"{base}/status.json", timeout=20)
    if raw is None:
        return None
    try:
        return json.loads(raw.decode("utf-8", "replace"))
    except json.JSONDecodeError:
        return {"phase": "unknown", "raw": raw[:200].decode("utf-8", "replace")}


def fetch_results(base: str, dest: Path) -> dict:
    """Save job.log, cmd.log and the result files; fall back to the job.log blocks."""
    dest.mkdir(parents=True, exist_ok=True)
    got: dict[str, str] = {}
    log = fetch(f"{base}/job.log", timeout=60)
    if log is not None:
        (dest / "job.log").write_bytes(log)
        got["job.log"] = "fetched"
    cmd = fetch(f"{base}/cmd.log", timeout=60)
    if cmd is not None:
        (dest / "cmd.log").write_bytes(cmd)
        got["cmd.log"] = "fetched"
    for name in RESULT_FILES:
        raw = fetch(f"{base}/out/{name}", timeout=60)
        if raw is not None:
            (dest / name).write_bytes(raw)
            got[name] = "fetched from /out/"
    if log is not None:
        blocks = blocks_from_job_log(log.decode("utf-8", "replace"))
        for name, body in blocks.items():
            if name in RESULT_FILES and name not in got:
                (dest / name).write_text(body + "\n")
                got[name] = "recovered from job.log"
    return got


def summary_line(dest: Path) -> str:
    p = dest / "summary.json"
    if not p.is_file():
        return "no summary.json"
    try:
        s = json.loads(p.read_text())
    except json.JSONDecodeError:
        return "summary.json unreadable"
    bits = [f"n_pairs={s.get('n_pairs')}", f"protocol={s.get('protocol')}", f"elapsed={s.get('elapsed_sec')}s"]
    for key in ("first_site_passing_h2c_bar", "edits_passing_h2c_bar", "stopped_after"):
        if key in s:
            bits.append(f"{key}={s[key]}")
    for table in ("directions", "sweep", "sites"):
        rows = s.get(table) or []
        if rows:
            best = max(rows, key=lambda r: (r.get("h2c_fraction_of_gap") or -1e9))
            bits.append(f"{table}: best={best.get('label')} h2c_fraction={best.get('h2c_fraction_of_gap')} flips={best.get('h2c_flips')}")
    return " | ".join(bits)


def terminate(pod_id: str) -> bool:
    key = os.environ.get("RUNPOD_API_KEY")
    if not key:
        print("[terminate] RUNPOD_API_KEY missing; not terminating", file=sys.stderr)
        return False
    body = json.dumps({"query": 'mutation { podTerminate(input: {podId: "%s"}) }' % pod_id}).encode()
    req = urllib.request.Request(
        "https://api.runpod.io/graphql",
        data=body,
        headers={"content-type": "application/json", "Authorization": "Bearer " + key, "User-Agent": UA},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            resp.read()
        return True
    except Exception as exc:
        print(f"[terminate] failed: {type(exc).__name__}", file=sys.stderr)
        return False


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--card", type=Path, default=None, help="launch card JSON written by the launcher")
    ap.add_argument("--pod", default=None, help="pod id (alternative to --card)")
    ap.add_argument("--dest", type=Path, required=True, help="where to save job.log and the result files")
    ap.add_argument("--follow", action="store_true", help="keep polling until the job ends or --max-minutes")
    ap.add_argument("--poll", type=int, default=60, help="seconds between polls with --follow")
    ap.add_argument("--max-minutes", type=int, default=120)
    ap.add_argument("--terminate-after-fetch", action="store_true", help="podTerminate once results are saved (needs RUNPOD_API_KEY)")
    args = ap.parse_args()

    pod_id = args.pod
    card = None
    if args.card is not None:
        card = json.loads(args.card.read_text())
        pod_id = pod_id or card.get("pod_id")
    if not pod_id or not re.fullmatch(r"[A-Za-z0-9]+", pod_id):
        print("ERROR: need --pod <id> or a launch card with pod_id", file=sys.stderr)
        return 1
    base = status_url_for(pod_id)
    deadline = time.time() + args.max_minutes * 60
    while True:
        status = read_status(base)
        stamp = datetime.now(timezone.utc).strftime("%H:%M:%SZ")
        if status is None:
            print(f"[{stamp}] {pod_id}: status unreachable", flush=True)
            if not args.follow or time.time() > deadline:
                return 30
        else:
            phase = status.get("phase", "unknown")
            detail = status.get("detail", "")
            print(f"[{stamp}] {pod_id}: phase={phase} {detail}".rstrip(), flush=True)
            if phase in TERMINAL:
                got = fetch_results(base, args.dest)
                (args.dest / "pod_status.json").write_text(
                    json.dumps(
                        {
                            "pod_id": pod_id,
                            "status_url": base,
                            "phase": phase,
                            "detail": detail,
                            "fetched_at": datetime.now(timezone.utc).isoformat(),
                            "files": got,
                            "launch_card": card,
                        },
                        indent=2,
                    )
                    + "\n"
                )
                print(f"[fetch] {', '.join(f'{k}: {v}' for k, v in got.items()) or 'nothing fetched'}", flush=True)
                print(f"[summary] {summary_line(args.dest)}", flush=True)
                if args.terminate_after_fetch and all(n in got for n in RESULT_FILES):
                    print(f"[terminate] {'ok' if terminate(pod_id) else 'not terminated'}", flush=True)
                return 0 if phase == "done" else 2
            if not args.follow or time.time() > deadline:
                return 10
        time.sleep(max(15, args.poll))


if __name__ == "__main__":
    raise SystemExit(main())
