#!/usr/bin/env python3
"""Poll Core pod lnqm4yawhv13on; on done, pull transcripts and launch separate full extract.

Does NOT modify/attach to the Core pod — only reads status/results.tgz then starts a NEW pod.
"""
from __future__ import annotations
import json, os, subprocess, sys, time, urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORE = "lnqm4yawhv13on"
STATUS = f"https://{CORE}-8765.proxy.runpod.net/status.json"
RESULTS = f"https://{CORE}-8765.proxy.runpod.net/results.tgz"
SECRETS = Path("/home/box/.secrets/interp-explorer.env")
STATE = ROOT / "results/stage1/core_watcher_state.json"
POLL_S = 120


def env():
    if SECRETS.is_file():
        for line in SECRETS.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def get_json(url: str, timeout: int = 30):
    req = urllib.request.Request(url, headers={"User-Agent": "nb-watch/1"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def save(state: dict):
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(state, indent=2) + "\n")


def pull_core():
    dest = ROOT / "upstream/scenarios/gemma2_9b/core"
    dest.mkdir(parents=True, exist_ok=True)
    tgz = Path("/tmp/narcbench_core_full_results.tgz")
    print(f"[watch] downloading {RESULTS}", flush=True)
    urllib.request.urlretrieve(RESULTS, tgz)
    print(f"[watch] tgz size={tgz.stat().st_size}", flush=True)
    # unpack — structure usually contains the run_id dir
    import tarfile
    with tarfile.open(tgz, "r:gz") as tf:
        tf.extractall(Path("/tmp/narcbench_core_full_unpacked"))
    # find run dir with batch_report
    unpacked = Path("/tmp/narcbench_core_full_unpacked")
    candidates = list(unpacked.rglob("batch_report.json"))
    if not candidates:
        # maybe flat run dirs
        print("[watch] no batch_report; listing", flush=True)
        for p in sorted(unpacked.rglob("*"))[:40]:
            print(" ", p, flush=True)
        raise SystemExit("cannot locate full Core results")
    run_dir = candidates[0].parent
    run_id = run_dir.name
    target = dest / run_id
    if target.exists():
        print(f"[watch] target exists {target}", flush=True)
    else:
        import shutil
        shutil.copytree(run_dir, target)
    # symlink latest
    latest = dest / "latest"
    if latest.is_symlink() or latest.exists():
        latest.unlink()
    latest.symlink_to(run_id)
    print(f"[watch] pulled → {target}", flush=True)
    return run_id


def launch_full_extract(run_id: str):
    cmd = [
        sys.executable,
        str(ROOT / "scripts/runpod_launch_extract_acts.py"),
        "--runs-rel", f"gemma2_9b/core/{run_id}",
        "--model", "google/gemma-2-9b-it",
        "--layers", "19-23",
        "--gen-only",
        "--kind", "full",
        "--max-minutes", "360",
        "--launch",
    ]
    print("[watch] launching full extract:", " ".join(cmd), flush=True)
    r = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
    print(r.stdout, flush=True)
    print(r.stderr, flush=True)
    if r.returncode != 0:
        raise SystemExit(f"full extract launch failed rc={r.returncode}")


def main():
    env()
    state = {"phase": "watching", "core": CORE, "started_pt": time.strftime("%Y-%m-%d %H:%M:%S")}
    save(state)
    print(f"[watch] polling {STATUS} every {POLL_S}s; will NOT touch Core process", flush=True)
    while True:
        try:
            st = get_json(STATUS)
        except Exception as e:
            print(f"[watch] status err: {e}", flush=True)
            time.sleep(POLL_S)
            continue
        phase = st.get("phase")
        print(f"[watch] {time.strftime('%H:%M:%S')} phase={phase} raw={st}", flush=True)
        state["last_status"] = st
        state["last_poll_pt"] = time.strftime("%Y-%m-%d %H:%M:%S")
        save(state)
        if phase == "done":
            state["phase"] = "pulling"
            save(state)
            run_id = pull_core()
            state["phase"] = "launching_full_extract"
            state["run_id"] = run_id
            save(state)
            launch_full_extract(run_id)
            state["phase"] = "full_extract_launched"
            save(state)
            print("[watch] done — full extract launched on separate pod", flush=True)
            return 0
        if phase == "failed":
            state["phase"] = "core_failed"
            save(state)
            print("[watch] Core failed — stop", flush=True)
            return 1
        time.sleep(POLL_S)


if __name__ == "__main__":
    raise SystemExit(main())
