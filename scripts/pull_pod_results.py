#!/usr/bin/env python3
"""Pull results.tgz from a finished narcbench job pod proxy; terminate if still up."""
from __future__ import annotations
import argparse, json, os, sys, tarfile, tempfile, urllib.request
from pathlib import Path

SECRETS = Path("/home/box/.secrets/interp-explorer.env")
UA = "Mozilla/5.0"

def read_env():
    if SECRETS.is_file():
        for line in SECRETS.read_text().splitlines():
            if "=" in line and not line.strip().startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

def gql(q, variables=None):
    body = {"query": q}
    if variables:
        body["variables"] = variables
    req = urllib.request.Request(
        "https://api.runpod.io/graphql",
        data=json.dumps(body).encode(),
        headers={"content-type": "application/json", "Authorization": "Bearer " + os.environ["RUNPOD_API_KEY"], "User-Agent": UA},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode())

def status(pid):
    url = f"https://{pid}-8765.proxy.runpod.net/status.json"
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode())

def download(pid, dest: Path):
    url = f"https://{pid}-8765.proxy.runpod.net/results.tgz"
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=600) as r:
        dest.write_bytes(r.read())
    return dest.stat().st_size

def terminate(pid):
    q = 'mutation { podTerminate(input: {podId: "%s"}) }' % pid
    try:
        return gql(q)
    except Exception as e:
        return {"error": str(e)}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pod", required=True)
    ap.add_argument("--extract-to", type=Path, required=True)
    ap.add_argument("--terminate", action="store_true")
    ap.add_argument("--require-done", action="store_true")
    args = ap.parse_args()
    read_env()
    st = status(args.pod)
    print("status", st)
    if args.require_done and st.get("phase") != "done":
        print("not done yet", file=sys.stderr)
        return 2
    args.extract_to.mkdir(parents=True, exist_ok=True)
    tgz = args.extract_to / f"{args.pod}_results.tgz"
    n = download(args.pod, tgz)
    print(f"downloaded {n} bytes -> {tgz}")
    with tarfile.open(tgz, "r:gz") as tar:
        tar.extractall(args.extract_to / "untar")
    print("extracted to", args.extract_to / "untar")
    if args.terminate:
        print("terminate", terminate(args.pod))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
