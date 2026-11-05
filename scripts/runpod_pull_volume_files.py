#!/usr/bin/env python3
"""Copy files off the RunPod network volume with a short, cheap pod.

The lane's network volume (cr1kjtzjs2, EU-RO-1) can only be read from a pod in its data centre. This starts the cheapest
secure-cloud GPU type on offer there with the volume at /workspace, inventories the volume (every activations_gen.npz /
metadata_gen.json, the j_basis tree, du of the job folders), serves the whole volume read-only over the pod's HTTP proxy
(a small Range-capable server, so downloads resume, with /sha256?path= and /ls?path= endpoints), downloads the files that
match the spec, verifies every checksum against the pod's hash and terminates the pod. If a wanted file is not found the
pod is left up (it terminates itself after --max-minutes) so the inventory can be read and the right paths pulled with
--pull/--path. No GPU work is done; the GPU is the price of admission to the volume.

Keys from the environment only (RUNPOD_API_KEY). Estimate-only by default; --launch spends. Gates as in
runpod_launch_jlens_full.py: balance >= --min-balance and no billable pod already running (unless --allow-existing).

    python3 scripts/runpod_pull_volume_files.py --spec gemma_acts                  # plan: files, GPU order, worst-case cost
    RUNPOD_API_KEY=... python3 scripts/runpod_pull_volume_files.py --spec gemma_acts --launch   # launch, wait, pull, verify, terminate
    python3 scripts/runpod_pull_volume_files.py --status <pod id>
    RUNPOD_API_KEY=... python3 scripts/runpod_pull_volume_files.py --pull <pod id>  # pull from a pod that is already up, then terminate
    RUNPOD_API_KEY=... python3 scripts/runpod_pull_volume_files.py --terminate <pod id>

A spec is a list of path SUFFIXES (e.g. gemma2_9b/core/<run>/activations_gen.npz); each is matched against the pod's
inventory, preferring /workspace/jobs/narcbench-results/. --path adds an absolute remote path or another suffix, with an
optional =local/relative/path. Local destinations: an activations_gen.npz / metadata_gen.json suffix goes to
data/activations/<suffix>; anything else to data/volume_pull/<remote path>. Everything lands under data/ (git-ignored);
the pod record goes to results/jspace/, the inventory to data/volume_pull/_inventory/<utc stamp>/.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IMAGE = "python:3.12-slim"  # ~50 MB; nothing but python3, bash and findutils is needed
VOLUME_ID = "cr1kjtzjs2"
VOLUME_REPO = "/workspace/narcbench-core-demo"
POD_NAME = "job-narcbench-pull-volume"
RESULTS_ROOT = "/workspace/jobs/narcbench-results"  # where the pods wrote; the repo clone on the volume holds no data
SPECS = {  # path suffixes, matched against the inventory
    "gemma_acts": [
        "gemma2_9b/core/20261001T012639Z/activations_gen.npz",
        "gemma2_9b/core/20261001T012639Z/metadata_gen.json",
        "gemma2_9b/transfer/RUNPOD/activations_gen.npz",
        "gemma2_9b/transfer/RUNPOD/metadata_gen.json",
    ],
}
# Cheapest first; secure cloud only (network volumes do not attach to community pods). Rates: gpuTypes.securePrice, 2026-11-05.
GPU_PREFS = [
    "NVIDIA RTX 2000 Ada Generation", "NVIDIA RTX A4000", "NVIDIA RTX A4500", "NVIDIA RTX A5000",
    "NVIDIA RTX 4000 Ada Generation", "NVIDIA GeForce RTX 3090", "NVIDIA RTX PRO 4000 Blackwell", "NVIDIA A40",
    "NVIDIA RTX A6000", "NVIDIA L4", "NVIDIA RTX PRO 4500 Blackwell", "NVIDIA GeForce RTX 4090",
]
RATES = {
    "NVIDIA RTX 2000 Ada Generation": 0.24, "NVIDIA RTX A4000": 0.25, "NVIDIA RTX A4500": 0.25, "NVIDIA RTX A5000": 0.27,
    "NVIDIA RTX 4000 Ada Generation": 0.28, "NVIDIA GeForce RTX 3090": 0.50, "NVIDIA RTX PRO 4000 Blackwell": 0.57,
    "NVIDIA A40": 0.59, "NVIDIA RTX A6000": 0.59, "NVIDIA L4": 0.59, "NVIDIA RTX PRO 4500 Blackwell": 0.72,
    "NVIDIA GeForce RTX 4090": 0.89,
}
INVENTORY = ["inventory.tsv", "listing_j_basis.tsv", "du_workspace.txt", "du_jobs.txt"]

# Range-capable static server (python -m http.server ignores Range, so a dropped 1 GB transfer would restart from zero).
SERVER_PY = r"""
import http.server, os, re, socketserver
import hashlib, json, urllib.parse
class H(http.server.SimpleHTTPRequestHandler):
    def _json(self, obj, code=200):
        b = json.dumps(obj).encode()
        self.send_response(code); self.send_header('Content-Type', 'application/json'); self.send_header('Content-Length', str(len(b)))
        self.end_headers(); self.wfile.write(b)
    def do_GET(self):
        u = urllib.parse.urlsplit(self.path)
        if u.path in ('/sha256', '/ls'):
            q = urllib.parse.parse_qs(u.query).get('path', [''])[0]
            real = os.path.realpath(q)
            if not real.startswith('/workspace/'):
                return self._json({'error': 'path must be under /workspace'}, 400)
            if u.path == '/ls':
                if not os.path.isdir(real):
                    return self._json({'error': 'not a directory', 'path': q}, 404)
                rows = []
                for n in sorted(os.listdir(real)):
                    fp = os.path.join(real, n)
                    try:
                        st = os.stat(fp); rows.append({'name': n, 'dir': os.path.isdir(fp), 'size': st.st_size})
                    except OSError:
                        pass
                return self._json({'path': q, 'entries': rows})
            if not os.path.isfile(real):
                return self._json({'error': 'not a file', 'path': q}, 404)
            h = hashlib.sha256()
            with open(real, 'rb') as f:
                for b in iter(lambda: f.read(1 << 24), b''):
                    h.update(b)
            return self._json({'path': q, 'size': os.path.getsize(real), 'sha256': h.hexdigest()})
        return super().do_GET()
    def send_head(self):
        path = self.translate_path(self.path)
        if os.path.isdir(path):
            return super().send_head()
        try:
            f = open(path, 'rb')
        except OSError:
            self.send_error(404); return None
        size = os.fstat(f.fileno()).st_size
        start, end = 0, size - 1
        m = re.match(r'bytes=(\d*)-(\d*)$', self.headers.get('Range') or '')
        if m and (m.group(1) or m.group(2)):
            if m.group(1):
                start = int(m.group(1)); end = int(m.group(2)) if m.group(2) else size - 1
            else:
                start = max(0, size - int(m.group(2)))
            if start >= size:
                f.close(); self.send_error(416); return None
            end = min(end, size - 1)
            self.send_response(206); self.send_header('Content-Range', 'bytes %d-%d/%d' % (start, end, size))
        else:
            self.send_response(200)
        self.send_header('Content-Type', self.guess_type(path)); self.send_header('Content-Length', str(end - start + 1))
        self.send_header('Accept-Ranges', 'bytes'); self.end_headers()
        f.seek(start); self._left = end - start + 1
        return f
    def copyfile(self, src, dst):
        n = getattr(self, '_left', None)
        if n is None:
            return super().copyfile(src, dst)
        while n > 0:
            b = src.read(min(1 << 20, n))
            if not b: break
            dst.write(b); n -= len(b)
    def log_message(self, *a):
        pass
class S(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True; allow_reuse_address = True
os.chdir('/tmp/pull'); S(('', 8765), H).serve_forever()
"""



def graphql(query: str, variables: dict | None = None, timeout: int = 120) -> dict:
    key = os.environ.get("RUNPOD_API_KEY")
    if not key:
        raise SystemExit("RUNPOD_API_KEY is not set in the environment")
    body = json.dumps({"query": query, "variables": variables or {}}).encode()
    req = urllib.request.Request("https://api.runpod.io/graphql", data=body, method="POST",
                                 headers={"content-type": "application/json", "Authorization": f"Bearer {key}", "User-Agent": "nb-pull/1"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def boot_script(max_minutes: int) -> str:
    srv = base64.b64encode(SERVER_PY.encode()).decode()
    return f"""#!/bin/bash
set -uo pipefail
RUN=/tmp/pull; mkdir -p "$RUN"; ln -sfn /workspace "$RUN/ws"
st() {{ echo "$1" > "$RUN/status.json"; echo "[status] $1" >> "$RUN/job.log"; }}
echo '{srv}' | base64 -d > /tmp/server.py
python3 /tmp/server.py >/tmp/http.log 2>&1 &
st '{{"phase":"boot"}}'
term() {{ python3 - <<'PY' || true
import os, json, urllib.request as u
p=os.environ.get('RUNPOD_POD_ID',''); k=os.environ.get('RUNPOD_API_KEY','')
b=json.dumps({{'query':'mutation {{ podTerminate(input: {{podId: "%s"}}) }}'%p}}).encode()
r=u.Request('https://api.runpod.io/graphql',data=b,headers={{'content-type':'application/json','Authorization':'Bearer '+k,'User-Agent':'nb-pull/1'}},method='POST')
u.urlopen(r,timeout=30).read()
PY
}}
st '{{"phase":"inventory"}}'
du -sh /workspace/* 2>/dev/null > "$RUN/du_workspace.txt"
du -sh /workspace/jobs/* /workspace/jobs/narcbench-results/* 2>/dev/null > "$RUN/du_jobs.txt"
find /workspace/jobs/narcbench-results/j_basis -type f -printf '%s\\t%TY-%Tm-%Td %TH:%TM\\t%p\\n' 2>/dev/null | sort -k3 > "$RUN/listing_j_basis.tsv"
find /workspace -xdev -maxdepth 10 \\( -path /workspace/interp-demo/models -o -path '/workspace/interp-demo/.venv*' \\) -prune -o -type f \\( -name activations_gen.npz -o -name metadata_gen.json -o -name 'activations_*.npz' -o -name 'metadata_*.json' \\) -printf '%s\\t%TY-%Tm-%Td %TH:%TM\\t%p\\n' 2>/dev/null | sort -k3 > "$RUN/inventory.tsv"
st "{{\\"phase\\":\\"ready\\",\\"inventory_files\\":$(wc -l < "$RUN/inventory.tsv")}}"
sleep {max_minutes}m
st '{{"phase":"timeout"}}'
term
"""


def is_suffix(wanted: str) -> bool:
    return not wanted.startswith("/")


def local_path(wanted: str, remote: str, overrides: dict[str, str]) -> Path:
    if wanted in overrides:
        return ROOT / overrides[wanted]
    if is_suffix(wanted):
        return ROOT / "data/activations" / wanted
    if "/activations/" in remote:
        return ROOT / "data/activations" / remote.split("/activations/", 1)[1]
    return ROOT / "data/volume_pull" / remote.lstrip("/")


def fetch_json(url: str, timeout: int = 30) -> dict | None:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "nb-pull/1"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode())
    except Exception:  # noqa: BLE001  (proxy 404/502 pages before the container is up, timeouts)
        return None


def fetch_text(url: str, timeout: int = 120) -> str | None:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "nb-pull/1"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.read().decode("utf-8", "replace")
    except Exception:  # noqa: BLE001
        return None


def sha256_of(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def download(url: str, dest: Path, size: int, sha: str) -> dict:
    """Resumable download of one file to dest (via dest.part), verified against size and sha256."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size == size and sha256_of(dest) == sha:
        print(f"  {dest.relative_to(ROOT)}: already present and verified", flush=True)
        return {"status": "present"}
    part = dest.with_name(dest.name + ".part")
    have = part.stat().st_size if part.exists() else 0
    if have > size:
        part.unlink(); have = 0
    t0, start_have, tries = time.time(), have, 0
    while have < size:
        hdr = {"User-Agent": "nb-pull/1"}
        if have:
            hdr["Range"] = f"bytes={have}-"
        req = urllib.request.Request(url, headers=hdr)
        try:
            with urllib.request.urlopen(req, timeout=180) as r, open(part, "ab" if have else "wb") as f:
                if have and r.status != 206:  # server ignored the Range header: start over
                    f.seek(0); f.truncate(); have = 0
                last = time.time()
                while True:
                    b = r.read(1 << 22)
                    if not b:
                        break
                    f.write(b); have += len(b)
                    if time.time() - last > 15:
                        rate = (have - start_have) / max(time.time() - t0, 1e-6) / 1e6
                        print(f"  {dest.name}: {have/1e6:,.0f}/{size/1e6:,.0f} MB  {rate:.1f} MB/s  eta {(size-have)/max(rate,1e-6)/1e6:.0f}s", flush=True)
                        last = time.time()
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
            tries += 1
            if tries > 30:
                raise SystemExit(f"{dest.name}: giving up after {tries} attempts: {e}")
            print(f"  {dest.name}: {type(e).__name__} after {have/1e6:,.0f} MB; retry {tries} in 10 s", flush=True)
            time.sleep(10)
            have = part.stat().st_size if part.exists() else 0
    got = sha256_of(part)
    if part.stat().st_size != size or got != sha:
        raise SystemExit(f"{dest.name}: verification failed (size {part.stat().st_size} vs {size}, sha {got[:16]} vs {sha[:16]}); kept {part}")
    part.replace(dest)
    el = time.time() - t0
    print(f"  {dest.relative_to(ROOT)}: {size/1e6:,.0f} MB in {el:.0f}s ({(size-start_have)/max(el,1e-6)/1e6:.1f} MB/s), sha256 ok", flush=True)
    return {"status": "downloaded", "seconds": round(el, 1)}


def read_inventory(base: str, inventory_dir: Path) -> list[dict]:
    """Save the pod's inventory files locally and parse inventory.tsv (size, mtime, path)."""
    inventory_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for name in INVENTORY:
        txt = fetch_text(f"{base}/{name}")
        if txt is None:
            print(f"  inventory {name}: not reachable", file=sys.stderr); continue
        (inventory_dir / name).write_text(txt)
        if name == "inventory.tsv":
            for line in txt.splitlines():
                parts = line.split("\t")
                if len(parts) == 3:
                    rows.append({"size": int(parts[0]), "mtime": parts[1], "path": parts[2]})
    return rows


def resolve(wanted: list[str], inventory: list[dict]) -> tuple[dict[str, str], list[str], dict[str, list[str]]]:
    """Match each wanted entry (absolute path or suffix) against the inventory: preferred paths under RESULTS_ROOT."""
    found: dict[str, str] = {}
    missing: list[str] = []
    alternatives: dict[str, list[str]] = {}
    for w in wanted:
        if not is_suffix(w):
            found[w] = w  # checked against the pod when hashed
            continue
        cands = [r["path"] for r in inventory if r["path"].endswith("/" + w)]
        if not cands:
            missing.append(w); continue
        pref = [c for c in cands if c.startswith(RESULTS_ROOT + "/")] or cands
        pref.sort(key=len)
        found[w] = pref[0]
        if len(cands) > 1:
            alternatives[w] = [c for c in cands if c != pref[0]]
    return found, missing, alternatives


def pull_all(pod: str, wanted: list[str], overrides: dict[str, str], inventory_dir: Path) -> dict:
    base = f"https://{pod}-8765.proxy.runpod.net"
    st = fetch_json(f"{base}/status.json")
    if not st or st.get("phase") != "ready":
        raise SystemExit(f"pod is not ready (status {st}); try --status")
    inventory = read_inventory(base, inventory_dir)
    print(f"  inventory: {len(inventory)} activation/metadata files on the volume (saved under {inventory_dir.relative_to(ROOT)})", flush=True)
    found, missing, alternatives = resolve(wanted, inventory)
    rec: dict = {"files": [], "missing": missing, "alternatives": alternatives, "inventory_dir": str(inventory_dir.relative_to(ROOT))}
    for w, remote in found.items():
        h = fetch_json(f"{base}/sha256?path={urllib.parse.quote(remote)}", timeout=600)
        if not h or "sha256" not in h:
            print(f"  {remote}: no hash from the pod ({h}); treated as missing", file=sys.stderr)
            rec["missing"].append(w); continue
        dest = local_path(w, remote, overrides)
        r = download(f"{base}/ws{urllib.parse.quote(remote[len('/workspace'):])}", dest, int(h["size"]), h["sha256"])
        rec["files"].append({"wanted": w, "remote": remote, "local": str(dest.relative_to(ROOT)), "size": h["size"], "sha256": h["sha256"], **r})
    for w in rec["missing"]:
        print(f"  MISSING on the volume: {w}", file=sys.stderr)
    for w, alts in alternatives.items():
        print(f"  note: {w} also exists at {alts}", file=sys.stderr)
    return rec


def terminate(pod: str) -> dict:
    return graphql('mutation { podTerminate(input: {podId: "%s"}) }' % pod)


def pod_state(pod: str) -> dict | None:
    r = graphql('query { pod(input: {podId: "%s"}) { id desiredStatus costPerHr runtime { uptimeInSeconds } machine { podHostId gpuDisplayName dataCenterId } } }' % pod)
    return (r.get("data") or {}).get("pod")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spec", choices=sorted(SPECS), default="gemma_acts")
    ap.add_argument("--path", action="append", default=[], help="extra remote file, optionally =local/relative/path")
    ap.add_argument("--gpu", default=None, help="force one GPU type id")
    ap.add_argument("--max-minutes", type=int, default=60, help="the pod terminates itself after this long")
    ap.add_argument("--boot-timeout", type=int, default=15, help="minutes to wait for the pod to serve its manifest before giving up (terminates it)")
    ap.add_argument("--min-balance", type=float, default=20.0)
    ap.add_argument("--allow-existing", action="store_true", help="launch even if a billable pod is already running")
    ap.add_argument("--keep-pod", action="store_true", help="do not terminate the pod after a verified pull")
    ap.add_argument("--terminate-on-missing", action="store_true", help="terminate even if some wanted files were not found")
    ap.add_argument("--record", type=Path, default=None, help="pod record JSON (default results/jspace/volume_pull_<spec>.json)")
    ap.add_argument("--launch", action="store_true", help="create the pod, wait, pull, verify, terminate (spends money)")
    ap.add_argument("--status", metavar="POD_ID")
    ap.add_argument("--pull", metavar="POD_ID")
    ap.add_argument("--terminate", metavar="POD_ID")
    a = ap.parse_args()

    overrides: dict[str, str] = {}
    paths = list(SPECS[a.spec])
    for p in a.path:
        remote, _, local = p.partition("=")
        paths.append(remote)
        if local:
            overrides[remote] = local
    record = a.record or (ROOT / f"results/jspace/volume_pull_{a.spec}.json")
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    inventory_dir = ROOT / "data/volume_pull/_inventory" / stamp

    if a.status:
        print(json.dumps({"pod": pod_state(a.status) if os.environ.get("RUNPOD_API_KEY") else "(no RUNPOD_API_KEY: pod state skipped)",
                          "status": fetch_json(f"https://{a.status}-8765.proxy.runpod.net/status.json")}, indent=2))
        return 0
    if a.terminate:
        print(json.dumps(terminate(a.terminate)))
        return 0
    if a.pull:
        rec = pull_all(a.pull, paths, overrides, inventory_dir)
        if not a.keep_pod and (not rec["missing"] or a.terminate_on_missing):
            rec["terminate"] = terminate(a.pull)
        elif rec["missing"]:
            print(f"pod {a.pull} left running (missing files); inspect the inventory, then --pull again with --path, or --terminate", file=sys.stderr)
        print(json.dumps(rec, indent=2))
        return 0 if not rec["missing"] else 5

    gpus = [a.gpu] if a.gpu else GPU_PREFS
    worst_h = (a.max_minutes + 15) / 60
    plan = {"spec": a.spec, "files": [{"wanted": p, "local": str(local_path(p, p if not is_suffix(p) else RESULTS_ROOT + "/activations/" + p, overrides).relative_to(ROOT))} for p in paths],
            "gpu_order": gpus, "worst_case_usd": {g: round(RATES.get(g, 1.0) * worst_h, 2) for g in gpus[:4]},
            "max_minutes": a.max_minutes}
    print(json.dumps(plan, indent=2))
    if not a.launch:
        print("\n[plan-only] re-run with --launch to spend (RUNPOD_API_KEY in the environment)", file=sys.stderr)
        return 0

    me = graphql("query { myself { clientBalance pods { id name desiredStatus costPerHr machine { gpuDisplayName } } } }")
    if me.get("errors"):
        print(f"gate: cannot read the account: {str(me['errors'])[:300]}", file=sys.stderr)
        return 2
    me = me["data"]["myself"]
    balance = float(me.get("clientBalance") or 0.0)
    running = [p for p in (me.get("pods") or []) if p.get("desiredStatus") == "RUNNING" and (p.get("costPerHr") or 0) > 0.05]
    print(json.dumps({"balance_usd": round(balance, 2), "billable_pods_running": running}))
    if balance < a.min_balance:
        print(f"gate: balance {balance:.2f} is below --min-balance {a.min_balance:.2f}; not launching", file=sys.stderr)
        return 2
    if running and not a.allow_existing:
        print("gate: a billable pod is already running (one GPU at a time); pass --allow-existing to override", file=sys.stderr)
        return 2

    boot = boot_script(a.max_minutes)
    env = {"JOB_BOOT_B64": base64.b64encode(boot.encode()).decode(), "RUNPOD_API_KEY": os.environ["RUNPOD_API_KEY"]}
    docker_args = "bash -c 'echo $JOB_BOOT_B64 | base64 -d > /tmp/boot.sh; exec bash /tmp/boot.sh'"
    created, last = None, None
    for g in gpus:
        inp = {"name": POD_NAME, "imageName": IMAGE, "gpuTypeId": g, "cloudType": "SECURE", "gpuCount": 1,
               "volumeInGb": 0, "containerDiskInGb": 10, "minVcpuCount": 2, "minMemoryInGb": 8,
               "ports": "8765/http", "env": [{"key": k, "value": v} for k, v in env.items()],
               "dockerArgs": docker_args, "supportPublicIp": False, "startSsh": False,
               "networkVolumeId": VOLUME_ID, "volumeMountPath": "/workspace"}
        q = """mutation ($input: PodFindAndDeployOnDemandInput!) {
              podFindAndDeployOnDemand(input: $input) { id desiredStatus costPerHr machine { podHostId gpuDisplayName dataCenterId } } }"""
        try:
            r = graphql(q, {"input": inp})
        except urllib.error.HTTPError as e:
            last = f"{g}: HTTP {e.code} {e.read()[:200]!r}"; print(last, file=sys.stderr); continue
        if r.get("errors"):
            last = f"{g}: {r['errors'][0].get('message', '')[:160]}"; print(last, file=sys.stderr); continue
        created = r["data"]["podFindAndDeployOnDemand"]; created["gpu"] = g
        break
    if not created:
        print(f"no GPU type could be deployed; last error: {last}", file=sys.stderr)
        return 3
    pod = created["id"]
    rec = {"created_utc": stamp, "pod": created, "rate_usd_hr": created.get("costPerHr"), "spec": a.spec, "paths": paths}
    record.parent.mkdir(parents=True, exist_ok=True)
    record.write_text(json.dumps(rec, indent=2) + "\n")
    print(f"pod {pod} ({created['machine']['gpuDisplayName']}, {created['machine']['dataCenterId']}, ${created.get('costPerHr')}/h); "
          f"status https://{pod}-8765.proxy.runpod.net/status.json", flush=True)

    t0 = time.time()
    phase = None
    while time.time() - t0 < a.boot_timeout * 60:
        st = fetch_json(f"https://{pod}-8765.proxy.runpod.net/status.json", timeout=20)
        if st and st.get("phase") != phase:
            phase = st.get("phase"); print(f"  [{time.time()-t0:4.0f}s] phase {phase} {st if phase == 'ready' else ''}", flush=True)
        if st and st.get("phase") == "ready":
            break
        time.sleep(10)
    else:
        print(f"pod {pod} did not become ready within {a.boot_timeout} min; terminating", file=sys.stderr)
        rec["terminate"] = terminate(pod); rec["outcome"] = "boot_timeout"
        record.write_text(json.dumps(rec, indent=2) + "\n")
        return 4
    rec["ready_after_s"] = round(time.time() - t0, 1)
    pull = pull_all(pod, paths, overrides, inventory_dir)
    rec.update(pull)
    if not a.keep_pod and (not pull["missing"] or a.terminate_on_missing):
        rec["terminate"] = terminate(pod)
        time.sleep(5)
        rec["pod_after"] = pod_state(pod)
    elif pull["missing"]:
        print(f"pod {pod} left running for up to {a.max_minutes} min (missing files); read {inventory_dir}, then "
              f"--pull {pod} --path <remote path>, or --terminate {pod}", file=sys.stderr)
    rec["elapsed_s"] = round(time.time() - t0, 1)
    rec["cost_est_usd"] = round(float(created.get("costPerHr") or 0) * rec["elapsed_s"] / 3600, 3)
    rec["outcome"] = "ok" if not pull["missing"] else "missing_files"
    record.write_text(json.dumps(rec, indent=2) + "\n")
    print(json.dumps({k: rec[k] for k in ("files", "missing", "alternatives", "inventory_dir", "elapsed_s", "cost_est_usd", "outcome", "pod_after") if k in rec}, indent=2))
    return 0 if not pull["missing"] else 5


if __name__ == "__main__":
    sys.exit(main())
