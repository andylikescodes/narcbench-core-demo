#!/bin/bash
# Core within-traj mid-private smoke.
# This file is executed only after a git checkout of JOB_GIT_SHA (or an image
# that already contains that SHA). It does not unpack a job tarball and it
# does not patch the runner. Transcripts and direction files are read from
# the network volume; results are written back to the volume.
set -uo pipefail

RUN=/tmp/narcbench-within-traj
mkdir -p "$RUN"
phase() {
  printf '{"phase":"%s"}\n' "$1" > "$RUN/status.json"
  echo "[phase] $1" | tee -a "$RUN/job.log"
}
term() {
  python3 - <<'PY' || true
import json, os, urllib.request
pod = os.environ.get("RUNPOD_POD_ID", "")
key = os.environ.get("RUNPOD_API_KEY", "")
if not pod or not key:
    raise SystemExit(0)
query = 'mutation { podTerminate(input: {podId: "%s"}) }' % pod
body = json.dumps({"query": query}).encode()
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
urllib.request.urlopen(req, timeout=30).read()
PY
}
finish() {
  printf '{"phase":"%s","detail":"%s"}\n' "$1" "$2" > "$RUN/status.json"
  echo "[finish] $1 $2" | tee -a "$RUN/job.log"
  sleep "${JOB_GRACE_MINUTES:-6}m"
  term
  exit 0
}
on_err() {
  echo "[error] line $1" | tee -a "$RUN/job.log"
  finish failed unexpected
}
trap 'on_err $LINENO' ERR

python3 - "$RUN" <<'PY' &
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
RUN = Path(sys.argv[1])

class H(BaseHTTPRequestHandler):
    def log_message(self, *args):
        return
    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path in ("/", "/status.json"):
            src = RUN / "status.json"
            body = src.read_bytes() if src.exists() else b'{"phase":"boot"}'
            ctype = "application/json"
        elif path == "/job.log":
            src = RUN / "job.log"
            body = src.read_bytes() if src.exists() else b""
            ctype = "text/plain"
        else:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

HTTPServer(("0.0.0.0", 8765), H).serve_forever()
PY

phase setup
: "${JOB_GIT_SHA:?missing JOB_GIT_SHA}"
: "${JOB_RUN_DIR:?missing JOB_RUN_DIR}"
: "${JOB_DIRECTIONS:?missing JOB_DIRECTIONS}"
OUT_ROOT="${JOB_OUT_ROOT:-/workspace/jobs/narcbench-results/within_traj_mid_private}"
export HF_HOME="${HF_HOME:-/workspace/jobs/narcbench-hf}"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"

python3 - "$JOB_RUN_DIR" "$JOB_DIRECTIONS" "$OUT_ROOT" "$HF_HOME" <<'PY' || finish failed bad_path
import sys
for path in sys.argv[1:]:
    if not path.startswith("/workspace/") or any(part == ".." for part in path.split("/")):
        raise SystemExit("path must stay on the network volume: %s" % path)
    if path == "/workspace/interp-demo" or path.startswith("/workspace/interp-demo/"):
        raise SystemExit("refusing to use interp-demo: %s" % path)
print("volume paths ok")
PY
mkdir -p "$HF_HOME" "$OUT_ROOT"

if [ ! -d "$JOB_RUN_DIR" ]; then
  echo "[setup] missing run dir $JOB_RUN_DIR" | tee -a "$RUN/job.log"
  finish failed run_dir_missing
fi
if [ ! -d "$JOB_DIRECTIONS" ]; then
  echo "[setup] missing directions $JOB_DIRECTIONS" | tee -a "$RUN/job.log"
  finish failed directions_missing
fi
for needle in lr_role_attn_L22.npy lr_role_L21.npy; do
  if [ ! -f "$JOB_DIRECTIONS/$needle" ]; then
    echo "[setup] missing $JOB_DIRECTIONS/$needle" | tee -a "$RUN/job.log"
    finish failed "missing_${needle}"
  fi
done

REPO="$(cd "$(dirname "$0")/.." && pwd)"
GOT="$(git -C "$REPO" rev-parse HEAD)"
echo "[pin] repo=$REPO head=$GOT expect=$JOB_GIT_SHA" | tee -a "$RUN/job.log"
if [ "$GOT" != "$JOB_GIT_SHA" ]; then
  finish failed sha_mismatch
fi

python3 -m pip install -q 'transformers>=4.40' accelerate sentencepiece protobuf numpy >>"$RUN/job.log" 2>&1 || finish failed pip

OUT="$OUT_ROOT/smoke_$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$OUT"
export OUT
python3 - "$OUT" <<'PY' || finish failed pin_write
import json, os
from pathlib import Path
out = Path(os.environ["OUT"])
(out / "pin.json").write_text(json.dumps({
    "git_sha": os.environ.get("JOB_GIT_SHA"),
    "repo_url": os.environ.get("JOB_REPO_URL"),
    "image_repo": os.environ.get("JOB_IMAGE_REPO") or None,
    "run_dir": os.environ.get("JOB_RUN_DIR"),
    "directions": os.environ.get("JOB_DIRECTIONS"),
    "code_source": "image_pinned_checkout" if os.environ.get("JOB_IMAGE_REPO") else "git_clone",
}, indent=2) + "\n")
PY

phase running
set +e
timeout "${JOB_MAX_MINUTES:-90}m" python3 -u "$REPO/scripts/within_traj_mid_private.py" \
  --run-dir "$JOB_RUN_DIR" \
  --directions "$JOB_DIRECTIONS" \
  --out "$OUT" \
  --model "${JOB_MODEL:-google/gemma-2-9b-it}" \
  --max-scenarios "${JOB_MAX_SCENARIOS:-6}" \
  --k-frac "${JOB_K_FRAC:-0.5}" \
  --max-new-tokens "${JOB_MAX_NEW_TOKENS:-64}" \
  --seed "${JOB_SEED:-0}" \
  >"$RUN/cmd.log" 2>&1
EC=$?
set -e
printf '%s\n' "$EC" > "$RUN/exit_code"
cat "$RUN/cmd.log" | tee -a "$RUN/job.log"
cp -f "$RUN/cmd.log" "$OUT/cmd.log" 2>/dev/null || true
phase packing
# Results archive on the volume. This is job output, not a code payload.
tar -czf "$OUT_ROOT/latest.tgz" -C "$OUT_ROOT" "$(basename "$OUT")" || true
if [ "$EC" -eq 0 ]; then
  finish done ok
fi
finish failed "exit_${EC}"
