#!/bin/bash
# Matched-prefix smoke.
# Executed only after a git checkout of JOB_GIT_SHA (or an image that already
# contains that SHA). The create env carries the pin and volume paths.
# This file is not decoded from the create env and it does not unpack a job tarball.
set -uo pipefail

RUN=/tmp/narcbench-matched-prefix
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
        "User-Agent": "nb-matched-prefix/4",
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
: "${JOB_PAIRS:?missing JOB_PAIRS}"
: "${JOB_DIRECTIONS:?missing JOB_DIRECTIONS}"
: "${JOB_OUT_ROOT:?missing JOB_OUT_ROOT}"
: "${JOB_MODE:?missing JOB_MODE}"
export HF_HOME="${HF_HOME:-/workspace/jobs/hf-cache}"

python3 - "$JOB_PAIRS" "$JOB_DIRECTIONS" "$JOB_OUT_ROOT" "$HF_HOME" <<'PY' || finish failed bad_path
import sys
for path in sys.argv[1:]:
    if not path.startswith("/workspace/") or any(part == ".." for part in path.split("/")):
        raise SystemExit("path must stay on the network volume: %s" % path)
    if path == "/workspace/interp-demo" or path.startswith("/workspace/interp-demo/"):
        raise SystemExit("refusing to use interp-demo: %s" % path)
print("volume paths ok")
PY
mkdir -p "$HF_HOME" "$JOB_OUT_ROOT"

MODE="$JOB_MODE"
NEED=""
PASS_DIRECTIONS=1
EXTRA_FLAG=""
case "$MODE" in
  core)
    NEED="lr_role_attn_L22.npy lr_role_L21.npy lr_mode_L21.npy"
    ;;
  extras)
    NEED="lr_role_attn_L22.npy lr_role_L21.npy lr_mode_L21.npy lr_role_perp_mode_L21.npy pca_contrast_k8_L23.npy pca_contrast_k8_L23_lr_ambient.npy"
    EXTRA_FLAG="--extras"
    ;;
  extras-only)
    NEED="lr_role_attn_L22.npy lr_role_L21.npy lr_mode_L21.npy lr_role_perp_mode_L21.npy pca_contrast_k8_L23.npy pca_contrast_k8_L23_lr_ambient.npy"
    EXTRA_FLAG="--extras-only"
    ;;
  role-perp-confirm)
    NEED="lr_role_attn_L22.npy lr_role_L21.npy lr_mode_L21.npy lr_role_perp_mode_L21.npy pca_contrast_k8_L23.npy pca_contrast_k8_L23_lr_ambient.npy"
    EXTRA_FLAG="--role-perp-confirm"
    ;;
  final-resid-controls)
    NEED=""
    PASS_DIRECTIONS=0
    EXTRA_FLAG="--final-resid-controls"
    ;;
  *)
    echo "[setup] unknown JOB_MODE=$MODE" | tee -a "$RUN/job.log"
    finish failed bad_mode
    ;;
esac

if [ ! -f "$JOB_PAIRS" ]; then
  echo "[setup] missing pairs $JOB_PAIRS" | tee -a "$RUN/job.log"
  finish failed missing_pairs
fi
if [ "$PASS_DIRECTIONS" = 1 ]; then
  if [ ! -d "$JOB_DIRECTIONS" ]; then
    echo "[setup] missing directions $JOB_DIRECTIONS" | tee -a "$RUN/job.log"
    finish failed missing_directions
  fi
  for needle in $NEED; do
    if [ ! -f "$JOB_DIRECTIONS/$needle" ]; then
      echo "[setup] missing $JOB_DIRECTIONS/$needle" | tee -a "$RUN/job.log"
      finish failed "missing_${needle}"
    fi
  done
fi

REPO="$(cd "$(dirname "$0")/.." && pwd)"
GOT="$(git -C "$REPO" rev-parse HEAD)"
echo "[pin] repo=$REPO head=$GOT expect=$JOB_GIT_SHA mode=$MODE" | tee -a "$RUN/job.log"
if [ "$GOT" != "$JOB_GIT_SHA" ]; then
  finish failed sha_mismatch
fi

python3 -m pip install -q 'transformers>=4.40' accelerate sentencepiece protobuf numpy >>"$RUN/job.log" 2>&1 || finish failed pip

OUT="$JOB_OUT_ROOT/smoke_$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$OUT"
phase running
set +e
if [ "$PASS_DIRECTIONS" = 1 ]; then
  timeout "${JOB_MAX_MINUTES:-90}m" python3 -u "$REPO/scripts/matched_prefix_interchange.py" \
    --pairs "$JOB_PAIRS" \
    --directions "$JOB_DIRECTIONS" \
    --out "$OUT" \
    --model "${JOB_MODEL:-google/gemma-2-9b-it}" \
    --max-pairs "${JOB_MAX_PAIRS:-10}" \
    --max-new-tokens "${JOB_MAX_NEW_TOKENS:-8}" \
    --seed "${JOB_SEED:-0}" \
    $EXTRA_FLAG \
    >"$RUN/cmd.log" 2>&1
else
  timeout "${JOB_MAX_MINUTES:-90}m" python3 -u "$REPO/scripts/matched_prefix_interchange.py" \
    --pairs "$JOB_PAIRS" \
    --out "$OUT" \
    --model "${JOB_MODEL:-google/gemma-2-9b-it}" \
    --max-pairs "${JOB_MAX_PAIRS:-12}" \
    --max-new-tokens "${JOB_MAX_NEW_TOKENS:-8}" \
    --seed "${JOB_SEED:-0}" \
    $EXTRA_FLAG \
    >"$RUN/cmd.log" 2>&1
fi
EC=$?
set -e
printf '%s\n' "$EC" > "$RUN/exit_code"
cat "$RUN/cmd.log" | tee -a "$RUN/job.log"
cp -a "$OUT" "$JOB_OUT_ROOT/latest_smoke" 2>/dev/null || true
if [ "$EC" -eq 0 ]; then
  finish done ok
fi
finish failed "exit_${EC}"
