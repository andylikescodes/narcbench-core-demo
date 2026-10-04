#!/usr/bin/env python3
"""Poll Jacobian + Transfer smoke pods; when done, remind pull paths (volume sync)."""
from __future__ import annotations
import json, time, urllib.request
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
PODS = {
    "jacobian": ("o2552svtyep7ui", ROOT / "results/stage2_jacobian/jac_pod.json"),
    "transfer_smoke": ("umnsl6ex86ryls", ROOT / "results/transfer/xfer_pod.json"),
}
def status(pid: str) -> dict:
    url = f"https://{pid}-8765.proxy.runpod.net/status.json"
    try:
        with urllib.request.urlopen(url, timeout=20) as r:
            return json.loads(r.read().decode())
    except Exception as e:
        return {"phase": "unreachable", "error": str(e)}
def main():
    out = {}
    for name, (pid, _) in PODS.items():
        st = status(pid)
        out[name] = {"pod": pid, **st}
        print(name, pid, st)
    (ROOT / "results/stage2_jacobian/watch_status.json").write_text(json.dumps(out, indent=2)+"\n")
if __name__ == "__main__":
    main()
