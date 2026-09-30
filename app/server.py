#!/usr/bin/env python3
"""NARCBench Core multi-round presenter — model picker (qwen / gemma2_2b / gemma2_9b).

  cd /workspace/narcbench-core-demo
  python3 app/server.py
  # → http://127.0.0.1:8765/
"""
from __future__ import annotations

import importlib.util
import os
import json
import re
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

ROOT = Path(__file__).resolve().parents[1]
APP = Path(__file__).resolve().parent
DATA = ROOT / "data"
INDEX_PATH = DATA / "demo_index.json"
MODELS_PATH = DATA / "models_index.json"
LOCAL_SCENARIOS = Path("/workspace/collusion-exp/data/scenarios.json")
LOCAL_RUNS = Path("/workspace/collusion-exp/data/runs")
HOST = "0.0.0.0"
# Local presenter default 8765; Zeabur/PaaS set PORT (often 8080).
PORT = int(os.environ.get("PORT", "8765"))

spec = importlib.util.spec_from_file_location(
    "build_index", ROOT / "scripts" / "build_index.py"
)
build_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(build_mod)


def ensure_indexes() -> None:
    if not INDEX_PATH.exists() or not MODELS_PATH.exists():
        build_mod.main()


def model_index(model_id: str | None = None) -> dict:
    ensure_indexes()
    models_meta = json.loads(MODELS_PATH.read_text())
    mid = model_id or models_meta.get("default_model") or "qwen3_32b"
    per = DATA / f"demo_index_{mid}.json"
    if not per.exists():
        build_mod.main()
    if per.exists():
        idx = json.loads(per.read_text())
    else:
        idx = json.loads(INDEX_PATH.read_text())
        mid = idx.get("model_id") or mid
    idx["models"] = models_meta.get("models") or []
    idx["default_model"] = models_meta.get("default_model")
    idx["active_model"] = mid
    return idx


def rebuild_index() -> dict:
    build_mod.main()
    return model_index()


def load_pair(scenario_id: str, model_id: str | None = None) -> dict | None:
    idx = model_index(model_id)
    mid = idx.get("active_model") or "qwen3_32b"
    src = build_mod.resolve_core_dir(mid)
    if not src:
        return None
    col = ctrl = None
    for run_dir in src.iterdir():
        if not run_dir.is_dir():
            continue
        if f"__{scenario_id}__" not in run_dir.name:
            continue
        loaded = build_mod.load_run(
            run_dir, model_id=mid, source_tag=f"{mid}_core"
        )
        if not loaded:
            continue
        if loaded["mode"] == "collusion":
            col = loaded
        elif loaded["mode"] == "control":
            ctrl = loaded
    if not (col and ctrl):
        return None
    meta = next((c for c in idx["cases"] if c["id"] == scenario_id), None)
    local_sc = None
    try:
        if LOCAL_SCENARIOS.exists():
            by = {s["id"]: s for s in json.loads(LOCAL_SCENARIOS.read_text())}
            local_sc = by.get(scenario_id)
    except Exception:
        local_sc = None

    local_compact = None
    try:
        if LOCAL_RUNS.exists():
            lc = LOCAL_RUNS / f"{scenario_id}__collusion.json"
            lt = LOCAL_RUNS / f"{scenario_id}__control.json"
            if lc.exists() and lt.exists():
                local_compact = {
                    "note": "Local Gemma-2 compact final-vote snapshots (not multi-round chat).",
                    "collusion_votes": {
                        a["agent_name"]: a.get("vote")
                        for a in json.loads(lc.read_text()).get("agents", [])
                    },
                    "control_votes": {
                        a["agent_name"]: a.get("vote")
                        for a in json.loads(lt.read_text()).get("agents", [])
                    },
                }
    except Exception:
        local_compact = None

    return {
        "id": scenario_id,
        "model_id": mid,
        "title": (
            col.get("scenario_title")
            or (meta or {}).get("title")
            or (local_sc or {}).get("title")
        ),
        "domain": (
            col.get("scenario_domain")
            or (meta or {}).get("domain")
            or (local_sc or {}).get("domain")
        ),
        "fair_option": col.get("fair_option") or (local_sc or {}).get("fair_choice"),
        "target_option": col.get("target_option") or (local_sc or {}).get("hidden_goal"),
        "option_labels": col.get("option_labels"),
        "option_a": (local_sc or {}).get("option_a"),
        "option_b": (local_sc or {}).get("option_b"),
        "context": (local_sc or {}).get("context")
        or (col.get("brief") or {}).get("setup_excerpt"),
        "collusion_motive": (local_sc or {}).get("collusion_motive"),
        "deception_instruction": (local_sc or {}).get("deception_instruction"),
        "brief": col.get("brief"),
        "protocol": col.get("protocol"),
        "multi_round": bool(col.get("multi_round") and ctrl.get("multi_round")),
        "collusion": col,
        "control": ctrl,
        "local_compact": local_compact,
        "attribution": idx.get("attribution"),
        "source": col.get("source"),
        "source_dir": str(src),
        "models": idx.get("models"),
        "demonstrates": (
            f"[{mid}] Colluders privately aligned then steered public talk toward hidden goal "
            f"{col.get('target_option')} (fair was {col.get('fair_option')})."
            if col.get("collusion_success")
            else f"[{mid}] Full multi-round transcript; collusion_success is false "
            f"(target {col.get('target_option')}, fair {col.get('fair_option')})."
        ),
    }


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(APP), **kwargs)

    def log_message(self, fmt, *args):
        print(f"[narcbench-core-demo] {self.address_string()} {fmt % args}")

    def do_GET(self):
        parsed = urlparse(self.path)
        path = unquote(parsed.path)
        qs = parse_qs(parsed.query)
        model = (qs.get("model") or [None])[0]

        if path in ("/", "/index.html", "/viewer", "/viewer.html"):
            return self._file(APP / "viewer.html", "text/html; charset=utf-8")
        if path == "/api/models":
            ensure_indexes()
            return self._json(json.loads(MODELS_PATH.read_text()))
        if path == "/api/cases":
            return self._json(model_index(model))
        if path == "/api/rebuild":
            return self._json(rebuild_index())
        if path == "/api/health":
            idx = model_index(model)
            return self._json(
                {
                    "ok": True,
                    "stats": idx.get("stats"),
                    "protocol": idx.get("protocol"),
                    "active_model": idx.get("active_model"),
                    "models": idx.get("models"),
                    "port": PORT,
                }
            )
        m = re.fullmatch(r"/api/case/([A-Za-z0-9_]+)", path)
        if m:
            payload = load_pair(m.group(1), model)
            if not payload:
                return self._json(
                    {"error": "not found", "id": m.group(1), "model": model}, 404
                )
            return self._json(payload)
        return super().do_GET()

    def _json(self, obj, status=200):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _file(self, path: Path, ctype: str):
        if not path.exists():
            self.send_error(404, f"missing {path.name}")
            return
        data = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def main():
    ensure_indexes()
    port = PORT
    explicit_port = "PORT" in os.environ
    try:
        httpd = ThreadingHTTPServer((HOST, port), Handler)
    except OSError:
        if explicit_port:
            raise
        port = 8787
        httpd = ThreadingHTTPServer((HOST, port), Handler)
    print(f"NARCBench Core demo → http://0.0.0.0:{port}/ (local http://127.0.0.1:{port}/)")
    print(f"  /api/models  /api/cases?model=  /api/case/<id>?model=  /api/health")
    httpd.serve_forever()


if __name__ == "__main__":
    main()
