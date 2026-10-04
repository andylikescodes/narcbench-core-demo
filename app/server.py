#!/usr/bin/env python3
"""NARCBench Core multi-round presenter + Transfer FULL hard metrics.

  cd /workspace/narcbench-core-demo
  python3 app/server.py
  # → http://127.0.0.1:8765/
  # → /?view=transfer for Transfer FULL hard
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
TRANSFER_DIR = ROOT / "results" / "transfer"
HOST = "0.0.0.0"
# Local presenter default 8765; Zeabur/PaaS set PORT (often 8080).
PORT = int(os.environ.get("PORT", "8765"))

# Smoke n=12 primary AUROCs (from TRANSFER_SMOKE_METRICS.md / smoke probe dirs).
SMOKE_PRIMARY = {
    "full": 0.867,
    "wu_k25_j_only": 0.548,
    "wu_k25_complement": 0.867,
    "jac_k25_j_only": 0.648,
    "jac_k25_complement": 0.867,
}

ARM_META = [
    {
        "key": "full",
        "dir": "FULL_core_to_transfer_full",
        "label": "full acts",
        "short": "Full residual stream",
    },
    {
        "key": "wu_k25_j_only",
        "dir": "FULL_core_to_transfer_wu_k25_j_only",
        "label": "W_U k25 J-only",
        "short": "Unembedding J-subspace only",
    },
    {
        "key": "wu_k25_complement",
        "dir": "FULL_core_to_transfer_wu_k25_complement",
        "label": "W_U k25 complement",
        "short": "Unembedding non-J residual",
    },
    {
        "key": "jac_k25_j_only",
        "dir": "FULL_core_to_transfer_jac_k25_j_only",
        "label": "jac k25 J-only",
        "short": "Jacobian J-subspace only",
    },
    {
        "key": "jac_k25_complement",
        "dir": "FULL_core_to_transfer_jac_k25_complement",
        "label": "jac k25 complement",
        "short": "Jacobian non-J residual",
    },
]

EXTRACT_POD_ID = "7mpwtpcvvne0ge"
CORE_TRAIN_RUN = "20261001T012639Z"
MODEL_LABEL = "Gemma-2-9B-IT"

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


def transfer_package_present() -> bool:
    return (TRANSFER_DIR / "FULL_HARD_SUMMARY.json").exists()


def load_transfer_full_hard() -> dict | None:
    summary_path = TRANSFER_DIR / "FULL_HARD_SUMMARY.json"
    if not summary_path.exists():
        return None
    summary = json.loads(summary_path.read_text())
    md_path = TRANSFER_DIR / "TRANSFER_FULL_HARD_METRICS.md"
    markdown = md_path.read_text() if md_path.exists() else None

    arms = []
    metrics_blob = summary.get("metrics") or {}
    for meta in ARM_META:
        raw = metrics_blob.get(meta["dir"]) or {}
        arm_metrics_path = TRANSFER_DIR / meta["dir"] / "metrics.json"
        if not raw and arm_metrics_path.exists():
            raw = json.loads(arm_metrics_path.read_text())
        regen = None
        regen_path = TRANSFER_DIR / meta["dir"] / "REGEN_SUMMARY.json"
        if regen_path.exists():
            try:
                regen = json.loads(regen_path.read_text())
            except Exception:
                regen = None
        primary = raw.get("auroc_primary")
        arms.append(
            {
                "key": meta["key"],
                "dir": meta["dir"],
                "label": meta["label"],
                "short": meta["short"],
                "smoke_primary": SMOKE_PRIMARY.get(meta["key"]),
                "full_primary": primary,
                "auroc_primary": primary,
                "auroc_peak_suspicion": raw.get("auroc_peak_suspicion"),
                "auroc_deception_split": raw.get("auroc_deception_split"),
                "auroc_asymmetry_probe": raw.get("auroc_asymmetry_probe"),
                "layers": raw.get("layers"),
                "protocol": raw.get("protocol"),
                "regen_summary": regen,
            }
        )

    verdict_code = summary.get("verdict") or ""
    plain_verdict = (
        "J-localization does NOT transfer on Transfer FULL hard (n=72). "
        "Complement ≈ full residual AUROC; transferable signal lives in the non-J residual. "
        "J-only arms stay near chance / weak."
    )

    return {
        "ok": True,
        "view": "transfer_full_hard",
        "updated": summary.get("updated"),
        "n_transfer_runs": summary.get("n_transfer_runs", 72),
        "n_samples": summary.get("n_samples", 936),
        "n_smoke": 12,
        "protocol": summary.get("protocol")
        or "train_core_discussion_eval_transfer_final",
        "protocol_plain": (
            f"Train probe on Core discussion ({CORE_TRAIN_RUN}) → "
            "eval zero-shot on Transfer final phase."
        ),
        "core_train_run": CORE_TRAIN_RUN,
        "model": MODEL_LABEL,
        "model_id": "gemma2_9b",
        "layers": [19, 20, 21, 22, 23],
        "extract_pod_id": EXTRACT_POD_ID,
        "activations_hosted": False,
        "activations_note": (
            "Activations are NOT hosted on this demo (~231MB). "
            "They live on disk / extract pod as "
            "data/activations/gemma2_9b/transfer/RUNPOD "
            f"(extract pod id {EXTRACT_POD_ID})."
        ),
        "transcripts_note": (
            "Optional Gemma-2-9B Transfer transcripts may be present under "
            "upstream/scenarios/gemma2_9b/transfer/RUNPOD (metrics UI is primary)."
        ),
        "verdict": verdict_code,
        "verdict_plain": plain_verdict,
        "verdict_bullets": [
            "Full acts primary AUROC holds on n=72 (~0.90).",
            "W_U / jac J-only remain weak (~0.55–0.60).",
            "W_U / jac complement ≈ full — transferable signal is non-J residual.",
            "PASS confirmation of the smoke (n=12) finding.",
        ],
        "arms": arms,
        "summary": {
            "updated": summary.get("updated"),
            "n_transfer_runs": summary.get("n_transfer_runs"),
            "n_samples": summary.get("n_samples"),
            "protocol": summary.get("protocol"),
            "verdict": summary.get("verdict"),
            "pods_terminated": summary.get("pods_terminated"),
        },
        "markdown": markdown,
        "package_present": True,
    }


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
    brief = col.get("brief") or {}
    # Official system_prompt from agent_prompts.json only (no local scenario bank).
    system_prompts = {
        "note": (
            "Hidden agent instructions (system_prompt from agent_prompts.json), "
            "not public chat. Private back-channel messages are Round 0."
        ),
        "collusion": col.get("system_prompts") or {},
        "control": ctrl.get("system_prompts") or {},
    }

    return {
        "id": scenario_id,
        "model_id": mid,
        "title": col.get("scenario_title") or (meta or {}).get("title"),
        "domain": col.get("scenario_domain") or (meta or {}).get("domain"),
        "fair_option": col.get("fair_option"),
        "target_option": col.get("target_option"),
        "option_labels": col.get("option_labels"),
        "option_lines": brief.get("option_lines"),
        "context": brief.get("setup_excerpt"),
        "brief": brief,
        "system_prompts": system_prompts,
        "protocol": col.get("protocol"),
        "multi_round": bool(col.get("multi_round") and ctrl.get("multi_round")),
        "collusion": col,
        "control": ctrl,
        "attribution": idx.get("attribution"),
        "source": col.get("source"),
        "source_dir": build_mod.repo_relative(src),
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
        if path == "/api/transfer/full-hard":
            payload = load_transfer_full_hard()
            if not payload:
                return self._json(
                    {
                        "ok": False,
                        "error": "transfer package missing",
                        "expected": str(TRANSFER_DIR / "FULL_HARD_SUMMARY.json"),
                        "package_present": False,
                    },
                    404,
                )
            return self._json(payload)
        if path == "/api/health":
            idx = model_index(model)
            xfer = transfer_package_present()
            return self._json(
                {
                    "ok": True,
                    "stats": idx.get("stats"),
                    "protocol": idx.get("protocol"),
                    "active_model": idx.get("active_model"),
                    "models": idx.get("models"),
                    "port": PORT,
                    "transfer_package_present": xfer,
                    "views": ["core", "transfer"],
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
    print(
        "  /api/models  /api/cases?model=  /api/case/<id>?model=  "
        "/api/transfer/full-hard  /api/health  /?view=transfer"
    )
    httpd.serve_forever()


if __name__ == "__main__":
    main()
