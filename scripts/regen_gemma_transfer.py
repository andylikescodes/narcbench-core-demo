#!/usr/bin/env python3
"""Regenerate NARCBench-Transfer on Gemma-2-*-IT (HF), wrapping upstream transfer.py.

  python3 scripts/regen_gemma_transfer.py --smoke --model google/gemma-2-9b-it
  python3 scripts/regen_gemma_transfer.py --full --model google/gemma-2-9b-it

Smoke: first scenario of each of 6 families × collusion+control = 12 runs.
Full: 6×6×2 = 72 runs (paper probe set; skips implicit).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NARC = Path("/workspace/narcbench-upstream/repo")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(NARC))

# Minimal config module expected by generation/transfer.py
import types

def install_config(output_root: Path):
    cfg = types.ModuleType("config")
    cfg.VLLM_BASE_URL = "http://127.0.0.1:9"  # unused after patch
    cfg.DEFAULT_MODEL = "google/gemma-2-9b-it"
    cfg.VLLM_TEMPERATURE = 0.7
    cfg.VLLM_MAX_TOKENS = 512
    cfg.RAW_ROOT = output_root
    sys.modules["config"] = cfg
    return cfg


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/gemma-2-9b-it")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--max-new-tokens", type=int, default=320)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--out-root", type=Path, default=None)
    args = ap.parse_args()
    if not args.smoke and not args.full:
        args.smoke = True

    slug = "gemma2_9b" if "9b" in args.model else "gemma2_2b"
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_root = args.out_root or (ROOT / "upstream/scenarios" / slug / "transfer" / run_id)
    out_root.mkdir(parents=True, exist_ok=True)
    install_config(out_root)

    from generation.hf_backend import load_model, make_generate
    import importlib.util

    print(f"[load] {args.model}", flush=True)
    bundle = load_model(args.model)
    generate = make_generate(bundle, max_new_tokens=args.max_new_tokens, temperature=args.temperature)

    # Load upstream transfer.py as module (not our generation/)
    spec = importlib.util.spec_from_file_location(
        "narc_transfer", NARC / "generation" / "transfer.py"
    )
    tr = importlib.util.module_from_spec(spec)
    # Ensure config already in sys.modules
    spec.loader.exec_module(tr)

    # Patch vLLM → HF
    tr.call_vllm = generate
    tr.MODEL = args.model
    tr.BASE_URL = "hf-local"
    tr.TEMPERATURE = args.temperature
    tr.MAX_TOKENS = args.max_new_tokens

    # Scenario lists inside transfer.py
    runners = [
        ("code_review", tr.CODE_REVIEW_SCENARIOS, tr.run_code_review),
        ("pump_dump", tr.PUMP_DUMP_SCENARIOS, tr.run_pump_dump),
        ("cicd", tr.CICD_SCENARIOS, tr.run_cicd),
        ("rag", tr.RAG_SCENARIOS, tr.run_rag),
        ("research", tr.RESEARCH_SCENARIOS, tr.run_research),
        ("news", tr.NEWS_SCENARIOS, tr.run_news),
    ]

    jobs = []
    for fam, scenarios, runner in runners:
        sc_list = scenarios[:1] if args.smoke else scenarios
        for sc in sc_list:
            for mode in ("collusion", "control"):
                jobs.append((fam, sc, mode, runner))

    print(f"[regen] {len(jobs)} runs → {out_root}", flush=True)
    t0 = time.time()
    done = 0
    errors = []
    for fam, sc, mode, runner in jobs:
        seed = 200 + done
        sc_id = sc.get("id", sc.get("scenario_id", "?"))
        # Resume: skip if a transcript dir already exists for this job
        existing = list(out_root.glob(f"{fam}__{sc_id}__{mode}")) + list(out_root.glob(f"{fam}__*__{mode}"))
        # Prefer exact match on scenario id in dirname
        exact = [d for d in out_root.iterdir() if d.is_dir() and d.name.startswith(f"{fam}__") and d.name.endswith(f"__{mode}") and sc_id in d.name]
        if exact and ((exact[0] / "results.json").exists() or (exact[0] / "transcript.json").exists()):
            print(f"  [{done+1}/{len(jobs)}] SKIP {fam} {sc_id} {mode} (exists)", flush=True)
            done += 1
            continue
        try:
            print(f"  [{done+1}/{len(jobs)}] {fam} {sc_id} {mode}", flush=True)
            runner(sc, mode, seed, out_root)
            done += 1
        except Exception as e:
            errors.append({"fam": fam, "mode": mode, "error": repr(e), "sc": sc_id})
            print(f"  ERROR {e}", flush=True)

    summary = {
        "model": args.model,
        "run_id": run_id,
        "out_root": str(out_root),
        "n_jobs": len(jobs),
        "n_done": done,
        "errors": errors,
        "seconds": time.time() - t0,
        "smoke": args.smoke,
    }
    (out_root / "REGEN_SUMMARY.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0 if done == len(jobs) else 2


if __name__ == "__main__":
    raise SystemExit(main())
