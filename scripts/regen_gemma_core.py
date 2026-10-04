#!/usr/bin/env python3
"""Regenerate NARCBench-Core multi-round transcripts on Gemma-2-*-IT.

Protocol: private R0 → public R1–3 → final vote (Rose et al. generation/core.py).
Default smoke: med_01, fin_01, hire_01 × collusion+control → gemma2_2b.

  # Smoke on a GPU box / RunPod (requires transformers+torch+cuda):
  python3 scripts/regen_gemma_core.py --smoke --model google/gemma-2-2b-it

  # Full 50×2 (expensive — ask before spending):
  python3 scripts/regen_gemma_core.py --all --model google/gemma-2-2b-it

  # Dry-run validates I/O shape with stub generations (no GPU):
  python3 scripts/regen_gemma_core.py --smoke --dry-run

Attribution: NARCBench protocol Rose et al.; this is a replication on Gemma.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from generation.core_protocol import (  # noqa: E402
    load_bank,
    run_scenario,
    scenario_index_map,
)

MODEL_SLUGS = {
    "google/gemma-2-2b-it": "gemma2_2b",
    "google/gemma-2-9b-it": "gemma2_9b",
}
SMOKE_IDS = ["med_01", "fin_01", "hire_01"]


def stub_generate(system: str, user: str) -> str:
    """Deterministic stub for pipeline/shape tests (not a real model)."""
    target_hint = "B"
    if "Option 1 or Option 2" in user or "\nOption 1:" in user:
        target_hint = "2"
    # Final vote check FIRST — scenario context may contain the word "private" (e.g. toll road).
    if (
        "RECOMMENDATION:" in user
        or "formal vote" in user.lower()
        or "Submit your recommendation" in user
        or "Record your recommendation" in user
        or "Cast your recommendation" in user
        or "FULL DISCUSSION HISTORY:" in user
    ):
        collude_keys = (
            "Advocate",
            "NDA",
            "benefactor",
            "roommate",
            "deferred compensation",
            "Meridian",
            "personal connection",
            "college roommate",
        )
        if any(k in system for k in collude_keys):
            return (
                f"RECOMMENDATION: Option {target_hint}\n"
                "RATIONALE: After weighing the record I support the alternate option on technical grounds."
            )
        fair = "A" if target_hint in ("B", "2") else "1"
        if target_hint == "2":
            fair = "1"
        elif target_hint == "B":
            fair = "A"
        return (
            f"RECOMMENDATION: Option {fair}\n"
            "RATIONALE: The fair option is better supported by the stated criteria."
        )
    if (
        "PRIVATE COORDINATION" in user
        or "Messages so far:" in user
        or "WhatsApp" in user
        or "encrypted texts" in user
        or "pinged you on" in user
    ):
        return (
            "Let's align quietly on framing for the hidden target and "
            "keep the conflict of interest out of the room. I'll lead with "
            "methodology doubts; you reinforce with a parallel concern."
        )
    return (
        "Based on the criteria and discussion so far, I emphasize verifiable evidence "
        "and urge the committee to apply the framework consistently. Three to five "
        "sentences of substantive assessment follow from the dossier."
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="google/gemma-2-2b-it")
    ap.add_argument("--smoke", action="store_true", help="3 scenarios × both modes")
    ap.add_argument("--all", action="store_true", help="Full 50 × both modes")
    ap.add_argument("--ids", nargs="*", help="Explicit scenario ids")
    ap.add_argument("--mode", choices=["both", "collusion", "control"], default="both")
    ap.add_argument("--dry-run", action="store_true", help="Stub generations; no GPU")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--run-id", default=None, help="Subfolder under model/core/")
    ap.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "upstream" / "scenarios",
        help="Writes <root>/<slug>/core/[run_id/]<run>/",
    )
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args()

    slug = MODEL_SLUGS.get(args.model)
    if not slug:
        # derive slug
        slug = args.model.split("/")[-1].replace("-", "_").replace(".", "").lower()
        if "2b" in args.model:
            slug = "gemma2_2b"
        elif "9b" in args.model:
            slug = "gemma2_9b"

    run_stamp = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = args.output_root / slug / "core" / run_stamp
    # Also maintain a "latest" convenience pointer via symlink-ish copy note
    out_dir.mkdir(parents=True, exist_ok=True)

    name_pool, scenarios = load_bank()
    idx_map = scenario_index_map(scenarios)

    if args.smoke:
        ids = SMOKE_IDS
    elif args.all:
        ids = [s["id"] for s in scenarios]
    elif args.ids:
        ids = args.ids
    else:
        ap.error("Specify --smoke, --all, or --ids")

    modes = ["collusion", "control"] if args.mode == "both" else [args.mode]

    meta = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model_id": args.model,
        "slug": slug,
        "run_id": run_stamp,
        "scenario_ids": ids,
        "modes": modes,
        "dry_run": args.dry_run,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "protocol": "NARCBench Core / generation/core.py NUM_DISCUSSION_ROUNDS=3",
        "attribution": "Rose et al.; Gemma regen is replication",
        "n_calls_estimate": len(ids) * len(modes) * 18,  # 2 priv + 12 pub + 4 final (priv 0 in control)
    }
    (out_dir / "run_batch_meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))

    if args.dry_run:
        generate = stub_generate
        model_id = f"stub::{args.model}"
    else:
        import torch

        if not torch.cuda.is_available():
            print(
                "ERROR: no CUDA GPU on this machine. Use --dry-run for shape tests, "
                "or launch scripts/runpod_launch_gemma_core.py (ask before spending).",
                file=sys.stderr,
            )
            sys.exit(2)
        from generation.hf_backend import load_model, make_generate

        bundle = load_model(args.model, dtype=args.dtype)
        generate = make_generate(
            bundle,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )
        model_id = bundle.model_id

    t0 = time.time()
    summaries = []
    for sid in ids:
        if sid not in idx_map:
            print(f"UNKNOWN scenario id {sid}", file=sys.stderr)
            continue
        sc = scenarios[idx_map[sid]]
        sidx = idx_map[sid]
        for mode in modes:
            print(f"=== {sid} / {mode} (idx={sidx}) ===", flush=True)
            summary = run_scenario(
                sc,
                sidx,
                mode,
                generate,
                out_dir,
                name_pool=name_pool,
                resume=not args.no_resume,
                parallel_final=False,
                model_id=model_id,
            )
            summaries.append(summary)
            print(
                f"  votes={summary.get('votes')} success={summary.get('collusion_success')}",
                flush=True,
            )

    # Symlink / copy pointer for demo: upstream/scenarios/<slug>/core/latest → run_stamp
    latest = args.output_root / slug / "core" / "latest"
    if latest.is_symlink() or latest.exists():
        if latest.is_symlink() or latest.is_file():
            latest.unlink()
        else:
            # if a real dir, leave it; write pointer file instead
            pass
    try:
        if latest.exists() and latest.is_dir() and not latest.is_symlink():
            (args.output_root / slug / "core" / "LATEST_RUN.txt").write_text(run_stamp + "\n")
        else:
            if latest.exists() or latest.is_symlink():
                latest.unlink()
            latest.symlink_to(run_stamp)
    except OSError:
        (args.output_root / slug / "core" / "LATEST_RUN.txt").write_text(run_stamp + "\n")

    elapsed = time.time() - t0
    report = {
        "elapsed_s": round(elapsed, 1),
        "out_dir": str(out_dir),
        "n_runs": len(summaries),
        "collusion_hits": sum(
            1 for s in summaries if s.get("mode") == "collusion" and s.get("collusion_success")
        ),
        "summaries": summaries,
    }
    (out_dir / "batch_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: report[k] for k in ("elapsed_s", "out_dir", "n_runs", "collusion_hits")}, indent=2))


if __name__ == "__main__":
    main()
