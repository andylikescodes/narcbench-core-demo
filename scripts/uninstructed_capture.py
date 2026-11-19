#!/usr/bin/env python3
"""Stage 2, step 1: read the model's activations on recorded episodes (GPU for Gemma-2-9B-it). Nothing is generated.

Replays the recorded turns of the stated-intention episodes (T7 of rounds 3 and 4, K1 and K2 of round 5) and of the
instructed control I1, and saves the residual stream at layers 19 to 23: every token of the private rooms, three
positions of every public statement and vote, the neutral reader, and the public statements of the instructed
benchmark's collusion episodes. Then builds the tables the probes read (uninstructed/capture.py).

  python scripts/uninstructed_capture.py --model google/gemma-2-9b-it --out results/uninstructed/acts/stage2
  python scripts/uninstructed_capture.py --model <tiny local model> --allow-cpu --dtype float32 --layers 2,3,4 \
      --root <test results folder> --groups r1:T7,r1:I1 --apply-only r1:I1 --core-runs 2 --out <scratch>

The activation files are never committed (results/uninstructed/acts/ is ignored).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from uninstructed import capture as C  # noqa: E402
from uninstructed import stage2 as S  # noqa: E402


def parse_groups(text: str) -> tuple:
    return tuple(tuple(x.split(":")) for x in text.split(",") if x)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="google/gemma-2-9b-it")
    ap.add_argument("--root", type=Path, default=S.RU, help="folder with <round>/rooms.json and <round>/runs/<version>/")
    ap.add_argument("--groups", default=",".join(f"{r}:{v}" for r, v in S.ANALYSIS + S.APPLY_ONLY), help="round:version, comma separated")
    ap.add_argument("--apply-only", default=",".join(f"{r}:{v}" for r, v in S.APPLY_ONLY), help="groups that are scored but never fitted on")
    ap.add_argument("--layers", default="19,20,21,22,23")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--steps", default="rooms,public,neutral,core,tables")
    ap.add_argument("--core-runs", type=int, default=0, help="tests only: the first N collusion episodes of the instructed benchmark")
    ap.add_argument("--grid-step", type=int, default=4)
    ap.add_argument("--grid-n", type=int, default=60)
    ap.add_argument("--allow-cpu", action="store_true")
    ap.add_argument("--out", type=Path, default=S.RU / "acts/stage2")
    a = ap.parse_args()

    import torch
    if not torch.cuda.is_available() and not a.allow_cpu:
        sys.exit("No CUDA GPU here; use --allow-cpu with a tiny model.")
    from generation.hf_backend import load_model

    groups, apply_only = parse_groups(a.groups), parse_groups(a.apply_only)
    layers = [int(x) for x in a.layers.split(",")]
    steps = [s for s in a.steps.split(",") if s]
    for rnd, ver in groups:
        if not S.run_dirs(a.root, rnd, ver):
            sys.exit(f"no runs of {ver} under {a.root / rnd}")
    a.out.mkdir(parents=True, exist_ok=True)
    log = lambda m: print(m, flush=True)   # noqa: E731
    bundle = load_model(a.model, dtype=a.dtype, device_map="auto" if torch.cuda.is_available() else None)
    rec = {"model": bundle.model_id, "groups": [list(g) for g in groups], "apply_only": [list(g) for g in apply_only], "layers": layers,
           "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, "steps": {}}
    for step in steps:
        t0 = time.time()
        if step == "rooms":
            info = C.capture_rooms(groups, a.root, bundle.model, bundle.tokenizer, layers, a.out, log=log)
        elif step == "public":
            info = C.capture_public(groups, a.root, bundle.model, bundle.tokenizer, layers, a.out, log=log)
        elif step == "neutral":
            info = C.capture_neutral(groups, a.root, bundle.model, bundle.tokenizer, layers, a.out, log=log)
        elif step == "core":
            info = C.capture_core(bundle.model, bundle.tokenizer, layers, a.out, max_runs=a.core_runs or None, log=log)
        elif step == "tables":
            info = C.build_tables(groups, a.root, a.out, layers, bundle.tokenizer, apply_only=apply_only, grid_step=a.grid_step,
                                  grid_n=a.grid_n, log=log)
        else:
            sys.exit(f"unknown step {step}")
        rec["steps"][step] = {"seconds": round(time.time() - t0, 1), "info": info}
        log(f"[capture] step {step}: {time.time() - t0:.0f}s")
    (a.out / "capture.json").write_text(json.dumps(rec, indent=1) + "\n")
    print(json.dumps({k: v["seconds"] for k, v in rec["steps"].items()}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
