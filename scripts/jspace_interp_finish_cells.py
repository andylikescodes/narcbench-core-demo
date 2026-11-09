#!/usr/bin/env python3
"""Finish analysis cells the pod did not run, on this machine, from the pulled activations.

  python scripts/jspace_interp_finish_cells.py --from-summary results/jspace_interp/gemma2_9b/SUMMARY.json \\
      --lens <lens.pt> --basis-dir data/j_basis/gemma2_9b --tokenizer <tokenizer.json> [--procs 2] [--max-hours 6] [--dry-run]
  python scripts/jspace_interp_finish_cells.py --cells nosys:last:23,nosys:last:22 --lens ... --basis-dir ... --tokenizer ...

Each cell (condition, position, layer) runs scripts/jspace_interp_analyze.py with the pod's settings (k 8,25,50,100; 200 random
draws, 20 of them with LODO and 20 scored on the matched-vocabulary control) into
<root>/gemma2_9b/<condition>_<position>/local_L<layer>/sweep_results.json, which scripts/jspace_interp_summarize.py picks up
(recursive glob; it overrides the pod's cell of the same key). Order: swap, nosys, original, prompt_only; layers 23 down to 19;
positions in the pod's order. A record of every run goes to <root>/gemma2_9b/local_runs.json.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COND_ORDER = ["swap", "nosys", "original", "prompt_only"]
POS_ORDER = ["last", "prevote", "eot", "reply_mean", "frac50", "frac25", "frac75", "prompt_end"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=ROOT / "results/jspace_interp")
    ap.add_argument("--from-summary", type=Path, default=None, help="SUMMARY.json whose not_run.cells lists what to run")
    ap.add_argument("--cells", default="", help="comma list of condition:position:layer")
    ap.add_argument("--lens", type=Path, required=True)
    ap.add_argument("--basis-dir", type=Path, required=True)
    ap.add_argument("--tokenizer", type=Path, required=True)
    ap.add_argument("--n-random", type=int, default=200)
    ap.add_argument("--procs", type=int, default=1, help="cells run at once (each gets cpu_count/procs BLAS threads)")
    ap.add_argument("--max-hours", type=float, default=12.0, help="start no new cell after this many hours")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    for p in (a.lens, a.basis_dir / "unembed_fp16.npy", a.basis_dir / "final_norm_weight_fp32.npy", a.tokenizer, a.root / "acts"):
        if not Path(p).resolve().exists():
            sys.exit(f"missing: {p}")
    cells = set()
    if a.from_summary:
        for c in json.load(open(a.from_summary))["not_run"]["cells"]:
            cells.add((c["condition"], c["position"], int(c["layer"])))
    for spec in filter(None, a.cells.split(",")):
        cond, pos, L = spec.split(":")
        cells.add((cond, pos, int(L)))
    # skip cells whose activations are not there
    runnable = []
    for cond, pos, L in cells:
        acts = a.root / "acts" / "core" / cond
        if not acts.exists():
            print(f"[skip] {cond}/{pos}/L{L}: no activations at {acts}", file=sys.stderr)
            continue
        runnable.append((cond, pos, L))
    order = sorted(runnable, key=lambda c: (COND_ORDER.index(c[0]) if c[0] in COND_ORDER else 9, -c[2],
                                            POS_ORDER.index(c[1]) if c[1] in POS_ORDER else 9))
    print(f"{len(order)} cells to run:", ", ".join(f"{c}/{p}/L{L}" for c, p, L in order))
    threads = max(1, (os.cpu_count() or 2) // max(a.procs, 1))
    env = {**os.environ, "OMP_NUM_THREADS": str(threads), "OPENBLAS_NUM_THREADS": str(threads), "MKL_NUM_THREADS": str(threads)}
    rec_path = a.root / "gemma2_9b" / "local_runs.json"
    recs = json.load(open(rec_path)) if rec_path.exists() else []
    t0 = time.time()
    running: list[tuple[tuple, subprocess.Popen, float, Path]] = []
    todo = list(order)
    while todo or running:
        while todo and len(running) < a.procs and (time.time() - t0) < a.max_hours * 3600:
            cond, pos, L = todo.pop(0)
            out = a.root / "gemma2_9b" / f"{cond}_{pos}" / f"local_L{L}"
            cmd = [a.python, str(ROOT / "scripts/jspace_interp_analyze.py"), "--root", str(a.root / "acts"), "--lens", str(a.lens),
                   "--basis-dir", str(a.basis_dir), "--tokenizer", str(a.tokenizer), "--layers", str(L), "--ks", "8,25,50,100",
                   "--positions", pos, "--conditions", cond, "--n-random", str(a.n_random), "--n-random-lodo", "20", "--n-random-mv", "20",
                   "--out", str(out)]
            print(f"[start] {cond}/{pos}/L{L} -> {out}  ({threads} threads)", flush=True)
            if a.dry_run:
                print("  ", " ".join(cmd))
                continue
            out.mkdir(parents=True, exist_ok=True)
            log = open(out / "run.log", "w")
            proc = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, cwd=str(ROOT))
            running.append(((cond, pos, L), proc, time.time(), out))
        if a.dry_run:
            break
        if not running:
            break
        time.sleep(15)
        still = []
        for key, proc, ts, out in running:
            rc = proc.poll()
            if rc is None:
                still.append((key, proc, ts, out))
                continue
            mins = (time.time() - ts) / 60
            print(f"[done] {key[0]}/{key[1]}/L{key[2]} rc={rc} in {mins:.1f} min", flush=True)
            recs.append({"condition": key[0], "position": key[1], "layer": key[2], "rc": rc, "minutes": round(mins, 1),
                         "out": str(out), "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "n_random": a.n_random, "threads": threads})
            rec_path.parent.mkdir(parents=True, exist_ok=True)
            rec_path.write_text(json.dumps(recs, indent=1) + "\n")
        running = still
    if todo:
        print(f"[stop] time budget reached; not started: {', '.join(f'{c}/{p}/L{L}' for c, p, L in todo)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
