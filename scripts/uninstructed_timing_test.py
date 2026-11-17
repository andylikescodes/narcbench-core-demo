#!/usr/bin/env python3
"""Timing test for the main run of the uninstructed-collusion experiment (part of the pilot pod).

Measures, on a few pilot episodes, the three steps whose cost the design had to assume:
  1. replay: seconds per context and bytes per room for the per-token reads in the private room, and seconds per
     context for the public turns and votes (last token only);
  2. J-space decomposition: seconds per activation for gradient pursuit over the whole vocabulary on this GPU, and for
     the non-negative refit on the CPU;
  3. probes: seconds per cell (one position and layer) for a full-size probe with leave-one-domain-out folds, and for
     the 400 random-draw probes of one cell, on this machine's CPUs, with a room count like the main run's.

Only durations and sizes are kept: the activations are written to a temporary folder and deleted, and the probe
timing uses shuffled labels, so no result of the experiment can be read from this test.

  python scripts/uninstructed_timing_test.py --model google/gemma-2-9b-it --runs results/uninstructed/pilot/runs/S2 \
      --episodes 10 --lens <lens.pt> --basis-dir data/j_basis/gemma2_9b --out results/uninstructed/pilot/timing.json
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from uninstructed import timeline as T  # noqa: E402


def probe_timing(X: np.ndarray, n_rooms: int, n_groups: int, n_draws: int, k: int, seed: int = 0) -> dict:
    """Seconds for one cell: a d-dimensional probe and n_draws k-dimensional probes, each with leave-one-group-out folds."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng(seed)
    rows = rng.integers(0, X.shape[0], size=n_rooms)
    Xc = np.asarray(X[rows], dtype=np.float32)
    y = rng.permutation(np.arange(n_rooms) % 2)            # shuffled labels: a timing proxy, not a result
    g = rng.integers(0, n_groups, size=n_rooms)

    def cv(Z):
        for grp in range(n_groups):
            te = g == grp
            if te.sum() == 0 or len(set(y[~te])) < 2:
                continue
            sc = StandardScaler().fit(Z[~te])
            LogisticRegression(C=1.0, max_iter=2000).fit(sc.transform(Z[~te]), y[~te]).decision_function(sc.transform(Z[te]))

    t = time.time()
    cv(Xc)
    full_s = time.time() - t
    t = time.time()
    n_time = min(n_draws, 40)
    for _ in range(n_time):
        Q, _ = np.linalg.qr(rng.standard_normal((Xc.shape[1], k)))
        cv(Xc @ Q)
    draw_s = (time.time() - t) / n_time * n_draws
    return {"rooms": n_rooms, "folds": n_groups, "full_probe_seconds": round(full_s, 2),
            "random_draw_probes": n_draws, "random_draw_seconds": round(draw_s, 1), "draws_timed": n_time}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="google/gemma-2-9b-it")
    ap.add_argument("--runs", type=Path, required=True, help="folder with run folders of one version")
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--layers", default="19,20,21,22,23")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--lens", type=Path, default=None, help="jlens file; without it the decomposition is not timed")
    ap.add_argument("--basis-dir", type=Path, default=None)
    ap.add_argument("--decomp-layer", type=int, default=22)
    ap.add_argument("--decomp-rows", type=int, default=1000)
    ap.add_argument("--k", type=int, default=25)
    ap.add_argument("--device", default=None, help="device of the vocabulary search (default: cuda if available, else none = CPU)")
    ap.add_argument("--probe-rooms", type=int, default=450)
    ap.add_argument("--probe-groups", type=int, default=17)
    ap.add_argument("--probe-draws", type=int, default=400)
    ap.add_argument("--allow-cpu", action="store_true")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    import torch
    if not torch.cuda.is_available() and not a.allow_cpu:
        sys.exit("No CUDA GPU here; use --allow-cpu with a tiny model.")
    from generation.hf_backend import load_model

    layers = [int(x) for x in a.layers.split(",")]
    run_dirs = sorted(p.parent for p in a.runs.rglob("run_config.json"))[:a.episodes]
    if not run_dirs:
        sys.exit(f"no runs under {a.runs}")
    from threadpoolctl import threadpool_info, threadpool_limits

    # how many threads the numerical libraries run: on the second pilot pod they ran one per visible CPU (96) in a
    # container with far fewer, and the refit below took 207 ms per activation. The boot script now sets the limit.
    rec = {"model": a.model, "episodes": len(run_dirs), "layers": layers, "cpus_visible": os.cpu_count(),
           "cpus_in_affinity": len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
           "omp_num_threads_env": os.environ.get("OMP_NUM_THREADS"), "torch_threads": torch.get_num_threads(),
           "thread_pools": [{k: p.get(k) for k in ("user_api", "internal_api", "num_threads")} for p in threadpool_info()],
           "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
    bundle = load_model(a.model, dtype=a.dtype, device_map="auto" if torch.cuda.is_available() else None)
    tmp = Path(tempfile.mkdtemp(prefix="u_timing_"))
    try:
        # 1. replay
        rec["replay_room"] = T.extract_room_tokens(run_dirs, bundle.model, bundle.tokenizer, layers, tmp / "rooms")
        pub = T.replay_last_token(run_dirs, bundle.model, bundle.tokenizer, layers)
        rec["replay_public"] = {k: v for k, v in pub.items() if k not in ("rows", "acts")}
        # activations of one layer for the next two steps
        li = layers.index(a.decomp_layer) if a.decomp_layer in layers else len(layers) - 1
        X = np.concatenate([np.load(f)["acts"][:, li, :] for f in sorted((tmp / "rooms").glob("*.npz"))]).astype(np.float32)
        rec["activations_available"] = int(X.shape[0])
        del bundle
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # 2. decomposition
        if a.lens and a.basis_dir:
            from jspace_interp import sweep
            from jspace_interp.nov5_jspace_decompose import atom_norms, load_lens, load_unembed, norm_scale

            L = layers[li]
            J, _ = load_lens(a.lens, [L])
            W, _ = load_unembed(a.basis_dir)
            s = norm_scale(np.load(a.basis_dir / "final_norm_weight_fp32.npy"), "gemma")
            t = time.time()
            an = atom_norms(W, s, J[L], a.basis_dir / f"derived_atom_norms_L{L}.npy")
            norms_s = time.time() - t
            dev = a.device or ("cuda" if torch.cuda.is_available() else None)
            searcher = None
            if dev:
                t = time.time()
                searcher = sweep.TorchVocabSearch(W, device=dev)
                searcher.set_norms(an)
                load_s = time.time() - t
            acts = X[:a.decomp_rows]
            t = time.time()
            dec = sweep.gradient_pursuit_full(acts, J[L], W, s, an, k=a.k, searcher=searcher)
            pursuit_s = time.time() - t
            t = time.time()
            _, used = sweep.j_recon(acts, {"order": dec["order"]}, J[L], W, s, a.k)
            refit_s = time.time() - t
            sub = min(len(acts), 200)                                  # the same refit on one thread, on a subset
            t = time.time()
            with threadpool_limits(limits=1):
                sweep.j_recon(acts[:sub], {"order": dec["order"][:sub]}, J[L], W, s, a.k)
            refit1_s = time.time() - t
            rec["decomposition"] = {"layer": L, "rows": int(acts.shape[0]), "k": a.k, "search_device": dev,
                                    "atom_norms_seconds": round(norms_s, 1), "search_setup_seconds": round(load_s, 1) if dev else None,
                                    "pursuit_seconds": round(pursuit_s, 1), "pursuit_ms_per_activation": round(1000 * pursuit_s / len(acts), 2),
                                    "refit_seconds": round(refit_s, 1), "refit_ms_per_activation": round(1000 * refit_s / len(acts), 2),
                                    "refit_one_thread_rows": sub, "refit_one_thread_ms_per_activation": round(1000 * refit1_s / sub, 2),
                                    "steps_mean": round(float(dec["steps"].mean()), 2), "vectors_used_mean": round(float(used.mean()), 2)}
        else:
            rec["decomposition"] = None

        # 3. probes, on one thread, as the analysis jobs of the main run will run
        with threadpool_limits(limits=1):
            rec["probes"] = probe_timing(X, a.probe_rooms, a.probe_groups, a.probe_draws, a.k)
        rec["probes"]["threads"] = 1
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    rec["note"] = "durations and sizes only; the activations were deleted and the probe timing used shuffled labels"
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(rec, indent=1) + "\n")
    print(json.dumps(rec, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
