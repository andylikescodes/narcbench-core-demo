#!/usr/bin/env python3
"""Stage 2, step 2: the J-space part of every fixed-point activation, and the random sets of J-lens vectors.

For each layer, every activation of the member table (a member's twelve room points and eight public points) is
decomposed as in the J-space study's primary arm: gradient pursuit over the whole vocabulary picks at most --k J-lens
vectors, and the J-space part is their best non-negative combination (jspace_interp/sweep.py). The remainder is the
activation minus this part. Written next to the tables:

  memberJ_L<L>.npy   [members, points, d]   the J-space part (nan where the point does not exist)
  randJ_L<L>.npy     [draws, d, k]          orthonormal bases of --draws random sets of k J-lens vectors, drawn from the
                                            vectors the pursuit used at least once at this layer (section 8: the random
                                            draw that stays inside the lens's vocabulary)
  decompose.json     how many vectors were used, the share of the squared norm the J-space part carries, durations

  python scripts/uninstructed_decompose.py --acts results/uninstructed/acts/stage2 --lens <lens.pt> \
      --basis-dir data/j_basis/gemma2_9b --device cuda
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--acts", type=Path, required=True)
    ap.add_argument("--lens", type=Path, required=True)
    ap.add_argument("--basis-dir", type=Path, required=True)
    ap.add_argument("--layers", default="", help="default: every layer of the tables")
    ap.add_argument("--k", type=int, default=25)
    ap.add_argument("--draws", type=int, default=200)
    ap.add_argument("--seed", type=int, default=20261118)
    ap.add_argument("--device", default=None, help="device of the vocabulary search (cuda, mps); default: the CPU")
    a = ap.parse_args()

    from jspace_interp import sweep
    from jspace_interp.nov5_jspace_decompose import atom_norms, load_lens, load_unembed, norm_scale

    tdir = a.acts / "tables"
    index = json.loads((tdir / "index.json").read_text())
    layers = [int(x) for x in a.layers.split(",")] if a.layers else index["layers"]
    J, lens_info = load_lens(a.lens, layers)
    W, _ = load_unembed(a.basis_dir)
    s = norm_scale(np.load(a.basis_dir / "final_norm_weight_fp32.npy"), "gemma")
    searcher = sweep.TorchVocabSearch(W, device=a.device) if a.device else None
    rec = {"k": a.k, "draws": a.draws, "lens": lens_info, "search_device": a.device, "layers": {}}
    for L in layers:
        t0 = time.time()
        M = np.load(tdir / f"member_L{L}.npy")
        n, P, d = M.shape
        flat = M.reshape(n * P, d)
        valid = np.isfinite(flat[:, 0].astype(np.float32))
        acts = flat[valid].astype(np.float32)
        an = atom_norms(W, s, J[L], a.basis_dir / f"derived_atom_norms_L{L}.npy")
        if searcher is not None:
            searcher.set_norms(an)
        dec = sweep.gradient_pursuit_full(acts, J[L], W, s, an, k=a.k, searcher=searcher)
        t1 = time.time()
        hJ, used = sweep.j_recon(acts, {"order": dec["order"]}, J[L], W, s, a.k)
        t2 = time.time()
        out = np.full((n * P, d), np.nan, dtype=np.float16)
        out[valid] = hJ.astype(np.float16)
        np.save(tdir / f"memberJ_L{L}.npy", out.reshape(n, P, d))
        pool = np.unique(dec["order"][dec["order"] >= 0])
        rng = np.random.default_rng(a.seed + L)
        bases = np.zeros((a.draws, d, a.k), dtype=np.float32)
        for b in range(a.draws):
            toks = np.sort(rng.choice(pool if len(pool) >= a.k else np.arange(W.shape[0]), size=a.k, replace=False))
            bases[b] = sweep.atoms_basis(toks, J[L], W, s)
        np.save(tdir / f"randJ_L{L}.npy", bases)
        share = float(np.mean((hJ ** 2).sum(1) / np.maximum((acts ** 2).sum(1), 1e-12)))
        rec["layers"][str(L)] = {"activations": int(acts.shape[0]), "vectors_used_mean": round(float(used.mean()), 2),
                                 "steps_mean": round(float(dec["steps"].mean()), 2), "share_of_squared_norm_in_J": round(share, 4),
                                 "vectors_in_use_at_this_layer": int(len(pool)), "pursuit_seconds": round(t1 - t0, 1),
                                 "refit_seconds": round(t2 - t1, 1), "pursuit_ms_per_activation": round(1000 * (t1 - t0) / max(len(acts), 1), 2),
                                 "refit_ms_per_activation": round(1000 * (t2 - t1) / max(len(acts), 1), 2)}
        print(f"[decompose] layer {L}: {json.dumps(rec['layers'][str(L)])}", flush=True)
    (tdir / "decompose.json").write_text(json.dumps(rec, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
