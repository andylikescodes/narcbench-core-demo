#!/usr/bin/env python3
"""Role directions for Experiment 3 without waiting for the full sweep (CPU, a few minutes per layer).

For each layer: the paper's decomposition at k=25 (pursuit filled to k) of the Core discussion rows of the given
condition and position, then u = mean h_J of colluders minus mean h_J of honest agents in collusion episodes, unit norm,
and the median ||h_J|| (the steering scale). Writes <out>/role_dir_J25_<cond>_<pos>_L<L>.npy and role_dirs.json.

  python scripts/jspace_interp_role_dir.py --root results/jspace_interp/acts --condition swap --position last \\
      --layers 22,23 --lens <lens.pt> --basis-dir data/j_basis/gemma2_9b --out results/jspace_interp/role_dirs
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
from jspace_interp import sweep  # noqa: E402
from jspace_interp.nov5_jspace_decompose import atom_norms, load_lens, load_unembed, norm_scale, parse_layers  # noqa: E402
from jspace_interp.rows import load_extraction, select_rows  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--condition", default="swap")
    ap.add_argument("--position", default="last")
    ap.add_argument("--layers", default="22,23")
    ap.add_argument("--k", type=int, default=25)
    ap.add_argument("--candidates", type=int, default=1024)
    ap.add_argument("--lens", type=Path, required=True)
    ap.add_argument("--basis-dir", type=Path, required=True)
    ap.add_argument("--norm-kind", default="gemma")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    layers = parse_layers(a.layers)
    J, lens_info = load_lens(a.lens, layers)
    W, _ = load_unembed(a.basis_dir)
    s = norm_scale(np.load(a.basis_dir / "final_norm_weight_fp32.npy"), a.norm_kind)
    cm, cz = load_extraction(a.root / "core" / a.condition)
    ci = select_rows(cm, "discussion", a.condition == "prompt_only")
    pos = "prompt_end" if a.condition == "prompt_only" else a.position
    recs_p = a.out / "role_dirs.json"
    recs = json.loads(recs_p.read_text()) if recs_p.exists() else []
    for L in layers:
        t0 = time.time()
        X = cz[f"{pos}__layer_{L}"][ci].astype(np.float32)
        an = atom_norms(W, s, J[L], a.basis_dir / f"derived_atom_norms_L{L}.npy")
        dec = sweep.decompose_nested(X, J[L], W, s, an, k_max=a.k, n_cand=a.candidates)
        hJ, _ = sweep.j_arms(X, dec, J[L], W, s, a.k, True)
        roles = np.array([cm[i]["role"] for i in ci])
        modes = np.array([cm[i]["mode"] for i in ci])
        col = (roles == "colluder") & (modes == "collusion")
        hon = (roles == "honest") & (modes == "collusion")
        u = hJ[col].mean(0) - hJ[hon].mean(0)
        u = (u / max(np.linalg.norm(u), 1e-8)).astype(np.float32)
        f = a.out / f"role_dir_J25_{a.condition}_{pos}_L{L}.npy"
        np.save(f, u)
        rec = {"condition": a.condition, "position": pos, "layer": L, "k": a.k, "file": str(f.relative_to(ROOT)) if f.is_relative_to(ROOT) else str(f),
               "hj_norm_median": float(np.median(np.linalg.norm(hJ, axis=1))), "n_colluder_rows": int(col.sum()),
               "n_honest_rows": int(hon.sum()), "pursuit_steps_mean": float(dec["steps"].mean()), "seconds": round(time.time() - t0, 1),
               "lens": lens_info}
        recs = [r for r in recs if not (r["condition"] == a.condition and r["position"] == pos and r["layer"] == L)] + [rec]
        recs_p.write_text(json.dumps(recs, indent=1) + "\n")
        print(json.dumps({k: v for k, v in rec.items() if k != "lens"}), flush=True)


if __name__ == "__main__":
    main()
