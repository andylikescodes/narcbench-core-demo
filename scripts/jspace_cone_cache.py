#!/usr/bin/env python3
"""Build the cone-fit arm of a jspace_decompose.py run without redoing the pursuit.

jspace_decompose.py saves each activation's chosen atoms and nonnegative coefficients (decomp_tokens.npz). The paper's prose
describes J-space as a nonnegative cone, so the registered variant "cone fit in place of the projection" is
    cone_i = sum_j c_ij v_{t_ij},   v_t = J_l^T (s * W_U[t]),
rebuilt here from those arrays and written as <variant dir>/cone/activations_gen.npz (+ metadata) in the paper layout, ready
for jspace_separability.py --external-arm.

    python3 scripts/jspace_cone_cache.py --variant-dir data/activations/gemma2_9b/core/20261001T012639Z/variants/jpaper_nnomp_k25 \
        --lens data/j_basis/neuronpedia/gemma-2-9b-it/jlens/Salesforce-wikitext/gemma-2-9b-it_jacobian_lens.pt
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("jd", ROOT / "scripts/jspace_decompose.py")
jd = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(jd)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant-dir", type=Path, required=True, help="a jspace_decompose.py --out-root")
    ap.add_argument("--lens", type=Path, required=True)
    ap.add_argument("--basis-dir", type=Path, default=ROOT / "data/j_basis/gemma2_9b")
    ap.add_argument("--norm-kind", choices=["gemma", "standard"], default="gemma")
    a = ap.parse_args()
    vdir = a.variant_dir if a.variant_dir.is_absolute() else ROOT / a.variant_dir
    lens_p = a.lens if a.lens.is_absolute() else ROOT / a.lens
    basis = a.basis_dir if a.basis_dir.is_absolute() else ROOT / a.basis_dir
    meta = json.loads((vdir / "decomp_meta.json").read_text())
    layers = meta["settings"]["layers"]
    tk = np.load(vdir / "decomp_tokens.npz")
    J, _ = jd.load_lens(lens_p, layers)
    W, _ = jd.load_unembed(basis)
    s = jd.norm_scale(np.load(basis / "final_norm_weight_fp32.npy"), a.norm_kind)
    out_dir = vdir / "cone"
    out_dir.mkdir(exist_ok=True)
    shutil.copy2(vdir / "j_only" / "metadata_gen.json", out_dir / "metadata_gen.json")
    arrays, stats = {}, {}
    for L in layers:
        t0 = time.time()
        toks, coefs = tk[f"tokens_L{L}"], tk[f"coefs_L{L}"]
        n = toks.shape[0]
        cone = np.zeros((n, W.shape[1]), dtype=np.float32)
        for i in range(n):
            m = toks[i] >= 0
            if not m.any():
                continue
            atoms = (np.asarray(W[toks[i][m]], dtype=np.float32) * s[None, :]) @ J[L]  # [m, d]
            cone[i] = coefs[i][m] @ atoms
        arrays[f"layer_{L}"] = cone
        hJ = np.load(vdir / "j_only" / "activations_gen.npz")[f"layer_{L}"]
        rel = float(np.linalg.norm(cone - hJ) / max(np.linalg.norm(hJ), 1e-12))
        stats[str(L)] = {"rows": int(n), "mean_norm": float(np.linalg.norm(cone, axis=1).mean()), "rel_diff_to_projection": rel,
                         "seconds": round(time.time() - t0, 1)}
        print(f"[layer {L}] cone rebuilt for {n} rows; ‖cone − h_J‖/‖h_J‖ = {rel:.3f} ({stats[str(L)]['seconds']}s)", flush=True)
    np.savez_compressed(out_dir / "activations_gen.npz", **arrays)
    (out_dir / "cone_meta.json").write_text(json.dumps({"source": str(vdir), "definition": "sum_j c_ij v_t_ij from decomp_tokens.npz",
                                                        "layers": stats}, indent=1) + "\n")
    print(f"wrote {out_dir / 'activations_gen.npz'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
