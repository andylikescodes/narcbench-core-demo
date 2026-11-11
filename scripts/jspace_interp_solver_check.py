#!/usr/bin/env python3
"""The J-space decomposition as the workspace paper states it, scored on the headline cell (CPU).

Anthropic's `jlens` package ships the lens fit and the readout, not the J-space decomposition, so the
decomposition in this repo is our implementation of the paper's text. The paper (Gurnee et al., 2026) gives:
J-lens vectors = rows of W_U J_l; J-space = sparse non-negative combinations of them (k about 25);
"gradient pursuit" as the solver, its reconstruction being the J-space component (Methods); and, in the
formalisation appendix, the projection onto the span of exactly k vectors. It leaves open the stopping rule,
any candidate pre-selection, the normalisation of the vectors and whether the final-norm gain is part of them.

This script scores, on the same activations and with the same probe and episode bootstrap as the main sweep,
the variants that differ in those choices:

  nnomp_span    non-negative OMP (NNLS refit), filled to exactly k, projection onto the span   = the sweep's "J fill"
  nnomp_cone    the same pursuit, J-space component = its non-negative reconstruction sum c_i v_i
  gp_span       gradient pursuit (the solver the paper names), filled to exactly k, projection onto the span
  gp_cone       gradient pursuit, J-space component = its non-negative reconstruction (the paper's operational form)
  nogain_nnomp_span, nogain_gp_cone
                the same two with J-lens vectors taken literally as rows of W_U J_l (no final-norm gain)

  python scripts/jspace_interp_solver_check.py --root results/jspace_interp/acts --lens <lens.pt> \\
      --basis-dir data/j_basis/gemma2_9b --layers 22,23 --out results/jspace_interp/solver_check
  python scripts/jspace_interp_solver_check.py --merge --out results/jspace_interp/solver_check
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
from jspace_interp.nov5_jspace_decompose import (atom_norms, candidates_and_readout, load_lens, load_unembed,  # noqa: E402
                                                 norm_scale, pursuit_one)
from jspace_interp.rows import load_extraction, select_rows, to_agents  # noqa: E402

VARIANTS = ("nnomp_span", "nnomp_cone", "gp_span", "gp_cone", "nogain_nnomp_span", "nogain_gp_cone")


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def decompose(X, J, W, s, an, k, n_cand, methods):
    """For each method: (span projection, cone reconstruction, steps, filled) over the rows of X."""
    cand, _, _ = candidates_and_readout(X, J, W, s, an, n_cand=n_cand, k_read=8)
    out = {m: {"span": np.zeros_like(X), "cone": np.zeros_like(X), "steps": [], "filled": []} for m in methods}
    for i in range(X.shape[0]):
        for m in methods:
            r = pursuit_one(X[i], cand[i], J, W, s, an, k=k, method=m, select="unit", fill=True)
            out[m]["span"][i], out[m]["cone"][i] = r["h_J"], r["cone"]
            out[m]["steps"].append(r["steps"])
            out[m]["filled"].append(r["filled"])
    return out


def shared_boot_idx(xfer_agents, B, seed):
    rids = sorted({a["run_id"] for a in xfer_agents})
    lab = {a["run_id"]: int(a["mode"] == "collusion") for a in xfer_agents}
    fam = {a["run_id"]: a["domain"] for a in xfer_agents}
    strata = np.array([f"{fam[r]}|{lab[r]}" for r in rids])
    rng = np.random.default_rng(seed)
    groups = [np.where(strata == g)[0] for g in np.unique(strata)]
    return np.vstack([np.concatenate([rng.choice(g, len(g), replace=True) for g in groups]) for _ in range(B)])


def run_layer(a, L):
    cm, cz = load_extraction(a.root / "core" / a.condition)
    tm, tz = load_extraction(a.root / "transfer" / a.condition)
    ci, ti = select_rows(cm, "discussion"), select_rows(tm, "final")
    key = f"{a.position}__layer_{L}"
    Xc, Xt = cz[key][ci].astype(np.float32), tz[key][ti].astype(np.float32)
    J = load_lens(a.lens, [L])[0][L]
    W, _ = load_unembed(a.basis_dir)
    gain = norm_scale(np.load(a.basis_dir / "final_norm_weight_fp32.npy"), "gemma")
    base_c, base_t = to_agents(cm, ci, Xc, "core"), to_agents(tm, ti, Xt, "transfer")
    boot_idx = shared_boot_idx(base_t, a.boot, a.seed + 1)   # the main sweep's resamples (same seed)

    def score(Fc, Ft):
        ca = [dict(x) for x in base_c]
        ta = [dict(x) for x in base_t]
        for x, y in zip(ca, to_agents(cm, ci, Fc, "core")):
            x["x"] = y["x"]
        for x, y in zip(ta, to_agents(tm, ti, Ft, "transfer")):
            x["x"] = y["x"]
        return sweep.protocol_scores(ca, ta, lodo=False, lofo=False, boot_idx=boot_idx)

    res = {"layer": L, "condition": a.condition, "position": a.position, "k": a.k, "n_core_rows": int(len(ci)),
           "n_transfer_rows": int(len(ti)), "arms": {}}
    full = score(Xc, Xt)
    boots = {"full": full["_boot"]}
    res["arms"]["full"] = {"core_to_transfer": full["core_to_transfer"], "ci95": full["core_to_transfer_ci95"]}
    for tag, s in (("", gain), ("nogain_", np.ones_like(gain))):
        t0 = time.time()
        an = atom_norms(W, s, J, None)
        methods = ("nnomp", "gp")
        dc = decompose(Xc, J, W, s, an, a.k, a.candidates, methods)
        dt = decompose(Xt, J, W, s, an, a.k, a.candidates, methods)
        for m in methods:
            for form in ("span", "cone"):
                name = f"{tag}{m}_{form}"
                if name not in VARIANTS:
                    continue
                r = score(dc[m][form], dt[m][form])
                boots[name] = r["_boot"]
                fve = float(np.mean((dc[m][form] ** 2).sum(1) / (Xc ** 2).sum(1)))
                res["arms"][name] = {"core_to_transfer": r["core_to_transfer"], "ci95": r["core_to_transfer_ci95"],
                                     "fve_core": fve, "steps_core_mean": float(np.mean(dc[m]["steps"])),
                                     "filled_core_mean": float(np.mean(dc[m]["filled"]))}
                log(f"L{L} {name}: C->T {r['core_to_transfer']:.3f} {np.round(r['core_to_transfer_ci95'], 3).tolist()} "
                    f"fve {fve:.4f} steps {np.mean(dc[m]['steps']):.1f}")
        log(f"L{L} dictionary '{tag or 'gain'}' done in {time.time() - t0:.0f}s")
    ref = boots["nnomp_span"]
    for name, b in boots.items():
        if name not in ("nnomp_span",):
            res["arms"][name]["minus_nnomp_span_ci95"] = [float(x) for x in np.nanpercentile(b - ref, [2.5, 97.5])]
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / f"layer_{L}.json").write_text(json.dumps(res, indent=1) + "\n")
    np.save(a.out / f"boot_layer_{L}.npy", np.stack([boots[n] for n in ("full",) + VARIANTS]).astype(np.float32))
    return res


EXTRA = ("gp_recon", "gp_lit", "gpfull_recon", "gpfull_lit", "gpfull_span")


def run_layer_extra(a, L):
    """Two choices the paper's text leaves open, scored on the same cell with the same resamples.

      gp_recon      gradient pursuit among the 1,024 candidates, no fill; J-space component = the best non-negative
                    combination (NNLS) of the vectors it picks.  This is the sweep's ``J25_recon`` arm.
      gp_lit        the same pursuit; J-space component = its own output sum c_i v_i (no refit of the coefficients)
      gpfull_*      gradient pursuit with the selection step over the whole vocabulary (no candidate pre-selection):
                    _recon = NNLS combination of its 25 vectors, _lit = its own output, _span = projection onto their span
    """
    from jspace_interp.nov5_jspace_decompose import project_onto_tokens

    cm, cz = load_extraction(a.root / "core" / a.condition)
    tm, tz = load_extraction(a.root / "transfer" / a.condition)
    ci, ti = select_rows(cm, "discussion"), select_rows(tm, "final")
    key = f"{a.position}__layer_{L}"
    Xc, Xt = cz[key][ci].astype(np.float32), tz[key][ti].astype(np.float32)
    X = np.concatenate([Xc, Xt])
    nc = Xc.shape[0]
    J = load_lens(a.lens, [L])[0][L]
    W, _ = load_unembed(a.basis_dir)
    s = norm_scale(np.load(a.basis_dir / "final_norm_weight_fp32.npy"), "gemma")
    an = atom_norms(W, s, J, a.basis_dir / f"derived_atom_norms_L{L}.npy")
    base_c, base_t = to_agents(cm, ci, Xc, "core"), to_agents(tm, ti, Xt, "transfer")
    boot_idx = shared_boot_idx(base_t, a.boot, a.seed + 1)

    def score(F):
        ca = [dict(x) for x in base_c]
        ta = [dict(x) for x in base_t]
        for x, y in zip(ca, to_agents(cm, ci, F[:nc], "core")):
            x["x"] = y["x"]
        for x, y in zip(ta, to_agents(tm, ti, F[nc:], "transfer")):
            x["x"] = y["x"]
        return sweep.protocol_scores(ca, ta, lodo=False, lofo=False, boot_idx=boot_idx)

    feats, info = {}, {}
    t0 = time.time()
    # (1) the sweep's pursuit: 1,024 candidates, no fill
    cand, _, _ = candidates_and_readout(X, J, W, s, an, n_cand=a.candidates, k_read=8)
    order = np.full((X.shape[0], a.k), -1, dtype=np.int64)
    lit = np.zeros_like(X)
    steps = np.zeros(X.shape[0], dtype=np.int32)
    for i in range(X.shape[0]):
        r = pursuit_one(X[i], cand[i], J, W, s, an, k=a.k, method="gp", select="unit", fill=False)
        order[i, :len(r["tokens"])] = r["tokens"]
        lit[i], steps[i] = r["cone"], r["steps"]
    feats["gp_recon"], used = sweep.j_recon(X, {"order": order}, J, W, s, a.k)
    feats["gp_lit"] = lit
    info["gp_recon"] = info["gp_lit"] = {"steps_core_mean": float(steps[:nc].mean()), "candidates": int(a.candidates)}
    info["gp_recon"] = dict(info["gp_recon"], atoms_used_core_mean=float(used[:nc].mean()))
    log(f"L{L} 1,024-candidate gradient pursuit done in {time.time() - t0:.0f}s (steps {steps[:nc].mean():.1f})")
    # (2) the same pursuit with the whole vocabulary as candidates
    t0 = time.time()
    full = sweep.gradient_pursuit_full(X, J, W, s, an, k=a.k, batch=a.batch, log=log)
    feats["gpfull_lit"] = full["recon_gp"]
    feats["gpfull_recon"], used_f = sweep.j_recon(X, {"order": full["order"]}, J, W, s, a.k)
    feats["gpfull_span"] = np.stack([project_onto_tokens(X[i], full["order"][i], J, W, s) for i in range(X.shape[0])])
    for n_ in ("gpfull_recon", "gpfull_lit", "gpfull_span"):
        info[n_] = {"steps_core_mean": float(full["steps"][:nc].mean()), "candidates": int(W.shape[0])}
    info["gpfull_recon"] = dict(info["gpfull_recon"], atoms_used_core_mean=float(used_f[:nc].mean()))
    log(f"L{L} full-vocabulary gradient pursuit done in {time.time() - t0:.0f}s (steps {full['steps'][:nc].mean():.1f})")

    def cos(A, B):
        na, nb = np.linalg.norm(A, axis=1), np.linalg.norm(B, axis=1)
        ok = (na > 0) & (nb > 0)
        return float(np.mean((A[ok] * B[ok]).sum(1) / (na[ok] * nb[ok])))

    jac, first5, outside = [], [], []
    for i in range(X.shape[0]):
        a1 = [int(x) for x in order[i] if x >= 0]
        a2 = [int(x) for x in full["order"][i] if x >= 0]
        if a1 and a2:
            jac.append(len(set(a1) & set(a2)) / len(set(a1) | set(a2)))
            first5.append(float(a1[:5] == a2[:5]))
            outside.append(float(np.mean([x not in set(cand[i].tolist()) for x in a2])))
    res = {"layer": L, "condition": a.condition, "position": a.position, "k": a.k, "arms": {},
           "overlap": {"token_set_jaccard_mean": float(np.mean(jac)), "first5_identical_share": float(np.mean(first5)),
                       "share_of_full_vocab_picks_outside_the_1024_candidates": float(np.mean(outside)),
                       "cos_gp_recon_vs_gp_lit": cos(feats["gp_recon"], feats["gp_lit"]),
                       "cos_gp_recon_vs_gpfull_recon": cos(feats["gp_recon"], feats["gpfull_recon"]),
                       "cos_gpfull_recon_vs_gpfull_lit": cos(feats["gpfull_recon"], feats["gpfull_lit"])}}
    boots = {}
    for name in EXTRA:
        r = score(feats[name])
        boots[name] = r["_boot"]
        fve = float(np.mean((feats[name][:nc] ** 2).sum(1) / (Xc ** 2).sum(1)))
        res["arms"][name] = dict(info[name], core_to_transfer=r["core_to_transfer"], ci95=r["core_to_transfer_ci95"], fve_core=fve)
        log(f"L{L} {name}: C->T {r['core_to_transfer']:.3f} {np.round(r['core_to_transfer_ci95'], 3).tolist()} fve {fve:.4f}")
    main_boot = a.out / f"boot_layer_{L}.npy"
    if main_boot.exists():                                  # paired differences to the sweep's earlier primary
        ref = np.load(main_boot)[(("full",) + VARIANTS).index("nnomp_span")]
        for name in EXTRA:
            res["arms"][name]["minus_nnomp_span_ci95"] = [float(x) for x in np.nanpercentile(boots[name] - ref, [2.5, 97.5])]
    for name in EXTRA:
        if name != "gp_recon":
            res["arms"][name]["minus_gp_recon_ci95"] = [float(x) for x in np.nanpercentile(boots[name] - boots["gp_recon"], [2.5, 97.5])]
    log(f"L{L} overlap {json.dumps({k: round(v, 3) for k, v in res['overlap'].items()})}")
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / f"extra_layer_{L}.json").write_text(json.dumps(res, indent=1) + "\n")
    np.save(a.out / f"extra_boot_layer_{L}.npy", np.stack([boots[n] for n in EXTRA]).astype(np.float32))
    return res


def merge_extra(a, out):
    files = sorted(a.out.glob("extra_layer_*.json"))
    if not files:
        return
    per = [json.loads(f.read_text()) for f in files]
    layers = [p["layer"] for p in per]
    ex = {"layers": layers, "per_layer": {str(p["layer"]): p["arms"] for p in per},
          "overlap": {str(p["layer"]): p["overlap"] for p in per}, "band_mean": {}, "band_ci95": {},
          "band_minus_nnomp_span_ci95": {}, "band_minus_gp_recon_ci95": {}}
    B = np.mean(np.stack([np.load(a.out / f"extra_boot_layer_{L}.npy") for L in layers]), axis=0)   # [arm, draw]
    main = [a.out / f"boot_layer_{L}.npy" for L in layers]
    ref = None
    if all(p.exists() for p in main):
        ref = np.mean(np.stack([np.load(p)[(("full",) + VARIANTS).index("nnomp_span")] for p in main]), axis=0)
    for i, n in enumerate(EXTRA):
        ex["band_mean"][n] = float(np.mean([p["arms"][n]["core_to_transfer"] for p in per]))
        ex["band_ci95"][n] = [float(x) for x in np.nanpercentile(B[i], [2.5, 97.5])]
        ex["band_minus_gp_recon_ci95"][n] = [float(x) for x in np.nanpercentile(B[i] - B[EXTRA.index("gp_recon")], [2.5, 97.5])]
        if ref is not None:
            ex["band_minus_nnomp_span_ci95"][n] = [float(x) for x in np.nanpercentile(B[i] - ref, [2.5, 97.5])]
    # every reading against the whole-vocabulary one (gradient pursuit over all tokens, non-negative combination), which
    # the later sweep takes as its J-space arm: paired differences of the band mean, for the earlier arms as well
    full_ref = B[EXTRA.index("gpfull_recon")]
    ex["band_minus_gpfull_recon_ci95"] = {n: [float(x) for x in np.nanpercentile(B[i] - full_ref, [2.5, 97.5])] for i, n in enumerate(EXTRA)}
    if all(p.exists() for p in main):
        M = np.mean(np.stack([np.load(p) for p in main]), axis=0)
        for i, n in enumerate(("full",) + VARIANTS):
            ex["band_minus_gpfull_recon_ci95"][n] = [float(x) for x in np.nanpercentile(M[i] - full_ref, [2.5, 97.5])]
    # the earlier arms over the same layers, so that the two sets of rows are comparable when fewer layers have run
    ex["band_mean_main_arms_same_layers"] = {n: float(np.mean([out["per_layer"][str(L)][n]["core_to_transfer"] for L in layers]))
                                             for n in ("full",) + VARIANTS if all(str(L) in out["per_layer"] for L in layers)}
    out["extra"] = ex


def merge(a):
    files = sorted(a.out.glob("layer_*.json"))
    per = [json.loads(f.read_text()) for f in files]
    layers = [p["layer"] for p in per]
    names = ("full",) + VARIANTS
    out = {"layers": layers, "condition": per[0]["condition"], "position": per[0]["position"], "k": per[0]["k"],
           "per_layer": {str(p["layer"]): p["arms"] for p in per}, "band_mean": {}, "band_minus_nnomp_span_ci95": {}}
    for n in names:
        out["band_mean"][n] = float(np.mean([p["arms"][n]["core_to_transfer"] for p in per]))
    # paired band-mean differences: the bootstrap resamples are the same episodes at every layer
    B = [np.load(a.out / f"boot_layer_{L}.npy") for L in layers]
    band = np.mean(np.stack(B), axis=0)                       # [arm, draw]
    ref = band[names.index("nnomp_span")]
    for i, n in enumerate(names):
        out["band_minus_nnomp_span_ci95"][n] = [float(x) for x in np.nanpercentile(band[i] - ref, [2.5, 97.5])]
        out.setdefault("band_ci95", {})[n] = [float(x) for x in np.nanpercentile(band[i], [2.5, 97.5])]
    merge_extra(a, out)
    (a.out / "solver_check.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({"band_mean": {k: round(v, 3) for k, v in out["band_mean"].items()},
                      "band_minus_nnomp_span_ci95": {k: [round(x, 3) for x in v] for k, v in out["band_minus_nnomp_span_ci95"].items()}}, indent=1))
    if "extra" in out:
        print(json.dumps({"extra_layers": out["extra"]["layers"],
                          "extra_band_mean": {k: round(v, 3) for k, v in out["extra"]["band_mean"].items()}}, indent=1))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=ROOT / "results/jspace_interp/acts")
    ap.add_argument("--lens", type=Path)
    ap.add_argument("--basis-dir", type=Path, default=ROOT / "data/j_basis/gemma2_9b")
    ap.add_argument("--condition", default="original")
    ap.add_argument("--position", default="last")
    ap.add_argument("--layers", default="19,20,21,22,23")
    ap.add_argument("--k", type=int, default=25)
    ap.add_argument("--candidates", type=int, default=1024)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--merge", action="store_true", help="combine the per-layer files into solver_check.json")
    ap.add_argument("--extra", action="store_true",
                    help="score the coefficient refit and the candidate pre-selection instead (gradient pursuit only; see run_layer_extra)")
    ap.add_argument("--batch", type=int, default=496, help="rows per batch of the full-vocabulary pursuit")
    ap.add_argument("--out", type=Path, default=ROOT / "results/jspace_interp/solver_check")
    a = ap.parse_args()
    if a.merge:
        return merge(a)
    for L in [int(x) for x in a.layers.split(",")]:
        (run_layer_extra if a.extra else run_layer)(a, L)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
