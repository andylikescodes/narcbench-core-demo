#!/usr/bin/env python3
"""Experiments 1 and 2 on saved activations (CPU): J-space readouts and the k x position sweep.

Inputs are extraction folders written by scripts/jspace_interp_extract.py (one per split x condition),
a lens in the jlens save format and the unembedding basis (nov5 layout). For each (condition, position,
layer, k) it scores, on Core leave-one-domain-out, Core -> Transfer (with a shared stratified episode
bootstrap) and within-Transfer leave-one-family-out:

  full, J_k (pursuit only), J_k_fill (filled to exactly k, the Nov 5 primary at k=25), C_k_fill,
  fixedframe_k (the k atoms most often chosen on Core rows, same for every row),
  shuffled_k (another row's atoms), randsub_k (N Haar subspaces), randatoms_k (N random sets of k
  J-lens atoms), pca_k (top-k principal components of Core rows).

Random controls are scored on Core -> Transfer (N draws) and on Core LODO for the first
``--n-random-lodo`` draws. Experiment 1 readouts use the k=25 filled atoms and the lens readout.

  python scripts/jspace_interp_analyze.py --root data/jspace_interp/gemma2_9b \\
      --lens <lens.pt> --basis-dir data/j_basis/gemma2_9b --tokenizer <tokenizer.json> \\
      --layers 19-23 --ks 8,25,50,100 --positions last,eot,prevote,frac50,reply_mean \\
      --conditions original,swap --n-random 200 --out results/jspace_interp/gemma2_9b
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from jspace_interp import concepts, detectors, stats, sweep  # noqa: E402
from jspace_interp.nov5_jspace_decompose import (atom_norms, load_lens, load_unembed, norm_scale,  # noqa: E402
                                                 parse_layers, shuffled_control)
from jspace_interp.rows import load_extraction, select_rows, to_agents  # noqa: E402


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def vocab_from_tokenizer(path: Path) -> dict[str, int]:
    tj = json.loads(Path(path).read_text())
    v = tj["model"]["vocab"]
    if isinstance(v, list):  # unigram
        return {t: i for i, (t, _) in enumerate(v)}
    return dict(v)


def shared_boot_idx(xfer_agents, B, seed):
    """Stratified (family x label) episode resamples, shared by every arm, layer and k."""
    rids = sorted({a["run_id"] for a in xfer_agents})
    lab = {a["run_id"]: int(a["mode"] == "collusion") for a in xfer_agents}
    fam = {a["run_id"]: a["domain"] for a in xfer_agents}
    strata = np.array([f"{fam[r]}|{lab[r]}" for r in rids])
    rng = np.random.default_rng(seed)
    groups = [np.where(strata == g)[0] for g in np.unique(strata)]
    return np.vstack([np.concatenate([rng.choice(g, len(g), replace=True) for g in groups]) for _ in range(B)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True, help="has core/<condition>/ and transfer/<condition>/")
    ap.add_argument("--lens", type=Path, required=True)
    ap.add_argument("--basis-dir", type=Path, required=True)
    ap.add_argument("--tokenizer", type=Path, required=True)
    ap.add_argument("--layers", default="19-23")
    ap.add_argument("--ks", default="8,25,50,100")
    ap.add_argument("--positions", default="last,eot,prevote,frac25,frac50,frac75,reply_mean")
    ap.add_argument("--conditions", default="original,swap")
    ap.add_argument("--n-random", type=int, default=200)
    ap.add_argument("--n-random-lodo", type=int, default=20)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--candidates", type=int, default=1024)
    ap.add_argument("--norm-kind", default="gemma")
    ap.add_argument("--no-lodo", action="store_true", help="skip Core LODO for every arm (fast smoke)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    layers = parse_layers(args.layers)
    ks = [int(k) for k in args.ks.split(",")]
    kmax = max(ks)
    positions = args.positions.split(",")
    J, lens_info = load_lens(args.lens, layers)
    W, _ = load_unembed(args.basis_dir)
    s = norm_scale(np.load(args.basis_dir / "final_norm_weight_fp32.npy"), args.norm_kind)
    an = {L: atom_norms(W, s, J[L], args.basis_dir / f"derived_atom_norms_L{L}.npy") for L in layers}
    vocab = vocab_from_tokenizer(args.tokenizer)
    inv_vocab = {i: t for t, i in vocab.items()}
    gids = concepts.group_token_ids(vocab)
    rng = np.random.default_rng(args.seed)
    results = {"lens": lens_info, "layers": layers, "ks": ks, "positions": positions,
               "concept_token_counts": {g: len(v) for g, v in gids.items()}, "cells": [], "exp1": []}

    for cond in args.conditions.split(","):
        cm, cz = load_extraction(args.root / "core" / cond)
        tm, tz = load_extraction(args.root / "transfer" / cond)
        prompt_only = cond == "prompt_only"
        ci = select_rows(cm, "discussion", prompt_only)
        ti = select_rows(tm, "final", prompt_only)
        pos_list = ["prompt_end"] if prompt_only else positions
        for pos in pos_list:
            for L in layers:
                key = f"{pos}__layer_{L}"
                if key not in cz.files:
                    log("missing", cond, key)
                    continue
                Xc = cz[key][ci].astype(np.float32)
                Xt = tz[key][ti].astype(np.float32)
                d = Xc.shape[1]
                base_c = to_agents(cm, ci, Xc, "core")
                base_t = to_agents(tm, ti, Xt, "transfer")
                boot_idx = shared_boot_idx(base_t, args.boot, args.seed + 1)
                lodo = not args.no_lodo

                def score(Fc, Ft, lodo_on=lodo):
                    ca = [dict(a) for a in base_c]
                    ta = [dict(a) for a in base_t]
                    for a, x in zip(ca, to_agents(cm, ci, Fc, "core")):
                        a["x"] = x["x"]
                    for a, x in zip(ta, to_agents(tm, ti, Ft, "transfer")):
                        a["x"] = x["x"]
                    return sweep.protocol_scores(ca, ta, lodo=lodo_on, lofo=lodo_on, boot_idx=boot_idx)

                cell = {"condition": cond, "position": pos, "layer": L, "n_core_rows": len(ci),
                        "n_transfer_rows": len(ti)}
                full = score(Xc, Xt)
                cell["full"] = {k: v for k, v in full.items() if not k.startswith("_")}
                log(cond, pos, L, "full", cell["full"])
                dc = sweep.decompose_nested(Xc, J[L], W, s, an[L], k_max=kmax, n_cand=args.candidates)
                dt = sweep.decompose_nested(Xt, J[L], W, s, an[L], k_max=kmax, n_cand=args.candidates)
                cell["pursuit_steps_core_mean"] = float(dc["steps"].mean())
                cell["pursuit_steps_transfer_mean"] = float(dt["steps"].mean())
                Fc_mean = np.linalg.norm(Xc, axis=1) ** 2
                for k in ks:
                    arms = {}
                    for fill in (False, True):
                        hc, tc = sweep.j_arms(Xc, dc, J[L], W, s, k, fill)
                        ht, tt = sweep.j_arms(Xt, dt, J[L], W, s, k, fill)
                        name = f"J{k}_fill" if fill else f"J{k}"
                        r = score(hc, ht)
                        arms[name] = r
                        arms[name]["fve_core"] = float(np.mean((hc ** 2).sum(1) / Fc_mean))
                        if fill:
                            arms[f"C{k}_fill"] = score(Xc - hc, Xt - ht)
                            # shuffled: another row's (filled) atoms
                            shc, _ = shuffled_control(Xc, tc, J[L], W, s, args.seed + L)
                            sht, _ = shuffled_control(Xt, tt, J[L], W, s, args.seed + L + 1000)
                            arms[f"shuffled{k}"] = score(shc, sht, lodo_on=False)
                            # fixed frame: most frequent atoms over Core rows
                            freq = Counter(int(t) for row in tc for t in row if t >= 0)
                            fixed = np.array([t for t, _ in freq.most_common(k)], dtype=np.int64)
                            Qf = sweep.atoms_basis(fixed, J[L], W, s)
                            arms[f"fixedframe{k}"] = score(sweep.project_fixed(Xc, Qf), sweep.project_fixed(Xt, Qf))
                            if k == 25 and pos in ("last", "prevote", "frac50", "prompt_end"):
                                results["exp1"].append(exp1_readouts(cond, pos, L, cm, ci, tm, ti, tc, tt,
                                                                     dc, dt, gids, inv_vocab))
                    # PCA control
                    Xm = Xc - Xc.mean(0)
                    _, _, Vt = np.linalg.svd(Xm, full_matrices=False)
                    Qp = Vt[:k].T
                    arms[f"pca{k}"] = score(sweep.project_fixed(Xc, Qp), sweep.project_fixed(Xt, Qp))
                    # many-draw random controls (Core -> Transfer; LODO for the first few draws)
                    rs_c2t, rs_lodo, ra_c2t = [], [], []
                    V = W.shape[0]
                    for b in range(args.n_random):
                        Q = sweep.haar_subspace(d, k, rng)
                        r = score(sweep.project_fixed(Xc, Q), sweep.project_fixed(Xt, Q),
                                  lodo_on=lodo and b < args.n_random_lodo)
                        rs_c2t.append(r["core_to_transfer"])
                        if "core_lodo" in r:
                            rs_lodo.append(r["core_lodo"])
                        toks = rng.choice(V, size=k, replace=False)
                        Qa = sweep.atoms_basis(toks, J[L], W, s)
                        r2 = score(sweep.project_fixed(Xc, Qa), sweep.project_fixed(Xt, Qa), lodo_on=False)
                        ra_c2t.append(r2["core_to_transfer"])
                    clean = {a: {kk: vv for kk, vv in r.items() if not kk.startswith("_")} for a, r in arms.items()}
                    jf = arms[f"J{k}_fill"]["core_to_transfer"]
                    jp = arms[f"J{k}"]["core_to_transfer"]
                    clean["randsub"] = {"n": len(rs_c2t), "c2t_mean": float(np.mean(rs_c2t)),
                                        "c2t_p95": float(np.percentile(rs_c2t, 95)),
                                        "c2t_draws": [float(x) for x in rs_c2t],
                                        "lodo_draws": [float(x) for x in rs_lodo],
                                        "J_fill_percentile": stats.empirical_percentile(jf, rs_c2t),
                                        "J_pursuit_percentile": stats.empirical_percentile(jp, rs_c2t)}
                    clean["randatoms"] = {"n": len(ra_c2t), "c2t_mean": float(np.mean(ra_c2t)),
                                          "c2t_p95": float(np.percentile(ra_c2t, 95)),
                                          "J_fill_percentile": stats.empirical_percentile(jf, ra_c2t)}
                    fullc = full["core_to_transfer"]
                    clean["recovery_J_fill"] = float((jf - 0.5) / max(fullc - 0.5, 1e-9))
                    # paired bootstrap difference J_fill - full on shared episodes
                    clean["J_fill_minus_full_ci95"] = list(np.nanpercentile(arms[f"J{k}_fill"]["_boot"] - full["_boot"],
                                                                            [2.5, 97.5]).astype(float))
                    cell[f"k{k}"] = clean
                    log(cond, pos, L, f"k={k}", "J_fill", round(jf, 3), "J", round(jp, 3),
                        "randsub p95", round(clean["randsub"]["c2t_p95"], 3), "full", round(fullc, 3))
                results["cells"].append(cell)
                (args.out / "sweep_results.json").write_text(json.dumps(results, indent=1, default=float) + "\n")
    log("wrote", args.out / "sweep_results.json")


def exp1_readouts(cond, pos, L, cm, ci, tm, ti, tc, tt, dc, dt, gids, inv_vocab):
    """Experiment 1 at k=25 (filled atoms) and the lens readout (top 50): concept-group shares per agent,
    colluder vs honest (within-episode permutation) and colluder vs control agents (by scenario)."""
    out = {"condition": cond, "position": pos, "layer": L}
    for split, meta, idx, toks, dec in (("core", cm, ci, tc, dc), ("transfer", tm, ti, tt, dt)):
        res = {}
        for src, T in (("atoms", toks), ("readout50", dec["readout"])):
            gm = sweep.group_mass(T, None, gids)
            G = np.stack([gm[g] for g in gids], axis=1)
            agents = to_agents(meta, idx, G, split)
            for gi, g in enumerate(gids):
                col = [a for a in agents if a["mode"] == "collusion"]
                v = [a["x"][gi] for a in col]
                obs, p, _ = stats.within_episode_role_permutation(
                    v, [a["run_id"] for a in col], [a["role"] == "colluder" for a in col], n_perm=5000)
                sel = [a for a in agents if a["role"] == "colluder" or a["mode"] == "control"]
                obs2, p2, _ = stats.stratified_permutation(
                    [a["x"][gi] for a in sel], [a["role"] == "colluder" for a in sel],
                    [a["scenario_id"] or a["run_id"].split("__")[1] for a in sel], n_perm=5000)
                res[f"{src}:{g}"] = {"colluder_minus_honest": obs, "p_within": p,
                                     "colluder_minus_control": obs2, "p_strat": p2}
            # per-token frequency differences (atoms only)
            if src == "atoms":
                rowsets = {}
                for j, i in enumerate(idx):
                    m = meta[i]
                    rowsets.setdefault((m["run_id"], m["agent_name"], m["role"], m["mode"]), set()).update(
                        int(t) for t in T[j] if t >= 0)
                keys = [k for k in rowsets if k[3] == "collusion"]
                tests = sweep.token_frequency_tests([rowsets[k] for k in keys], [k[2] == "colluder" for k in keys],
                                                    [k[0] for k in keys], n_perm=1000)
                res["top_tokens"] = [dict(t, text=inv_vocab.get(t["token"], "?")) for t in tests[:30]]
                cnt = Counter(t for k in keys for t in rowsets[k])
                res["most_common_atoms"] = [(inv_vocab.get(t, "?"), c) for t, c in cnt.most_common(20)]
        pv = [v[k] for v in res.values() if isinstance(v, dict) for k in ("p_within", "p_strat")]
        adj, _ = stats.benjamini_hochberg(pv) if pv else ([], [])
        res["bh_adjusted_p_in_order"] = [float(x) for x in adj]
        out[split] = res
    return out


if __name__ == "__main__":
    main()
