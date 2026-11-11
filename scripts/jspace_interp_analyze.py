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
``--n-random-lodo`` draws; the episode bootstrap is not run on the random draws (their spread is the null).
Experiment 1 readouts use the k=25 filled atoms and the lens readout.

Matched-vocabulary control (control 2): when <root>/warned/<condition>/ exists (extraction of the warned
control episodes, scripts/jspace_interp_gen_matched_vocab.py), every arm is also scored on the warned
agents with probes fit out of fold on Core (jspace_interp/matched_vocab.py): warned honest vs unwarned
honest within the warned episodes, colluders vs warned honest, the detectors on warned vs control
episodes, and the concept-atom mass of warned agents. The first --n-random-mv random subspaces give the
null for those readings.

Role directions for Experiment 3: at k=25 the mean filled J component of colluders minus honest agents
(Core discussion rows, collusion episodes) and the median ||h_J|| are written to <out>/role_dirs/.

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
from jspace_interp import concepts, detectors, matched_vocab, stats, sweep  # noqa: E402
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


def clean(r: dict) -> dict:
    return {kk: vv for kk, vv in r.items() if not kk.startswith("_")}


def mv_summary(draws: list[dict]) -> dict:
    """Spread of the matched-vocabulary readings over random draws."""
    out = {"n": len(draws)}
    for key in ("auroc_colluder_vs_warned", "auroc_warned_vs_unwarned", "auroc_warned_vs_control_honest", "echo_ratio"):
        vals = [d[key] for d in draws if d.get(key) is not None]
        if vals:
            out[key] = {"mean": float(np.mean(vals)), "p5": float(np.percentile(vals, 5)), "p95": float(np.percentile(vals, 95)),
                        "draws": [float(v) for v in vals]}
    ep = [d["episode"].get("warned_vs_control_primary") for d in draws if d.get("episode")]
    ep = [v for v in ep if v is not None]
    if ep:
        out["episode_warned_vs_control_primary"] = {"mean": float(np.mean(ep)), "p5": float(np.percentile(ep, 5)),
                                                    "p95": float(np.percentile(ep, 95))}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, required=True, help="has core/<condition>/ and transfer/<condition>/ (and warned/<condition>/)")
    ap.add_argument("--warned-root", type=Path, default=None, help="folder with <condition>/ extractions of the warned episodes (default <root>/warned)")
    ap.add_argument("--lens", type=Path, required=True)
    ap.add_argument("--basis-dir", type=Path, required=True)
    ap.add_argument("--tokenizer", type=Path, required=True)
    ap.add_argument("--layers", default="19-23")
    ap.add_argument("--ks", default="8,25,50,100")
    ap.add_argument("--positions", default="last,eot,prevote,frac25,frac50,frac75,reply_mean")
    ap.add_argument("--conditions", default="original,swap")
    ap.add_argument("--n-random", type=int, default=200)
    ap.add_argument("--n-random-lodo", type=int, default=20)
    ap.add_argument("--n-random-mv", type=int, default=20, help="random subspaces also scored on the matched-vocabulary control")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--boot-random", action="store_true", help="also bootstrap every random draw (slow; off by default)")
    ap.add_argument("--candidates", type=int, default=1024)
    ap.add_argument("--norm-kind", default="gemma",
                    help="gain of the final norm inside the J-lens vectors: gemma = (1 + weight), the default; none = no gain "
                         "(the vectors are then literally the rows of W_U J_l)")
    ap.add_argument("--no-lodo", action="store_true", help="skip Core LODO for every arm (fast smoke)")
    ap.add_argument("--no-warned", action="store_true", help="ignore warned extractions even if present")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--method", default="nnomp", choices=("nnomp", "gp"),
                    help="pursuit solver: nnomp (non-negative OMP, NNLS refit) or gp (gradient pursuit, the solver Gurnee et al. name)")
    ap.add_argument("--reuse-random-from", type=Path, default=None,
                    help="root of an earlier sweep (<root>/<condition>_<position>/sweep_results.json): copy its random-subspace and "
                         "random-atom draws instead of redrawing them (they do not depend on the decomposition)")
    ap.add_argument("--redraw-random-atoms", action="store_true",
                    help="with --reuse-random-from: reuse the random subspaces but draw the random lens-atom sets again "
                         "(needed when the J-lens vectors differ from the earlier sweep's, e.g. --norm-kind none)")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "role_dirs").mkdir(exist_ok=True)

    layers = parse_layers(args.layers)
    ks = [int(k) for k in args.ks.split(",")]
    kmax = max(ks)
    positions = args.positions.split(",")
    J, lens_info = load_lens(args.lens, layers)
    W, _ = load_unembed(args.basis_dir)
    s = norm_scale(np.load(args.basis_dir / "final_norm_weight_fp32.npy"), args.norm_kind)
    an_tag = "" if args.norm_kind == "gemma" else f"_{args.norm_kind}"      # the cache depends on the gain
    an = {L: atom_norms(W, s, J[L], args.basis_dir / f"derived_atom_norms{an_tag}_L{L}.npy") for L in layers}
    vocab = vocab_from_tokenizer(args.tokenizer)
    inv_vocab = {i: t for t, i in vocab.items()}
    gids = concepts.group_token_ids(vocab)
    gnames = list(gids)
    rng = np.random.default_rng(args.seed)
    warned_root = args.warned_root or (args.root / "warned")
    results = {"lens": lens_info, "layers": layers, "ks": ks, "positions": positions,
               "concept_token_counts": {g: len(v) for g, v in gids.items()},
               "settings": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
               "cells": [], "exp1": [], "role_dirs": []}

    for cond in args.conditions.split(","):
        cm, cz = load_extraction(args.root / "core" / cond)
        tm, tz = load_extraction(args.root / "transfer" / cond)
        prompt_only = cond == "prompt_only"
        ci = select_rows(cm, "discussion", prompt_only)
        ti = select_rows(tm, "final", prompt_only)
        have_warned = (not args.no_warned) and (warned_root / cond / "metadata.json").exists()
        if have_warned:
            wm, wz = load_extraction(warned_root / cond)
            wi = select_rows(wm, "discussion", prompt_only)
            log(cond, "warned episodes present:", len({wm[i]["run_id"] for i in wi}), "episodes,", len(wi), "rows")
        pos_list = ["prompt_end"] if prompt_only else positions
        for pos in pos_list:
            prev_random = {}
            if args.reuse_random_from is not None:
                pf = args.reuse_random_from / f"{cond}_{pos}" / "sweep_results.json"
                for pc in json.loads(pf.read_text())["cells"]:
                    if pc["condition"] == cond and pc["position"] == pos:
                        prev_random[pc["layer"]] = pc
                log(cond, pos, "random draws reused from", pf, "layers", sorted(prev_random))
            for L in layers:
                key = f"{pos}__layer_{L}"
                if key not in cz.files:
                    log("missing", cond, key)
                    continue
                Xc = cz[key][ci].astype(np.float32)
                Xt = tz[key][ti].astype(np.float32)
                Xw = wz[key][wi].astype(np.float32) if have_warned and key in wz.files else None
                d = Xc.shape[1]
                base_c = to_agents(cm, ci, Xc, "core")
                base_t = to_agents(tm, ti, Xt, "transfer")
                base_w = to_agents(wm, wi, Xw, "warned") if Xw is not None else None
                boot_idx = shared_boot_idx(base_t, args.boot, args.seed + 1)
                lodo = not args.no_lodo

                def with_x(base, meta, idx, F, split):
                    out = [dict(a) for a in base]
                    for a, x in zip(out, to_agents(meta, idx, F, split)):
                        a["x"] = x["x"]
                    return out

                def score(Fc, Ft, Fw=None, lodo_on=lodo, boot=True, mv=True):
                    ca = with_x(base_c, cm, ci, Fc, "core")
                    ta = with_x(base_t, tm, ti, Ft, "transfer")
                    r = sweep.protocol_scores(ca, ta, lodo=lodo_on, lofo=lodo_on, boot_idx=boot_idx if boot else None)
                    if Fw is not None and mv:
                        wa = with_x(base_w, wm, wi, Fw, "warned")
                        r["matched_vocab"] = matched_vocab.score(ca, wa, seed=args.seed)
                    return r

                cell = {"condition": cond, "position": pos, "layer": L, "n_core_rows": len(ci),
                        "n_transfer_rows": len(ti), "n_warned_rows": int(len(wi)) if Xw is not None else 0}
                t_cell = time.time()
                full = score(Xc, Xt, Xw)
                cell["full"] = clean(full)
                log(cond, pos, L, "full", {k: v for k, v in cell["full"].items() if k != "matched_vocab"},
                    "| warned:", {k: round(v, 3) for k, v in cell["full"].get("matched_vocab", {}).items()
                                  if isinstance(v, float)} if Xw is not None else "-")
                dc = sweep.decompose_nested(Xc, J[L], W, s, an[L], k_max=kmax, n_cand=args.candidates, method=args.method)
                dt = sweep.decompose_nested(Xt, J[L], W, s, an[L], k_max=kmax, n_cand=args.candidates, method=args.method)
                dw = (sweep.decompose_nested(Xw, J[L], W, s, an[L], k_max=kmax, n_cand=args.candidates, method=args.method)
                      if Xw is not None else None)
                cell["pursuit_steps_core_mean"] = float(dc["steps"].mean())
                cell["pursuit_steps_transfer_mean"] = float(dt["steps"].mean())
                cell["decompose_seconds"] = round(time.time() - t_cell, 1)
                Fc_mean = np.linalg.norm(Xc, axis=1) ** 2
                for k in ks:
                    t_k = time.time()
                    arms = {}
                    for fill in (False, True):
                        hc, tc = sweep.j_arms(Xc, dc, J[L], W, s, k, fill)
                        ht, tt = sweep.j_arms(Xt, dt, J[L], W, s, k, fill)
                        hw, tw = sweep.j_arms(Xw, dw, J[L], W, s, k, fill) if Xw is not None else (None, None)
                        name = f"J{k}_fill" if fill else f"J{k}"
                        r = score(hc, ht, hw)
                        arms[name] = r
                        arms[name]["fve_core"] = float(np.mean((hc ** 2).sum(1) / Fc_mean))
                        if fill:
                            arms[f"C{k}_fill"] = score(Xc - hc, Xt - ht, (Xw - hw) if Xw is not None else None)
                            # shuffled: another row's (filled) atoms
                            shc, _ = shuffled_control(Xc, tc, J[L], W, s, args.seed + L)
                            sht, _ = shuffled_control(Xt, tt, J[L], W, s, args.seed + L + 1000)
                            arms[f"shuffled{k}"] = score(shc, sht, lodo_on=False, mv=False)
                            # fixed frame: most frequent atoms over Core rows
                            freq = Counter(int(t) for row in tc for t in row if t >= 0)
                            fixed = np.array([t for t, _ in freq.most_common(k)], dtype=np.int64)
                            Qf = sweep.atoms_basis(fixed, J[L], W, s)
                            arms[f"fixedframe{k}"] = score(sweep.project_fixed(Xc, Qf), sweep.project_fixed(Xt, Qf),
                                                           sweep.project_fixed(Xw, Qf) if Xw is not None else None)
                            arms[f"fixedframe{k}"]["atoms"] = [inv_vocab.get(int(t), "?") for t in fixed[:k]]
                            if k == 25:
                                # role direction for Experiment 3: mean filled J component, colluders minus honest (collusion episodes)
                                roles = np.array([cm[i]["role"] for i in ci])
                                modes = np.array([cm[i]["mode"] for i in ci])
                                col = (roles == "colluder") & (modes == "collusion")
                                hon = (roles == "honest") & (modes == "collusion")
                                if col.any() and hon.any():
                                    u = hc[col].mean(0) - hc[hon].mean(0)
                                    u = u / max(np.linalg.norm(u), 1e-8)
                                    rd = {"condition": cond, "position": pos, "layer": L, "k": k,
                                          "hj_norm_median": float(np.median(np.linalg.norm(hc, axis=1))),
                                          "file": f"role_dirs/role_dir_J25_{cond}_{pos}_L{L}.npy"}
                                    np.save(args.out / rd["file"], u.astype(np.float32))
                                    results["role_dirs"].append(rd)
                                if pos in ("last", "prevote", "frac50", "prompt_end"):
                                    results["exp1"].append(exp1_readouts(cond, pos, L, cm, ci, tm, ti, tc, tt, dc, dt, gids,
                                                                         inv_vocab, warned=(wm, wi, tw) if Xw is not None else None))
                    # the J-space component as Gurnee et al. operationalise it: the non-negative combination of the (at most k)
                    # pursuit atoms; the remainder is their non-J-space component
                    rc, uc = sweep.j_recon(Xc, dc, J[L], W, s, k)
                    rt, _ = sweep.j_recon(Xt, dt, J[L], W, s, k)
                    rw = sweep.j_recon(Xw, dw, J[L], W, s, k)[0] if Xw is not None else None
                    arms[f"J{k}_recon"] = score(rc, rt, rw)
                    arms[f"J{k}_recon"]["fve_core"] = float(np.mean((rc ** 2).sum(1) / Fc_mean))
                    arms[f"J{k}_recon"]["atoms_used_core_mean"] = float(uc.mean())
                    arms[f"C{k}_recon"] = score(Xc - rc, Xt - rt, (Xw - rw) if Xw is not None else None)
                    # PCA control
                    Xm = Xc - Xc.mean(0)
                    _, _, Vt = np.linalg.svd(Xm, full_matrices=False)
                    Qp = Vt[:k].T
                    arms[f"pca{k}"] = score(sweep.project_fixed(Xc, Qp), sweep.project_fixed(Xt, Qp),
                                            sweep.project_fixed(Xw, Qp) if Xw is not None else None)
                    # many-draw random controls (Core -> Transfer; LODO for the first few draws; matched-vocab for the first few)
                    rs_c2t, rs_lodo, ra_c2t, rs_mv = [], [], [], []
                    V = W.shape[0]
                    reuse = (prev_random.get(L) or {}).get(f"k{k}")
                    if reuse is not None:  # the random arms do not depend on the decomposition
                        rs_c2t = list(reuse["randsub"]["c2t_draws"])
                        rs_lodo = list(reuse["randsub"].get("lodo_draws", []))
                        if args.redraw_random_atoms:
                            for b in range(args.n_random):
                                toks = rng.choice(V, size=k, replace=False)
                                Qa = sweep.atoms_basis(toks, J[L], W, s)
                                r2 = score(sweep.project_fixed(Xc, Qa), sweep.project_fixed(Xt, Qa), lodo_on=False,
                                           boot=args.boot_random, mv=False)
                                ra_c2t.append(r2["core_to_transfer"])
                        else:
                            ra_c2t = list(reuse["randatoms"]["c2t_draws"])
                    for b in range(0 if reuse is not None else args.n_random):
                        Q = sweep.haar_subspace(d, k, rng)
                        do_mv = Xw is not None and b < args.n_random_mv
                        r = score(sweep.project_fixed(Xc, Q), sweep.project_fixed(Xt, Q),
                                  sweep.project_fixed(Xw, Q) if do_mv else None,
                                  lodo_on=lodo and b < args.n_random_lodo, boot=args.boot_random, mv=do_mv)
                        rs_c2t.append(r["core_to_transfer"])
                        if "core_lodo" in r:
                            rs_lodo.append(r["core_lodo"])
                        if "matched_vocab" in r:
                            rs_mv.append(r["matched_vocab"])
                        toks = rng.choice(V, size=k, replace=False)
                        Qa = sweep.atoms_basis(toks, J[L], W, s)
                        r2 = score(sweep.project_fixed(Xc, Qa), sweep.project_fixed(Xt, Qa), lodo_on=False, boot=args.boot_random, mv=False)
                        ra_c2t.append(r2["core_to_transfer"])
                    out_k = {a: clean(r) for a, r in arms.items()}
                    jf = arms[f"J{k}_fill"]["core_to_transfer"]
                    jp = arms[f"J{k}"]["core_to_transfer"]
                    out_k["randsub"] = {"n": len(rs_c2t), "c2t_mean": float(np.mean(rs_c2t)),
                                        "c2t_p95": float(np.percentile(rs_c2t, 95)),
                                        "c2t_draws": [float(x) for x in rs_c2t],
                                        "lodo_draws": [float(x) for x in rs_lodo],
                                        "J_fill_percentile": stats.empirical_percentile(jf, rs_c2t),
                                        "J_pursuit_percentile": stats.empirical_percentile(jp, rs_c2t)}
                    if rs_mv:
                        out_k["randsub"]["matched_vocab"] = mv_summary(rs_mv)
                    if reuse is not None:
                        out_k["randsub"]["reused"] = True
                        if "matched_vocab" in reuse["randsub"]:
                            out_k["randsub"]["matched_vocab"] = reuse["randsub"]["matched_vocab"]
                    jr = arms[f"J{k}_recon"]["core_to_transfer"]
                    out_k["randsub"]["J_recon_percentile"] = stats.empirical_percentile(jr, rs_c2t)
                    out_k["randatoms"] = {"n": len(ra_c2t), "c2t_mean": float(np.mean(ra_c2t)),
                                          "c2t_p95": float(np.percentile(ra_c2t, 95)),
                                          "c2t_draws": [float(x) for x in ra_c2t],
                                          "J_fill_percentile": stats.empirical_percentile(jf, ra_c2t),
                                          "J_pursuit_percentile": stats.empirical_percentile(jp, ra_c2t),
                                          "J_recon_percentile": stats.empirical_percentile(jr, ra_c2t)}
                    fullc = full["core_to_transfer"]
                    out_k["recovery_J_fill"] = float((jf - 0.5) / max(fullc - 0.5, 1e-9))
                    out_k["recovery_J_pursuit"] = float((jp - 0.5) / max(fullc - 0.5, 1e-9))
                    # paired bootstrap differences on shared episodes
                    out_k["J_fill_minus_full_ci95"] = list(np.nanpercentile(arms[f"J{k}_fill"]["_boot"] - full["_boot"], [2.5, 97.5]).astype(float))
                    out_k["J_pursuit_minus_full_ci95"] = list(np.nanpercentile(arms[f"J{k}"]["_boot"] - full["_boot"], [2.5, 97.5]).astype(float))
                    out_k["J_fill_minus_pca_ci95"] = list(np.nanpercentile(arms[f"J{k}_fill"]["_boot"] - arms[f"pca{k}"]["_boot"], [2.5, 97.5]).astype(float))
                    out_k["C_fill_minus_J_fill_ci95"] = list(np.nanpercentile(arms[f"C{k}_fill"]["_boot"] - arms[f"J{k}_fill"]["_boot"], [2.5, 97.5]).astype(float))
                    rb = arms[f"J{k}_recon"]["_boot"]
                    out_k["recovery_J_recon"] = float((jr - 0.5) / max(fullc - 0.5, 1e-9))
                    out_k["J_recon_minus_full_ci95"] = list(np.nanpercentile(rb - full["_boot"], [2.5, 97.5]).astype(float))
                    out_k["J_recon_minus_pca_ci95"] = list(np.nanpercentile(rb - arms[f"pca{k}"]["_boot"], [2.5, 97.5]).astype(float))
                    out_k["J_recon_minus_J_fill_ci95"] = list(np.nanpercentile(rb - arms[f"J{k}_fill"]["_boot"], [2.5, 97.5]).astype(float))
                    out_k["C_recon_minus_J_recon_ci95"] = list(np.nanpercentile(arms[f"C{k}_recon"]["_boot"] - rb, [2.5, 97.5]).astype(float))
                    out_k["seconds"] = round(time.time() - t_k, 1)
                    cell[f"k{k}"] = out_k
                    mvs = arms[f"J{k}_fill"].get("matched_vocab", {})
                    log(cond, pos, L, f"k={k}", "[" + args.method + "]", "J_recon", round(jr, 3), "J_fill", round(jf, 3), "J", round(jp, 3),
                        "randsub p95", round(out_k["randsub"]["c2t_p95"], 3), "pca", round(arms[f"pca{k}"]["core_to_transfer"], 3),
                        "full", round(fullc, 3),
                        "| mv col-vs-warned", round(mvs["auroc_colluder_vs_warned"], 3) if mvs.get("auroc_colluder_vs_warned") is not None else "-",
                        "warned-vs-unwarned", round(mvs["auroc_warned_vs_unwarned"], 3) if mvs.get("auroc_warned_vs_unwarned") is not None else "-",
                        f"({out_k['seconds']}s)")
                cell["seconds"] = round(time.time() - t_cell, 1)
                results["cells"].append(cell)
                (args.out / "sweep_results.json").write_text(json.dumps(results, indent=1, default=float) + "\n")
    log("wrote", args.out / "sweep_results.json")


def exp1_readouts(cond, pos, L, cm, ci, tm, ti, tc, tt, dc, dt, gids, inv_vocab, warned=None):
    """Experiment 1 at k=25 (filled atoms) and the lens readout (top 50): concept-group shares per agent,
    colluder vs honest (within-episode permutation) and colluder vs control agents (by scenario).
    With ``warned`` = (meta, idx, atoms) the matched-vocabulary concept readings are added for Core."""
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
                                     "colluder_minus_control": obs2, "p_strat": p2,
                                     "mean_colluder": float(np.mean([a["x"][gi] for a in agents if a["role"] == "colluder"])) if any(a["role"] == "colluder" for a in agents) else None,
                                     "mean_honest_in_collusion": float(np.mean(v)) if v else None,
                                     "mean_control": float(np.mean([a["x"][gi] for a in agents if a["mode"] == "control"])) if any(a["mode"] == "control" for a in agents) else None}
            if src == "atoms" and split == "core" and warned is not None:
                wmeta, widx, wtoks = warned
                gw = sweep.group_mass(wtoks, None, gids)
                Gw = np.stack([gw[g] for g in gids], axis=1)
                wagents = to_agents(wmeta, widx, Gw, "warned")
                res["matched_vocab_concepts"] = matched_vocab.concept_mass_readings(agents, wagents, list(gids))
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
                if warned is not None and split == "core":
                    wmeta, widx, wtoks = warned
                    wsets = {}
                    for j, i in enumerate(widx):
                        m = wmeta[i]
                        wsets.setdefault((m["run_id"], m["agent_name"], bool(m.get("warned"))), set()).update(int(t) for t in wtoks[j] if t >= 0)
                    cw = Counter(t for k, sset in wsets.items() if k[2] for t in sset)
                    res["most_common_atoms_warned"] = [(inv_vocab.get(t, "?"), c) for t, c in cw.most_common(20)]
        pv = [v[k] for v in res.values() if isinstance(v, dict) and "p_within" in v for k in ("p_within", "p_strat")]
        adj, _ = stats.benjamini_hochberg(pv) if pv else ([], [])
        res["bh_adjusted_p_in_order"] = [float(x) for x in adj]
        out[split] = res
    return out


if __name__ == "__main__":
    main()
