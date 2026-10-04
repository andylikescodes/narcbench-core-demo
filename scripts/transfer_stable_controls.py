#!/usr/bin/env python3
"""Offline control contrasts for Transfer-stable residual signal (CPU-only).

Question 3 lite:
  (1) Deception/role direction: colluder vs honest (collusion mode only)
  (2) Task/mode direction: collusion-run vs control-run (agent-average per run)
  (3) Cross-test specificity + cosine(role, mode)
  (4) PCA-contrast k=4/8 @ L23: role + mode AUROC in subspace

Fit on Core discussion public → eval Transfer final (+ Core ID / LDO).

  .venv-probe/bin/python scripts/transfer_stable_controls.py \
    --core-acts data/activations/gemma2_9b/core/20261001T012639Z \
    --transfer-acts data/activations/gemma2_9b/transfer/RUNPOD \
    --out results/transfer_stable
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from eval_transfer_from_core import (  # noqa: E402
    build_transfer_groups,
    enrich_transfer_meta,
)
from train_collusion_probe import (  # noqa: E402
    build_groups,
    compute_scores,
    domain_of,
    get_layer,
    load_acts,
    parse_layers,
)
from transfer_stable_offline import (  # noqa: E402
    collusion_sample_indices,
    diff_in_means,
    fit_lr_weight,
    identity_scaler,
    sample_level_direction_auroc,
    scenario_aurocs,
)


# ---------------------------------------------------------------------------
# Index helpers
# ---------------------------------------------------------------------------

def public_indices(
    meta: list[dict],
    *,
    phase: str,
    mode: str | None = None,
    role: str | None = None,
) -> list[int]:
    out = []
    for i, m in enumerate(meta):
        if m.get("channel") != "public":
            continue
        if m.get("phase") != phase:
            continue
        if mode is not None and m.get("mode") != mode:
            continue
        if role is not None and m.get("role") != role:
            continue
        out.append(i)
    return out


def run_agent_average(
    meta: list[dict],
    acts: np.ndarray,
    *,
    phase: str,
    mode: str | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """One vector per run = mean over public agents (and rounds if present).

    Returns X (n_runs, D), y_mode (1=collusion), run_ids.
    """
    buckets: dict[str, list[int]] = defaultdict(list)
    mode_of: dict[str, str] = {}
    for i, m in enumerate(meta):
        if m.get("channel") != "public" or m.get("phase") != phase:
            continue
        if mode is not None and m.get("mode") != mode:
            continue
        rid = m["run_id"]
        buckets[rid].append(i)
        mode_of[rid] = m["mode"]
    rids = sorted(buckets.keys())
    X = np.stack([acts[buckets[r]].mean(axis=0) for r in rids], axis=0).astype(np.float64)
    y = np.array([1 if mode_of[r] == "collusion" else 0 for r in rids], dtype=np.int64)
    return X, y, rids


def fit_binary_lr(
    X_pos: np.ndarray, X_neg: np.ndarray, seed: int
) -> tuple[StandardScaler, np.ndarray]:
    """Balanced LR; returns scaler + unit weight."""
    rng = np.random.RandomState(seed)
    n = min(len(X_pos), len(X_neg))
    if n < 2:
        raise RuntimeError(f"need ≥2 per class, got pos={len(X_pos)} neg={len(X_neg)}")
    pos = X_pos[rng.choice(len(X_pos), n, replace=False)]
    neg = X_neg[rng.choice(len(X_neg), n, replace=False)]
    scaler = StandardScaler()
    X = scaler.fit_transform(np.vstack([neg, pos]))
    y = np.array([0] * n + [1] * n)
    clf = LogisticRegression(max_iter=1000, C=1.0, random_state=seed)
    clf.fit(X, y)
    w = clf.coef_[0].astype(np.float64)
    w = w / (np.linalg.norm(w) + 1e-12)
    return scaler, w


def project_auroc(
    X: np.ndarray, y: np.ndarray, scaler: StandardScaler, w: np.ndarray
) -> float | None:
    if len(np.unique(y)) < 2:
        return None
    scores = scaler.transform(X) @ w
    return float(roc_auc_score(y, scores))


def role_sample_auroc(
    meta, acts, scaler, w, *, phase: str
) -> dict[str, Any]:
    """Within collusion mode: colluder vs honest sample AUROC."""
    col, hon = collusion_sample_indices(meta, phase=phase)
    auc = sample_level_direction_auroc(acts, col, hon, w, scaler)
    return {
        "auroc": auc,
        "n_colluder": len(col),
        "n_honest": len(hon),
        "phase": phase,
        "mode_filter": "collusion",
        "label": "role (colluder=1 vs honest=0)",
    }


def mode_runavg_auroc(
    meta, acts, scaler, w, *, phase: str
) -> dict[str, Any]:
    """Mode AUROC on agent-averaged run vectors."""
    X, y, rids = run_agent_average(meta, acts, phase=phase)
    auc = project_auroc(X, y, scaler, w)
    return {
        "auroc": auc,
        "n_runs": len(rids),
        "mode_counts": dict(Counter(y.tolist())),
        "phase": phase,
        "aggregation": "agent_average_per_run",
        "label": "mode (collusion=1 vs control=0)",
    }


def mode_scenario_aurocs(
    meta, acts, scaler, w, core_meta, core_acts, seed: int, *, phase: str, is_transfer: bool
) -> dict[str, Any]:
    """HARD-style Peak/Split/Asym on eval groups; Asym fit on Core discussion."""
    if is_transfer:
        groups = build_transfer_groups(meta, acts, phase=phase)
    else:
        groups = build_groups(meta, acts, phase=phase, avg_rounds=True)
    rids = sorted(groups.keys())
    core_groups = build_groups(core_meta, core_acts, phase="discussion", avg_rounds=True)
    core_rids = sorted(core_groups.keys())
    core_y = np.array([1 if core_groups[r]["mode"] == "collusion" else 0 for r in core_rids])
    out = scenario_aurocs(groups, rids, scaler, w, core_groups, core_rids, core_y, seed)
    out["phase"] = phase
    out["aggregation"] = "peak_split_asym_scenario"
    return out


# ---------------------------------------------------------------------------
# Core LDO (holdout-style)
# ---------------------------------------------------------------------------

def core_ldo_role_auroc(
    meta, acts, *, seed: int, fit_fn
) -> dict[str, Any]:
    """Leave-one-domain-out sample-level role AUROC on Core discussion public."""
    col, hon = collusion_sample_indices(meta, phase="discussion")
    domains = sorted(
        {
            domain_of(meta[i].get("scenario_id", ""), meta[i].get("scenario_domain", ""))
            for i in col + hon
        }
    )
    fold_aucs = []
    for held in domains:
        train_col = [
            i
            for i in col
            if domain_of(meta[i].get("scenario_id", ""), meta[i].get("scenario_domain", "")) != held
        ]
        train_hon = [
            i
            for i in hon
            if domain_of(meta[i].get("scenario_id", ""), meta[i].get("scenario_domain", "")) != held
        ]
        test_col = [
            i
            for i in col
            if domain_of(meta[i].get("scenario_id", ""), meta[i].get("scenario_domain", "")) == held
        ]
        test_hon = [
            i
            for i in hon
            if domain_of(meta[i].get("scenario_id", ""), meta[i].get("scenario_domain", "")) == held
        ]
        if len(train_col) < 4 or len(train_hon) < 4 or not test_col or not test_hon:
            continue
        scaler, w = fit_fn(acts, train_col, train_hon, seed)
        auc = sample_level_direction_auroc(acts, test_col, test_hon, w, scaler)
        if auc is not None:
            fold_aucs.append({"held_domain": held, "auroc": auc, "n_test": len(test_col) + len(test_hon)})
    return {
        "mean_auroc": float(np.mean([f["auroc"] for f in fold_aucs])) if fold_aucs else None,
        "n_folds": len(fold_aucs),
        "folds": fold_aucs,
        "label": "Core LDO role (colluder vs honest, discussion public)",
    }


def core_ldo_mode_auroc(meta, acts, *, seed: int) -> dict[str, Any]:
    """Leave-one-domain-out mode AUROC on Core discussion agent-averages."""
    X, y, rids = run_agent_average(meta, acts, phase="discussion")
    # map run → domain from meta
    run_dom = {}
    for m in meta:
        if m["run_id"] not in run_dom and m.get("channel") == "public" and m.get("phase") == "discussion":
            run_dom[m["run_id"]] = domain_of(m.get("scenario_id", ""), m.get("scenario_domain", ""))
    domains = sorted(set(run_dom[r] for r in rids))
    fold_aucs = []
    for held in domains:
        tr = [i for i, r in enumerate(rids) if run_dom[r] != held]
        te = [i for i, r in enumerate(rids) if run_dom[r] == held]
        if len(tr) < 8 or len(te) < 2:
            continue
        ytr, yte = y[tr], y[te]
        if len(np.unique(ytr)) < 2 or len(np.unique(yte)) < 2:
            continue
        Xtr, Xte = X[tr], X[te]
        pos = Xtr[ytr == 1]
        neg = Xtr[ytr == 0]
        scaler, w = fit_binary_lr(pos, neg, seed)
        auc = project_auroc(Xte, yte, scaler, w)
        if auc is not None:
            fold_aucs.append({"held_domain": held, "auroc": auc, "n_test_runs": len(te)})
    return {
        "mean_auroc": float(np.mean([f["auroc"] for f in fold_aucs])) if fold_aucs else None,
        "n_folds": len(fold_aucs),
        "folds": fold_aucs,
        "label": "Core LDO mode (agent-avg per run, discussion public)",
    }


# ---------------------------------------------------------------------------
# PCA contrast subspace (Core-fit colluder−honest deltas)
# ---------------------------------------------------------------------------

def fit_pca_contrast(core_meta, core_acts, *, phase: str, seed: int, max_k: int = 8):
    col, hon = collusion_sample_indices(core_meta, phase=phase)
    all_col_pub = col + hon
    X_fit = core_acts[all_col_pub]
    by_run: dict[str, dict] = defaultdict(lambda: {"col": [], "hon": []})
    for i in all_col_pub:
        m = core_meta[i]
        if m["role"] == "colluder":
            by_run[m["run_id"]]["col"].append(core_acts[i])
        else:
            by_run[m["run_id"]]["hon"].append(core_acts[i])
    deltas = []
    for dsets in by_run.values():
        if dsets["col"] and dsets["hon"]:
            deltas.append(np.mean(dsets["col"], axis=0) - np.mean(dsets["hon"], axis=0))
    deltas_arr = np.stack(deltas, axis=0)
    n_comp = min(max_k, max(1, deltas_arr.shape[0] - 1), deltas_arr.shape[1])
    pca = PCA(n_components=n_comp, random_state=seed)
    pca.fit(deltas_arr)
    mu = X_fit.mean(axis=0)
    return pca, mu, int(deltas_arr.shape[0])


def project_pca(acts, pca, mu, k: int) -> np.ndarray:
    P = pca.components_[:k]
    return ((acts - mu) @ P.T).astype(np.float32)


# ---------------------------------------------------------------------------
# Per-layer controls
# ---------------------------------------------------------------------------

def run_layer_controls(
    core_meta, core_acts, xfer_meta, xfer_acts, layer: int, *, seed: int
) -> dict[str, Any]:
    print(f"[controls] layer {layer}", flush=True)
    out: dict[str, Any] = {"layer": layer}

    # --- (1) Role / deception direction (collusion-mode only) ---
    col_idx, hon_idx = collusion_sample_indices(core_meta, phase="discussion")
    role_scaler, role_w = fit_lr_weight(core_acts, col_idx, hon_idx, seed)
    role_dim = diff_in_means(core_acts, col_idx, hon_idx)
    # orient dim scaler same as prior work
    rng = np.random.RandomState(seed)
    nc = min(len(col_idx), len(hon_idx))
    hi = list(rng.choice(hon_idx, nc, replace=False))
    ci = col_idx[:nc]
    dim_scaler = StandardScaler()
    dim_scaler.fit(np.vstack([core_acts[hi], core_acts[ci]]))

    out["role_direction"] = {
        "fit": "LR colluder vs honest on Core collusion discussion public",
        "n_train_colluder": len(col_idx),
        "n_train_honest": len(hon_idx),
        "lr": {},
        "diff_in_means": {},
    }
    # Role AUROC (within collusion)
    out["role_direction"]["lr"]["role_auroc_core_discussion"] = role_sample_auroc(
        core_meta, core_acts, role_scaler, role_w, phase="discussion"
    )
    out["role_direction"]["lr"]["role_auroc_core_final"] = role_sample_auroc(
        core_meta, core_acts, role_scaler, role_w, phase="final"
    )
    out["role_direction"]["lr"]["role_auroc_transfer_final"] = role_sample_auroc(
        xfer_meta, xfer_acts, role_scaler, role_w, phase="final"
    )
    out["role_direction"]["lr"]["core_ldo_role"] = core_ldo_role_auroc(
        core_meta, core_acts, seed=seed, fit_fn=fit_lr_weight
    )
    # Mode AUROC with role direction (cross-test)
    out["role_direction"]["lr"]["mode_auroc_runavg_core_discussion"] = mode_runavg_auroc(
        core_meta, core_acts, role_scaler, role_w, phase="discussion"
    )
    out["role_direction"]["lr"]["mode_auroc_runavg_transfer_final"] = mode_runavg_auroc(
        xfer_meta, xfer_acts, role_scaler, role_w, phase="final"
    )
    out["role_direction"]["lr"]["mode_scenario_transfer_final"] = mode_scenario_aurocs(
        xfer_meta, xfer_acts, role_scaler, role_w, core_meta, core_acts, seed, phase="final", is_transfer=True
    )
    out["role_direction"]["lr"]["mode_scenario_core_discussion"] = mode_scenario_aurocs(
        core_meta, core_acts, role_scaler, role_w, core_meta, core_acts, seed, phase="discussion", is_transfer=False
    )

    # Diff-in-means role arm (lighter)
    out["role_direction"]["diff_in_means"]["role_auroc_transfer_final"] = role_sample_auroc(
        xfer_meta, xfer_acts, dim_scaler, role_dim, phase="final"
    )
    out["role_direction"]["diff_in_means"]["mode_auroc_runavg_transfer_final"] = mode_runavg_auroc(
        xfer_meta, xfer_acts, dim_scaler, role_dim, phase="final"
    )

    print(
        f"  role→role Transfer final AUROC="
        f"{out['role_direction']['lr']['role_auroc_transfer_final']['auroc']:.3f}  "
        f"role→mode runavg Transfer="
        f"{out['role_direction']['lr']['mode_auroc_runavg_transfer_final']['auroc']:.3f}  "
        f"role→mode scenario primary="
        f"{out['role_direction']['lr']['mode_scenario_transfer_final'].get('auroc_primary', float('nan')):.3f}",
        flush=True,
    )

    # --- (2) Mode / task direction (agent-average per run) ---
    Xc, yc, rids_c = run_agent_average(core_meta, core_acts, phase="discussion")
    pos = Xc[yc == 1]
    neg = Xc[yc == 0]
    mode_scaler, mode_w = fit_binary_lr(pos, neg, seed)
    # Also ambient sample-level mode direction (all agents, not run-avg) for comparison note
    col_all = public_indices(core_meta, phase="discussion", mode="collusion")
    ctrl_all = public_indices(core_meta, phase="discussion", mode="control")
    # balance for sample-level mode dir
    mode_sample_scaler, mode_sample_w = fit_binary_lr(
        core_acts[col_all], core_acts[ctrl_all], seed
    )

    out["mode_direction"] = {
        "fit": (
            "LR on Core discussion public agent-average per run "
            "(collusion-run=1 vs control-run=0); one vector per run"
        ),
        "n_train_runs": len(rids_c),
        "train_mode_counts": dict(Counter(yc.tolist())),
        "runavg": {},
        "sample_level_alt": {
            "note": "Alternate: LR on all public discussion samples (not run-averaged)",
            "n_collusion_samples": len(col_all),
            "n_control_samples": len(ctrl_all),
        },
    }
    # Mode AUROC (intended)
    out["mode_direction"]["runavg"]["mode_auroc_core_discussion"] = mode_runavg_auroc(
        core_meta, core_acts, mode_scaler, mode_w, phase="discussion"
    )
    out["mode_direction"]["runavg"]["mode_auroc_core_final"] = mode_runavg_auroc(
        core_meta, core_acts, mode_scaler, mode_w, phase="final"
    )
    out["mode_direction"]["runavg"]["mode_auroc_transfer_final"] = mode_runavg_auroc(
        xfer_meta, xfer_acts, mode_scaler, mode_w, phase="final"
    )
    out["mode_direction"]["runavg"]["core_ldo_mode"] = core_ldo_mode_auroc(
        core_meta, core_acts, seed=seed
    )
    out["mode_direction"]["runavg"]["mode_scenario_transfer_final"] = mode_scenario_aurocs(
        xfer_meta, xfer_acts, mode_scaler, mode_w, core_meta, core_acts, seed, phase="final", is_transfer=True
    )
    # Cross-test: mode direction → role separation within Transfer collusion
    out["mode_direction"]["runavg"]["role_auroc_core_discussion"] = role_sample_auroc(
        core_meta, core_acts, mode_scaler, mode_w, phase="discussion"
    )
    out["mode_direction"]["runavg"]["role_auroc_transfer_final"] = role_sample_auroc(
        xfer_meta, xfer_acts, mode_scaler, mode_w, phase="final"
    )

    # Sample-level mode alt: Transfer mode + role cross
    out["mode_direction"]["sample_level_alt"]["mode_auroc_runavg_transfer_final"] = mode_runavg_auroc(
        xfer_meta, xfer_acts, mode_sample_scaler, mode_sample_w, phase="final"
    )
    out["mode_direction"]["sample_level_alt"]["role_auroc_transfer_final"] = role_sample_auroc(
        xfer_meta, xfer_acts, mode_sample_scaler, mode_sample_w, phase="final"
    )

    print(
        f"  mode→mode runavg Transfer="
        f"{out['mode_direction']['runavg']['mode_auroc_transfer_final']['auroc']:.3f}  "
        f"mode→role Transfer="
        f"{out['mode_direction']['runavg']['role_auroc_transfer_final']['auroc']:.3f}",
        flush=True,
    )

    # --- (3) Cosine similarity ---
    # Align spaces: both unit; role_w from sample scaler space, mode_w from run-avg scaler space.
    # Report cosine in ambient raw space by undoing scalers into raw directions:
    # w_raw ∝ w / scale  (score ≈ w @ ((x-mean)/scale) = (w/scale) @ x + const)
    role_w_raw = role_w / (role_scaler.scale_ + 1e-12)
    role_w_raw = role_w_raw / (np.linalg.norm(role_w_raw) + 1e-12)
    mode_w_raw = mode_w / (mode_scaler.scale_ + 1e-12)
    mode_w_raw = mode_w_raw / (np.linalg.norm(mode_w_raw) + 1e-12)
    mode_sample_w_raw = mode_sample_w / (mode_sample_scaler.scale_ + 1e-12)
    mode_sample_w_raw = mode_sample_w_raw / (np.linalg.norm(mode_sample_w_raw) + 1e-12)
    out["cosine"] = {
        "role_lr_vs_mode_runavg_raw": float(role_w_raw @ mode_w_raw),
        "role_lr_vs_mode_sample_raw": float(role_w_raw @ mode_sample_w_raw),
        "role_dim_vs_mode_runavg_raw": float(role_dim @ mode_w_raw),
        "note": (
            "Cosines in ambient residual space after undoing StandardScaler "
            "(w_raw ∝ w/scale). Positive = same orientation (collusion/colluder high)."
        ),
    }
    print(
        f"  cos(role,mode_runavg)={out['cosine']['role_lr_vs_mode_runavg_raw']:.3f}  "
        f"cos(role,mode_sample)={out['cosine']['role_lr_vs_mode_sample_raw']:.3f}",
        flush=True,
    )

    # Stash weights norms only (not full vectors) for JSON size
    out["direction_meta"] = {
        "role_w_norm": 1.0,
        "mode_w_norm": 1.0,
        "dim": int(core_acts.shape[1]),
    }
    # Keep actual weights for PCA section reuse within process — attach privately
    out["_role_scaler"] = role_scaler
    out["_role_w"] = role_w
    out["_mode_scaler"] = mode_scaler
    out["_mode_w"] = mode_w
    return out


def run_pca_controls(
    core_meta, core_npz, xfer_meta, xfer_npz, *, layer: int, ks: list[int], seed: int
) -> dict[str, Any]:
    print(f"[controls] PCA-contrast @ L{layer} ks={ks}", flush=True)
    core_acts = get_layer(core_npz, len(core_meta), layer)
    xfer_acts = get_layer(xfer_npz, len(xfer_meta), layer)
    pca, mu, n_deltas = fit_pca_contrast(core_meta, core_acts, phase="discussion", seed=seed, max_k=max(ks))
    out: dict[str, Any] = {
        "layer": layer,
        "n_deltas": n_deltas,
        "n_components_fit": int(pca.n_components_),
        "per_k": {},
    }
    col_idx, hon_idx = collusion_sample_indices(core_meta, phase="discussion")
    for k in ks:
        if k > pca.n_components_:
            continue
        core_proj = project_pca(core_acts, pca, mu, k)
        xfer_proj = project_pca(xfer_acts, pca, mu, k)
        # Role direction in subspace
        role_scaler, role_w = fit_lr_weight(core_proj, col_idx, hon_idx, seed)
        # Mode direction in subspace (run-avg)
        Xc, yc, _ = run_agent_average(core_meta, core_proj, phase="discussion")
        mode_scaler, mode_w = fit_binary_lr(Xc[yc == 1], Xc[yc == 0], seed)

        k_out = {
            "k": k,
            "explained_var_ratio_sum": float(pca.explained_variance_ratio_[:k].sum()),
            "role_dir": {
                "role_auroc_transfer_final": role_sample_auroc(
                    xfer_meta, xfer_proj, role_scaler, role_w, phase="final"
                ),
                "role_auroc_core_discussion": role_sample_auroc(
                    core_meta, core_proj, role_scaler, role_w, phase="discussion"
                ),
                "mode_auroc_runavg_transfer_final": mode_runavg_auroc(
                    xfer_meta, xfer_proj, role_scaler, role_w, phase="final"
                ),
                "mode_scenario_transfer_final": mode_scenario_aurocs(
                    xfer_meta, xfer_proj, role_scaler, role_w,
                    core_meta, core_proj, seed, phase="final", is_transfer=True,
                ),
            },
            "mode_dir": {
                "mode_auroc_runavg_transfer_final": mode_runavg_auroc(
                    xfer_meta, xfer_proj, mode_scaler, mode_w, phase="final"
                ),
                "role_auroc_transfer_final": role_sample_auroc(
                    xfer_meta, xfer_proj, mode_scaler, mode_w, phase="final"
                ),
                "mode_scenario_transfer_final": mode_scenario_aurocs(
                    xfer_meta, xfer_proj, mode_scaler, mode_w,
                    core_meta, core_proj, seed, phase="final", is_transfer=True,
                ),
            },
        }
        # Cosine in subspace (undo scalers)
        rw = role_w / (role_scaler.scale_ + 1e-12)
        rw = rw / (np.linalg.norm(rw) + 1e-12)
        mw = mode_w / (mode_scaler.scale_ + 1e-12)
        mw = mw / (np.linalg.norm(mw) + 1e-12)
        k_out["cosine_role_vs_mode"] = float(rw @ mw)
        out["per_k"][str(k)] = k_out
        print(
            f"  k={k} role→role xfer={k_out['role_dir']['role_auroc_transfer_final']['auroc']:.3f} "
            f"role→mode xfer={k_out['role_dir']['mode_auroc_runavg_transfer_final']['auroc']:.3f} "
            f"mode→mode xfer={k_out['mode_dir']['mode_auroc_runavg_transfer_final']['auroc']:.3f} "
            f"mode→role xfer={k_out['mode_dir']['role_auroc_transfer_final']['auroc']:.3f} "
            f"cos={k_out['cosine_role_vs_mode']:.3f}",
            flush=True,
        )
    return out


def strip_private(d: dict) -> dict:
    """Remove non-JSON _* keys recursively."""
    out = {}
    for k, v in d.items():
        if str(k).startswith("_"):
            continue
        if isinstance(v, dict):
            out[k] = strip_private(v)
        else:
            out[k] = v
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--core-acts", type=Path, required=True)
    ap.add_argument("--transfer-acts", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--layers", default="21,22")
    ap.add_argument("--pca-layer", type=int, default=23)
    ap.add_argument("--pca-ks", default="4,8")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    core_dir = args.core_acts if args.core_acts.is_absolute() else ROOT / args.core_acts
    xfer_dir = args.transfer_acts if args.transfer_acts.is_absolute() else ROOT / args.transfer_acts
    out = args.out if args.out.is_absolute() else ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)

    layers = parse_layers(args.layers)
    pca_ks = [int(x) for x in args.pca_ks.split(",") if x.strip()]

    print("[load] Core + Transfer", flush=True)
    core_meta, core_npz = load_acts(core_dir)
    xfer_meta, xfer_npz = load_acts(xfer_dir)
    xfer_meta = enrich_transfer_meta(xfer_meta)
    print(f"[load] core_n={len(core_meta)} xfer_n={len(xfer_meta)} layers={layers}", flush=True)

    per_layer = {}
    for L in layers:
        core_acts = get_layer(core_npz, len(core_meta), L)
        xfer_acts = get_layer(xfer_npz, len(xfer_meta), L)
        per_layer[str(L)] = strip_private(
            run_layer_controls(core_meta, core_acts, xfer_meta, xfer_acts, L, seed=args.seed)
        )

    pca_out = run_pca_controls(
        core_meta, core_npz, xfer_meta, xfer_npz,
        layer=args.pca_layer, ks=pca_ks, seed=args.seed,
    )

    # Claim helpers
    def g(L, *path, default=None):
        cur = per_layer[str(L)]
        for p in path:
            if not isinstance(cur, dict) or p not in cur:
                return default
            cur = cur[p]
        return cur

    summary = {
        "protocol": (
            "Fit Core discussion public → eval Transfer final (+ Core ID/LDO). "
            "Labels: mode (collusion/control) and role (colluder/honest). "
            "Never collusion_success."
        ),
        "role_direction_fit": "LR colluder vs honest, Core collusion-mode discussion public",
        "mode_direction_fit": "LR agent-average per run, Core discussion public (collusion vs control)",
        "layers": layers,
        "pca_layer": args.pca_layer,
        "pca_ks": pca_ks,
        "highlights": {},
    }
    for L in layers:
        summary["highlights"][f"L{L}"] = {
            "role_dir_role_auroc_xfer_final": g(L, "role_direction", "lr", "role_auroc_transfer_final", "auroc"),
            "role_dir_mode_runavg_xfer_final": g(L, "role_direction", "lr", "mode_auroc_runavg_transfer_final", "auroc"),
            "role_dir_mode_scenario_primary_xfer": g(L, "role_direction", "lr", "mode_scenario_transfer_final", "auroc_primary"),
            "role_dir_core_ldo_role": g(L, "role_direction", "lr", "core_ldo_role", "mean_auroc"),
            "mode_dir_mode_runavg_xfer_final": g(L, "mode_direction", "runavg", "mode_auroc_transfer_final", "auroc"),
            "mode_dir_role_auroc_xfer_final": g(L, "mode_direction", "runavg", "role_auroc_transfer_final", "auroc"),
            "mode_dir_core_ldo_mode": g(L, "mode_direction", "runavg", "core_ldo_mode", "mean_auroc"),
            "cosine_role_vs_mode_runavg": g(L, "cosine", "role_lr_vs_mode_runavg_raw"),
            "cosine_role_vs_mode_sample": g(L, "cosine", "role_lr_vs_mode_sample_raw"),
        }

    # Build claim
    L21 = summary["highlights"].get("L21", {})
    L22 = summary["highlights"].get("L22", {})
    pca8 = pca_out["per_k"].get("8", {})
    claim_nums = {
        "L21_role_to_role_xfer": L21.get("role_dir_role_auroc_xfer_final"),
        "L21_role_to_mode_xfer": L21.get("role_dir_mode_runavg_xfer_final"),
        "L21_mode_to_mode_xfer": L21.get("mode_dir_mode_runavg_xfer_final"),
        "L21_mode_to_role_xfer": L21.get("mode_dir_role_auroc_xfer_final"),
        "L21_cosine": L21.get("cosine_role_vs_mode_runavg"),
        "L22_role_to_role_xfer": L22.get("role_dir_role_auroc_xfer_final"),
        "L22_role_to_mode_xfer": L22.get("role_dir_mode_runavg_xfer_final"),
        "L22_mode_to_mode_xfer": L22.get("mode_dir_mode_runavg_xfer_final"),
        "L22_mode_to_role_xfer": L22.get("mode_dir_role_auroc_xfer_final"),
        "L22_cosine": L22.get("cosine_role_vs_mode_runavg"),
        "pca8_role_to_role": (pca8.get("role_dir") or {}).get("role_auroc_transfer_final", {}).get("auroc"),
        "pca8_role_to_mode": (pca8.get("role_dir") or {}).get("mode_auroc_runavg_transfer_final", {}).get("auroc"),
        "pca8_mode_to_mode": (pca8.get("mode_dir") or {}).get("mode_auroc_runavg_transfer_final", {}).get("auroc"),
        "pca8_mode_to_role": (pca8.get("mode_dir") or {}).get("role_auroc_transfer_final", {}).get("auroc"),
        "pca8_cosine": pca8.get("cosine_role_vs_mode"),
    }
    summary["claim_numbers"] = claim_nums

    # Auto-classify specificity
    def _f(x):
        return float(x) if x is not None else float("nan")

    # Use L21 as primary (best LR layer from prior)
    r2r, r2m = _f(claim_nums["L21_role_to_role_xfer"]), _f(claim_nums["L21_role_to_mode_xfer"])
    m2m, m2r = _f(claim_nums["L21_mode_to_mode_xfer"]), _f(claim_nums["L21_mode_to_role_xfer"])
    cos = _f(claim_nums["L21_cosine"])
    # Heuristic:
    # deception-specific: role→role high, role→mode ~chance, mode→role ~chance, |cos| low
    # mode-generic: role→mode high and ≈ role→role scenario, |cos| high, mode→role also decent
    # mixed: otherwise
    role_specific = (r2r >= 0.70) and (r2m < 0.65) and (abs(m2r - 0.5) < 0.12) and (abs(cos) < 0.35)
    mode_generic = (r2m >= 0.75) and (m2m >= 0.75) and (abs(cos) >= 0.45)
    if role_specific and not mode_generic:
        verdict = "deception-specific"
    elif mode_generic and r2r < 0.65:
        verdict = "task/mode-generic"
    else:
        verdict = "mixed"
    summary["verdict"] = verdict
    summary["verdict_rule"] = (
        "deception-specific if role→role≥0.70 & role→mode<0.65 & |mode→role−0.5|<0.12 & |cos|<0.35; "
        "task/mode-generic if role→mode≥0.75 & mode→mode≥0.75 & |cos|≥0.45 & role→role<0.65; "
        "else mixed. Primary layer L21."
    )

    results = {
        "core_acts": str(core_dir),
        "transfer_acts": str(xfer_dir),
        "seed": args.seed,
        "summary": summary,
        "per_layer": per_layer,
        "pca_contrast": pca_out,
    }
    (out / "controls_offline.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    print(f"[done] wrote {out / 'controls_offline.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
