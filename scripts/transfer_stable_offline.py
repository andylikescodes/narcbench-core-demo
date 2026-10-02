#!/usr/bin/env python3
"""Transfer-stable representation analysis (CPU-only).

Where does Core→Transfer collusion signal live?
  (1A) Per-layer residual sweep L19–23 + mean-pooled residual
  (1B) Low-dim subspace: diff-in-means, LR weight, PCA(k)
  (2)  Shared single direction vs full / J-only baselines

Protocol matches HARD metrics: train Core discussion public → eval Transfer final.

  .venv-probe/bin/python scripts/transfer_stable_offline.py \
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
    evaluate_transfer,
)
from train_collusion_probe import (  # noqa: E402
    build_groups,
    compute_scores,
    get_layer,
    load_acts,
    parse_layers,
    train_direction,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def identity_scaler(dim: int) -> StandardScaler:
    """No-op scaler (mean=0, scale=1) so raw projections work with compute_scores."""
    s = StandardScaler()
    s.mean_ = np.zeros(dim, dtype=np.float64)
    s.scale_ = np.ones(dim, dtype=np.float64)
    s.var_ = np.ones(dim, dtype=np.float64)
    s.n_features_in_ = dim
    s.n_samples_seen_ = 1
    return s


def collusion_sample_indices(meta: list[dict], *, phase: str = "discussion") -> tuple[list[int], list[int]]:
    col, hon = [], []
    for i, m in enumerate(meta):
        if m.get("mode") != "collusion":
            continue
        if m.get("channel") != "public":
            continue
        if m.get("phase") != phase:
            continue
        if m.get("role") == "colluder":
            col.append(i)
        elif m.get("role") == "honest":
            hon.append(i)
    return col, hon


def control_honest_indices(meta: list[dict], *, phase: str = "discussion") -> list[int]:
    out = []
    for i, m in enumerate(meta):
        if m.get("mode") != "control":
            continue
        if m.get("channel") != "public":
            continue
        if m.get("phase") != phase:
            continue
        if m.get("role") == "honest":
            out.append(i)
    return out


def diff_in_means(acts: np.ndarray, col_idx: list[int], hon_idx: list[int]) -> np.ndarray:
    d = acts[col_idx].mean(axis=0) - acts[hon_idx].mean(axis=0)
    n = np.linalg.norm(d) + 1e-12
    return (d / n).astype(np.float64)


def scenario_aurocs(groups, rids, scaler, w, core_groups, core_rids, core_y, seed: int) -> dict[str, float]:
    """Peak / Split / Asym AUROCs on Transfer (or any eval) groups; Asym fit on Core."""
    y = np.array([1 if groups[r]["mode"] == "collusion" else 0 for r in rids])
    if len(np.unique(y)) < 2:
        return {"error": "single_class", "n": len(rids)}  # type: ignore
    scores = compute_scores(groups, rids, scaler, w)
    core_scores = compute_scores(core_groups, core_rids, scaler, w)
    out: dict[str, Any] = {
        "auroc_peak_suspicion": float(roc_auc_score(y, scores["peak_suspicion"])),
        "auroc_deception_split": float(roc_auc_score(y, scores["deception_split"])),
        "n_runs": len(rids),
        "mode_counts": dict(Counter(y.tolist())),
    }
    fs = StandardScaler()
    clf = LogisticRegression(max_iter=2000, C=0.01, random_state=seed)
    clf.fit(fs.fit_transform(core_scores["asymmetry_probe"]), core_y)
    out["auroc_asymmetry_probe"] = float(
        roc_auc_score(y, clf.predict_proba(fs.transform(scores["asymmetry_probe"]))[:, 1])
    )
    out["auroc_primary"] = float(
        np.mean(
            [
                out["auroc_peak_suspicion"],
                out["auroc_deception_split"],
                out["auroc_asymmetry_probe"],
            ]
        )
    )
    # per-family peak/split
    fam_auc = {}
    for fam in sorted({groups[r]["domain"] for r in rids}):
        idx = [i for i, r in enumerate(rids) if groups[r]["domain"] == fam]
        yf = y[idx]
        if len(np.unique(yf)) < 2 or len(idx) < 4:
            continue
        fam_auc[fam] = {
            "peak": float(roc_auc_score(yf, scores["peak_suspicion"][idx])),
            "split": float(roc_auc_score(yf, scores["deception_split"][idx])),
            "n": len(idx),
        }
    out["per_family"] = fam_auc
    return out


def mean_pool_layers(npz, n: int, layers: list[int]) -> np.ndarray:
    stacks = [get_layer(npz, n, L) for L in layers]
    return np.mean(np.stack(stacks, axis=0), axis=0).astype(np.float32)


def wrap_single_layer(arr: np.ndarray, fake_layer: int = 0):
    """Minimal npz-like object with one layer key for evaluate_transfer reuse."""

    class _Fake:
        def __init__(self, a, L):
            self._a = a
            self._L = L
            self.files = [f"layer_{L}"]

        def __getitem__(self, k):
            return self._a

    return _Fake(arr, fake_layer)


# ---------------------------------------------------------------------------
# (1A) Layer sweep
# ---------------------------------------------------------------------------

def run_layer_sweep(
    core_meta, core_npz, xfer_meta, xfer_npz, layers: list[int], *, seed: int
) -> dict[str, Any]:
    per = []
    for L in layers:
        print(f"[1A] full residual layer {L}", flush=True)
        m = evaluate_transfer(
            core_meta, core_npz, xfer_meta, xfer_npz, L,
            core_phase="discussion", transfer_phase="final", seed=seed,
        )
        per.append(m)
        if "auroc_primary" in m:
            print(
                f"  Peak={m['auroc_peak_suspicion']:.3f} Split={m['auroc_deception_split']:.3f} "
                f"Asym={m['auroc_asymmetry_probe']:.3f} primary={m['auroc_primary']:.3f}",
                flush=True,
            )

    # mean-pooled residual across layers
    print("[1A] mean-pooled residual L19–23", flush=True)
    core_mean = mean_pool_layers(core_npz, len(core_meta), layers)
    xfer_mean = mean_pool_layers(xfer_npz, len(xfer_meta), layers)
    # train direction on mean-pooled
    scaler, w = train_direction(core_meta, core_mean, phase="discussion", seed=seed)
    core_groups = build_groups(core_meta, core_mean, phase="discussion", avg_rounds=True)
    core_rids = sorted(core_groups.keys())
    core_y = np.array([1 if core_groups[r]["mode"] == "collusion" else 0 for r in core_rids])
    xfer_groups = build_transfer_groups(xfer_meta, xfer_mean, phase="final")
    xfer_rids = sorted(xfer_groups.keys())
    mean_pool = scenario_aurocs(
        xfer_groups, xfer_rids, scaler, w, core_groups, core_rids, core_y, seed
    )
    mean_pool["layer"] = "mean_pool_19_23"
    mean_pool["n_core_runs"] = len(core_rids)
    mean_pool["n_transfer_runs"] = len(xfer_rids)
    print(
        f"  Peak={mean_pool['auroc_peak_suspicion']:.3f} Split={mean_pool['auroc_deception_split']:.3f} "
        f"Asym={mean_pool['auroc_asymmetry_probe']:.3f} primary={mean_pool['auroc_primary']:.3f}",
        flush=True,
    )

    def nanmean(key):
        vs = [m[key] for m in per if m.get(key) is not None]
        return float(np.mean(vs)) if vs else float("nan")

    return {
        "protocol": "core_discussion_to_transfer_final_full_residual",
        "layers": layers,
        "band_mean_primary": nanmean("auroc_primary"),
        "band_mean_peak": nanmean("auroc_peak_suspicion"),
        "band_mean_split": nanmean("auroc_deception_split"),
        "band_mean_asym": nanmean("auroc_asymmetry_probe"),
        "per_layer": per,
        "mean_pool_residual": mean_pool,
    }


# ---------------------------------------------------------------------------
# (1B)+(2) Directions + PCA subspaces
# ---------------------------------------------------------------------------

def fit_lr_weight(acts: np.ndarray, col_idx: list[int], hon_idx: list[int], seed: int):
    rng = np.random.RandomState(seed)
    nc = min(len(col_idx), len(hon_idx))
    hi = list(rng.choice(hon_idx, nc, replace=False))
    ci = col_idx[:nc]
    scaler = StandardScaler()
    X = scaler.fit_transform(np.vstack([acts[hi], acts[ci]]))
    y = np.array([0] * nc + [1] * nc)
    clf = LogisticRegression(max_iter=1000, C=1.0, random_state=seed)
    clf.fit(X, y)
    w = clf.coef_[0].astype(np.float64)
    w = w / (np.linalg.norm(w) + 1e-12)
    return scaler, w


def eval_direction_arm(
    name: str,
    core_meta, core_acts, xfer_meta, xfer_acts,
    scaler, w, seed: int,
) -> dict[str, Any]:
    core_groups = build_groups(core_meta, core_acts, phase="discussion", avg_rounds=True)
    core_rids = sorted(core_groups.keys())
    core_y = np.array([1 if core_groups[r]["mode"] == "collusion" else 0 for r in core_rids])
    xfer_groups = build_transfer_groups(xfer_meta, xfer_acts, phase="final")
    xfer_rids = sorted(xfer_groups.keys())
    out = scenario_aurocs(xfer_groups, xfer_rids, scaler, w, core_groups, core_rids, core_y, seed)
    out["arm"] = name
    out["n_core_runs"] = len(core_rids)
    out["n_transfer_runs"] = len(xfer_rids)
    return out


def sample_level_direction_auroc(
    acts: np.ndarray, pos_idx: list[int], neg_idx: list[int], w: np.ndarray, scaler: StandardScaler | None
) -> float | None:
    """AUROC of projection onto w: pos vs neg samples (token/agent rows)."""
    if not pos_idx or not neg_idx:
        return None
    if scaler is not None:
        Xp = scaler.transform(acts[pos_idx])
        Xn = scaler.transform(acts[neg_idx])
    else:
        Xp = acts[pos_idx]
        Xn = acts[neg_idx]
    scores = np.concatenate([Xp @ w, Xn @ w])
    y = np.array([1] * len(pos_idx) + [0] * len(neg_idx))
    if len(np.unique(y)) < 2:
        return None
    return float(roc_auc_score(y, scores))


def run_subspace_and_direction(
    core_meta, core_npz, xfer_meta, xfer_npz, layers: list[int], *, seed: int, pca_ks: list[int]
) -> dict[str, Any]:
    results: dict[str, Any] = {"per_layer": {}, "pca_ks": pca_ks}

    for L in layers:
        print(f"[1B/2] layer {L} directions + PCA", flush=True)
        core_acts = get_layer(core_npz, len(core_meta), L)
        xfer_acts = get_layer(xfer_npz, len(xfer_meta), L)
        col_idx, hon_idx = collusion_sample_indices(core_meta, phase="discussion")
        ctrl_idx = control_honest_indices(core_meta, phase="discussion")

        layer_out: dict[str, Any] = {"layer": L, "n_col": len(col_idx), "n_hon": len(hon_idx), "n_ctrl": len(ctrl_idx)}

        # --- Diff-in-means ---
        d = diff_in_means(core_acts, col_idx, hon_idx)
        # score with identity scaler (raw) vs centered by collusion-sample mean
        # Use StandardScaler fit on balanced col+hon (same as LR) for fair Peak/Split
        rng = np.random.RandomState(seed)
        nc = min(len(col_idx), len(hon_idx))
        hi = list(rng.choice(hon_idx, nc, replace=False))
        ci = col_idx[:nc]
        dim_scaler = StandardScaler()
        dim_scaler.fit(np.vstack([core_acts[hi], core_acts[ci]]))
        dim_arm = eval_direction_arm(
            "diff_in_means", core_meta, core_acts, xfer_meta, xfer_acts, dim_scaler, d, seed
        )
        # also raw (no scale) projection
        id_scaler = identity_scaler(core_acts.shape[1])
        dim_raw = eval_direction_arm(
            "diff_in_means_raw", core_meta, core_acts, xfer_meta, xfer_acts, id_scaler, d, seed
        )
        layer_out["diff_in_means"] = dim_arm
        layer_out["diff_in_means_raw"] = dim_raw
        print(f"  diff-in-means primary={dim_arm['auroc_primary']:.3f} (raw={dim_raw['auroc_primary']:.3f})", flush=True)

        # --- LR weight (same as full probe direction) ---
        lr_scaler, lr_w = fit_lr_weight(core_acts, col_idx, hon_idx, seed)
        lr_arm = eval_direction_arm(
            "lr_weight", core_meta, core_acts, xfer_meta, xfer_acts, lr_scaler, lr_w, seed
        )
        layer_out["lr_weight"] = lr_arm
        # cosine alignment between dim and lr
        layer_out["cosine_dim_vs_lr"] = float(np.abs(d @ lr_w))
        print(
            f"  lr_weight primary={lr_arm['auroc_primary']:.3f}  |cos(dim,lr)|={layer_out['cosine_dim_vs_lr']:.3f}",
            flush=True,
        )

        # --- Control light check: sample-level ---
        # collusion colluders vs control honest (discussion public)
        layer_out["control_sample_auroc"] = {
            "dim_col_vs_ctrl_honest": sample_level_direction_auroc(
                core_acts, col_idx, ctrl_idx, d, dim_scaler
            ),
            "lr_col_vs_ctrl_honest": sample_level_direction_auroc(
                core_acts, col_idx, ctrl_idx, lr_w, lr_scaler
            ),
            "dim_col_vs_hon_collusion": sample_level_direction_auroc(
                core_acts, col_idx, hon_idx, d, dim_scaler
            ),
            "note": (
                "Sample-level AUROC on Core discussion public. "
                "col_vs_ctrl_honest: collusion-run colluders vs control-run honest agents "
                "(control has no colluder role). col_vs_hon_collusion: within-collusion contrast."
            ),
        }
        # Core scenario-level with same direction (in-distribution sanity)
        core_groups = build_groups(core_meta, core_acts, phase="discussion", avg_rounds=True)
        core_rids = sorted(core_groups.keys())
        core_y = np.array([1 if core_groups[r]["mode"] == "collusion" else 0 for r in core_rids])
        core_sc = compute_scores(core_groups, core_rids, lr_scaler, lr_w)
        layer_out["core_id_scenario_auroc"] = {
            "peak": float(roc_auc_score(core_y, core_sc["peak_suspicion"])),
            "split": float(roc_auc_score(core_y, core_sc["deception_split"])),
            "n_runs": len(core_rids),
            "note": "In-distribution Core scenario AUROC with LR direction (not LDO CV).",
        }

        # --- PCA on Core collusion discussion public residuals ---
        all_col_pub = col_idx + hon_idx  # all collusion-run public discussion
        X_fit = core_acts[all_col_pub]
        # center for PCA
        pca_full = PCA(n_components=min(max(pca_ks), X_fit.shape[0] - 1, X_fit.shape[1]), random_state=seed)
        pca_full.fit(X_fit)

        # PCA on colluder−honest deltas (per-run paired if possible, else global)
        # Use per-sample: for each collusion run, mean(colluder)-mean(honest) if both present
        deltas = []
        by_run: dict[str, dict] = defaultdict(lambda: {"col": [], "hon": []})
        for i in all_col_pub:
            m = core_meta[i]
            if m["role"] == "colluder":
                by_run[m["run_id"]]["col"].append(core_acts[i])
            else:
                by_run[m["run_id"]]["hon"].append(core_acts[i])
        for rid, dsets in by_run.items():
            if dsets["col"] and dsets["hon"]:
                deltas.append(np.mean(dsets["col"], axis=0) - np.mean(dsets["hon"], axis=0))
        deltas_arr = np.stack(deltas, axis=0) if deltas else X_fit[:1] * 0
        n_delta_comp = min(max(pca_ks), max(1, deltas_arr.shape[0] - 1), deltas_arr.shape[1])
        pca_delta = PCA(n_components=n_delta_comp, random_state=seed)
        if deltas_arr.shape[0] >= 2:
            pca_delta.fit(deltas_arr)
            delta_ok = True
        else:
            delta_ok = False

        pca_results = []
        for k in pca_ks:
            if k > pca_full.n_components_:
                continue
            # Project Core & Transfer onto top-k residual PCA; train LR in subspace on Core
            P = pca_full.components_[:k]  # (k, D)
            # project: (N, k)
            # For Peak/Split we need agent vectors in ambient space OR score in subspace.
            # Train LR in k-dim on Core collusion samples; lift weight back to ambient: w_amb = P.T @ w_k
            Xc = (core_acts[ci] - pca_full.mean_) @ P.T
            Xh = (core_acts[hi] - pca_full.mean_) @ P.T
            Xs = np.vstack([Xh, Xc])
            ys = np.array([0] * nc + [1] * nc)
            sub_scaler = StandardScaler()
            Xs_s = sub_scaler.fit_transform(Xs)
            clf = LogisticRegression(max_iter=1000, C=1.0, random_state=seed)
            clf.fit(Xs_s, ys)
            w_k = clf.coef_[0].astype(np.float64)
            # ambient direction (in original residual space, accounting for sub_scaler):
            # score ambient: ((x - pca_mean) @ P.T - sub_mean) / sub_scale @ w_k
            # Equivalent ambient linear: w_amb such that w_amb @ x ≈ ...
            # For compute_scores we need scaler + w in ambient dim.
            # Build effective ambient: first center by pca_mean, project, then sub_scaler, then w_k.
            # Simpler approach: project all agent vectors to k-dim, wrap as k-dim "acts", use StandardScaler+w in k-dim.
            core_proj = ((core_acts - pca_full.mean_) @ P.T).astype(np.float32)
            xfer_proj = ((xfer_acts - pca_full.mean_) @ P.T).astype(np.float32)
            # re-fit scaler+w in projected space (same as above)
            proj_scaler = StandardScaler()
            Xp = proj_scaler.fit_transform(np.vstack([core_proj[hi], core_proj[ci]]))
            clf2 = LogisticRegression(max_iter=1000, C=1.0, random_state=seed)
            clf2.fit(Xp, ys)
            w_proj = clf2.coef_[0].astype(np.float64)
            w_proj = w_proj / (np.linalg.norm(w_proj) + 1e-12)
            arm = eval_direction_arm(
                f"pca_residual_k{k}",
                core_meta, core_proj, xfer_meta, xfer_proj, proj_scaler, w_proj, seed,
            )
            arm["k"] = k
            arm["explained_var_ratio_sum"] = float(pca_full.explained_variance_ratio_[:k].sum())
            pca_results.append(arm)
            print(
                f"  PCA residual k={k} primary={arm['auroc_primary']:.3f} "
                f"var={arm['explained_var_ratio_sum']:.3f}",
                flush=True,
            )

        # PCA on contrast deltas only
        pca_delta_results = []
        if delta_ok:
            for k in pca_ks:
                if k > pca_delta.n_components_:
                    continue
                P = pca_delta.components_[:k]
                # Project ambient residuals onto contrast PCA (no pca_delta.mean_ subtract on residuals —
                # components are directions in residual space; use mean of deltas for centering of deltas only)
                # For scoring agents: project (x - mean_all_col_pub) or just x @ P.T
                # Use: project centered by global collusion-public mean
                mu = X_fit.mean(axis=0)
                core_proj = ((core_acts - mu) @ P.T).astype(np.float32)
                xfer_proj = ((xfer_acts - mu) @ P.T).astype(np.float32)
                proj_scaler = StandardScaler()
                Xp = proj_scaler.fit_transform(np.vstack([core_proj[hi], core_proj[ci]]))
                ys = np.array([0] * nc + [1] * nc)
                clf2 = LogisticRegression(max_iter=1000, C=1.0, random_state=seed)
                clf2.fit(Xp, ys)
                w_proj = clf2.coef_[0].astype(np.float64)
                w_proj = w_proj / (np.linalg.norm(w_proj) + 1e-12)
                arm = eval_direction_arm(
                    f"pca_contrast_k{k}",
                    core_meta, core_proj, xfer_meta, xfer_proj, proj_scaler, w_proj, seed,
                )
                arm["k"] = k
                arm["n_deltas"] = int(deltas_arr.shape[0])
                arm["explained_var_ratio_sum"] = float(pca_delta.explained_variance_ratio_[:k].sum())
                pca_delta_results.append(arm)
                print(
                    f"  PCA contrast k={k} primary={arm['auroc_primary']:.3f} "
                    f"var={arm['explained_var_ratio_sum']:.3f}",
                    flush=True,
                )

        # Also: pure top-1 PCA component as direction (no LR retrain) — residual PC1
        if pca_full.n_components_ >= 1:
            pc1 = pca_full.components_[0].astype(np.float64)
            # orient toward colluders
            if (dim_scaler.transform(core_acts[ci]).mean(0) @ pc1) < (
                dim_scaler.transform(core_acts[hi]).mean(0) @ pc1
            ):
                pc1 = -pc1
            pc1_arm = eval_direction_arm(
                "pca_residual_pc1_direct",
                core_meta, core_acts, xfer_meta, xfer_acts, dim_scaler, pc1, seed,
            )
            layer_out["pca_residual_pc1_direct"] = pc1_arm

        layer_out["pca_residual"] = pca_results
        layer_out["pca_contrast"] = pca_delta_results
        results["per_layer"][str(L)] = layer_out

    # Aggregate bests
    best_dim = max(
        (results["per_layer"][str(L)]["diff_in_means"] for L in layers),
        key=lambda x: x["auroc_primary"],
    )
    best_lr = max(
        (results["per_layer"][str(L)]["lr_weight"] for L in layers),
        key=lambda x: x["auroc_primary"],
    )
    all_pca = []
    for L in layers:
        for arm in results["per_layer"][str(L)]["pca_residual"]:
            all_pca.append({**arm, "layer": L})
        for arm in results["per_layer"][str(L)]["pca_contrast"]:
            all_pca.append({**arm, "layer": L})
    best_pca = max(all_pca, key=lambda x: x["auroc_primary"]) if all_pca else None

    # Band-mean of single-direction arms
    def band_mean(arm_key):
        vs = [results["per_layer"][str(L)][arm_key]["auroc_primary"] for L in layers]
        return float(np.mean(vs))

    results["summary"] = {
        "best_diff_in_means": {
            "layer": next(
                L
                for L in layers
                if results["per_layer"][str(L)]["diff_in_means"]["auroc_primary"]
                == best_dim["auroc_primary"]
            ),
            **{k: best_dim[k] for k in ("auroc_primary", "auroc_peak_suspicion", "auroc_deception_split", "auroc_asymmetry_probe")},
        },
        "best_lr_weight": {
            "layer": next(
                L
                for L in layers
                if results["per_layer"][str(L)]["lr_weight"]["auroc_primary"]
                == best_lr["auroc_primary"]
            ),
            **{k: best_lr[k] for k in ("auroc_primary", "auroc_peak_suspicion", "auroc_deception_split", "auroc_asymmetry_probe")},
        },
        "best_pca": best_pca,
        "band_mean_diff_in_means_primary": band_mean("diff_in_means"),
        "band_mean_lr_weight_primary": band_mean("lr_weight"),
        "band_mean_cosine_dim_vs_lr": float(
            np.mean([results["per_layer"][str(L)]["cosine_dim_vs_lr"] for L in layers])
        ),
    }
    return results


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--core-acts", type=Path, required=True)
    ap.add_argument("--transfer-acts", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--layers", default="19-23")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--pca-ks", default="1,2,4,8,16,32")
    args = ap.parse_args()

    core_dir = args.core_acts if args.core_acts.is_absolute() else ROOT / args.core_acts
    xfer_dir = args.transfer_acts if args.transfer_acts.is_absolute() else ROOT / args.transfer_acts
    out = args.out if args.out.is_absolute() else ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)

    layers = parse_layers(args.layers)
    pca_ks = [int(x) for x in args.pca_ks.split(",") if x.strip()]

    print("[load] Core + Transfer activations", flush=True)
    core_meta, core_npz = load_acts(core_dir)
    xfer_meta, xfer_npz = load_acts(xfer_dir)
    xfer_meta = enrich_transfer_meta(xfer_meta)

    avail_c = sorted(int(k.split("_")[1]) for k in core_npz.files if k.startswith("layer_"))
    avail_t = sorted(int(k.split("_")[1]) for k in xfer_npz.files if k.startswith("layer_"))
    layers = [L for L in layers if L in avail_c and L in avail_t]
    print(f"[load] layers={layers} core_n={len(core_meta)} xfer_n={len(xfer_meta)}", flush=True)

    # (1A)
    layer_sweep = run_layer_sweep(
        core_meta, core_npz, xfer_meta, xfer_npz, layers, seed=args.seed
    )
    (out / "layer_sweep.json").write_text(json.dumps(layer_sweep, indent=2) + "\n")

    # (1B)+(2)
    subspace = run_subspace_and_direction(
        core_meta, core_npz, xfer_meta, xfer_npz, layers, seed=args.seed, pca_ks=pca_ks
    )
    (out / "subspace_and_direction.json").write_text(json.dumps(subspace, indent=2) + "\n")

    summary = {
        "core_acts": str(core_dir),
        "transfer_acts": str(xfer_dir),
        "layers": layers,
        "layer_sweep_band_mean_primary": layer_sweep["band_mean_primary"],
        "mean_pool_primary": layer_sweep["mean_pool_residual"]["auroc_primary"],
        "best_diff_in_means": subspace["summary"]["best_diff_in_means"],
        "best_lr_weight": subspace["summary"]["best_lr_weight"],
        "best_pca": {
            k: subspace["summary"]["best_pca"][k]
            for k in ("arm", "layer", "k", "auroc_primary", "auroc_peak_suspicion", "auroc_deception_split", "auroc_asymmetry_probe", "explained_var_ratio_sum")
            if subspace["summary"]["best_pca"] and k in subspace["summary"]["best_pca"]
        }
        if subspace["summary"]["best_pca"]
        else None,
        "band_mean_diff_in_means_primary": subspace["summary"]["band_mean_diff_in_means_primary"],
        "band_mean_lr_weight_primary": subspace["summary"]["band_mean_lr_weight_primary"],
        "band_mean_cosine_dim_vs_lr": subspace["summary"]["band_mean_cosine_dim_vs_lr"],
    }
    (out / "offline_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
