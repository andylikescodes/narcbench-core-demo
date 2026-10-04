#!/usr/bin/env python3
"""Residualize mode out of role (and role out of mode) — Transfer-stable offline.

Fit on Core discussion public:
  role: LR colluder vs honest (collusion mode only)
  mode: LR agent-average per run (collusion vs control)

Ambient residual unit vectors (scaler-undone):
  role_perp = role − proj_mode(role)
  mode_perp = mode − proj_role(mode)

Re-eval Transfer final: role_perp → role / mode AUROC; same for mode_perp;
baseline role/mode for comparison; cosines before/after.

Also exports intervention directions to results/transfer_stable/directions/.

  .venv-probe/bin/python scripts/transfer_stable_residualize.py \
    --core-acts data/activations/gemma2_9b/core/20261001T012639Z \
    --transfer-acts data/activations/gemma2_9b/transfer/RUNPOD \
    --out results/transfer_stable
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from eval_transfer_from_core import enrich_transfer_meta  # noqa: E402
from train_collusion_probe import get_layer, load_acts, parse_layers  # noqa: E402
from transfer_stable_controls import (  # noqa: E402
    fit_binary_lr,
    fit_pca_contrast,
    mode_runavg_auroc,
    project_pca,
    role_sample_auroc,
    run_agent_average,
)
from transfer_stable_offline import (  # noqa: E402
    collusion_sample_indices,
    diff_in_means,
    fit_lr_weight,
)


def unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64).reshape(-1)
    return (v / (np.linalg.norm(v) + 1e-12)).astype(np.float64)


def ambient_from_scaled(w: np.ndarray, scaler: StandardScaler) -> np.ndarray:
    """Undo StandardScaler so score ≈ (w/scale)·x + const → unit ambient direction."""
    w_raw = w / (scaler.scale_ + 1e-12)
    return unit(w_raw)


def project_out(v: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """v − proj_basis(v); both assumed unit (or near-unit)."""
    b = unit(basis)
    return unit(v - float(v @ b) * b)


def identity_scaler(dim: int) -> StandardScaler:
    s = StandardScaler()
    s.mean_ = np.zeros(dim, dtype=np.float64)
    s.scale_ = np.ones(dim, dtype=np.float64)
    s.var_ = np.ones(dim, dtype=np.float64)
    s.n_features_in_ = dim
    s.n_samples_seen_ = 1
    return s


def eval_dir(
    name: str,
    w_ambient: np.ndarray,
    xfer_meta,
    xfer_acts,
) -> dict[str, Any]:
    """Project ambient unit direction with identity scaler (raw residual space)."""
    sc = identity_scaler(xfer_acts.shape[1])
    role = role_sample_auroc(xfer_meta, xfer_acts, sc, w_ambient, phase="final")
    mode = mode_runavg_auroc(xfer_meta, xfer_acts, sc, w_ambient, phase="final")
    return {
        "name": name,
        "role_auroc_transfer_final": role,
        "mode_auroc_runavg_transfer_final": mode,
    }


def fit_layer_directions(core_meta, core_acts, *, seed: int) -> dict[str, Any]:
    col_idx, hon_idx = collusion_sample_indices(core_meta, phase="discussion")
    role_scaler, role_w = fit_lr_weight(core_acts, col_idx, hon_idx, seed)
    role_dim = diff_in_means(core_acts, col_idx, hon_idx)

    Xc, yc, rids = run_agent_average(core_meta, core_acts, phase="discussion")
    mode_scaler, mode_w = fit_binary_lr(Xc[yc == 1], Xc[yc == 0], seed)

    role_amb = ambient_from_scaled(role_w, role_scaler)
    mode_amb = ambient_from_scaled(mode_w, mode_scaler)
    role_perp = project_out(role_amb, mode_amb)
    mode_perp = project_out(mode_amb, role_amb)

    return {
        "n_train_colluder": len(col_idx),
        "n_train_honest": len(hon_idx),
        "n_train_runs": len(rids),
        "train_mode_counts": dict(Counter(yc.tolist())),
        "role_scaler": role_scaler,
        "role_w_scaled": role_w,
        "mode_scaler": mode_scaler,
        "mode_w_scaled": mode_w,
        "role_ambient": role_amb,
        "mode_ambient": mode_amb,
        "role_perp": role_perp,
        "mode_perp": mode_perp,
        "role_dim": unit(role_dim),
        "cosine_before": float(role_amb @ mode_amb),
        "cosine_role_perp_vs_mode": float(role_perp @ mode_amb),
        "cosine_mode_perp_vs_role": float(mode_perp @ role_amb),
        "cosine_role_perp_vs_mode_perp": float(role_perp @ mode_perp),
        "norm_proj_role_on_mode": float(abs(role_amb @ mode_amb)),  # |cos|
    }


def run_pca_contrast_export(
    core_meta, core_acts, xfer_meta, xfer_acts, *, seed: int, k: int = 8
) -> dict[str, Any]:
    """PCA-contrast subspace @ this layer; return (k,d) components + ambient LR-in-contrast."""
    pca, mu, n_deltas = fit_pca_contrast(
        core_meta, core_acts, phase="discussion", seed=seed, max_k=k
    )
    k_use = min(k, int(pca.n_components_))
    P = pca.components_[:k_use].astype(np.float64)  # (k, d)
    core_proj = project_pca(core_acts, pca, mu, k_use)
    xfer_proj = project_pca(xfer_acts, pca, mu, k_use)
    col_idx, hon_idx = collusion_sample_indices(core_meta, phase="discussion")
    role_scaler, role_w = fit_lr_weight(core_proj, col_idx, hon_idx, seed)
    # Ambient direction = P.T @ (w / scale) then unit
    w_sub = ambient_from_scaled(role_w, role_scaler)  # in k-dim
    role_amb_contrast = unit(P.T @ w_sub)  # (d,)
    sc = identity_scaler(xfer_acts.shape[1])
    role_auc = role_sample_auroc(xfer_meta, xfer_acts, sc, role_amb_contrast, phase="final")
    mode_auc = mode_runavg_auroc(xfer_meta, xfer_acts, sc, role_amb_contrast, phase="final")
    # Also role/mode AUROC inside subspace (for documentation)
    role_in = role_sample_auroc(xfer_meta, xfer_proj, role_scaler, role_w, phase="final")
    Xc, yc, _ = run_agent_average(core_meta, core_proj, phase="discussion")
    mode_scaler, mode_w = fit_binary_lr(Xc[yc == 1], Xc[yc == 0], seed)
    mode_in = mode_runavg_auroc(xfer_meta, xfer_proj, mode_scaler, mode_w, phase="final")
    return {
        "k": k_use,
        "n_deltas": n_deltas,
        "explained_var_ratio_sum": float(pca.explained_variance_ratio_[:k_use].sum()),
        "components": P,  # (k, d)
        "mu": mu.astype(np.float64),
        "lr_in_contrast_ambient": role_amb_contrast,
        "role_auroc_ambient_transfer_final": role_auc,
        "mode_auroc_ambient_transfer_final": mode_auc,
        "role_auroc_in_subspace_transfer_final": role_in,
        "mode_auroc_in_subspace_transfer_final": mode_in,
    }


def export_directions(out_dir: Path, layer_pack: dict, pca_pack: dict | None, meta: dict) -> None:
    ddir = out_dir / "directions"
    ddir.mkdir(parents=True, exist_ok=True)

    # Required exports
    np.save(ddir / "lr_role_L21.npy", layer_pack["21"]["role_ambient"].astype(np.float32))
    np.save(ddir / "lr_role_L22.npy", layer_pack["22"]["role_ambient"].astype(np.float32))
    np.save(ddir / "diff_means_L22.npy", layer_pack["22"]["role_dim"].astype(np.float32))
    np.save(ddir / "lr_role_perp_mode_L21.npy", layer_pack["21"]["role_perp"].astype(np.float32))

    # Extra useful vectors
    np.save(ddir / "lr_mode_L21.npy", layer_pack["21"]["mode_ambient"].astype(np.float32))
    np.save(ddir / "lr_mode_perp_role_L21.npy", layer_pack["21"]["mode_perp"].astype(np.float32))
    np.save(ddir / "lr_role_perp_mode_L22.npy", layer_pack["22"]["role_perp"].astype(np.float32))

    pca_doc = None
    if pca_pack is not None:
        # Save (k, d) contrast components; also ambient LR-in-contrast as cleaner single dir
        np.save(ddir / "pca_contrast_k8_L23.npy", pca_pack["components"].astype(np.float32))
        np.save(
            ddir / "pca_contrast_k8_L23_lr_ambient.npy",
            pca_pack["lr_in_contrast_ambient"].astype(np.float32),
        )
        pca_doc = {
            "file": "pca_contrast_k8_L23.npy",
            "shape": list(pca_pack["components"].shape),
            "layout": "(k, d) PCA components on Core colluder−honest per-run deltas @ L23",
            "also_exported": "pca_contrast_k8_L23_lr_ambient.npy — unit ambient = P.T @ LR_sub (cleaner single steer dir)",
            "k": pca_pack["k"],
            "explained_var_ratio_sum": pca_pack["explained_var_ratio_sum"],
            "role_auroc_ambient_transfer_final": pca_pack["role_auroc_ambient_transfer_final"]["auroc"],
            "role_auroc_in_subspace_transfer_final": pca_pack["role_auroc_in_subspace_transfer_final"]["auroc"],
        }

    files = {
        "lr_role_L21.npy": {
            "layer": 21,
            "kind": "lr_role_ambient_unit",
            "shape": list(layer_pack["21"]["role_ambient"].shape),
            "auroc_role_xfer": meta["per_layer"]["21"]["role"]["role_auroc_transfer_final"]["auroc"],
            "auroc_mode_xfer": meta["per_layer"]["21"]["role"]["mode_auroc_runavg_transfer_final"]["auroc"],
        },
        "lr_role_L22.npy": {
            "layer": 22,
            "kind": "lr_role_ambient_unit",
            "shape": list(layer_pack["22"]["role_ambient"].shape),
            "auroc_role_xfer": meta["per_layer"]["22"]["role"]["role_auroc_transfer_final"]["auroc"],
            "auroc_mode_xfer": meta["per_layer"]["22"]["role"]["mode_auroc_runavg_transfer_final"]["auroc"],
        },
        "diff_means_L22.npy": {
            "layer": 22,
            "kind": "diff_means_colluder_minus_honest_ambient_unit",
            "shape": list(layer_pack["22"]["role_dim"].shape),
        },
        "lr_role_perp_mode_L21.npy": {
            "layer": 21,
            "kind": "role_minus_proj_mode_ambient_unit",
            "shape": list(layer_pack["21"]["role_perp"].shape),
            "auroc_role_xfer": meta["per_layer"]["21"]["role_perp"]["role_auroc_transfer_final"]["auroc"],
            "auroc_mode_xfer": meta["per_layer"]["21"]["role_perp"]["mode_auroc_runavg_transfer_final"]["auroc"],
        },
    }
    if pca_doc:
        files["pca_contrast_k8_L23.npy"] = pca_doc

    meta_out = {
        "train_protocol": (
            "Fit Core discussion public → eval Transfer final. "
            "role = LR colluder vs honest (collusion mode only); "
            "mode = LR agent-average per run (collusion vs control). "
            "Ambient unit = (w/scale) L2-normalized. "
            "role_perp = role − proj_mode(role)."
        ),
        "labels": ["mode", "role"],
        "core_acts": meta["core_acts"],
        "transfer_acts": meta["transfer_acts"],
        "seed": meta["seed"],
        "files": files,
        "residualize_claim": meta.get("one_line_claim"),
        "residualize_highlights": meta.get("highlights"),
    }
    (ddir / "meta.json").write_text(json.dumps(meta_out, indent=2) + "\n")
    print(f"[export] directions → {ddir}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--core-acts", type=Path, required=True)
    ap.add_argument("--transfer-acts", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--layers", default="21,22")
    ap.add_argument("--pca-layer", type=int, default=23)
    ap.add_argument("--pca-k", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip-export", action="store_true")
    args = ap.parse_args()

    core_dir = args.core_acts if args.core_acts.is_absolute() else ROOT / args.core_acts
    xfer_dir = args.transfer_acts if args.transfer_acts.is_absolute() else ROOT / args.transfer_acts
    out = args.out if args.out.is_absolute() else ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)

    layers = parse_layers(args.layers)
    print("[load] Core + Transfer", flush=True)
    core_meta, core_npz = load_acts(core_dir)
    xfer_meta, xfer_npz = load_acts(xfer_dir)
    xfer_meta = enrich_transfer_meta(xfer_meta)
    print(f"[load] core_n={len(core_meta)} xfer_n={len(xfer_meta)} layers={layers}", flush=True)

    per_layer: dict[str, Any] = {}
    layer_pack: dict[str, Any] = {}

    for L in layers:
        print(f"[residualize] layer {L}", flush=True)
        core_acts = get_layer(core_npz, len(core_meta), L)
        xfer_acts = get_layer(xfer_npz, len(xfer_meta), L)
        pack = fit_layer_directions(core_meta, core_acts, seed=args.seed)
        layer_pack[str(L)] = pack

        role_e = eval_dir("role", pack["role_ambient"], xfer_meta, xfer_acts)
        mode_e = eval_dir("mode", pack["mode_ambient"], xfer_meta, xfer_acts)
        role_perp_e = eval_dir("role_perp", pack["role_perp"], xfer_meta, xfer_acts)
        mode_perp_e = eval_dir("mode_perp", pack["mode_perp"], xfer_meta, xfer_acts)

        # Also eval with original scalers for parity with controls_offline (scaled space)
        role_scaled = {
            "role_auroc_transfer_final": role_sample_auroc(
                xfer_meta, xfer_acts, pack["role_scaler"], pack["role_w_scaled"], phase="final"
            ),
            "mode_auroc_runavg_transfer_final": mode_runavg_auroc(
                xfer_meta, xfer_acts, pack["role_scaler"], pack["role_w_scaled"], phase="final"
            ),
        }
        mode_scaled = {
            "mode_auroc_runavg_transfer_final": mode_runavg_auroc(
                xfer_meta, xfer_acts, pack["mode_scaler"], pack["mode_w_scaled"], phase="final"
            ),
            "role_auroc_transfer_final": role_sample_auroc(
                xfer_meta, xfer_acts, pack["mode_scaler"], pack["mode_w_scaled"], phase="final"
            ),
        }

        per_layer[str(L)] = {
            "layer": L,
            "cosine_before_role_vs_mode": pack["cosine_before"],
            "cosine_after_role_perp_vs_mode": pack["cosine_role_perp_vs_mode"],
            "cosine_after_mode_perp_vs_role": pack["cosine_mode_perp_vs_role"],
            "cosine_role_perp_vs_mode_perp": pack["cosine_role_perp_vs_mode_perp"],
            "role": role_e,
            "mode": mode_e,
            "role_perp": role_perp_e,
            "mode_perp": mode_perp_e,
            "role_scaled_space_parity": role_scaled,
            "mode_scaled_space_parity": mode_scaled,
            "n_train_colluder": pack["n_train_colluder"],
            "n_train_honest": pack["n_train_honest"],
            "n_train_runs": pack["n_train_runs"],
        }
        print(
            f"  cos before={pack['cosine_before']:.3f}  "
            f"role→role={role_e['role_auroc_transfer_final']['auroc']:.3f}  "
            f"role→mode={role_e['mode_auroc_runavg_transfer_final']['auroc']:.3f}  "
            f"role_perp→role={role_perp_e['role_auroc_transfer_final']['auroc']:.3f}  "
            f"role_perp→mode={role_perp_e['mode_auroc_runavg_transfer_final']['auroc']:.3f}",
            flush=True,
        )
        print(
            f"  mode→mode={mode_e['mode_auroc_runavg_transfer_final']['auroc']:.3f}  "
            f"mode→role={mode_e['role_auroc_transfer_final']['auroc']:.3f}  "
            f"mode_perp→mode={mode_perp_e['mode_auroc_runavg_transfer_final']['auroc']:.3f}  "
            f"mode_perp→role={mode_perp_e['role_auroc_transfer_final']['auroc']:.3f}",
            flush=True,
        )

    # Optional PCA-contrast @ L23
    pca_pack = None
    pca_json = None
    if args.pca_layer:
        print(f"[residualize] PCA-contrast k={args.pca_k} @ L{args.pca_layer}", flush=True)
        core_acts = get_layer(core_npz, len(core_meta), args.pca_layer)
        xfer_acts = get_layer(xfer_npz, len(xfer_meta), args.pca_layer)
        pca_pack = run_pca_contrast_export(
            core_meta, core_acts, xfer_meta, xfer_acts, seed=args.seed, k=args.pca_k
        )
        # Residualize role vs mode inside subspace too
        pca_obj, mu, _ = fit_pca_contrast(
            core_meta, core_acts, phase="discussion", seed=args.seed, max_k=args.pca_k
        )
        core_proj = project_pca(core_acts, pca_obj, mu, pca_pack["k"])
        xfer_proj = project_pca(xfer_acts, pca_obj, mu, pca_pack["k"])
        sub = fit_layer_directions(core_meta, core_proj, seed=args.seed)
        pca_json = {
            "layer": args.pca_layer,
            "k": pca_pack["k"],
            "explained_var_ratio_sum": pca_pack["explained_var_ratio_sum"],
            "role_auroc_ambient_lr_in_contrast": pca_pack["role_auroc_ambient_transfer_final"],
            "mode_auroc_ambient_lr_in_contrast": pca_pack["mode_auroc_ambient_transfer_final"],
            "in_subspace": {
                "cosine_before": sub["cosine_before"],
                "cosine_after_role_perp_vs_mode": sub["cosine_role_perp_vs_mode"],
                "role": eval_dir("role", sub["role_ambient"], xfer_meta, xfer_proj),
                "mode": eval_dir("mode", sub["mode_ambient"], xfer_meta, xfer_proj),
                "role_perp": eval_dir("role_perp", sub["role_perp"], xfer_meta, xfer_proj),
                "mode_perp": eval_dir("mode_perp", sub["mode_perp"], xfer_meta, xfer_proj),
            },
        }
        print(
            f"  PCA k={pca_pack['k']} subspace cos before={sub['cosine_before']:.3f}  "
            f"role→role={pca_json['in_subspace']['role']['role_auroc_transfer_final']['auroc']:.3f}  "
            f"role_perp→role={pca_json['in_subspace']['role_perp']['role_auroc_transfer_final']['auroc']:.3f}  "
            f"role_perp→mode={pca_json['in_subspace']['role_perp']['mode_auroc_runavg_transfer_final']['auroc']:.3f}",
            flush=True,
        )

    # Highlights + claim (primary L21)
    L21 = per_layer.get("21", {})
    L22 = per_layer.get("22", {})

    def auc(block, arm, key):
        return (block.get(arm) or {}).get(key, {}).get("auroc")

    highlights = {
        "L21": {
            "cos_before": L21.get("cosine_before_role_vs_mode"),
            "cos_after_role_perp_vs_mode": L21.get("cosine_after_role_perp_vs_mode"),
            "role_to_role": auc(L21, "role", "role_auroc_transfer_final"),
            "role_to_mode": auc(L21, "role", "mode_auroc_runavg_transfer_final"),
            "role_perp_to_role": auc(L21, "role_perp", "role_auroc_transfer_final"),
            "role_perp_to_mode": auc(L21, "role_perp", "mode_auroc_runavg_transfer_final"),
            "mode_to_mode": auc(L21, "mode", "mode_auroc_runavg_transfer_final"),
            "mode_to_role": auc(L21, "mode", "role_auroc_transfer_final"),
            "mode_perp_to_mode": auc(L21, "mode_perp", "mode_auroc_runavg_transfer_final"),
            "mode_perp_to_role": auc(L21, "mode_perp", "role_auroc_transfer_final"),
        },
        "L22": {
            "cos_before": L22.get("cosine_before_role_vs_mode"),
            "cos_after_role_perp_vs_mode": L22.get("cosine_after_role_perp_vs_mode"),
            "role_to_role": auc(L22, "role", "role_auroc_transfer_final"),
            "role_to_mode": auc(L22, "role", "mode_auroc_runavg_transfer_final"),
            "role_perp_to_role": auc(L22, "role_perp", "role_auroc_transfer_final"),
            "role_perp_to_mode": auc(L22, "role_perp", "mode_auroc_runavg_transfer_final"),
            "mode_to_mode": auc(L22, "mode", "mode_auroc_runavg_transfer_final"),
            "mode_to_role": auc(L22, "mode", "role_auroc_transfer_final"),
            "mode_perp_to_mode": auc(L22, "mode_perp", "mode_auroc_runavg_transfer_final"),
            "mode_perp_to_role": auc(L22, "mode_perp", "role_auroc_transfer_final"),
        },
    }

    r2r = highlights["L21"]["role_to_role"]
    rp2r = highlights["L21"]["role_perp_to_role"]
    rp2m = highlights["L21"]["role_perp_to_mode"]
    r2m = highlights["L21"]["role_to_mode"]
    cos_b = highlights["L21"]["cos_before"]
    cos_a = highlights["L21"]["cos_after_role_perp_vs_mode"]

    still_transfers = (rp2r is not None) and (rp2r >= 0.70)
    mode_leak_dropped = (
        r2m is not None and rp2m is not None and (rp2m < r2m - 0.05)
    )
    if still_transfers and mode_leak_dropped:
        verdict = "YES — after removing mode, role still transfers (mode leak reduced)"
    elif still_transfers:
        verdict = "YES — after removing mode, role still transfers (mode leak similar)"
    else:
        verdict = "NO — after removing mode, role transfer collapses"

    one_line = (
        f"{verdict}: L21 role→role {r2r:.3f} → role_perp→role {rp2r:.3f}; "
        f"role→mode {r2m:.3f} → role_perp→mode {rp2m:.3f}; "
        f"cos(role,mode) {cos_b:.3f} → cos(role_perp,mode) {cos_a:.3f}."
    )

    results = {
        "protocol": (
            "Fit Core discussion public → eval Transfer final. "
            "Labels: mode / role only. "
            "role_perp = role − proj_mode(role); mode_perp = mode − proj_role(mode) "
            "in ambient residual space (scaler-undone unit vectors)."
        ),
        "core_acts": str(core_dir),
        "transfer_acts": str(xfer_dir),
        "seed": args.seed,
        "layers": layers,
        "highlights": highlights,
        "one_line_claim": one_line,
        "verdict_short": "YES" if still_transfers else "NO",
        "per_layer": per_layer,
        "pca_contrast": pca_json,
    }

    # JSON-safe: strip non-serializable from per_layer already clean
    (out / "residualize_mode.json").write_text(json.dumps(results, indent=2) + "\n")

    # Markdown writeup
    md = f"""# Residualize mode out of role — Transfer-stable

Updated: 2026-10-01 ~1:30 PM PT  
JSON: `results/transfer_stable/residualize_mode.json`  
Script: `scripts/transfer_stable_residualize.py`

## Protocol

| Piece | Choice |
| --- | --- |
| Fit | Core **discussion public** |
| Eval | Transfer **final** |
| Role | LR colluder vs honest, collusion mode only |
| Mode | LR agent-average per run, collusion vs control |
| Residualize | ambient unit: `role_perp = role − proj_mode(role)`; `mode_perp = mode − proj_role(mode)` |
| Labels | **mode** / **role** only |

## One-line claim

> {one_line}

## Table — Transfer final AUROCs (ambient unit directions)

| Layer | Dir → target | AUROC |
| ---: | --- | ---: |
| 21 | role → role | {highlights['L21']['role_to_role']:.3f} |
| 21 | role → mode (run-avg) | {highlights['L21']['role_to_mode']:.3f} |
| 21 | **role_perp → role** | **{highlights['L21']['role_perp_to_role']:.3f}** |
| 21 | **role_perp → mode** | **{highlights['L21']['role_perp_to_mode']:.3f}** |
| 21 | mode → mode | {highlights['L21']['mode_to_mode']:.3f} |
| 21 | mode → role | {highlights['L21']['mode_to_role']:.3f} |
| 21 | mode_perp → mode | {highlights['L21']['mode_perp_to_mode']:.3f} |
| 21 | mode_perp → role | {highlights['L21']['mode_perp_to_role']:.3f} |
| 22 | role → role | {highlights['L22']['role_to_role']:.3f} |
| 22 | role → mode | {highlights['L22']['role_to_mode']:.3f} |
| 22 | role_perp → role | {highlights['L22']['role_perp_to_role']:.3f} |
| 22 | role_perp → mode | {highlights['L22']['role_perp_to_mode']:.3f} |
| 22 | mode → mode | {highlights['L22']['mode_to_mode']:.3f} |
| 22 | mode → role | {highlights['L22']['mode_to_role']:.3f} |
| 22 | mode_perp → mode | {highlights['L22']['mode_perp_to_mode']:.3f} |
| 22 | mode_perp → role | {highlights['L22']['mode_perp_to_role']:.3f} |

## Cosines (ambient)

| Pair | L21 | L22 |
| --- | ---: | ---: |
| role vs mode (before) | {highlights['L21']['cos_before']:.3f} | {highlights['L22']['cos_before']:.3f} |
| role_perp vs mode (after) | {highlights['L21']['cos_after_role_perp_vs_mode']:.3f} | {highlights['L22']['cos_after_role_perp_vs_mode']:.3f} |

## Read

- If **role_perp → role** stays high while **role_perp → mode** drops, deception geometry is not reducible to mode.
- Cosine after residualize should be ≈ 0 by construction (role_perp ⟂ mode).
- Directions for causal smoke: `results/transfer_stable/directions/` (see `meta.json`).

## Artifacts

| File | Path |
| --- | --- |
| This writeup | `results/transfer_stable/RESIDUALIZE_MODE.md` |
| Raw JSON | `results/transfer_stable/residualize_mode.json` |
| Directions | `results/transfer_stable/directions/` |
"""
    (out / "RESIDUALIZE_MODE.md").write_text(md)
    print(f"\n[claim] {one_line}", flush=True)

    if not args.skip_export:
        export_directions(out, layer_pack, pca_pack, results)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
