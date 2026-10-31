#!/usr/bin/env python3
"""J-space separability: can a probe restricted to J-space tell the colluders apart?

Pre-registered design: docs/jspace/JSPACE_SEPARABILITY_DESIGN.md (read it first; the analysis
labels A1-A5 and the arm names below are defined there).

The probe pipeline is the repository's own (train_collusion_probe.py / eval_transfer_from_core.py,
which mirror probes/reproduce.py of the NARCBench repository). Every arm is a map from activations
to vectors in R^d; the pipeline is re-run from scratch on each arm's vectors.

Inputs: the paper layout (activations_gen.npz keyed layer_<L> + metadata_gen.json) for Core and
Transfer, and an unembedding directory holding unembed_fp16.npy [V, d] (+ unembed_row_norms_fp32.npy).

  python3 scripts/jspace_separability.py \
      --core-acts <dir> --transfer-acts <dir> --unembed <dir> --layers 26-30 \
      --ks 8,16,25,50,100 --out results/jspace/qwen3_32b --tag qwen3_32b \
      [--tokenizer tokenizer.json] [--n-boot 2000] [--n-perm 2000]

CPU only. Nothing here launches a pod.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import rankdata
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from eval_transfer_from_core import build_transfer_groups, enrich_transfer_meta  # noqa: E402
from train_collusion_probe import (  # noqa: E402
    build_groups,
    compute_scores,
    domain_of,
    load_acts,
    parse_layers,
    train_direction,
)
from transfer_stable_residualize import ambient_from_scaled, unit  # noqa: E402

DESIGN = "docs/jspace/JSPACE_SEPARABILITY_DESIGN.md"
FRAME_ARMS = ("J", "shuffled", "randrows")
FIXED_ARMS = ("randsub", "pca")
ARM_ORDER = [
    "full", "J", "C_J", "randsub", "C_randsub", "shuffled", "C_shuffled",
    "randrows", "C_randrows", "pca", "C_pca", "contrast",
]
METRICS = ("peak", "split", "asym")
PRIMARY_K = 25


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_split(acts_dir: Path, layers: list[int], *, transfer: bool) -> tuple[list[dict], dict[int, np.ndarray], dict]:
    meta, npz = load_acts(acts_dir)
    if transfer:
        meta = enrich_transfer_meta(meta)
    avail = sorted(int(k.split("_")[1]) for k in npz.files if k.startswith("layer_"))
    layers = [L for L in layers if L in avail]
    if not layers:
        raise SystemExit(f"no requested layer in {acts_dir}; available {avail}")
    n = len(meta)
    first = npz[f"layer_{layers[0]}"][:n]
    keep = np.abs(first).sum(axis=1) > 0
    acts = {L: np.ascontiguousarray(npz[f"layer_{L}"][:n][keep], dtype=np.float32) for L in layers}
    meta = [m for m, k in zip(meta, keep) if k]
    info = {
        "dir": str(acts_dir),
        "rows": int(keep.sum()),
        "zero_rows_dropped": int((~keep).sum()),
        "runs": len({m["run_id"] for m in meta}),
        "mode_phase_channel_role": {
            " / ".join(k): v for k, v in sorted(Counter((m["mode"], m["phase"], m["channel"], m["role"]) for m in meta).items())
        },
        "domains_or_families": sorted({domain_of(m.get("scenario_id", ""), m.get("scenario_domain", "") or "") for m in meta}),
    }
    return meta, acts, info


def has_discussion_groups(meta: list[dict], acts: np.ndarray) -> bool:
    groups = build_groups(meta, acts, phase="discussion", avg_rounds=True)
    modes = Counter(g["mode"] for g in groups.values())
    col = any(m["mode"] == "collusion" and m["phase"] == "discussion" and m["channel"] == "public" and m["role"] == "colluder" for m in meta)
    hon = any(m["mode"] == "collusion" and m["phase"] == "discussion" and m["channel"] == "public" and m["role"] == "honest" for m in meta)
    return len(groups) >= 8 and len(modes) == 2 and col and hon


# ---------------------------------------------------------------------------
# Frames and projections
# ---------------------------------------------------------------------------

def load_unembed(unembed_dir: Path, scale: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """W_U [V, d] (memmap) and its row norms. With `scale` (the final RMSNorm weight gamma, [d]), the dictionary is
    W_U diag(gamma): the directions the logit lens actually reads after the final norm (exploratory variant E1)."""
    W = np.load(unembed_dir / "unembed_fp16.npy", mmap_mode="r")
    if scale is not None:
        scale = np.asarray(scale, dtype=np.float32).reshape(-1)
        assert scale.shape[0] == W.shape[1], (scale.shape, W.shape)
        Ws = np.lib.format.open_memmap(unembed_dir / "_scaled_tmp.npy", mode="w+", dtype=np.float16, shape=W.shape)
        for r0 in range(0, W.shape[0], 8192):
            Ws[r0:r0 + 8192] = (np.asarray(W[r0:r0 + 8192], dtype=np.float32) * scale[None, :]).astype(np.float16)
        Ws.flush()
        W = np.load(unembed_dir / "_scaled_tmp.npy", mmap_mode="r")
        norms = np.concatenate([np.linalg.norm(np.asarray(W[r0:r0 + 8192], dtype=np.float32), axis=1) for r0 in range(0, W.shape[0], 8192)]).astype(np.float32)
        return W, norms
    norms_p = unembed_dir / "unembed_row_norms_fp32.npy"
    if norms_p.exists():
        norms = np.load(norms_p).astype(np.float32)
    else:
        norms = np.linalg.norm(np.asarray(W, dtype=np.float32), axis=1).astype(np.float32)
    return W, norms


def topk_frames(acts: np.ndarray, W: np.ndarray, norms: np.ndarray, kmax: int, *, vocab_chunk: int = 8192) -> tuple[np.ndarray, np.ndarray]:
    """[n, kmax] indices (descending score) of each row's most aligned unit unembedding rows, and the scores."""
    n, d = acts.shape
    V = W.shape[0]
    A = np.ascontiguousarray(acts, dtype=np.float32)
    best_val = np.full((n, kmax), -np.inf, dtype=np.float32)
    best_idx = np.zeros((n, kmax), dtype=np.int32)
    for v0 in range(0, V, vocab_chunk):
        v1 = min(V, v0 + vocab_chunk)
        Wc = np.asarray(W[v0:v1], dtype=np.float32) / np.maximum(norms[v0:v1, None], 1e-12)
        sc = A @ Wc.T
        c = v1 - v0
        if c > kmax:
            part = np.argpartition(sc, -kmax, axis=1)[:, -kmax:]
            pv = np.take_along_axis(sc, part, axis=1)
        else:
            part = np.broadcast_to(np.arange(c, dtype=np.int32), (n, c)).copy()
            pv = sc
        cand_v = np.concatenate([best_val, pv], axis=1)
        cand_i = np.concatenate([best_idx, (part + v0).astype(np.int32)], axis=1)
        keep = np.argpartition(cand_v, -kmax, axis=1)[:, -kmax:]
        best_val = np.take_along_axis(cand_v, keep, axis=1)
        best_idx = np.take_along_axis(cand_i, keep, axis=1)
    order = np.argsort(-best_val, axis=1, kind="stable")
    return np.take_along_axis(best_idx, order, axis=1), np.take_along_axis(best_val, order, axis=1)


def nested_frame_projections(acts: np.ndarray, idx: np.ndarray, W: np.ndarray, ks: list[int]) -> dict[int, np.ndarray]:
    """h_J(k) = Q_k Q_k^T h for every k in ks, where Q_k spans the first k rows of W[idx[i]].

    One Householder QR per row at kmax: its leading k columns span the leading k frame rows, so the
    k-nested projections come from one decomposition (identical to project_j_complement.project_one
    applied to each prefix, up to floating point)."""
    n, d = acts.shape
    kmax = idx.shape[1]
    assert max(ks) <= kmax
    out = {k: np.zeros((n, d), dtype=np.float32) for k in ks}
    for i in range(n):
        rows = np.asarray(W[idx[i]], dtype=np.float64)  # [kmax, d], descending score order
        Q, R = np.linalg.qr(rows.T, mode="reduced")  # Q [d, kmax]
        diag = np.abs(np.diag(R))
        coef = Q.T @ acts[i].astype(np.float64)
        coef[diag <= 1e-8 * max(float(diag.max()), 1e-12)] = 0.0  # dependent rows add nothing
        for k in ks:
            out[k][i] = (Q[:, :k] @ coef[:k]).astype(np.float32)
    return out


def haar_basis(d: int, kmax: int, rng: np.random.RandomState) -> np.ndarray:
    Q, _ = np.linalg.qr(rng.randn(d, kmax), mode="reduced")
    return Q.astype(np.float64)


def fixed_projection(acts: np.ndarray, Q: np.ndarray, k: int) -> np.ndarray:
    Qk = Q[:, :k].astype(np.float32)
    return ((acts @ Qk) @ Qk.T).astype(np.float32)


def core_discussion_rows(meta: list[dict]) -> list[int]:
    return [i for i, m in enumerate(meta) if m.get("phase") == "discussion" and m.get("channel") == "public"]


def pca_basis(acts_rows: np.ndarray, kmax: int, seed: int) -> np.ndarray:
    k = min(kmax, acts_rows.shape[0] - 1, acts_rows.shape[1])
    pca = PCA(n_components=k, random_state=seed).fit(acts_rows.astype(np.float64))
    return pca.components_.T  # [d, k], orthonormal columns


def contrast_basis(meta: list[dict], acts: np.ndarray, kmax: int, seed: int) -> tuple[np.ndarray, int]:
    """PCA of per-episode (mean colluder - mean honest) discussion vectors over Core collusion episodes."""
    groups = build_groups(meta, acts, phase="discussion", avg_rounds=True)
    deltas = []
    for rid, g in sorted(groups.items()):
        if g["mode"] != "collusion":
            continue
        col = [g["agents"][a] for a in g["agents"] if g["roles"][a] == "colluder"]
        hon = [g["agents"][a] for a in g["agents"] if g["roles"][a] == "honest"]
        if col and hon:
            deltas.append(np.mean(col, axis=0) - np.mean(hon, axis=0))
    D = np.stack(deltas).astype(np.float64)
    k = min(kmax, D.shape[0] - 1)
    pca = PCA(n_components=k, random_state=seed).fit(D)
    return pca.components_.T, k


def fve(arm: np.ndarray, h: np.ndarray) -> float:
    num = np.einsum("ij,ij->i", arm, arm)
    den = np.einsum("ij,ij->i", h, h) + 1e-12
    return float(np.mean(num / den))


# ---------------------------------------------------------------------------
# Probe evaluations (per-episode scores are kept)
# ---------------------------------------------------------------------------

def asym_probabilities(tr_scores: dict, tr_y: np.ndarray, te_scores: dict, seed: int) -> np.ndarray:
    fs = StandardScaler()
    clf = LogisticRegression(max_iter=2000, C=0.01, random_state=seed)
    clf.fit(fs.fit_transform(tr_scores["asymmetry_probe"]), tr_y)
    return clf.predict_proba(fs.transform(te_scores["asymmetry_probe"]))[:, 1]


def labels_of(groups: dict, rids: list[str]) -> np.ndarray:
    return np.array([1 if groups[r]["mode"] == "collusion" else 0 for r in rids], dtype=int)


def aurocs_from(y: np.ndarray, peak: np.ndarray, split: np.ndarray, asym: np.ndarray) -> dict[str, float]:
    if len(np.unique(y)) < 2:
        return {m: float("nan") for m in METRICS} | {"primary": float("nan")}
    out = {
        "peak": float(roc_auc_score(y, peak)),
        "split": float(roc_auc_score(y, split)),
        "asym": float(roc_auc_score(y, asym)),
    }
    out["primary"] = float(np.mean([out[m] for m in METRICS]))
    return out


def per_family(y: np.ndarray, fams: list[str], peak: np.ndarray, split: np.ndarray) -> dict[str, dict]:
    out = {}
    for f in sorted(set(fams)):
        idx = [i for i, g in enumerate(fams) if g == f]
        yf = y[idx]
        if len(np.unique(yf)) < 2 or len(idx) < 4:
            continue
        out[f] = {"peak": float(roc_auc_score(yf, peak[idx])), "split": float(roc_auc_score(yf, split[idx])), "n": len(idx)}
    return out


def transfer_eval(core_meta, core_arr, xfer_meta, xfer_arr, *, seed: int, xfer_phase: str = "final") -> dict[str, Any]:
    """A2 / A4b: fit on every Core discussion turn, score the Transfer episodes once (eval_transfer_from_core protocol)."""
    scaler, w = train_direction(core_meta, core_arr, exclude_domain=None, phase="discussion", seed=seed)
    core_groups = build_groups(core_meta, core_arr, phase="discussion", avg_rounds=True)
    core_rids = sorted(core_groups)
    core_y = labels_of(core_groups, core_rids)
    core_scores = compute_scores(core_groups, core_rids, scaler, w)
    xfer_groups = build_transfer_groups(xfer_meta, xfer_arr, phase=xfer_phase)
    rids = sorted(xfer_groups)
    y = labels_of(xfer_groups, rids)
    sc = compute_scores(xfer_groups, rids, scaler, w)
    asym = asym_probabilities(core_scores, core_y, sc, seed)
    fams = [xfer_groups[r]["domain"] for r in rids]
    out = aurocs_from(y, sc["peak_suspicion"], sc["deception_split"], asym)
    out["per_family"] = per_family(y, fams, sc["peak_suspicion"], sc["deception_split"])
    out["n_runs"] = len(rids)
    out["_episodes"] = {
        "rids": rids, "y": y.tolist(), "family": fams,
        "peak": sc["peak_suspicion"].tolist(), "split": sc["deception_split"].tolist(), "asym": asym.tolist(),
    }
    out["_direction"] = (scaler, w)
    return out


def core_ldo(core_meta, core_arr, *, seed: int, eval_phase: str = "discussion") -> dict[str, Any]:
    """A1 (eval_phase='discussion', = train_collusion_probe.run_cv) and A4a (eval_phase='final').

    Direction and Asymmetry weights are fit on the training domains' discussion turns / episodes;
    the held-out domain's episodes are scored from their eval_phase vectors."""
    disc_groups = build_groups(core_meta, core_arr, phase="discussion", avg_rounds=True)
    eval_groups = disc_groups if eval_phase == "discussion" else build_groups(core_meta, core_arr, phase=eval_phase, avg_rounds=False)
    rids = sorted(disc_groups)
    domains = sorted({disc_groups[r]["domain"] for r in rids if disc_groups[r]["domain"]})
    folds: dict[str, dict] = {}
    pooled = {"rids": [], "y": [], "family": [], "peak": [], "split": [], "asym": []}
    for hold in domains:
        tr = [r for r in rids if disc_groups[r]["domain"] != hold]
        te = [r for r in rids if disc_groups[r]["domain"] == hold and r in eval_groups]
        tr_y, te_y = labels_of(disc_groups, tr), labels_of(disc_groups, te)
        if len(np.unique(tr_y)) < 2 or len(np.unique(te_y)) < 2:
            continue
        try:
            scaler, w = train_direction(core_meta, core_arr, exclude_domain=hold, phase="discussion", seed=seed)
        except RuntimeError:
            continue
        tr_sc = compute_scores(disc_groups, tr, scaler, w)
        te_sc = compute_scores(eval_groups, te, scaler, w)
        asym = asym_probabilities(tr_sc, tr_y, te_sc, seed)
        folds[hold] = aurocs_from(te_y, te_sc["peak_suspicion"], te_sc["deception_split"], asym) | {"n": len(te)}
        pooled["rids"] += te
        pooled["y"] += te_y.tolist()
        pooled["family"] += [hold] * len(te)
        pooled["peak"] += te_sc["peak_suspicion"].tolist()
        pooled["split"] += te_sc["deception_split"].tolist()
        pooled["asym"] += asym.tolist()
    out = {m: float(np.mean([f[m] for f in folds.values()])) for m in METRICS}
    out["primary"] = float(np.mean([f["primary"] for f in folds.values()]))
    out["fold_sd_primary"] = float(np.std([f["primary"] for f in folds.values()]))
    out["folds"] = folds
    out["n_folds"] = len(folds)
    out["_episodes"] = pooled
    return out


def transfer_lofo(xfer_meta, xfer_arr, *, seed: int, phase: str = "final") -> dict[str, Any]:
    """A3: within-Transfer leave-one-family-out, direction fit on the other families' final turns."""
    groups = build_transfer_groups(xfer_meta, xfer_arr, phase=phase)
    rids = sorted(groups)
    fams = sorted({groups[r]["domain"] for r in rids})
    folds: dict[str, dict] = {}
    pooled = {"rids": [], "y": [], "family": [], "peak": [], "split": [], "asym": []}
    for hold in fams:
        tr = [r for r in rids if groups[r]["domain"] != hold]
        te = [r for r in rids if groups[r]["domain"] == hold]
        tr_y, te_y = labels_of(groups, tr), labels_of(groups, te)
        if len(np.unique(tr_y)) < 2 or len(np.unique(te_y)) < 2:
            continue
        try:
            scaler, w = train_direction(xfer_meta, xfer_arr, exclude_domain=hold, phase=phase, seed=seed)
        except RuntimeError:
            continue
        tr_sc = compute_scores(groups, tr, scaler, w)
        te_sc = compute_scores(groups, te, scaler, w)
        asym = asym_probabilities(tr_sc, tr_y, te_sc, seed)
        folds[hold] = aurocs_from(te_y, te_sc["peak_suspicion"], te_sc["deception_split"], asym) | {"n": len(te)}
        pooled["rids"] += te
        pooled["y"] += te_y.tolist()
        pooled["family"] += [hold] * len(te)
        pooled["peak"] += te_sc["peak_suspicion"].tolist()
        pooled["split"] += te_sc["deception_split"].tolist()
        pooled["asym"] += asym.tolist()
    y = np.array(pooled["y"])
    out = aurocs_from(y, np.array(pooled["peak"]), np.array(pooled["split"]), np.array(pooled["asym"]))
    out["fold_mean_primary"] = float(np.mean([f["primary"] for f in folds.values()])) if folds else float("nan")
    out["folds"] = folds
    out["n_folds"] = len(folds)
    out["_episodes"] = pooled
    return out


# ---------------------------------------------------------------------------
# Inference on stored per-episode scores
# ---------------------------------------------------------------------------

def strata(y: np.ndarray, fams: list[str]) -> list[np.ndarray]:
    keys = defaultdict(list)
    for i, (yy, f) in enumerate(zip(y, fams)):
        keys[(int(yy), f)].append(i)
    return [np.array(v) for v in keys.values()]


def bootstrap_indices(y: np.ndarray, fams: list[str], B: int, rng: np.random.RandomState) -> list[np.ndarray]:
    st = strata(y, fams)
    out = []
    for _ in range(B):
        out.append(np.concatenate([rng.choice(s, len(s), replace=True) for s in st]))
    return out


def fast_auc(y: np.ndarray, s: np.ndarray) -> float:
    """Mann-Whitney AUROC with average ranks for ties; equals sklearn's roc_auc_score."""
    n1 = int(y.sum())
    n2 = len(y) - n1
    if n1 == 0 or n2 == 0:
        return float("nan")
    r = rankdata(s)
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n2))


def primary_on(ep: dict, idx: np.ndarray) -> float:
    y = np.asarray(ep["y"])[idx]
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(np.mean([fast_auc(y, np.asarray(ep[m])[idx]) for m in METRICS]))


def bootstrap_band(ep_by_layer: dict[int, dict], boots: list[np.ndarray]) -> np.ndarray:
    """Band-mean primary on each resample; the episode sets are identical across layers."""
    vals = np.zeros(len(boots))
    for b, idx in enumerate(boots):
        vals[b] = np.mean([primary_on(ep, idx) for ep in ep_by_layer.values()])
    return vals


def permuted_labels(y: np.ndarray, fams: list[str], B: int, rng: np.random.RandomState) -> np.ndarray:
    """[B, n] label matrix, each row a permutation of y within family (class counts preserved per family)."""
    groups = defaultdict(list)
    for i, f in enumerate(fams):
        groups[f].append(i)
    Y = np.tile(y, (B, 1))
    for g in groups.values():
        g = np.array(g)
        for b in range(B):
            Y[b, g] = y[rng.permutation(g)]
    return Y


def permutation_pvalue(ep_by_layer: dict[int, dict], Y: np.ndarray) -> float:
    """One-sided p for band-mean primary > chance, labels permuted within family (rows of Y).

    The direction was fit without the evaluated labels, so the per-episode scores are fixed under
    the null and AUROC = (sum of positive ranks - n1(n1+1)/2) / (n1 n2) vectorizes over permutations."""
    ep0 = next(iter(ep_by_layer.values()))
    y = np.asarray(ep0["y"])
    n1 = int(y.sum())
    n2 = len(y) - n1
    obs = np.mean([primary_on(ep, np.arange(len(y))) for ep in ep_by_layer.values()])
    null = np.zeros(Y.shape[0])
    for ep in ep_by_layer.values():
        for m in METRICS:
            r = rankdata(np.asarray(ep[m]))
            null += (Y @ r - n1 * (n1 + 1) / 2) / (n1 * n2)
    null /= len(ep_by_layer) * len(METRICS)
    return float((1 + np.sum(null >= obs - 1e-12)) / (Y.shape[0] + 1))


def ci(vals: np.ndarray) -> list[float]:
    v = vals[np.isfinite(vals)]
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] if len(v) else [float("nan")] * 2


# ---------------------------------------------------------------------------
# Geometry (A5)
# ---------------------------------------------------------------------------

def decode_tokens(tokenizer_path: Path | None, ids: list[int]) -> list[str]:
    if tokenizer_path is None or not tokenizer_path.exists():
        return [str(i) for i in ids]
    try:
        from tokenizers import Tokenizer
        tok = Tokenizer.from_file(str(tokenizer_path))
        return [tok.decode([int(i)]) for i in ids]
    except Exception:
        return [str(i) for i in ids]


def geometry(core_meta, core_h, core_idx, W, directions: dict[str, tuple], k: int, rng, tokenizer_path, n_sample=400) -> dict[str, Any]:
    d = core_h.shape[1]
    w_full = ambient_from_scaled(directions["full"][1], directions["full"][0])
    rows_disc = core_discussion_rows(core_meta)
    pick = rng.choice(rows_disc, min(n_sample, len(rows_disc)), replace=False)
    rand_dir = unit(rng.randn(d))
    share_w, share_r = [], []
    for i in pick:
        Q, _ = np.linalg.qr(np.asarray(W[core_idx[i, :k]], dtype=np.float64).T, mode="reduced")
        share_w.append(float(np.sum((Q.T @ w_full) ** 2)))
        share_r.append(float(np.sum((Q.T @ rand_dir) ** 2)))
    out = {
        "k": k,
        "share_of_full_direction_in_frames": {"median": float(np.median(share_w)), "mean": float(np.mean(share_w)), "max": float(np.max(share_w))},
        "share_of_random_direction_in_frames": {"median": float(np.median(share_r)), "mean": float(np.mean(share_r))},
        "k_over_d": k / d,
        "cos_full_vs": {},
    }
    for name, (sc, w) in directions.items():
        if name == "full":
            continue
        out["cos_full_vs"][name] = float(abs(ambient_from_scaled(w, sc) @ w_full))
    # frame tokens by role (Core discussion, collusion episodes)
    cnt = {"colluder": Counter(), "honest": Counter()}
    for i in rows_disc:
        m = core_meta[i]
        if m.get("mode") != "collusion":
            continue
        cnt[m["role"]].update(core_idx[i, :k].tolist())
    top = {}
    for role, c in cnt.items():
        ids = [t for t, _ in c.most_common(25)]
        toks = decode_tokens(tokenizer_path, ids)
        top[role] = [{"id": int(t), "token": s, "share_of_frames": round(c[t] / max(1, sum(1 for i in rows_disc if core_meta[i].get("mode") == "collusion" and core_meta[i]["role"] == role)), 3)} for t, s in zip(ids, toks)]
    col_set = set(t for t, _ in cnt["colluder"].most_common(100))
    hon_set = set(t for t, _ in cnt["honest"].most_common(100))
    out["frame_tokens_top25_by_role"] = top
    out["jaccard_top100_frame_tokens_colluder_vs_honest"] = float(len(col_set & hon_set) / max(1, len(col_set | hon_set)))
    return out


# ---------------------------------------------------------------------------
# Readings (section 6 of the design)
# ---------------------------------------------------------------------------

def readings(summary: dict) -> dict[str, Any]:
    k = str(PRIMARY_K)
    out: dict[str, Any] = {}
    a1 = summary.get("A1_core_ldo", {}).get("band_mean", {}).get(k, {})
    if "full" in a1 and "J" in a1:
        diff = a1["full"]["primary"] - a1["J"]["primary"]
        out["R1_in_distribution"] = {
            "J_primary": a1["J"]["primary"], "full_minus_J": diff,
            "reading": "J-space separates colluders in distribution" if a1["J"]["primary"] >= 0.90 and diff < 0.05
            else "in distribution J-space is a weaker carrier than the full residual",
        }
    inf = summary.get("A2_inference", {}).get(k, {})
    arms = summary.get("A2_transfer", {}).get("band_mean", {}).get(k, {})
    if inf and arms:
        dif = inf["differences"]
        c_j = dif["C_J-J"]; j_r = dif["J-randsub"]; c_f = dif["C_J-full"]; j_s = dif["J-shuffled"]
        cond_i = c_j["ci"][0] > 0 and c_j["est"] >= 0.15
        cond_ii = j_r["ci"][0] <= 0
        cond_iii = c_f["ci"][0] <= 0 <= c_f["ci"][1]
        if cond_i and cond_ii and cond_iii:
            txt = "the transferable signal is not in J-space"
        elif cond_i and not cond_ii:
            txt = "J-space carries a part of the transferable signal; the complement carries most of it"
        elif not cond_i:
            txt = "J-space and its complement are not clearly separated on Transfer (condition i fails)"
        else:
            txt = "mixed: C minus J clear, J vs random unclear, complement vs full differs"
        out["R2_transfer"] = {
            "J": arms["J"]["primary"], "C_J": arms["C_J"]["primary"], "randsub": arms["randsub"]["primary"],
            "shuffled": arms["shuffled"]["primary"], "full": arms["full"]["primary"],
            "conditions": {"i_C_minus_J": cond_i, "ii_J_not_above_random": cond_ii, "iii_C_equals_full": cond_iii},
            "J_minus_shuffled": j_s, "p_perm_J": inf["arms"]["J"]["p_perm"], "p_perm_randsub": inf["arms"]["randsub"]["p_perm"],
            "reading": txt,
        }
        cr = dif["C_randsub-full"]
        # Amendment of 31 Oct 2026 (design section 6, item 3): equality means the interval contains zero OR |difference| < 0.02.
        out["R3_complement_uninformative"] = {
            "C_randsub_minus_full": cr,
            "reading": "complement-of-random equals full (as expected); 'complement equals full' says nothing about J-space by itself"
            if (cr["ci"][0] <= 0 <= cr["ci"][1]) or abs(cr["est"]) < 0.02 else "complement of a random subspace differs from full (unexpected; inspect)",
        }
    a3 = summary.get("A3_transfer_lofo", {}).get("band_mean", {}).get(k, {})
    a3inf = summary.get("A3_inference", {}).get(k, {})
    if a3 and arms:
        jw = a3["J"]["primary"]; jx = arms["J"]["primary"]
        p = a3inf.get("arms", {}).get("J", {}).get("p_perm")
        if jw >= 0.80 and jx <= 0.65:
            txt = "role information is present in J-space on Transfer, but the Core-fit J direction does not transfer"
        elif p is not None and p > 0.05:
            txt = "role information is absent from J-space on Transfer finals (within-Transfer J probe at its permutation null)"
        else:
            txt = "intermediate: within-Transfer J probe above chance but below 0.80, or Core-fit J above 0.65"
        out["R4_within_transfer"] = {"J_within_transfer": jw, "J_core_to_transfer": jx, "p_perm_within": p,
                                     "C_J_within_transfer": a3["C_J"]["primary"], "full_within_transfer": a3["full"]["primary"], "reading": txt}
    a4a = summary.get("A4a_core_disc_to_final", {}).get("band_mean", {}).get(k, {})
    a4b = summary.get("A4b_core_to_transfer_discussion", {}).get("band_mean", {}).get(k, {})
    if a4a and arms and "J" in a4a:
        out["R5_phase_vs_family"] = {
            "J_core_disc_to_core_final": a4a["J"]["primary"], "full_core_disc_to_core_final": a4a["full"]["primary"],
            "J_core_disc_to_transfer_final": arms["J"]["primary"],
            "J_core_disc_to_transfer_discussion": a4b["J"]["primary"] if a4b and "J" in a4b else None,
            "full_core_disc_to_transfer_discussion": a4b["full"]["primary"] if a4b and "full" in a4b else None,
        }
        ja, jx = a4a["J"]["primary"], arms["J"]["primary"]
        if a1 and "J" in a1:
            drop_phase = a1["J"]["primary"] - ja
            drop_total = a1["J"]["primary"] - jx
            out["R5_phase_vs_family"]["drop_phase_only"] = drop_phase
            out["R5_phase_vs_family"]["drop_total"] = drop_total
            out["R5_phase_vs_family"]["reading"] = (
                "the J drop is mostly the phase change (discussion to final)" if drop_total > 0.05 and drop_phase >= 0.6 * drop_total
                else "the J drop is mostly the family change" if drop_total > 0.05 and drop_phase <= 0.4 * drop_total
                else "the J drop is shared between phase and family" if drop_total > 0.05 else "no large J drop to decompose"
            )
    if arms:
        small = {kk: v for kk, v in summary.get("A2_transfer", {}).get("band_mean", {}).items() if int(kk) <= 16}
        best_contrast = max((v["contrast"]["primary"] for v in small.values() if "contrast" in v), default=float("nan"))
        best_pca = max((v["pca"]["primary"] for v in small.values() if "pca" in v), default=float("nan"))
        j_small = max((v["J"]["primary"] for v in small.values() if "J" in v), default=float("nan"))
        out["R6_dimension"] = {
            "best_contrast_k_le_16": best_contrast, "best_pca_k_le_16": best_pca, "best_J_k_le_16": j_small, "full": arms["full"]["primary"],
            "reading": "dimension is not the explanation: a supervised or PCA subspace of at most 16 dimensions reaches the full transfer AUROC while J does not"
            if (best_contrast >= arms["full"]["primary"] - 0.03 or best_pca >= arms["full"]["primary"] - 0.03) and j_small < arms["full"]["primary"] - 0.10
            else "small supervised/PCA subspaces do not reach the full probe either; dimension may matter",
        }
    return out


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def f3(x) -> str:
    try:
        return f"{float(x):.3f}" if x is not None and np.isfinite(float(x)) else "—"
    except (TypeError, ValueError):
        return "—"


def write_results_md(summary: dict, out: Path) -> None:
    tag = summary["tag"]
    L = [f"# J-space separability — {tag}", "",
         f"Design: `{DESIGN}`. Script: `scripts/jspace_separability.py`. Written {summary['written']}. "
         f"Layers {summary['layers']} (centre {summary['centre_layer']}); frame sizes k = {summary['ks']}; "
         f"bootstrap B = {summary['n_boot']}, permutations B = {summary['n_perm']}; seed {summary['seed']}.", "",
         "## Data", "",
         "| Split | Rows used | Zero rows dropped | Runs | Domains / families |", "|---|---:|---:|---:|---|"]
    for name in ("core", "transfer"):
        i = summary["data"][name]
        L.append(f"| {name} | {i['rows']} | {i['zero_rows_dropped']} | {i['runs']} | {len(i['domains_or_families'])}: {', '.join(i['domains_or_families'][:8])}{'…' if len(i['domains_or_families']) > 8 else ''} |")
    L += ["", "## A2. Core discussion → Transfer final, band mean (primary AUROC)", "",
          "| arm | " + " | ".join(f"k={k}" for k in summary["ks"]) + " |", "|---|" + "---:|" * len(summary["ks"])]
    bm = summary["A2_transfer"]["band_mean"]
    for arm in ARM_ORDER:
        cells = []
        for k in summary["ks"]:
            v = bm.get(str(k), {}).get(arm)
            cells.append(f3(v["primary"]) if v else "—")
        if arm == "full":
            cells = [f3(bm[str(summary['ks'][0])]["full"]["primary"])] + ["(same)"] * (len(summary["ks"]) - 1)
        L.append(f"| {arm} | " + " | ".join(cells) + " |")
    k = str(PRIMARY_K)
    if k in summary.get("A2_inference", {}):
        inf = summary["A2_inference"][k]
        L += ["", f"### A2 inference at k = {PRIMARY_K} (band mean; 95% bootstrap intervals over episodes; permutation p for primary > 0.5)", "",
              "| arm | primary | 95% CI | mean FVE core | mean FVE transfer | p (perm) |", "|---|---:|---|---:|---:|---:|"]
        for arm in ARM_ORDER:
            a = inf["arms"].get(arm)
            if not a:
                continue
            fv = bm[k][arm]
            L.append(f"| {arm} | {f3(a['est'])} | [{f3(a['ci'][0])}, {f3(a['ci'][1])}] | {f3(fv.get('fve_core'))} | {f3(fv.get('fve_xfer'))} | {a['p_perm']:.4f} |")
        L += ["", "| difference | estimate | 95% CI |", "|---|---:|---|"]
        for name, dct in inf["differences"].items():
            L.append(f"| {name} | {dct['est']:+.3f} | [{dct['ci'][0]:+.3f}, {dct['ci'][1]:+.3f}] |")
        L += ["", "Per layer, J vs complement vs full at k = 25:", "", "| layer | full | J | C_J | randsub | shuffled | pca | contrast |", "|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for Lr in summary["layers"]:
            row = summary["A2_transfer"]["per_layer"][str(Lr)][k]
            L.append(f"| {Lr} | " + " | ".join(f3(row[a]["primary"]) for a in ("full", "J", "C_J", "randsub", "shuffled", "pca", "contrast")) + " |")
        pf = summary["A2_transfer"]["per_layer"][str(summary["centre_layer"])][k]
        fams = sorted(set().union(*[set(pf[a].get("per_family", {})) for a in ("full", "J", "C_J")]))
        if fams:
            L += ["", f"Per family at the centre layer L{summary['centre_layer']}, k = 25 (Split AUROC):", "",
                  "| family | full | J | C_J | randsub |", "|---|---:|---:|---:|---:|"]
            for f in fams:
                L.append(f"| {f} | " + " | ".join(f3(pf[a].get("per_family", {}).get(f, {}).get("split")) for a in ("full", "J", "C_J", "randsub")) + " |")
    for key, title in (("A1_core_ldo", "A1. Core leave-one-domain-out (fit and eval on discussion turns)"),
                       ("A3_transfer_lofo", "A3. Within-Transfer leave-one-family-out (fit and eval on final turns; pooled held-out AUROC)"),
                       ("A4a_core_disc_to_final", "A4a. Core discussion → Core final, leave-one-domain-out (phase changes, family does not)"),
                       ("A4b_core_to_transfer_discussion", "A4b. Core discussion → Transfer discussion (family changes, phase does not)")):
        sec = summary.get(key)
        if not sec or not sec.get("band_mean"):
            L += ["", f"## {title}", "", "not available for this model (see `data` above)."]
            continue
        ks_here = sorted(sec["band_mean"], key=int)
        L += ["", f"## {title}", "", "Band mean over layers that were run (per layer in `summary.json`); primary AUROC.", "",
              "| arm | " + " | ".join(f"k={kk}" for kk in ks_here) + " |", "|---|" + "---:|" * len(ks_here)]
        for arm in ARM_ORDER:
            cells = [f3(sec["band_mean"][kk][arm]["primary"]) if arm in sec["band_mean"][kk] else "—" for kk in ks_here]
            L.append(f"| {arm} | " + " | ".join(cells) + " |")
        infk = summary.get(key.replace("_transfer_lofo", "_inference").replace("A1_core_ldo", "A1_inference"), {}).get(k) if key in ("A3_transfer_lofo",) else None
        if key == "A3_transfer_lofo" and summary.get("A3_inference", {}).get(k):
            inf3 = summary["A3_inference"][k]
            L += ["", f"Inference at k = {PRIMARY_K} (pooled held-out episodes, bootstrap over episodes, permutation within family):", "",
                  "| arm | primary | 95% CI | p (perm) |", "|---|---:|---|---:|"]
            for arm in ARM_ORDER:
                a = inf3["arms"].get(arm)
                if a:
                    L.append(f"| {arm} | {f3(a['est'])} | [{f3(a['ci'][0])}, {f3(a['ci'][1])}] | {a['p_perm']:.4f} |")
            L += ["", "| difference | estimate | 95% CI |", "|---|---:|---|"]
            for name, dct in inf3["differences"].items():
                L.append(f"| {name} | {dct['est']:+.3f} | [{dct['ci'][0]:+.3f}, {dct['ci'][1]:+.3f}] |")
        if key == "A1_core_ldo" and sec.get("fold_paired"):
            L += ["", f"Fold-paired differences at k = {PRIMARY_K}, centre layer (full minus arm; folds = held-out domains):", "",
                  "| arm | mean diff | folds where full > arm | folds | sign-test p |", "|---|---:|---:|---:|---:|"]
            for arm, v in sec["fold_paired"].items():
                L.append(f"| {arm} | {v['mean_diff']:+.3f} | {v['n_full_greater']} | {v['n_folds']} | {v['sign_p']:.3f} |")
    g = summary.get("A5_geometry")
    if g:
        L += ["", "## A5. Geometry (centre layer, k = 25)", "",
              f"Share of the full probe direction inside each activation's own frame: median {f3(g['share_of_full_direction_in_frames']['median'])}, "
              f"max {f3(g['share_of_full_direction_in_frames']['max'])}; a random direction: median {f3(g['share_of_random_direction_in_frames']['median'])}; k/d = {g['k_over_d']:.4f}.",
              "", "|cos| between the full probe direction and the direction each arm learns: " + ", ".join(f"{a} {f3(v)}" for a, v in g["cos_full_vs"].items()) + ".",
              "", f"Frame tokens shared between colluder and honest turns (Jaccard of the 100 most frequent): {f3(g['jaccard_top100_frame_tokens_colluder_vs_honest'])}.", "",
              "| role | 25 most frequent frame tokens (share of that role's discussion turns whose frame holds the token) |", "|---|---|"]
        for role, lst in g["frame_tokens_top25_by_role"].items():
            L.append(f"| {role} | " + ", ".join(f"`{t['token']!r}` {t['share_of_frames']:.2f}" for t in lst) + " |")
    L += ["", "## Readings (rules fixed in the design, section 6)", ""]
    for key, r in summary.get("readings", {}).items():
        L.append(f"- **{key}**: {r.get('reading', '')}")
        L.append("  " + "; ".join(f"{kk} = {f3(v) if isinstance(v, (int, float)) else v}" for kk, v in r.items() if kk != "reading" and not isinstance(v, (dict, list))))
    L += ["", "## Baseline reproduction", "",
          f"Full-activation Core LDO primary {f3(summary['A1_core_ldo'].get('band_mean', {}).get(k, {}).get('full', {}).get('primary'))} "
          f"(paper: 1.00 for every model in distribution); full-activation Core → Transfer band mean {f3(bm[k]['full']['primary'])}.", ""]
    (out / "RESULTS.md").write_text("\n".join(L))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def strip_private(d):
    if isinstance(d, dict):
        return {k: strip_private(v) for k, v in d.items() if not str(k).startswith("_")}
    if isinstance(d, list):
        return [strip_private(v) for v in d]
    if isinstance(d, (np.floating, np.integer)):
        return d.item()
    return d


def band_mean(per_layer: dict[str, dict]) -> dict[str, dict]:
    """per_layer[L][k][arm] -> band[k][arm] with mean primary/peak/split/asym (+ mean FVEs)."""
    band: dict[str, dict] = {}
    layers = list(per_layer)
    for k in per_layer[layers[0]]:
        band[k] = {}
        for arm in per_layer[layers[0]][k]:
            vals = [per_layer[L][k][arm] for L in layers if arm in per_layer[L].get(k, {})]
            entry = {m: float(np.nanmean([v[m] for v in vals])) for m in METRICS + ("primary",)}
            for extra in ("fve_core", "fve_xfer"):
                if all(extra in v for v in vals):
                    entry[extra] = float(np.mean([v[extra] for v in vals]))
            entry["n_layers"] = len(vals)
            band[k][arm] = entry
    return band


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--core-acts", type=Path)
    ap.add_argument("--transfer-acts", type=Path)
    ap.add_argument("--unembed", type=Path, help="dir with unembed_fp16.npy (+ unembed_row_norms_fp32.npy)")
    ap.add_argument("--layers")
    ap.add_argument("--recompute-readings", action="store_true",
                    help="reload <out>/summary.json, recompute the readings (design section 6) and rewrite summary.json + RESULTS.md; no activations needed")
    ap.add_argument("--unembed-scale", type=Path, default=None,
                    help="exploratory E1: a [d] .npy (the final RMSNorm weight) that scales the unembedding columns before frames are chosen")
    ap.add_argument("--ks", default="8,16,25,50,100")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--tokenizer", type=Path, default=None, help="tokenizer.json for decoding frame tokens (optional)")
    ap.add_argument("--centre-layer", type=int, default=None)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--n-perm", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--vocab-chunk", type=int, default=8192)
    ap.add_argument("--ldo-all-k-all-layers", action="store_true", help="run the Core LDO k sweep at every layer (slow); default: k=25 everywhere, sweep at the centre layer")
    ap.add_argument("--skip-ldo", action="store_true")
    ap.add_argument("--skip-lofo", action="store_true")
    args = ap.parse_args()

    if args.recompute_readings:
        out = args.out if args.out.is_absolute() else ROOT / args.out
        summary = json.loads((out / "summary.json").read_text())
        summary["readings"] = readings(summary)
        summary["readings_recomputed"] = time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime())
        (out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
        write_results_md(summary, out)
        for key, r in summary["readings"].items():
            print(f"  {key}: {r.get('reading')}")
        return 0
    missing = [n for n in ("core_acts", "transfer_acts", "unembed", "layers") if getattr(args, n) is None]
    if missing:
        ap.error("missing " + ", ".join("--" + m.replace("_", "-") for m in missing))

    t_start = time.time()
    rng = np.random.RandomState(args.seed)
    layers = parse_layers(args.layers)
    ks = sorted({int(x) for x in args.ks.split(",") if x.strip()} | {PRIMARY_K})
    kmax = max(ks)
    out = args.out if args.out.is_absolute() else ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)

    core_meta, core_acts, core_info = load_split(args.core_acts if args.core_acts.is_absolute() else ROOT / args.core_acts, layers, transfer=False)
    xfer_meta, xfer_acts, xfer_info = load_split(args.transfer_acts if args.transfer_acts.is_absolute() else ROOT / args.transfer_acts, layers, transfer=True)
    layers = [L for L in layers if L in core_acts and L in xfer_acts]
    centre = args.centre_layer if args.centre_layer is not None else layers[len(layers) // 2]
    scale = np.load(args.unembed_scale) if args.unembed_scale else None
    W, norms = load_unembed(args.unembed if args.unembed.is_absolute() else ROOT / args.unembed, scale=scale)
    V, d = W.shape
    n_core, n_xfer = len(core_meta), len(xfer_meta)
    xfer_has_disc = has_discussion_groups(xfer_meta, xfer_acts[layers[0]])
    print(f"[{args.tag}] core rows {n_core} ({core_info['runs']} runs), transfer rows {n_xfer} ({xfer_info['runs']} runs), "
          f"d={d} V={V}, layers {layers}, centre {centre}, ks {ks}, transfer discussion groups: {xfer_has_disc}", flush=True)

    summary: dict[str, Any] = {
        "tag": args.tag, "design": DESIGN, "written": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()),
        "data": {"core": core_info, "transfer": xfer_info, "unembed": str(args.unembed), "V": int(V), "d": int(d),
                 "unembed_scale": str(args.unembed_scale) if args.unembed_scale else None},
        "layers": layers, "centre_layer": centre, "ks": ks, "seed": args.seed, "n_boot": args.n_boot, "n_perm": args.n_perm,
        "A1_core_ldo": {"per_layer": {}}, "A2_transfer": {"per_layer": {}}, "A3_transfer_lofo": {"per_layer": {}},
        "A4a_core_disc_to_final": {"per_layer": {}}, "A4b_core_to_transfer_discussion": {"per_layer": {}},
    }
    episodes: dict[str, dict] = {"A2": {}, "A3": {}, "A4b": {}}  # [analysis][layer][k][arm] -> per-episode scores
    directions_centre: dict[str, tuple] = {}
    core_idx_centre = None

    # fixed subspaces are shared across layers only in seed; they are refit per layer (PCA is layer-specific)
    for L in layers:
        tL = time.time()
        hc, hx = core_acts[L], xfer_acts[L]
        H = np.vstack([hc, hx])
        print(f"[L{L}] frames (top-{kmax} of {V} unit unembedding rows) …", flush=True)
        idx, _vals = topk_frames(H, W, norms, kmax, vocab_chunk=args.vocab_chunk)
        # shuffled frames: a different activation from the same split
        perm_c = rng.permutation(n_core)
        perm_x = rng.permutation(n_xfer)
        for arr, n in ((perm_c, n_core), (perm_x, n_xfer)):
            fixed = np.where(arr == np.arange(n))[0]
            if len(fixed) > 1:
                arr[fixed] = np.roll(arr[fixed], 1)
            elif len(fixed) == 1 and n > 1:
                j = (fixed[0] + 1) % n
                arr[fixed[0]], arr[j] = arr[j], arr[fixed[0]]
        idx_shuf = np.vstack([idx[:n_core][perm_c], idx[n_core:][perm_x]])
        idx_rand = np.stack([rng.choice(V, kmax, replace=False) for _ in range(len(H))]).astype(np.int32)
        print(f"[L{L}] projections …", flush=True)
        projJ = nested_frame_projections(H, idx, W, ks)
        projS = nested_frame_projections(H, idx_shuf, W, ks)
        projR = nested_frame_projections(H, idx_rand, W, ks)
        Q_rand = haar_basis(d, kmax, rng)
        Q_pca = pca_basis(hc[core_discussion_rows(core_meta)], kmax, args.seed)
        Q_con, k_con = contrast_basis(core_meta, hc, kmax, args.seed)
        if L == centre:
            core_idx_centre = idx[:n_core]
        fve_rows = {"core": slice(0, n_core), "xfer": slice(n_core, None)}

        for k in ks:
            tk = time.time()
            arms: dict[str, np.ndarray] = {"full": H}
            arms["J"] = projJ[k]; arms["C_J"] = H - projJ[k]
            arms["shuffled"] = projS[k]; arms["C_shuffled"] = H - projS[k]
            arms["randrows"] = projR[k]; arms["C_randrows"] = H - projR[k]
            arms["randsub"] = fixed_projection(H, Q_rand, k); arms["C_randsub"] = H - arms["randsub"]
            kp = min(k, Q_pca.shape[1])
            arms["pca"] = fixed_projection(H, Q_pca, kp); arms["C_pca"] = H - arms["pca"]
            arms["contrast"] = fixed_projection(H, Q_con, min(k, k_con))
            res_a2, res_a1, res_a3, res_a4a, res_a4b = {}, {}, {}, {}, {}
            run_ldo_here = (not args.skip_ldo) and (k == PRIMARY_K or L == centre or args.ldo_all_k_all_layers)
            for arm, A in arms.items():
                Ac, Ax = A[:n_core], A[n_core:]
                r = transfer_eval(core_meta, Ac, xfer_meta, Ax, seed=args.seed)
                r["fve_core"] = fve(Ac, hc); r["fve_xfer"] = fve(Ax, hx)
                if L == centre and k == PRIMARY_K:
                    directions_centre[arm] = r.pop("_direction")
                else:
                    r.pop("_direction")
                episodes["A2"].setdefault(str(L), {}).setdefault(str(k), {})[arm] = r.pop("_episodes")
                res_a2[arm] = r
                if run_ldo_here:
                    r1 = core_ldo(core_meta, Ac, seed=args.seed, eval_phase="discussion"); r1.pop("_episodes")
                    res_a1[arm] = r1
                    r4 = core_ldo(core_meta, Ac, seed=args.seed, eval_phase="final"); r4.pop("_episodes")
                    res_a4a[arm] = r4
                if not args.skip_lofo:
                    r3 = transfer_lofo(xfer_meta, Ax, seed=args.seed)
                    episodes["A3"].setdefault(str(L), {}).setdefault(str(k), {})[arm] = r3.pop("_episodes")
                    res_a3[arm] = r3
                if xfer_has_disc:
                    r4b = transfer_eval(core_meta, Ac, xfer_meta, Ax, seed=args.seed, xfer_phase="discussion"); r4b.pop("_direction")
                    episodes["A4b"].setdefault(str(L), {}).setdefault(str(k), {})[arm] = r4b.pop("_episodes")
                    res_a4b[arm] = r4b
            summary["A2_transfer"]["per_layer"].setdefault(str(L), {})[str(k)] = res_a2
            if res_a1:
                summary["A1_core_ldo"]["per_layer"].setdefault(str(L), {})[str(k)] = res_a1
                summary["A4a_core_disc_to_final"]["per_layer"].setdefault(str(L), {})[str(k)] = res_a4a
            if res_a3:
                summary["A3_transfer_lofo"]["per_layer"].setdefault(str(L), {})[str(k)] = res_a3
            if res_a4b:
                summary["A4b_core_to_transfer_discussion"]["per_layer"].setdefault(str(L), {})[str(k)] = res_a4b
            print(f"[L{L} k={k}] A2 primary: full {res_a2['full']['primary']:.3f} | J {res_a2['J']['primary']:.3f} | C_J {res_a2['C_J']['primary']:.3f} | "
                  f"randsub {res_a2['randsub']['primary']:.3f} | shuffled {res_a2['shuffled']['primary']:.3f} | pca {res_a2['pca']['primary']:.3f} | "
                  f"contrast {res_a2['contrast']['primary']:.3f}" + (f" || A1 J {res_a1['J']['primary']:.3f} C_J {res_a1['C_J']['primary']:.3f}" if res_a1 else "")
                  + (f" || A3 J {res_a3['J']['primary']:.3f} C_J {res_a3['C_J']['primary']:.3f}" if res_a3 else "") + f"  ({time.time() - tk:.0f}s)", flush=True)
        del projJ, projS, projR
        print(f"[L{L}] done in {time.time() - tL:.0f}s", flush=True)

    # band means
    for key in ("A1_core_ldo", "A2_transfer", "A3_transfer_lofo", "A4a_core_disc_to_final", "A4b_core_to_transfer_discussion"):
        per_layer = summary[key]["per_layer"]
        if not per_layer:
            continue
        if key in ("A1_core_ldo", "A4a_core_disc_to_final"):
            # k=25 at every layer that ran; other ks only where they ran (centre layer)
            band: dict[str, dict] = {}
            for L_, dk in per_layer.items():
                for k_, arms_ in dk.items():
                    for arm, v in arms_.items():
                        band.setdefault(k_, {}).setdefault(arm, []).append(v)
            summary[key]["band_mean"] = {
                k_: {arm: {m: float(np.mean([v[m] for v in vals])) for m in METRICS + ("primary",)} | {"n_layers": len(vals)} for arm, vals in arms_.items()}
                for k_, arms_ in band.items()
            }
        else:
            summary[key]["band_mean"] = band_mean(per_layer)

    # fold-paired A1 at the centre layer, k=25
    a1c = summary["A1_core_ldo"]["per_layer"].get(str(centre), {}).get(str(PRIMARY_K))
    if a1c:
        fp = {}
        full_f = a1c["full"]["folds"]
        for arm in ARM_ORDER:
            if arm == "full" or arm not in a1c:
                continue
            ds = [full_f[h]["primary"] - a1c[arm]["folds"][h]["primary"] for h in full_f if h in a1c[arm]["folds"]]
            n_pos = sum(x > 0 for x in ds); n_neg = sum(x < 0 for x in ds); n = n_pos + n_neg
            from math import comb
            p = min(1.0, 2 * sum(comb(n, i) for i in range(0, min(n_pos, n_neg) + 1)) / (2 ** n)) if n else 1.0
            fp[arm] = {"mean_diff": float(np.mean(ds)), "n_full_greater": int(n_pos), "n_folds": len(ds), "sign_p": float(p)}
        summary["A1_core_ldo"]["fold_paired"] = fp

    # inference on Transfer per-episode scores (A2, A3)
    def infer(block: dict, key_out: str, B_boot: int, B_perm: int):
        res = {}
        for k_ in sorted({k for L_ in block for k in block[L_]}, key=int):
            arms_here = {arm for L_ in block for arm in block[L_].get(k_, {})}
            ep_first = block[layers_s[0]][k_]["full"]
            y = np.asarray(ep_first["y"]); fams = ep_first["family"]
            boots = bootstrap_indices(y, fams, B_boot, np.random.RandomState(args.seed + 1))
            Yperm = permuted_labels(y, fams, B_perm, np.random.RandomState(args.seed + 2))
            dist = {}
            arms_out = {}
            for arm in ARM_ORDER:
                if arm not in arms_here:
                    continue
                ep_by_layer = {L_: block[L_][k_][arm] for L_ in layers_s if arm in block[L_].get(k_, {})}
                vals = bootstrap_band(ep_by_layer, boots)
                dist[arm] = vals
                est = float(np.mean([primary_on(ep, np.arange(len(y))) for ep in ep_by_layer.values()]))
                p = permutation_pvalue(ep_by_layer, Yperm)
                arms_out[arm] = {"est": est, "ci": ci(vals), "p_perm": p}
            diffs = {}
            for a, b in (("C_J", "J"), ("J", "randsub"), ("J", "shuffled"), ("J", "randrows"), ("J", "pca"), ("C_J", "full"), ("C_randsub", "full"), ("contrast", "J"), ("pca", "J")):
                if a in dist and b in dist:
                    diffs[f"{a}-{b}"] = {"est": arms_out[a]["est"] - arms_out[b]["est"], "ci": ci(dist[a] - dist[b])}
            res[k_] = {"arms": arms_out, "differences": diffs, "n_episodes": int(len(y))}
            print(f"[{key_out} k={k_}] " + " | ".join(f"{a} {v['est']:.3f} [{v['ci'][0]:.3f},{v['ci'][1]:.3f}] p={v['p_perm']:.3f}" for a, v in arms_out.items() if a in ("full", "J", "C_J", "randsub")), flush=True)
        summary[key_out] = res

    layers_s = [str(L) for L in layers]
    print("[inference] bootstrap + permutation on A2 …", flush=True)
    infer(episodes["A2"], "A2_inference", args.n_boot, args.n_perm)
    if episodes["A3"]:
        print("[inference] bootstrap + permutation on A3 …", flush=True)
        infer(episodes["A3"], "A3_inference", args.n_boot, args.n_perm)

    # geometry at the centre layer
    if core_idx_centre is not None and "full" in directions_centre:
        print("[geometry] …", flush=True)
        summary["A5_geometry"] = geometry(core_meta, core_acts[centre], core_idx_centre, W, directions_centre, PRIMARY_K,
                                          np.random.RandomState(args.seed + 3), args.tokenizer)

    summary["readings"] = readings(summary)
    summary["seconds"] = time.time() - t_start
    clean = strip_private(summary)
    (out / "summary.json").write_text(json.dumps(clean, indent=1) + "\n")
    (out / "episode_scores.json").write_text(json.dumps(episodes) + "\n")
    write_results_md(clean, out)
    print(f"[{args.tag}] wrote {out/'summary.json'}, {out/'RESULTS.md'}, {out/'episode_scores.json'} in {summary['seconds']:.0f}s", flush=True)
    for key, r in clean["readings"].items():
        print(f"  {key}: {r.get('reading')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
