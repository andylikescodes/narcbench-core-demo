#!/usr/bin/env python3
"""J-space decomposition of recorded activations with a Jacobian lens, the way Gurnee et al. (2026) define it.

Setting. h is a residual activation at layer l (the output of block l, where every recorded NARCBench activation was
read). J_l is the mean Jacobian of the final residual (output of the last block) with respect to h_l: here the published
lens for the model (Neuronpedia's fit with Anthropic's `jlens` package), or any file in that save format. The J-lens
readout is

    logits(h) = softcap( W_U . norm(J_l h) ),    norm(x) = x / rms(x) * s,   s = 1 + gamma (Gemma) or s = gamma

exactly as `jlens.hf.unembed` applies it. rms(.) is a positive scalar per activation and the softcap is monotone, so
logit_i is an increasing function of <v_i, h> with

    v_i = J_l^T (s * W_U[i])       (row i of W_U diag(s) J_l: the J-lens vector of token i, in layer-l residual space).

The ranking of tokens is the ranking of <v_i, h>, and the J-lens vectors are the dictionary over which the paper
decomposes activations.

Decomposition (paper, "Formalization" appendix and probe appendix). For each activation a greedy nonnegative sparse
pursuit picks k atoms S (k = 25 by default): at every step the atom with the largest positive correlation <v_i/|v_i|, r>
with the current residual r joins S, then the nonnegative coefficients are refitted on S (NNLS, `--method nnomp`) or
updated by one optimal gradient step clipped at zero (`--method gp`, gradient pursuit). `--method topk` skips the pursuit
and takes the k atoms with the largest <v_i/|v_i|, h>: the plain "top-k J-lens tokens" reading. Then

    h_J = P_S h   (orthogonal projection onto span{v_i : i in S}),    h_C = h - h_J,    FVE_J = |h_J|^2 / |h|^2.

The paper's formal definition takes the best span of exactly k atoms, so when the nonnegative pursuit stops early
(no candidate correlates positively with the residual) the remaining slots are filled with the next-best candidates by
correlation with h itself and the coefficients are refitted (`--no-fill` keeps the early stop); how many atoms were
filled is recorded per activation. The cone fit sum_{i in S} c_i v_i (c >= 0), the chosen tokens and their
coefficients are stored too. Candidate atoms are
the top `--candidates` tokens by <v_i/|v_i|, h>; the pursuit never leaves them. Correlations are computed as
(s * W_U[i]) . (J_l r) without materialising atoms; only chosen atoms are formed explicitly. Atom norms |v_i| are
computed once per layer and cached next to the lens.

Normalisation of the atoms is a choice the paper does not state. By default (`--select unit`) atoms are compared by
unit-normalised correlation, the standard for matching pursuit; on Gemma-2-9B-it the J-lens vectors have norms from
0.02 to 11 and the largest belong to tokens such as <end_of_turn> and <unused…>, so raw-score selection
(`--select raw`, the readout's own ordering) is dominated by them. Run both when the choice could matter.

Inputs: --acts-dir with activations_gen.npz (keys layer_<L>) and metadata_gen.json; --lens in the jlens save format
({"J": {layer: [d, d]}, "n_prompts", "source_layers", "d_model"}); --basis-dir with unembed_fp16.npy
(+ unembed_row_norms_fp32.npy) and final_norm_weight_fp32.npy (or --norm-weight). Outputs under --out-root:
j_only/ and complement/ in the paper layout (activations_gen.npz + metadata_gen.json), decomp_tokens.npz and
decomp_meta.json.

    python3 scripts/jspace_decompose.py --acts-dir data/activations/gemma2_9b/core/<run> \\
        --lens data/j_basis/neuronpedia/gemma-2-9b-it/jlens/Salesforce-wikitext/gemma-2-9b-it_jacobian_lens.pt \\
        --basis-dir data/j_basis/gemma2_9b --layers 19-23 --k 25 --out-root results/jspace/decomp/core_<run>

CPU only; numpy + scipy, torch only to read the .pt lens. No model is run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import nnls

ROOT = Path(__file__).resolve().parents[1]


# ----------------------------------------------------------------------------------------------------------------------
# inputs
# ----------------------------------------------------------------------------------------------------------------------
def parse_layers(spec: str) -> list[int]:
    if "-" in spec and "," not in spec:
        a, b = spec.split("-", 1)
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in spec.split(",") if x.strip()]


def load_lens(path: Path, layers: list[int]) -> tuple[dict[int, np.ndarray], dict]:
    """Read a jlens save file: J[l] as float32 [d, d] mapping a column activation h to J h."""
    import torch  # only for the .pt container

    ck = torch.load(str(path), map_location="cpu", weights_only=True)
    if "J" not in ck:
        raise SystemExit(f"{path} is not a JacobianLens file (keys {sorted(ck)})")
    missing = [L for L in layers if L not in ck["J"]]
    if missing:
        raise SystemExit(f"lens has no layers {missing}; it has {sorted(ck['J'])[:3]}..{max(ck['J'])}")
    J = {L: ck["J"][L].float().numpy().astype(np.float32) for L in layers}
    with open(path, "rb") as fh:
        head = hashlib.sha256(fh.read(1 << 24)).hexdigest()[:16]  # first 16 MiB: cheap identity for the record
    info = {"path": str(path), "n_prompts": int(ck.get("n_prompts", -1)), "d_model": int(ck.get("d_model", -1)),
            "source_layers": [int(x) for x in ck.get("source_layers", [])], "sha256_first16MiB": head,
            "stored_dtype": str(next(iter(ck["J"].values())).dtype)}
    return J, info


def load_unembed(basis_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    W = np.load(basis_dir / "unembed_fp16.npy", mmap_mode="r")
    p = basis_dir / "unembed_row_norms_fp32.npy"
    norms = np.load(p) if p.exists() else np.linalg.norm(np.asarray(W, dtype=np.float32), axis=1).astype(np.float32)
    return W, norms


def norm_scale(weight: np.ndarray, kind: str) -> np.ndarray:
    """Per-dimension scale of the final norm. Gemma's RMSNorm multiplies by (1 + weight); most others by weight."""
    w = np.asarray(weight, dtype=np.float32).reshape(-1)
    return (1.0 + w) if kind == "gemma" else w


# ----------------------------------------------------------------------------------------------------------------------
# readout
# ----------------------------------------------------------------------------------------------------------------------
def rms_norm(x: np.ndarray, s: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """norm(x) = x / sqrt(mean(x^2) + eps) * s, row-wise (Gemma2RMSNorm in float32)."""
    x = np.asarray(x, dtype=np.float32)
    r = np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps)
    return x / r * s


def softcap(logits: np.ndarray, cap: float | None) -> np.ndarray:
    if not cap:
        return logits
    return cap * np.tanh(logits / cap)


def readout_logits(h: np.ndarray, J: np.ndarray, W: np.ndarray, s: np.ndarray, cap: float | None) -> np.ndarray:
    """softcap(W_U . norm(J h)) for a batch of row activations h [n, d] -> [n, V]. Full vocab; use for small n."""
    z = rms_norm(np.asarray(h, dtype=np.float32) @ J.T, s)
    return softcap(z @ np.asarray(W, dtype=np.float32).T, cap)


def atom_norms(W: np.ndarray, s: np.ndarray, J: np.ndarray, cache: Path | None, chunk: int = 8192) -> np.ndarray:
    """|v_i| = |J^T (s * W_U[i])| for every token i, i.e. the row norms of W_U diag(s) J. Cached."""
    if cache is not None and cache.exists():
        a = np.load(cache)
        if a.shape[0] == W.shape[0]:
            return a.astype(np.float32)
    V = W.shape[0]
    out = np.zeros(V, dtype=np.float32)
    for r0 in range(0, V, chunk):
        r1 = min(V, r0 + chunk)
        rows = (np.asarray(W[r0:r1], dtype=np.float32) * s[None, :]) @ J  # [chunk, d] = rows of W diag(s) J
        out[r0:r1] = np.linalg.norm(rows, axis=1)
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache, out)
    return out


def candidates_and_readout(acts: np.ndarray, J: np.ndarray, W: np.ndarray, s: np.ndarray, an: np.ndarray, *,
                           n_cand: int, k_read: int, vocab_chunk: int = 8192, batch: int = 64,
                           select: str = "unit", log=None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """For every activation: the top-n_cand tokens by correlation (pursuit candidates; <v_i, h>/|v_i| with
    select="unit", <v_i, h> with select="raw"), the top-k_read tokens by raw score <v_i, h> (the J-lens readout
    ranking) and those raw scores. score_i = (s * W_U[i]) . (J h), computed as Z = (acts J^T) * s, scores = Z W_U^T."""
    n, d = acts.shape
    V = W.shape[0]
    cand = np.zeros((n, n_cand), dtype=np.int64)
    read = np.zeros((n, k_read), dtype=np.int64)
    read_s = np.zeros((n, k_read), dtype=np.float32)
    inv_an = (1.0 / np.maximum(an, 1e-12)).astype(np.float32) if select == "unit" else np.ones(V, dtype=np.float32)
    t0 = time.time()
    for b0 in range(0, n, batch):
        b1 = min(n, b0 + batch)
        Z = (np.asarray(acts[b0:b1], dtype=np.float32) @ J.T) * s[None, :]  # [b, d]
        # running top lists across vocab chunks
        best_c_val = np.full((b1 - b0, n_cand), -np.inf, dtype=np.float32)
        best_c_idx = np.zeros((b1 - b0, n_cand), dtype=np.int64)
        best_r_val = np.full((b1 - b0, k_read), -np.inf, dtype=np.float32)
        best_r_idx = np.zeros((b1 - b0, k_read), dtype=np.int64)
        for v0 in range(0, V, vocab_chunk):
            v1 = min(V, v0 + vocab_chunk)
            sc = Z @ np.asarray(W[v0:v1], dtype=np.float32).T  # [b, chunk] raw scores
            corr = sc * inv_an[None, v0:v1]
            for vals, idxs, cur, kk in ((best_c_val, best_c_idx, corr, n_cand), (best_r_val, best_r_idx, sc, k_read)):
                allv = np.concatenate([vals, cur], axis=1)
                alli = np.concatenate([idxs, np.arange(v0, v1)[None, :].repeat(b1 - b0, 0)], axis=1)
                top = np.argpartition(-allv, kk - 1, axis=1)[:, :kk]
                vals[:] = np.take_along_axis(allv, top, 1)
                idxs[:] = np.take_along_axis(alli, top, 1)
        order = np.argsort(-best_c_val, axis=1)
        cand[b0:b1] = np.take_along_axis(best_c_idx, order, 1)
        order = np.argsort(-best_r_val, axis=1)
        read[b0:b1] = np.take_along_axis(best_r_idx, order, 1)
        read_s[b0:b1] = np.take_along_axis(best_r_val, order, 1)
        if log and ((b1 // batch) % 10 == 0 or b1 == n):
            log(f"    readout {b1}/{n} ({time.time() - t0:.0f}s)")
    return cand, read, read_s


# ----------------------------------------------------------------------------------------------------------------------
# pursuit
# ----------------------------------------------------------------------------------------------------------------------
def pursuit_one(h: np.ndarray, cand: np.ndarray, J: np.ndarray, W: np.ndarray, s: np.ndarray, an: np.ndarray, *,
                k: int, method: str = "nnomp", select: str = "unit", fill: bool = True) -> dict:
    """Greedy nonnegative sparse coding of one activation over the candidate J-lens vectors.

    Returns S (token ids, in selection order), c (nonnegative coefficients), h_J (orthogonal projection onto span S),
    h_C = h - h_J, cone = sum c_i v_i, the number of pursuit steps (fewer than k if no candidate correlates positively
    with the residual) and the number of atoms filled in afterwards to reach k (0 when fill=False).
    select="unit" compares atoms by <v_i, r>/|v_i|, select="raw" by <v_i, r>.
    """
    h = np.asarray(h, dtype=np.float32)
    Wc = np.asarray(W[cand], dtype=np.float32) * s[None, :]  # [N, d]; atom i of this set is Wc[i] @ J
    inv_an = 1.0 / np.maximum(an[cand], 1e-12) if select == "unit" else np.ones(len(cand), dtype=np.float32)
    N = Wc.shape[0]
    chosen: list[int] = []
    atoms: list[np.ndarray] = []
    c = np.zeros(0, dtype=np.float32)
    r = h.copy()
    taken = np.zeros(N, dtype=bool)

    def correlations(res: np.ndarray) -> np.ndarray:
        u = J @ res  # J r  (column convention)
        return (Wc @ u) * inv_an  # <v_i, r>/|v_i| = (s*W_i).(J r)/|v_i|

    if method == "topk":
        corr = correlations(h)
        order = np.argsort(-corr)[:k]
        for j in order:
            chosen.append(int(j))
            atoms.append(Wc[j] @ J)
        A = np.stack(atoms, axis=1) if atoms else np.zeros((h.shape[0], 0), dtype=np.float32)
        c = nnls(A, h)[0].astype(np.float32) if atoms else c
        steps = len(chosen)
    else:
        steps = 0
        for _ in range(k):
            corr = correlations(r)
            corr[taken] = -np.inf
            j = int(np.argmax(corr))
            if not np.isfinite(corr[j]) or corr[j] <= 0:
                break
            taken[j] = True
            chosen.append(j)
            atoms.append(Wc[j] @ J)
            A = np.stack(atoms, axis=1)  # [d, |S|]
            if method == "nnomp":
                c = nnls(A, h)[0].astype(np.float32)
            elif method == "gp":  # gradient pursuit: one optimal step on the support, clipped at zero
                c = np.concatenate([c, np.zeros(1, dtype=np.float32)])
                g = A.T @ r
                Ag = A @ g
                alpha = float(g @ g) / max(float(Ag @ Ag), 1e-20)
                c = np.maximum(c + alpha * g, 0.0).astype(np.float32)
            else:
                raise ValueError(method)
            r = h - A @ c
            steps += 1
    filled = 0
    if fill and len(chosen) < k:
        # exactly k atoms, as in the paper's min over |S| = k: next-best candidates by correlation with h itself
        corr0 = correlations(h)
        corr0[taken] = -np.inf
        for j in np.argsort(-corr0)[: k - len(chosen)]:
            if not np.isfinite(corr0[j]):
                break
            taken[j] = True
            chosen.append(int(j))
            atoms.append(Wc[j] @ J)
            filled += 1
        if atoms:
            c = nnls(np.stack(atoms, axis=1), h)[0].astype(np.float32)
    if atoms:
        A = np.stack(atoms, axis=1)
        Q, _ = np.linalg.qr(A)
        h_J = Q @ (Q.T @ h)
        cone = A @ c
    else:
        h_J = np.zeros_like(h)
        cone = np.zeros_like(h)
    tokens = np.asarray([int(cand[j]) for j in chosen], dtype=np.int64)
    return {"tokens": tokens, "coefs": c, "h_J": h_J.astype(np.float32), "h_C": (h - h_J).astype(np.float32),
            "cone": cone.astype(np.float32), "steps": steps, "filled": filled}


def project_onto_tokens(h: np.ndarray, toks: np.ndarray, J: np.ndarray, W: np.ndarray, s: np.ndarray) -> np.ndarray:
    """Orthogonal projection of h onto span{v_t : t in toks}, v_t = J^T (s * W_U[t]) (toks: token ids, -1 = unused)."""
    toks = toks[toks >= 0]
    if len(toks) == 0:
        return np.zeros_like(h, dtype=np.float32)
    A = ((np.asarray(W[toks], dtype=np.float32) * s[None, :]) @ J).T  # [d, m]
    Q, _ = np.linalg.qr(A)
    return (Q @ (Q.T @ np.asarray(h, dtype=np.float32))).astype(np.float32)


def derangement(n: int, rng: np.random.Generator) -> np.ndarray:
    """A permutation of range(n) with no fixed point (n >= 2), as the probe script builds its shuffled frames."""
    perm = rng.permutation(n)
    fixed = np.where(perm == np.arange(n))[0]
    if len(fixed) > 1:
        perm[fixed] = np.roll(perm[fixed], 1)
    elif len(fixed) == 1 and n > 1:
        j = (fixed[0] + 1) % n
        perm[fixed[0]], perm[j] = perm[j], perm[fixed[0]]
    return perm


def shuffled_control(acts: np.ndarray, toks: np.ndarray, J: np.ndarray, W: np.ndarray, s: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """shuffled_k: each activation projected onto the span of ANOTHER activation's chosen atoms (a derangement of the
    rows; design section 2). Returns the projections and the permutation used."""
    n = acts.shape[0]
    perm = derangement(n, np.random.default_rng(seed)) if n > 1 else np.arange(n)
    out = np.zeros_like(acts, dtype=np.float32)
    for i in range(n):
        out[i] = project_onto_tokens(acts[i], toks[perm[i]], J, W, s)
    return out, perm


def decompose_layer(acts: np.ndarray, J: np.ndarray, W: np.ndarray, s: np.ndarray, an: np.ndarray, *, k: int,
                    n_cand: int, method: str, k_read: int = 25, vocab_chunk: int = 8192, batch: int = 64,
                    select: str = "unit", fill: bool = True, log=None) -> dict:
    n, d = acts.shape
    cand, read, read_s = candidates_and_readout(acts, J, W, s, an, n_cand=n_cand, k_read=k_read,
                                                vocab_chunk=vocab_chunk, batch=batch, select=select, log=log)
    hJ = np.zeros((n, d), dtype=np.float32)
    hC = np.zeros((n, d), dtype=np.float32)
    cone = np.zeros((n, d), dtype=np.float32)
    toks = np.full((n, k), -1, dtype=np.int64)
    coefs = np.zeros((n, k), dtype=np.float32)
    steps = np.zeros(n, dtype=np.int32)
    filled = np.zeros(n, dtype=np.int32)
    t0 = time.time()
    for i in range(n):
        out = pursuit_one(acts[i], cand[i], J, W, s, an, k=k, method=method, select=select, fill=fill)
        m = len(out["tokens"])
        hJ[i], hC[i], cone[i] = out["h_J"], out["h_C"], out["cone"]
        toks[i, :m] = out["tokens"]
        coefs[i, :m] = out["coefs"][:m]
        steps[i] = out["steps"]
        filled[i] = out["filled"]
        if log and ((i + 1) % 200 == 0 or i + 1 == n):
            log(f"    pursuit {i + 1}/{n} ({time.time() - t0:.0f}s)")
    hh = np.einsum("ij,ij->i", acts, acts) + 1e-12
    fve_j = np.einsum("ij,ij->i", hJ, hJ) / hh
    fve_cone = 1.0 - np.einsum("ij,ij->i", acts - cone, acts - cone) / hh
    n_tok = steps + filled
    overlap = np.mean([len(set(toks[i, :n_tok[i]].tolist()) & set(read[i].tolist())) / max(n_tok[i], 1) for i in range(n)])
    stats = {"n": int(n), "k": k, "method": method, "select": select, "fill": bool(fill), "candidates": n_cand,
             "mean_fve_j": float(fve_j.mean()), "median_fve_j": float(np.median(fve_j)),
             "mean_fve_cone": float(fve_cone.mean()), "mean_steps": float(steps.mean()),
             "frac_early_stop": float((steps < k).mean()), "mean_filled": float(filled.mean()),
             "mean_frac_chosen_in_readout_top25": float(overlap),
             "mean_j_norm": float(np.linalg.norm(hJ, axis=1).mean()), "mean_c_norm": float(np.linalg.norm(hC, axis=1).mean()),
             "pursuit_seconds": round(time.time() - t0, 1)}
    return {"h_J": hJ, "h_C": hC, "cone": cone, "tokens": toks, "coefs": coefs, "steps": steps, "filled": filled,
            "readout_top": read, "readout_scores": read_s, "fve_j": fve_j.astype(np.float32), "stats": stats}


# ----------------------------------------------------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--acts-dir", type=Path, required=True, help="activations_gen.npz + metadata_gen.json")
    ap.add_argument("--lens", type=Path, required=True, help="jlens save file (.pt)")
    ap.add_argument("--basis-dir", type=Path, default=ROOT / "data/j_basis/gemma2_9b",
                    help="unembed_fp16.npy, unembed_row_norms_fp32.npy, final_norm_weight_fp32.npy")
    ap.add_argument("--norm-weight", type=Path, default=None, help="override: final norm weight .npy [d]")
    ap.add_argument("--norm-kind", choices=["gemma", "standard"], default="gemma",
                    help="gemma: scale 1+weight (Gemma2RMSNorm); standard: scale weight")
    ap.add_argument("--softcap", type=float, default=30.0, help="final logit softcap (Gemma-2: 30); 0 disables")
    ap.add_argument("--layers", default="19-23")
    ap.add_argument("--k", type=int, default=25)
    ap.add_argument("--candidates", type=int, default=1024)
    ap.add_argument("--method", choices=["nnomp", "gp", "topk"], default="nnomp")
    ap.add_argument("--select", choices=["unit", "raw"], default="unit",
                    help="compare atoms by unit-normalised correlation (unit) or by raw score (raw, the readout's order)")
    ap.add_argument("--no-fill", action="store_true",
                    help="keep the early stop of the nonnegative pursuit instead of filling S up to exactly k atoms")
    ap.add_argument("--limit", type=int, default=None, help="only the first n activations (tests)")
    ap.add_argument("--vocab-chunk", type=int, default=8192)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--out-root", type=Path, required=True)
    ap.add_argument("--shuffled-seed", type=int, default=0,
                    help="also write shuffled/: each activation projected onto another activation's chosen atoms (a seeded "
                         "derangement of the rows, per layer); -1 disables")
    a = ap.parse_args()

    acts_dir = a.acts_dir if a.acts_dir.is_absolute() else ROOT / a.acts_dir
    basis_dir = a.basis_dir if a.basis_dir.is_absolute() else ROOT / a.basis_dir
    out_root = a.out_root if a.out_root.is_absolute() else ROOT / a.out_root
    lens_p = a.lens if a.lens.is_absolute() else ROOT / a.lens
    log = lambda m: print(m, flush=True)  # noqa: E731

    layers = parse_layers(a.layers)
    J, lens_info = load_lens(lens_p, layers)
    W, w_norms = load_unembed(basis_dir)
    nw_p = a.norm_weight or (basis_dir / "final_norm_weight_fp32.npy")
    s = norm_scale(np.load(nw_p), a.norm_kind)
    d = W.shape[1]
    assert s.shape[0] == d and all(J[L].shape == (d, d) for L in layers), (s.shape, d, {L: J[L].shape for L in layers})

    meta_p = acts_dir / "metadata_gen.json"
    npz = np.load(acts_dir / "activations_gen.npz")
    avail = sorted(int(k.split("_")[1]) for k in npz.files if k.startswith("layer_"))
    missing = [L for L in layers if L not in avail]
    if missing:
        raise SystemExit(f"activations have no layers {missing}; available {avail}")

    j_dir, c_dir, s_dir = out_root / "j_only", out_root / "complement", out_root / "shuffled"
    for p in (j_dir, c_dir) + ((s_dir,) if a.shuffled_seed >= 0 else ()):
        p.mkdir(parents=True, exist_ok=True)
        shutil.copy2(meta_p, p / "metadata_gen.json")
    cache_dir = lens_p.parent / "derived"

    j_arrays, c_arrays, s_arrays, tok_arrays, layer_stats = {}, {}, {}, {}, {}
    for L in layers:
        acts = np.ascontiguousarray(npz[f"layer_{L}"], dtype=np.float32)
        if a.limit:
            acts = acts[: a.limit]
        log(f"[layer {L}] {acts.shape[0]} activations, d={acts.shape[1]}")
        t0 = time.time()
        an = atom_norms(W, s, J[L], cache_dir / f"atom_norms_L{L}.npy", chunk=a.vocab_chunk)
        log(f"  atom norms ready ({time.time() - t0:.0f}s; mean {an.mean():.4f}, max {an.max():.4f})")
        res = decompose_layer(acts, J[L], W, s, an, k=a.k, n_cand=a.candidates, method=a.method,
                              vocab_chunk=a.vocab_chunk, batch=a.batch, select=a.select, fill=not a.no_fill, log=log)
        st = res["stats"]
        st["seconds"] = round(time.time() - t0, 1)
        layer_stats[str(L)] = st
        log(f"  FVE_J mean {st['mean_fve_j']:.4f} median {st['median_fve_j']:.4f}; cone FVE {st['mean_fve_cone']:.4f}; "
            f"steps {st['mean_steps']:.1f} filled {st['mean_filled']:.1f}; chosen∩readout-top25 {st['mean_frac_chosen_in_readout_top25']:.2f}")
        j_arrays[f"layer_{L}"] = res["h_J"]
        c_arrays[f"layer_{L}"] = res["h_C"]
        for key in ("tokens", "coefs", "steps", "filled", "readout_top", "readout_scores", "fve_j"):
            tok_arrays[f"{key}_L{L}"] = res[key]
        if a.shuffled_seed >= 0:
            t1 = time.time()
            h_S, perm = shuffled_control(acts, res["tokens"], J[L], W, s, a.shuffled_seed + L)
            s_arrays[f"layer_{L}"] = h_S
            tok_arrays[f"shuffled_perm_L{L}"] = perm
            fve_s = np.einsum("ij,ij->i", h_S, h_S) / (np.einsum("ij,ij->i", acts, acts) + 1e-12)
            st["mean_fve_shuffled"] = float(fve_s.mean())
            st["shuffled_seconds"] = round(time.time() - t1, 1)
            log(f"  shuffled control: FVE mean {st['mean_fve_shuffled']:.4f} ({st['shuffled_seconds']}s)")

    np.savez_compressed(j_dir / "activations_gen.npz", **j_arrays)
    np.savez_compressed(c_dir / "activations_gen.npz", **c_arrays)
    if a.shuffled_seed >= 0:
        np.savez_compressed(s_dir / "activations_gen.npz", **s_arrays)
    np.savez_compressed(out_root / "decomp_tokens.npz", **tok_arrays)
    meta = {
        "definition": (
            "Gurnee et al. (2026) J-space: per activation, greedy nonnegative pursuit over the J-lens vectors "
            "v_i = J_l^T (s * W_U[i]) (rows of W_U diag(s) J_l, s the final-norm scale) picks k atoms; "
            "h_J = orthogonal projection onto their span, h_C = h - h_J; cone = nonnegative fit."
        ),
        "readout": "softcap(W_U . norm(J_l h)), norm(x) = x/rms(x)*s, as jlens.hf.unembed",
        "settings": {"layers": layers, "k": a.k, "candidates": a.candidates, "method": a.method, "select": a.select, "shuffled_seed": a.shuffled_seed,
                     "fill": not a.no_fill, "norm_kind": a.norm_kind, "softcap": a.softcap, "limit": a.limit},
        "lens": lens_info,
        "basis_dir": str(basis_dir), "norm_weight": str(nw_p), "acts_dir": str(acts_dir),
        "layers": layer_stats,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (out_root / "decomp_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    log(f"wrote {j_dir}, {c_dir}, {out_root / 'decomp_tokens.npz'}, {out_root / 'decomp_meta.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
