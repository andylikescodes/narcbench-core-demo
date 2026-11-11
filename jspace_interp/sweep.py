"""Experiment 2 (k x position sweep with many-draw controls) and Experiment 1 (atom readouts).

All numpy/scipy on saved activations; uses the recovered Nov 5 decomposition code.
"""
from __future__ import annotations

import numpy as np

from . import detectors, stats
from .nov5_jspace_decompose import candidates_and_readout, pursuit_one, project_onto_tokens


def decompose_nested(acts, J, W, s, an, *, k_max: int, n_cand: int = 1024, select="unit", batch=64,
                     vocab_chunk=8192, k_read=50, method="nnomp", searcher=None, log=None):
    """One greedy non-negative pursuit to k_max without fill; the first k atoms of the selection order give J_k
    for every k <= k_max (the greedy order does not depend on k). ``method`` is "nnomp" (non-negative OMP with an
    NNLS refit, the solver of the Nov 5 and Nov 9 runs), "gp" (gradient pursuit, the solver Gurnee et al. name; both
    search the n_cand tokens best aligned with h) or "gpfull" (gradient pursuit whose selection step searches the
    whole vocabulary at every step, the literal procedure). Also returns fill candidates (next-best by correlation
    with h) so a filled-to-k arm can be built, and the lens readout. ``searcher`` (a TorchVocabSearch) moves the
    whole-vocabulary products to another device; it is used by "gpfull" and for the candidate list."""
    if searcher is not None:
        cand, read, read_s = searcher.candidates_and_readout(acts, J, s, n_cand=n_cand, k_read=k_read, select=select)
    else:
        cand, read, read_s = candidates_and_readout(acts, J, W, s, an, n_cand=n_cand, k_read=k_read,
                                                    vocab_chunk=vocab_chunk, batch=batch, select=select)
    n = acts.shape[0]
    if method == "gpfull":
        g = gradient_pursuit_full(acts, J, W, s, an, k=k_max, vocab_chunk=vocab_chunk, searcher=searcher, log=log)
        return {"order": g["order"], "steps": g["steps"], "cand": cand, "readout": read, "readout_scores": read_s,
                "coefs_gp": g["coefs"]}
    order = np.full((n, k_max), -1, dtype=np.int64)
    steps = np.zeros(n, dtype=np.int32)
    for i in range(n):
        out = pursuit_one(acts[i], cand[i], J, W, s, an, k=k_max, method=method, select=select, fill=False)
        m = len(out["tokens"])
        order[i, :m] = out["tokens"]
        steps[i] = out["steps"]
    return {"order": order, "steps": steps, "cand": cand, "readout": read, "readout_scores": read_s}


class TorchVocabSearch:
    """Whole-vocabulary products <v_t, x> for a batch of vectors on a torch device (the Mac's GPU, "mps").

    The unembedding sits on the device once, in the half precision it is stored in, and is widened to single
    precision chunk by chunk. For the pursuit's selection step the device only proposes the ``top`` best tokens
    per row; their correlations are then recomputed on the CPU in single precision from the stored unembedding and
    the largest of those decides, so the choice does not depend on the device's arithmetic."""

    def __init__(self, W, device="mps", chunk=16000, top=8):
        import torch

        self.torch, self.dev, self.W, self.chunk, self.top = torch, torch.device(device), W, chunk, top
        self.V, self.d = W.shape
        self.chunks = [torch.from_numpy(np.array(W[v0:v0 + chunk])).to(self.dev) for v0 in range(0, self.V, chunk)]
        self.inv_an_np = self.inv_an = None

    def set_norms(self, an):
        self.inv_an_np = (1.0 / np.maximum(an, 1e-12)).astype(np.float32)
        self.inv_an = self.torch.from_numpy(self.inv_an_np).to(self.dev)

    def _scores(self, Zt, ci):
        return Zt @ self.chunks[ci].to(self.torch.float32).T

    def best(self, Z, T):
        """Z [b, d]: s * (J r) per row; T [b, m]: tokens a row already holds (-1 = none). Returns the largest
        <v_t, r>/|v_t| over the tokens a row does not hold, and the token."""
        torch = self.torch
        b = Z.shape[0]
        Z = np.ascontiguousarray(Z, dtype=np.float32)
        Zt = torch.from_numpy(Z).to(self.dev)
        top_v = torch.full((b, self.top), -float("inf"), device=self.dev)
        top_i = torch.zeros((b, self.top), dtype=torch.long, device=self.dev)
        for ci in range(len(self.chunks)):
            v0 = ci * self.chunk
            v1 = v0 + self.chunks[ci].shape[0]
            corr = self._scores(Zt, ci) * self.inv_an[None, v0:v1]
            ri, cj = np.nonzero((T >= v0) & (T < v1))
            if len(ri):
                corr[torch.from_numpy(ri).to(self.dev), torch.from_numpy(T[ri, cj] - v0).to(self.dev)] = -float("inf")
            v, j = torch.topk(corr, self.top, dim=1)
            allv, alli = torch.cat([top_v, v], 1), torch.cat([top_i, j + v0], 1)
            top_v, o = torch.topk(allv, self.top, dim=1)
            top_i = torch.gather(alli, 1, o)
        cand = top_i.cpu().numpy()
        dev_val = top_v.cpu().numpy()
        flat = cand.reshape(-1)
        o = np.argsort(flat, kind="stable")
        Wsel = np.empty((flat.size, self.d), dtype=np.float32)
        Wsel[o] = np.asarray(self.W[flat[o]], dtype=np.float32)
        val = np.einsum("bjd,bd->bj", Wsel.reshape(b, self.top, self.d), Z) * self.inv_an_np[cand]
        val[~np.isfinite(dev_val)] = -np.inf
        j = val.argmax(1)
        rows = np.arange(b)
        return val[rows, j].astype(np.float32), cand[rows, j]

    def candidates_and_readout(self, acts, J, s, *, n_cand, k_read, select="unit", batch=496):
        """The device version of ``candidates_and_readout``: per row the n_cand tokens with the largest
        <v_t, h>/|v_t| (in that order) and the k_read tokens with the largest raw score <v_t, h>."""
        torch = self.torch
        n = acts.shape[0]
        cand = np.zeros((n, n_cand), dtype=np.int64)
        read = np.zeros((n, k_read), dtype=np.int64)
        read_s = np.zeros((n, k_read), dtype=np.float32)
        for b0 in range(0, n, batch):
            b1 = min(n, b0 + batch)
            b = b1 - b0
            Z = np.ascontiguousarray((np.asarray(acts[b0:b1], dtype=np.float32) @ J.T) * s[None, :], dtype=np.float32)
            Zt = torch.from_numpy(Z).to(self.dev)
            cv = torch.full((b, n_cand), -float("inf"), device=self.dev)
            ci_ = torch.zeros((b, n_cand), dtype=torch.long, device=self.dev)
            rv = torch.full((b, k_read), -float("inf"), device=self.dev)
            ri_ = torch.zeros((b, k_read), dtype=torch.long, device=self.dev)
            for ci in range(len(self.chunks)):
                v0 = ci * self.chunk
                v1 = v0 + self.chunks[ci].shape[0]
                sc = self._scores(Zt, ci)
                corr = sc * self.inv_an[None, v0:v1] if select == "unit" else sc
                for cur, kk, which in ((corr, n_cand, "c"), (sc, k_read, "r")):
                    v, j = torch.topk(cur, kk, dim=1)
                    pv, pi = (cv, ci_) if which == "c" else (rv, ri_)
                    allv, alli = torch.cat([pv, v], 1), torch.cat([pi, j + v0], 1)
                    nv, o = torch.topk(allv, kk, dim=1)
                    ni = torch.gather(alli, 1, o)
                    if which == "c":
                        cv, ci_ = nv, ni
                    else:
                        rv, ri_ = nv, ni
            cand[b0:b1] = ci_.cpu().numpy()
            read[b0:b1] = ri_.cpu().numpy()
            read_s[b0:b1] = rv.cpu().numpy()
        return cand, read, read_s


def gradient_pursuit_full(acts, J, W, s, an, *, k: int, batch: int = 496, vocab_chunk: int = 8192, log=None,
                          searcher=None):
    """Gradient pursuit whose selection step runs over the whole vocabulary (no candidate pre-selection).

    The same algorithm as ``pursuit_one(method="gp", select="unit", fill=False)`` with every token as a candidate,
    batched over activations so that one pass over the unembedding serves a whole batch: at each step every row
    picks the J-lens vector with the largest <v_t, r>/|v_t| among those it has not picked (the row stops if none is
    positive), takes one optimal gradient step on its support and clips the coefficients at zero. Rows that have
    stopped take no further part. ``searcher`` (a TorchVocabSearch with its norms set) runs the selection step on
    another device.

    Returns {"order": [n, k] token ids in selection order (-1 = unused), "coefs": [n, k] gradient pursuit's own
    coefficients, "steps": [n], "recon_gp": [n, d] sum_i c_i v_i with those coefficients}.
    """
    n, d = acts.shape
    V = W.shape[0]
    inv_an = (1.0 / np.maximum(an, 1e-12)).astype(np.float32)
    order = np.full((n, k), -1, dtype=np.int64)
    coefs = np.zeros((n, k), dtype=np.float32)
    steps = np.zeros(n, dtype=np.int32)
    recon = np.zeros((n, d), dtype=np.float32)
    grow = 32
    for b0 in range(0, n, batch):
        b1 = min(n, b0 + batch)
        b = b1 - b0
        H = np.asarray(acts[b0:b1], dtype=np.float32)
        R = H.copy()
        kc = min(k, grow)
        A = np.zeros((b, kc, d), dtype=np.float32)         # the chosen J-lens vectors of each row
        C = np.zeros((b, kc), dtype=np.float32)
        T = np.full((b, kc), -1, dtype=np.int64)
        m = np.zeros(b, dtype=np.int64)
        alive = np.ones(b, dtype=bool)
        for step in range(k):
            act = np.nonzero(alive)[0]
            if len(act) == 0:
                break
            if step >= A.shape[1]:
                e = min(k - A.shape[1], grow)
                A = np.concatenate([A, np.zeros((b, e, d), dtype=np.float32)], 1)
                C = np.concatenate([C, np.zeros((b, e), dtype=np.float32)], 1)
                T = np.concatenate([T, np.full((b, e), -1, dtype=np.int64)], 1)
            Z = (R[act] @ J.T) * s[None, :]                # row i: s * (J r_i); <v_t, r_i> = W_U[t] . Z_i
            Ta = T[act]
            if searcher is not None:
                best_val, best_idx = searcher.best(Z, Ta)
            else:
                na = len(act)
                rows = np.arange(na)
                best_val = np.full(na, -np.inf, dtype=np.float32)
                best_idx = np.zeros(na, dtype=np.int64)
                for v0 in range(0, V, vocab_chunk):
                    v1 = min(V, v0 + vocab_chunk)
                    corr = (Z @ np.asarray(W[v0:v1], dtype=np.float32).T) * inv_an[None, v0:v1]
                    ri, ci = np.nonzero((Ta >= v0) & (Ta < v1))  # vectors a row already holds are not candidates
                    corr[ri, Ta[ri, ci] - v0] = -np.inf
                    j = np.argmax(corr, axis=1)
                    val = corr[rows, j]
                    upd = val > best_val
                    best_val[upd] = val[upd]
                    best_idx[upd] = v0 + j[upd]
            ok = np.isfinite(best_val) & (best_val > 0)
            alive[act[~ok]] = False
            sel = act[ok]
            if len(sel) == 0:
                break
            tsel = best_idx[ok]
            # fancy indexing on the memory map wants sorted indices
            o = np.argsort(tsel)
            vnew = np.empty((len(sel), d), dtype=np.float32)
            vnew[o] = (np.asarray(W[tsel[o]], dtype=np.float32) * s[None, :]) @ J
            A[sel, m[sel]] = vnew
            T[sel, m[sel]] = tsel
            m[sel] += 1
            mm = int(m[sel].max())
            Asel = A[sel, :mm]
            G = np.einsum("bkd,bd->bk", Asel, R[sel])      # gradient on the support (zero beyond m)
            AG = np.einsum("bk,bkd->bd", G, Asel)
            alpha = (G * G).sum(1) / np.maximum((AG * AG).sum(1), 1e-20)
            Cs = np.maximum(C[sel, :mm] + alpha[:, None] * G, 0.0)
            C[sel, :mm] = Cs
            R[sel] = H[sel] - np.einsum("bk,bkd->bd", Cs, Asel)
            steps[b0 + sel] += 1
            if log and (step % 5 == 4 or step == k - 1):
                log(f"    full-vocabulary pursuit rows {b0}-{b1} step {step + 1}/{k} ({int(alive.sum())} rows active)")
        kc = T.shape[1]
        order[b0:b1, :kc], coefs[b0:b1, :kc] = T, C
        recon[b0:b1] = np.einsum("bk,bkd->bd", C, A)
    return {"order": order, "coefs": coefs, "steps": steps, "recon_gp": recon}


def j_arms(acts, dec, J, W, s, k: int, fill: bool):
    """h_J for the first k pursuit atoms; with fill, top up to k from the candidate list (correlation with h)."""
    n, d = acts.shape
    hJ = np.zeros_like(acts, dtype=np.float32)
    toks = np.full((n, k), -1, dtype=np.int64)
    for i in range(n):
        t = [x for x in dec["order"][i, :k] if x >= 0]
        if fill and len(t) < k:
            for c in dec["cand"][i]:
                if len(t) >= k:
                    break
                if c not in t:
                    t.append(int(c))
        toks[i, :len(t)] = t
        hJ[i] = project_onto_tokens(acts[i], np.array(t, dtype=np.int64), J, W, s)
    return hJ, toks


def j_recon(acts, dec, J, W, s, k: int):
    """The J-space component as Gurnee et al. operationalise it: the best non-negative combination of the (at most k)
    J-lens vectors the pursuit chose, sum_i c_i v_i with c >= 0 (NNLS on the first k atoms of the selection order).
    h minus this is their non-J-space component. Returns the reconstructions [n, d] and the number of atoms with a
    positive coefficient per row."""
    from scipy.optimize import nnls

    n, d = acts.shape
    out = np.zeros((n, d), dtype=np.float32)
    used = np.zeros(n, dtype=np.int32)
    for i in range(n):
        t = np.array([x for x in dec["order"][i, :k] if x >= 0], dtype=np.int64)
        if len(t) == 0:
            continue
        A = ((np.asarray(W[t], dtype=np.float32) * s[None, :]) @ J).T.astype(np.float64)  # [d, m] J-lens vectors
        c = nnls(A, np.asarray(acts[i], dtype=np.float64))[0]
        out[i] = (A @ c).astype(np.float32)
        used[i] = int((c > 0).sum())
    return out, used


def haar_subspace(d, k, rng):
    Q, R = np.linalg.qr(rng.standard_normal((d, k)))
    return Q * np.sign(np.diag(R))[None, :]


def project_fixed(acts, Q):
    return (acts @ Q) @ Q.T


def atoms_basis(toks, J, W, s):
    A = ((np.asarray(W[toks], dtype=np.float32) * s[None, :]) @ J).T
    Q, _ = np.linalg.qr(A)
    return Q


def protocol_scores(core_agents, xfer_agents, *, lodo=True, lofo=True, boot_idx=None):
    out = {}
    if lodo:
        r = detectors.core_lodo(core_agents)
        r.pop("agent_scores", None)
        out["core_lodo"] = r["primary"]
    r = detectors.evaluate(core_agents, xfer_agents)
    out["core_to_transfer"] = r["primary"]
    out["_c2t"] = r
    if boot_idx is not None:
        _, b = detectors.episode_bootstrap(r["y"], r["episode_scores"], strata=None, idx=boot_idx)
        out["core_to_transfer_ci95"] = [float(np.nanpercentile(b, 2.5)), float(np.nanpercentile(b, 97.5))]
        out["_boot"] = b
    if lofo:
        out["transfer_lofo"] = detectors.transfer_lofo(xfer_agents)["primary"]
    return out


def group_mass(toks, coefs_or_none, group_ids: dict[str, list[int]]):
    """Per-row share of atoms (or of coefficient mass) whose token is in each concept group."""
    n, k = toks.shape
    out = {}
    for g, ids in group_ids.items():
        ids = set(ids)
        hit = np.array([[t in ids for t in row] for row in toks], dtype=float)
        if coefs_or_none is None:
            valid = (toks >= 0).sum(1).clip(min=1)
            out[g] = hit.sum(1) / valid
        else:
            c = np.asarray(coefs_or_none, float)
            out[g] = (hit * c).sum(1) / np.maximum(c.sum(1), 1e-12)
    return out


def token_frequency_tests(agent_tok_sets: list[set], is_colluder: list[bool], run_ids, vocab_size_hint=None,
                          min_count=5, n_perm=2000, seed=0):
    """Per-token share of colluders vs honest agents whose atoms include the token; within-episode
    permutation null; BH across tested tokens."""
    from collections import Counter

    cnt = Counter(t for s in agent_tok_sets for t in s)
    tested = [t for t, c in cnt.items() if c >= min_count and c <= len(agent_tok_sets) - min_count]
    res = []
    for t in tested:
        v = [1.0 if t in s else 0.0 for s in agent_tok_sets]
        obs, p, _ = stats.within_episode_role_permutation(v, run_ids, is_colluder, n_perm=n_perm, seed=seed)
        res.append((t, obs, p))
    if not res:
        return []
    adj, rej = stats.benjamini_hochberg([r[2] for r in res])
    return sorted([{"token": int(t), "colluder_minus_honest": o, "p": p, "p_bh": float(a), "significant": bool(r)}
                   for (t, o, p), a, r in zip(res, adj, rej)], key=lambda x: x["p"])
