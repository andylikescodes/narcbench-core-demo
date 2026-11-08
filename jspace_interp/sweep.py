"""Experiment 2 (k x position sweep with many-draw controls) and Experiment 1 (atom readouts).

All numpy/scipy on saved activations; uses the recovered Nov 5 decomposition code.
"""
from __future__ import annotations

import numpy as np

from . import detectors, stats
from .nov5_jspace_decompose import candidates_and_readout, pursuit_one, project_onto_tokens


def decompose_nested(acts, J, W, s, an, *, k_max: int, n_cand: int = 1024, select="unit", batch=64,
                     vocab_chunk=8192, k_read=50):
    """One NNOMP pursuit to k_max without fill; the first k atoms of the selection order give J_k
    for every k <= k_max (greedy order does not depend on k). Also returns fill candidates
    (next-best by correlation with h) so a filled-to-k arm can be built, and the lens readout."""
    cand, read, read_s = candidates_and_readout(acts, J, W, s, an, n_cand=n_cand, k_read=k_read,
                                                vocab_chunk=vocab_chunk, batch=batch, select=select)
    n = acts.shape[0]
    order = np.full((n, k_max), -1, dtype=np.int64)
    steps = np.zeros(n, dtype=np.int32)
    for i in range(n):
        out = pursuit_one(acts[i], cand[i], J, W, s, an, k=k_max, method="nnomp", select=select, fill=False)
        m = len(out["tokens"])
        order[i, :m] = out["tokens"]
        steps[i] = out["steps"]
    return {"order": order, "steps": steps, "cand": cand, "readout": read, "readout_scores": read_s}


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
