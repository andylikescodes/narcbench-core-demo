"""Statistics for Experiment 1 (concept readouts) and the controls."""
from __future__ import annotations

import numpy as np


def within_episode_role_permutation(values, run_ids, is_colluder, n_perm=10000, seed=0):
    """Colluder-minus-honest mean difference inside collusion episodes, with a null that
    reshuffles the colluder labels within each episode (keeping the count per episode).

    values, run_ids, is_colluder: arrays over agents of collusion episodes only.
    Returns (observed difference, two-sided p, null array).
    """
    v = np.asarray(values, float)
    r = np.asarray(run_ids)
    c = np.asarray(is_colluder, bool)
    groups = [np.where(r == g)[0] for g in np.unique(r)]

    def diff(lbl):
        return v[lbl].mean() - v[~lbl].mean()

    obs = diff(c)
    rng = np.random.default_rng(seed)
    null = np.empty(n_perm)
    for b in range(n_perm):
        lbl = c.copy()
        for g in groups:
            lbl[g] = rng.permutation(c[g])
        null[b] = diff(lbl)
    p = (1 + np.sum(np.abs(null) >= abs(obs) - 1e-12)) / (1 + n_perm)
    return float(obs), float(p), null


def stratified_permutation(values, labels, strata, n_perm=10000, seed=0):
    """Mean difference label1 - label0, permuting labels within strata (e.g. scenario)."""
    v = np.asarray(values, float)
    lab = np.asarray(labels, bool)
    st = np.asarray(strata)
    groups = [np.where(st == g)[0] for g in np.unique(st)]
    obs = v[lab].mean() - v[~lab].mean()
    rng = np.random.default_rng(seed)
    null = np.empty(n_perm)
    for b in range(n_perm):
        l2 = lab.copy()
        for g in groups:
            l2[g] = rng.permutation(lab[g])
        null[b] = v[l2].mean() - v[~l2].mean()
    p = (1 + np.sum(np.abs(null) >= abs(obs) - 1e-12)) / (1 + n_perm)
    return float(obs), float(p), null


def benjamini_hochberg(p, q=0.05):
    p = np.asarray(p, float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order] * n / (np.arange(n) + 1)
    adj = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.minimum(adj, 1.0)
    return out, out <= q


def jensen_shannon(p, q, eps=1e-12):
    p = np.asarray(p, float) + eps
    q = np.asarray(q, float) + eps
    p /= p.sum()
    q /= q.sum()
    m = 0.5 * (p + q)
    return float(0.5 * np.sum(p * np.log(p / m)) + 0.5 * np.sum(q * np.log(q / m)))


def auroc_mwu(pos, neg):
    """AUROC of pos vs neg (Mann-Whitney), ties counted half."""
    pos = np.asarray(pos, float)
    neg = np.asarray(neg, float)
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    gt = (pos[:, None] > neg[None, :]).sum()
    eq = (pos[:, None] == neg[None, :]).sum()
    return float((gt + 0.5 * eq) / (len(pos) * len(neg)))


def permutation_auroc(pos, neg, n_perm=10000, seed=0):
    pos = np.asarray(pos, float)
    neg = np.asarray(neg, float)
    obs = auroc_mwu(pos, neg)
    allv = np.concatenate([pos, neg])
    rng = np.random.default_rng(seed)
    null = np.empty(n_perm)
    for b in range(n_perm):
        pp = rng.permutation(allv)
        null[b] = auroc_mwu(pp[: len(pos)], pp[len(pos):])
    p = (1 + np.sum(np.abs(null - 0.5) >= abs(obs - 0.5) - 1e-12)) / (1 + n_perm)
    return obs, float(p)


def empirical_percentile(value, null):
    null = np.asarray(null, float)
    pct = float(np.mean(null < value) * 100)
    p = float((1 + np.sum(null >= value)) / (1 + len(null)))
    return pct, p
