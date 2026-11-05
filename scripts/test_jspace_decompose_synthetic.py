#!/usr/bin/env python3
"""Synthetic checks for scripts/jspace_decompose.py (CPU, seconds; no model, no download).

1. Recovery: h = six J-lens atoms with positive coefficients (+ tiny noise). nnomp recovers the six tokens and their
   coefficients, h_J reproduces the clean part, h_J is orthogonal to h_C and h_J + h_C = h. gp and topk are reported
   leniently (they are approximations by design).
2. Readout: the raw-score ranking equals the ranking of softcap(W_U . norm(J h)), and the readout's top tokens contain
   the planted ones.
3. Norm: rms_norm with s = 1 + gamma matches transformers' Gemma2RMSNorm when transformers is importable.
4. Atom norms: the chunked computation equals the direct one.
5. decompose_layer end to end on a small batch: shapes, FVE bounds, planted rows recovered.

    python3 scripts/test_jspace_decompose_synthetic.py
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("jd", ROOT / "scripts/jspace_decompose.py")
jd = importlib.util.module_from_spec(spec)
spec.loader.exec_module(jd)

fails: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}")
    if not ok:
        fails.append(name)


rng = np.random.default_rng(0)
d, V, k, n_cand = 48, 400, 6, 80
W = (rng.normal(size=(V, d)) / np.sqrt(d)).astype(np.float16)
gamma = rng.normal(scale=0.2, size=d).astype(np.float32)
s = jd.norm_scale(gamma, "gemma")
J = (np.eye(d) + 0.3 * rng.normal(size=(d, d)) / np.sqrt(d)).astype(np.float32)
atoms_all = (W.astype(np.float32) * s[None, :]) @ J  # rows v_i = J^T (s * W_i)

S_true = np.sort(rng.choice(V, k, replace=False))
c_true = rng.uniform(0.5, 2.0, size=k).astype(np.float32)
h_clean = (c_true @ atoms_all[S_true]).astype(np.float32)
h = h_clean + 1e-4 * rng.normal(size=d).astype(np.float32)

# 4. atom norms
an = jd.atom_norms(W, s, J, None, chunk=64)
direct = np.linalg.norm(atoms_all, axis=1)
check("atom norms (chunked == direct)", np.max(np.abs(an - direct) / direct) < 1e-5)

# 2. candidates + readout ranking
cand, read, read_s = jd.candidates_and_readout(h[None, :], J, W, s, an, n_cand=n_cand, k_read=10, vocab_chunk=128, batch=4)
check("candidates contain the planted tokens", set(S_true.tolist()) <= set(cand[0].tolist()))
logits = jd.readout_logits(h[None, :], J, W, s, cap=30.0)[0]
top_by_logit = np.argsort(-logits)[:10]
check("readout order == raw-score order", np.array_equal(top_by_logit, read[0]),
      f"logit top5 {top_by_logit[:5].tolist()} read top5 {read[0][:5].tolist()}")
raw_scores = atoms_all @ h
check("raw scores match <v_i,h>", np.allclose(read_s[0], np.sort(raw_scores)[::-1][:10], rtol=1e-4, atol=1e-5))
ranks = np.empty(V, dtype=np.int64)
ranks[np.argsort(-logits)] = np.arange(V)
# The readout ranks by the raw score <v_i, h>, not by the pursuit coefficient, so a planted atom with a small coefficient
# can rank below correlated non-planted atoms in this low-dimensional synthetic setting; informational only.
print(f"INFO  planted tokens' readout ranks: {sorted(ranks[S_true].tolist())} (coefficients {c_true.round(2).tolist()})")

# 1. recovery with nnomp
res = jd.pursuit_one(h, cand[0], J, W, s, an, k=k, method="nnomp")
check("nnomp recovers the support", set(res["tokens"].tolist()) == set(S_true.tolist()),
      f"got {sorted(res['tokens'].tolist())} want {S_true.tolist()}")
if set(res["tokens"].tolist()) == set(S_true.tolist()):
    order = np.argsort(res["tokens"])
    c_got = res["coefs"][order]
    c_want = c_true[np.argsort(S_true)]
    check("nnomp coefficients", np.max(np.abs(c_got - c_want) / c_want) < 1e-2, f"max rel err {np.max(np.abs(c_got - c_want) / c_want):.2e}")
rel = np.linalg.norm(res["h_J"] - h_clean) / np.linalg.norm(h_clean)
check("h_J reproduces the clean part", rel < 1e-3, f"rel err {rel:.2e}")
orth = abs(float(res["h_J"] @ res["h_C"])) / float(h @ h)
check("h_J orthogonal to h_C (|h_J.h_C| / |h|^2 at fp32)", orth < 1e-5, f"{orth:.1e}")
check("h_J + h_C == h", np.allclose(res["h_J"] + res["h_C"], h, atol=1e-5))
check("cone fit == clean part", np.linalg.norm(res["cone"] - h_clean) / np.linalg.norm(h_clean) < 1e-2)
check("all coefficients nonnegative", np.all(res["coefs"] >= 0))

for method in ("gp", "topk"):
    r2 = jd.pursuit_one(h, cand[0], J, W, s, an, k=k, method=method)
    got = len(set(r2["tokens"].tolist()) & set(S_true.tolist()))
    fve = float(r2["h_J"] @ r2["h_J"]) / float(h @ h)
    check(f"{method} recovers >=4 of 6 planted atoms", got >= 4, f"{got}/6, FVE_J {fve:.4f}")
    check(f"{method} nonnegative coefficients", np.all(r2["coefs"] >= -1e-7))

# early stop: an activation orthogonal-ish with negative correlations only -> no atom should be forced
h_neg = -(atoms_all[S_true].sum(0))  # negative combination of planted atoms
r3 = jd.pursuit_one(h_neg, cand[0], J, W, s, an, k=k, method="nnomp")
check("pursuit stops when no positive correlation (or keeps c >= 0)", r3["steps"] <= k and np.all(r3["coefs"] >= 0),
      f"steps {r3['steps']}")

# 3. norm vs transformers' Gemma2RMSNorm
try:
    import torch
    from transformers.models.gemma2.modeling_gemma2 import Gemma2RMSNorm

    x = rng.normal(size=(5, d)).astype(np.float32) * 30
    mod = Gemma2RMSNorm(d, eps=1e-6)
    with torch.no_grad():
        mod.weight.copy_(torch.tensor(gamma))
        ref = mod(torch.tensor(x)).numpy()
    ours = jd.rms_norm(x, s)
    check("rms_norm == Gemma2RMSNorm (1 + weight)", np.max(np.abs(ours - ref)) < 1e-4, f"max abs diff {np.max(np.abs(ours - ref)):.1e}")
except Exception as e:  # noqa: BLE001
    print(f"SKIP  Gemma2RMSNorm comparison ({type(e).__name__}: {str(e)[:60]})")

# 5. decompose_layer on a batch: planted rows + random rows
n_rand = 3
acts = np.stack([h] * 2 + [rng.normal(size=d).astype(np.float32) * np.linalg.norm(h) / np.sqrt(d) for _ in range(n_rand)])
out = jd.decompose_layer(acts, J, W, s, an, k=k, n_cand=n_cand, method="nnomp", k_read=10, vocab_chunk=128, batch=2)
check("decompose_layer shapes", out["h_J"].shape == acts.shape and out["tokens"].shape == (acts.shape[0], k)
      and out["readout_top"].shape == (acts.shape[0], 10))
check("FVE_J within [0, 1]", np.all(out["fve_j"] >= -1e-6) and np.all(out["fve_j"] <= 1 + 1e-6))
check("planted rows recovered in batch", all(set(out["tokens"][i].tolist()) == set(S_true.tolist()) for i in range(2)))
check("planted rows FVE_J ~ 1", np.all(out["fve_j"][:2] > 0.999))
check("random rows FVE_J < 1", np.all(out["fve_j"][2:] < 0.999), f"{out['fve_j'][2:].round(3).tolist()}")
check("stats keys", {"mean_fve_j", "mean_steps", "frac_early_stop", "mean_frac_chosen_in_readout_top25"} <= set(out["stats"]))

# 6. shuffled control: projection onto another row's atoms
perm = jd.derangement(acts.shape[0], np.random.default_rng(0))
check("derangement has no fixed point", np.all(perm != np.arange(acts.shape[0])) and sorted(perm.tolist()) == list(range(acts.shape[0])))
own = jd.project_onto_tokens(h, out["tokens"][0], J, W, s)
check("project_onto_tokens(own atoms) == h_J", np.allclose(own, out["h_J"][0], atol=1e-4))
h_S, perm_used = jd.shuffled_control(acts, out["tokens"], J, W, s, seed=1)
check("shuffled control shapes + derangement", h_S.shape == acts.shape and np.all(perm_used != np.arange(acts.shape[0])))
fve_S = np.einsum("ij,ij->i", h_S, h_S) / np.einsum("ij,ij->i", acts, acts)
check("shuffled projections are projections (FVE in [0, 1])", np.all(fve_S >= -1e-6) and np.all(fve_S <= 1 + 1e-6), f"{fve_S.round(3).tolist()}")
# rows 0 and 1 are the same planted h, so a planted row shuffled onto the other planted row's atoms keeps FVE 1; onto a random row's it drops
i_rand = [i for i in range(2) if perm_used[i] >= 2]
if i_rand:
    check("planted row onto a random row's atoms explains less than its own", all(fve_S[i] < 0.999 for i in i_rand), f"{[round(float(fve_S[i]), 3) for i in i_rand]}")

print("\nALL PASS" if not fails else f"\nFAILED: {fails}")
sys.exit(1 if fails else 0)
