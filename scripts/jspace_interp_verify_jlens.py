#!/usr/bin/env python3
"""Check this repo's lens arithmetic against Anthropic's reference implementation (package `jlens`).

What comes from where
  * The lens (the matrices J_l) is Anthropic's: the published fit for google/gemma-2-9b-it, made with
    `jlens.fit` by Neuronpedia (HF neuronpedia/jacobian-lens, 453 WikiText prompts, layers 0-40).
  * `jlens` (github.com/anthropics/jacobian-lens, v0.1.0, commit 581d398) provides the fit, the transport
    `JacobianLens.transport` (J_l h) and the readout `HFLensModel.unembed` (final norm, LM head, softcap).
    It contains no J-space decomposition.
  * The analysis in this repo applies the lens with its own numpy code (jspace_interp/nov5_jspace_decompose.py)
    and builds the J-lens vectors v_i = J_l^T ((1 + gamma) * W_U[i]) from it.

This script runs the official code on real NARCBench activations and compares:
  (a) the matrices read by `jlens.JacobianLens.load` and by our loader;
  (b) `JacobianLens.transport` and our `h @ J.T`;
  (c) `HFLensModel.unembed` (executed on the real Gemma-2 final norm and unembedding weights) and our readout,
      over the full vocabulary: largest logit difference, top-1 agreement, top-50 overlap, and agreement of the
      top-50 ranking our pipeline uses (raw scores) with the official ranking;
  (d) that the official pre-softcap logit of token i equals <v_i, h> / rms(J h), i.e. that our J-lens vectors
      are exactly the readout directions of the official lens.

The official `unembed` needs the model's final norm and LM head. To avoid loading the 9B model, the real HF
`Gemma2RMSNorm` module is filled with the model's final-norm weight and a linear layer with the model's
unembedding rows (both exported from the google/gemma-2-9b-it safetensors), and `HFLensModel.unembed` is
called on that pair. A token's logit does not depend on the other rows, so the vocabulary is processed in chunks.

  pip install "git+https://github.com/anthropics/jacobian-lens@581d398"
  python scripts/jspace_interp_verify_jlens.py --lens <gemma-2-9b-it_jacobian_lens.pt> \\
      --basis-dir data/j_basis/gemma2_9b --root results/jspace_interp/acts
"""
from __future__ import annotations

import argparse
import importlib.metadata as md
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from jspace_interp.nov5_jspace_decompose import load_lens, load_unembed, norm_scale, rms_norm, softcap  # noqa: E402
from jspace_interp.rows import load_extraction, select_rows  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lens", type=Path, required=True)
    ap.add_argument("--basis-dir", type=Path, default=ROOT / "data/j_basis/gemma2_9b")
    ap.add_argument("--root", type=Path, default=ROOT / "results/jspace_interp/acts")
    ap.add_argument("--layers", default="19,20,21,22,23")
    ap.add_argument("--positions", default="last,frac50")
    ap.add_argument("--n", type=int, default=48, help="activations per layer and position (half Core, half Transfer)")
    ap.add_argument("--softcap", type=float, default=None, help="final_logit_softcapping (default: from --config, else Gemma2Config)")
    ap.add_argument("--rms-eps", type=float, default=None, help="rms_norm_eps (default: from --config, else Gemma2Config)")
    ap.add_argument("--config", type=Path, default=None, help="the model's config.json")
    ap.add_argument("--vocab-chunk", type=int, default=16384)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=ROOT / "results/jspace_interp/jlens_check/jlens_check.json")
    a = ap.parse_args()

    import jlens
    import torch
    from transformers.models.gemma2.configuration_gemma2 import Gemma2Config
    from transformers.models.gemma2.modeling_gemma2 import Gemma2RMSNorm

    layers = [int(x) for x in a.layers.split(",")]
    cfg_src = "Gemma2Config defaults"
    cfg = Gemma2Config()
    cap, eps = cfg.final_logit_softcapping, cfg.rms_norm_eps
    if a.config and a.config.exists():
        c = json.loads(a.config.read_text())
        cap, eps, cfg_src = c.get("final_logit_softcapping", cap), c.get("rms_norm_eps", eps), str(a.config.name) + " of google/gemma-2-9b-it"
    cap = a.softcap if a.softcap is not None else cap
    eps = a.rms_eps if a.rms_eps is not None else eps

    t0 = time.time()
    lens = jlens.JacobianLens.load(str(a.lens))                 # official loader
    J_ours, lens_info = load_lens(a.lens, layers)              # ours
    W, _ = load_unembed(a.basis_dir)
    gamma = np.load(a.basis_dir / "final_norm_weight_fp32.npy").astype(np.float32)
    s = norm_scale(gamma, "gemma")
    d, V = W.shape[1], W.shape[0]
    norm = Gemma2RMSNorm(d, eps=eps)
    with torch.no_grad():
        norm.weight.copy_(torch.from_numpy(gamma))
    rng = np.random.default_rng(a.seed)
    res = {"jlens": {"version": md.version("jlens"), "source": "github.com/anthropics/jacobian-lens @ 581d398 (Initial release)",
                     "lens": repr(lens)},
           "lens_file": lens_info, "model_config": {"final_logit_softcapping": cap, "rms_norm_eps": eps, "source": cfg_src},
           "n_per_cell": a.n, "cells": []}

    acts = {}
    for split, phase in (("core", "discussion"), ("transfer", "final")):
        meta, npz = load_extraction(a.root / split / "original")
        idx = select_rows(meta, phase)
        pick = np.sort(rng.choice(idx, size=a.n // 2, replace=False))
        for pos in a.positions.split(","):
            for L in layers:
                acts.setdefault((pos, L), []).append(npz[f"{pos}__layer_{L}"][pick].astype(np.float32))
    acts = {k: np.concatenate(v) for k, v in acts.items()}
    probe_tokens = np.sort(rng.choice(V, size=256, replace=False))

    for (pos, L), h in acts.items():
        Jo = J_ours[L]
        same_matrix = float(np.abs(lens.jacobians[L].numpy() - Jo).max())
        ht = torch.from_numpy(h)
        t_off = lens.transport(ht, L)                           # official: residual @ J.T
        t_ours = h @ Jo.T
        z_ours = rms_norm(t_ours, s, eps=eps)
        off = np.empty((h.shape[0], V), dtype=np.float32)
        ours = np.empty((h.shape[0], V), dtype=np.float32)
        raw = np.empty((h.shape[0], V), dtype=np.float32)       # the ranking score our pipeline uses
        Zraw = t_ours * s[None, :]
        for v0 in range(0, V, a.vocab_chunk):
            v1 = min(V, v0 + a.vocab_chunk)
            Wc = np.asarray(W[v0:v1], dtype=np.float32)
            head = torch.nn.Linear(d, v1 - v0, bias=False)
            with torch.no_grad():
                head.weight.copy_(torch.from_numpy(Wc))
                stub = SimpleNamespace(_lm_head=head, _final_norm=norm, _logit_softcap=cap)
                off[:, v0:v1] = jlens.hf.HFLensModel.unembed(stub, t_off).numpy()   # official readout code
            ours[:, v0:v1] = softcap(z_ours @ Wc.T, cap)
            raw[:, v0:v1] = Zraw @ Wc.T
        top_off = np.argsort(-off, axis=1)[:, :50]
        top_ours = np.argsort(-ours, axis=1)[:, :50]
        top_raw = np.argsort(-raw, axis=1)[:, :50]
        overlap = [len(set(x) & set(y)) / 50 for x, y in zip(top_off, top_ours)]
        overlap_raw = [len(set(x) & set(y)) / 50 for x, y in zip(top_off, top_raw)]
        # (d) official pre-softcap logit of token i  ==  <v_i, h> / rms(J h)   with v_i = J^T ((1+gamma) * W_U[i])
        Wp = np.asarray(W[probe_tokens], dtype=np.float32)
        v = (Wp * s[None, :]) @ Jo                              # rows: J-lens vectors of the probe tokens
        rms = np.sqrt(np.mean(t_ours * t_ours, axis=1, keepdims=True) + eps)
        ours_pre = (h @ v.T) / rms
        off_c = np.clip(off[:, probe_tokens] / cap, -0.999999, 0.999999)
        off_pre = cap * np.arctanh(off_c)
        ok = np.abs(off[:, probe_tokens]) < 0.98 * cap          # away from the saturated part of the softcap
        cell = {"position": pos, "layer": L, "n": int(h.shape[0]),
                "max_abs_diff_matrix": same_matrix,
                "max_abs_diff_transport": float(np.abs(t_off.numpy() - t_ours).max()),
                "transport_scale": float(np.abs(t_ours).max()),
                "max_abs_diff_logits": float(np.abs(off - ours).max()),
                "logit_scale": float(np.abs(off).max()),
                "top1_agreement": float(np.mean(top_off[:, 0] == top_ours[:, 0])),
                "top50_overlap_min": float(min(overlap)), "top50_overlap_mean": float(np.mean(overlap)),
                "top1_agreement_raw_ranking": float(np.mean(top_off[:, 0] == top_raw[:, 0])),
                "top50_overlap_raw_ranking_min": float(min(overlap_raw)),
                "max_abs_diff_prelogit_vs_lens_vector": float(np.abs(off_pre - ours_pre)[ok].max()),
                "prelogit_scale": float(np.abs(ours_pre[ok]).max())}
        res["cells"].append(cell)
        print(json.dumps(cell), flush=True)

    c = res["cells"]
    res["summary"] = {k: float(max(x[k] for x in c)) for k in ("max_abs_diff_matrix", "max_abs_diff_transport", "max_abs_diff_logits",
                                                              "max_abs_diff_prelogit_vs_lens_vector")}
    res["summary"].update({k: float(min(x[k] for x in c)) for k in ("top1_agreement", "top50_overlap_min", "top1_agreement_raw_ranking",
                                                                    "top50_overlap_raw_ranking_min")})
    res["summary"]["n_activations"] = int(sum(x["n"] for x in c))
    res["seconds"] = round(time.time() - t0, 1)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1) + "\n")
    print("summary", json.dumps(res["summary"]))
    print("wrote", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
