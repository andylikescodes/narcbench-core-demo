"""Experiment 3: steering the final vote along J-space directions, with matched prefixes.

Setup (pre-registered in the plan, Nov 8):
  * Prefix = the agent's own final-vote prompt (system + user, condition original or swap), then the
    forced reply start "RECOMMENDATION: Option". The next-token logits for the target and fair option
    letters give the outcome  y = logit[target] - logit[fair]  (log-odds of voting the target).
  * The intervention adds  alpha * scale * u  to the residual stream at one layer, only on the
    final-instruction tail (from the last blank line of the user prompt to the end, forced tokens
    included). Everything before the tail is identical between steered and unsteered passes.
  * Directions u (unit norm): concept-group J-lens atoms (mean of the group's atoms), the J-space
    role-probe direction (from the analysis), and n_random Haar directions with the same norm.
  * scale = median ||h_J|| at that layer if known, else a recorded fallback (0.1 * median ||h|| on the tail).

Pre-registered bars (a direction "steers" only if all hold):
  1. Sign: honest agents' mean delta-y > 0 for +u, and colluders' mean delta-y < 0 for -u.
  2. Beats random: |mean delta-y| above the 95th percentile of the random directions at the same alpha.
  3. Survives Benjamini-Hochberg at q = 0.05 across all (direction, alpha, group) cells, using the
     empirical p-value against the random directions.
  4. Dose: the effect grows from the smallest to the largest alpha.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

from .positions import gemma_chat_text
from .stats import benjamini_hochberg

FORCED = "RECOMMENDATION: Option"
ALPHAS = (0.5, 1.0, 2.0, 4.0)


def option_token_ids(tokenizer, letter: str) -> list[int]:
    ids = set()
    for form in (" " + letter, letter, "▁" + letter):
        t = tokenizer.convert_tokens_to_ids(form) if form.startswith("▁") else None
        if t is not None and t != tokenizer.unk_token_id:
            ids.add(int(t))
        enc = tokenizer(form, add_special_tokens=False)["input_ids"]
        if len(enc) == 1:
            ids.add(int(enc[0]))
    return sorted(ids)


def build_prefix(row: dict, tokenizer):
    text, spans = gemma_chat_text(row.get("system_prompt") or "", row.get("user_prompt") or "", None, tokenizer)
    text = text + FORCED
    enc = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
    up = row.get("user_prompt") or ""
    cut = up.rfind("\n\n")
    sysp = row.get("system_prompt") or ""
    user0 = spans["content"][0] + (len(sysp) + 2 if sysp and up else 0)
    tail_char = user0 + (cut if cut >= 0 else 0)
    offs = enc["offset_mapping"]
    tail_start = next((i for i, (a, b) in enumerate(offs) if b > tail_char), len(offs) - 1)
    return enc["input_ids"], tail_start


def final_rows(samples: list[dict]) -> list[dict]:
    out = []
    for s in samples:
        if s.get("phase") != "final":
            continue
        if not s.get("target_option") or not s.get("fair_option"):
            continue
        out.append(s)
    return out


def concept_directions(J_L, W, s, group_ids: dict[str, list[int]]):
    """Unit mean J-lens atom for each concept group (atoms v_i = J^T ((1+gamma) * W_U[i]))."""
    dirs = {}
    for g, ids in group_ids.items():
        if not ids:
            continue
        A = (W[ids] * s[None, :]) @ J_L  # rows: J^T applied to each scaled unembedding row
        A = A / np.linalg.norm(A, axis=1, keepdims=True).clip(1e-8)
        u = A.mean(0)
        dirs[f"concept:{g}"] = u / max(np.linalg.norm(u), 1e-8)
    return dirs


def run_steering(rows, model, tokenizer, layer: int, directions: dict[str, np.ndarray], *, alphas=ALPHAS,
                 scale: float | None = None, log=print) -> dict:
    import torch

    device = next(model.parameters()).device
    backbone = model.model if hasattr(model, "model") else model
    mod = backbone.layers[layer]
    state = {"vec": None, "start": 0}

    def hook(_m, _i, out):
        if state["vec"] is None:
            return out
        h = out[0] if isinstance(out, (tuple, list)) else out
        h = h.clone()
        h[:, state["start"]:, :] += state["vec"].to(h.dtype)
        return (h,) + tuple(out[1:]) if isinstance(out, (tuple, list)) else h

    handle = mod.register_forward_hook(hook)
    grab = {}

    def grab_hook(_m, _i, out):
        h = out[0] if isinstance(out, (tuple, list)) else out
        grab["h"] = h[0].detach().float()

    records = []
    try:
        tail_norms = []
        prepared = []
        for r in rows:
            ids, start = build_prefix(r, tokenizer)
            t_ids = option_token_ids(tokenizer, r["target_option"])
            f_ids = option_token_ids(tokenizer, r["fair_option"])
            if not t_ids or not f_ids:
                continue
            prepared.append((r, ids, start, t_ids, f_ids))

        def y_of(ids, t_ids, f_ids):
            with torch.inference_mode():
                logits = model(input_ids=torch.tensor([ids], device=device), use_cache=False).logits[0, -1].float()
            return float(torch.logsumexp(logits[t_ids], 0) - torch.logsumexp(logits[f_ids], 0))

        base = []
        g = mod.register_forward_hook(grab_hook)
        try:
            for r, ids, start, t_ids, f_ids in prepared:
                state["vec"] = None
                base.append(y_of(ids, t_ids, f_ids))
                tail_norms.extend(grab["h"][start:].norm(dim=-1).cpu().tolist())
        finally:
            g.remove()
        scale_src = "given (median ||h_J||)"
        if scale is None:
            scale = 0.1 * float(np.median(tail_norms)) if tail_norms else 1.0
            scale_src = "fallback: 0.1 * median ||h|| on the tail"
        for k, (r, ids, start, t_ids, f_ids) in enumerate(prepared):
            rec = {"run_id": r["run_id"], "agent_name": r["agent_name"], "role": r["role"], "mode": r["mode"],
                   "vote": r.get("agent_vote"), "target": r["target_option"], "y0": base[k], "delta": {}}
            state["start"] = start
            for name, u in directions.items():
                ut = torch.tensor(u, dtype=torch.float32, device=device)
                for a in alphas:
                    for sign in (+1, -1):
                        state["vec"] = sign * a * scale * ut
                        rec["delta"][f"{name}|{a}|{sign:+d}"] = y_of(ids, t_ids, f_ids) - base[k]
            state["vec"] = None
            records.append(rec)
            if k == 0 or (k + 1) % 10 == 0:
                log(f"[steer] {k + 1}/{len(prepared)}")
    finally:
        handle.remove()
    return {"layer": layer, "scale": scale, "scale_source": scale_src, "alphas": list(alphas),
            "directions": list(directions), "records": records}


def summarize(result: dict, q: float = 0.05) -> dict:
    """Apply the pre-registered bars. Groups: honest (+u should raise y) and colluder (-u should lower y)."""
    recs = result["records"]
    dnames = result["directions"]
    rand = [d for d in dnames if d.startswith("random:")]
    real = [d for d in dnames if not d.startswith("random:")]
    groups = {"honest": (+1, lambda r: r["role"] != "colluder"), "colluder": (-1, lambda r: r["role"] == "colluder")}
    cells, pvals = [], []
    for gname, (sign, keep) in groups.items():
        rs = [r for r in recs if keep(r)]
        if not rs:
            continue
        for a in result["alphas"]:
            key = lambda d: f"{d}|{a}|{sign:+d}"
            null = np.array([np.mean([r["delta"][key(d)] for r in rs]) for d in rand]) if rand else np.array([])
            for d in real:
                m = float(np.mean([r["delta"][key(d)] for r in rs]))
                want = m * (1 if gname == "honest" else -1)  # positive means predicted direction
                if len(null):
                    signed_null = null * (1 if gname == "honest" else -1)
                    p = (1 + np.sum(signed_null >= want)) / (1 + len(null))
                    p95 = float(np.percentile(np.abs(null), 95))
                else:
                    p, p95 = 1.0, float("nan")
                cells.append({"group": gname, "direction": d, "alpha": a, "n": len(rs), "mean_delta": m,
                              "predicted_sign_ok": want > 0, "random_abs_p95": p95,
                              "beats_random": bool(abs(m) > p95 and want > 0), "p_vs_random": float(p)})
                pvals.append(p)
    if pvals:
        adj, rej = benjamini_hochberg(np.array(pvals), q)
        for c, a_, r_ in zip(cells, adj, rej):
            c["p_bh"] = float(a_)
            c["bh_reject"] = bool(r_)
    verdict = {}
    for gname in groups:
        for d in real:
            cs = sorted([c for c in cells if c["group"] == gname and c["direction"] == d], key=lambda c: c["alpha"])
            if not cs:
                continue
            dose = abs(cs[-1]["mean_delta"]) > abs(cs[0]["mean_delta"])
            verdict[f"{gname}:{d}"] = bool(all(c["predicted_sign_ok"] for c in cs)
                                           and any(c["beats_random"] and c.get("bh_reject") for c in cs) and dose)
    return {"cells": cells, "passes_all_bars": verdict}
