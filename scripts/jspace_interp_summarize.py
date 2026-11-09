#!/usr/bin/env python3
"""Summarise the jspace-interp results into SUMMARY.json, RESULTS.md and LaTeX table fragments for the paper.

Usage
  python scripts/jspace_interp_summarize.py                       # reads results/jspace_interp/{gemma2_9b/*/sweep_results.json,steer/*.json}
  python scripts/jspace_interp_summarize.py --files A.json B.json --steer S.json --out DIR --summary-dir DIR

Writes <summary-dir>/SUMMARY.json and RESULTS.md, and <out>/tab_*.tex (fragments the paper \\input{}s). Every cell of the
planned grid (conditions x positions x layers x k, plus the prompt_only / nosys controls and the steering runs) that is
missing from the inputs is listed under "not_run" so the paper can say so.
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
POS_ORDER = ["last", "eot", "prevote", "reply_mean", "frac25", "frac50", "frac75", "prompt_end"]
GROUPS = ["deception_secrecy", "coordination", "steering_persuasion", "honesty_standards"]
GROUP_SHORT = {"deception_secrecy": "deception/secrecy", "coordination": "coordination",
               "steering_persuasion": "steering/persuasion", "honesty_standards": "honesty/standards"}
NOV5 = {"full_c2t": 0.902, "J25_c2t": 0.755, "pca25_c2t": 0.856, "randsub_one_draw": 0.647}


# ----------------------------------------------------------------------------------------------------------------- utils
def tex(s: str) -> str:
    return str(s).replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")


def tex_tok(t: str) -> str:
    """A token string inside \\texttt{}: escape TeX specials, show the SentencePiece space and newlines."""
    if t is None or t == "":
        return r"\textit{(empty)}"
    out = []
    for ch in str(t):
        if ch == "\u2581":
            out.append(r"\textvisiblespace{}")
        elif ch == "\n":
            out.append(r"\textbackslash{}n")
        elif ch in "\\{}$&#%_^~":
            out.append({"\\": r"\textbackslash{}", "^": r"\^{}", "~": r"\~{}"}.get(ch, "\\" + ch))
        elif ord(ch) < 32:
            out.append("?")
        else:
            out.append(ch)
    return "".join(out)


def f3(x, nd=3) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "--"
    return f"{x:.{nd}f}"


def ci(v) -> str:
    if not v or v[0] is None:
        return "--"
    return f"[{v[0]:.2f}, {v[1]:.2f}]"


def mark(b) -> str:
    return r"$\checkmark$" if b else "--"


def bh(pvals):
    """Benjamini-Hochberg adjusted p-values (same order as the input)."""
    p = np.asarray(pvals, float)
    n = len(p)
    if n == 0:
        return []
    order = np.argsort(p)
    ranked = p[order] * n / (np.arange(n) + 1)
    adj = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.minimum(adj, 1.0)
    return [float(x) for x in out]


def pos_key(p):
    return POS_ORDER.index(p) if p in POS_ORDER else 99


# ----------------------------------------------------------------------------------------------------------------- load
def load_sweeps(files):
    cells, exp1, role_dirs, settings = {}, {}, [], []
    ks, positions = set(), set()
    for f in files:
        try:
            d = json.load(open(f))
        except Exception as e:  # noqa: BLE001
            print(f"[skip] {f}: {e}", file=sys.stderr)
            continue
        settings.append({"file": str(f), **{k: d.get(k) for k in ("ks", "layers", "positions")},
                         "n_random": (d.get("settings") or {}).get("n_random"), "lens": (d.get("lens") or {}).get("path")})
        for c in d.get("cells", []):
            key = (c["condition"], c["position"], int(c["layer"]))
            cells[key] = c
            positions.add(c["position"])
            for kk in c:
                if kk.startswith("k") and kk[1:].isdigit():
                    ks.add(int(kk[1:]))
        for e in d.get("exp1", []):
            exp1[(e["condition"], e["position"], int(e["layer"]))] = e
        role_dirs += d.get("role_dirs", [])
    return cells, exp1, role_dirs, settings, sorted(ks), sorted(positions, key=pos_key)


def load_steer(files):
    out = {}
    for f in files:
        try:
            d = json.load(open(f))
        except Exception as e:  # noqa: BLE001
            print(f"[skip] {f}: {e}", file=sys.stderr)
            continue
        cfg = d.get("config") or {}
        cond = cfg.get("condition") or Path(f).stem.split("_")[-2]
        split = cfg.get("split") or (Path(f).stem.split("_")[1] if Path(f).stem.count("_") >= 3 else "core")
        out[(split, cond, int(d["layer"]))] = {"file": str(f), **d}
    return out


# ----------------------------------------------------------------------------------------------------------------- rows
def arm(cell, k, name):
    return (cell.get(f"k{k}") or {}).get(name)


def cell_row(cell, k):
    kb = cell.get(f"k{k}")
    if not kb:
        return None
    full = cell["full"]
    J, Jf, Cf = kb.get(f"J{k}", {}), kb.get(f"J{k}_fill", {}), kb.get(f"C{k}_fill", {})
    pca, ff, sh = kb.get(f"pca{k}", {}), kb.get(f"fixedframe{k}", {}), kb.get(f"shuffled{k}", {})
    rs, ra = kb.get("randsub", {}), kb.get("randatoms", {})
    jf, jp = Jf.get("core_to_transfer"), J.get("core_to_transfer")
    rs95, ra95 = rs.get("c2t_p95"), ra.get("c2t_p95")
    r = {
        "condition": cell["condition"], "position": cell["position"], "layer": cell["layer"], "k": k,
        "n_core_rows": cell.get("n_core_rows"), "n_transfer_rows": cell.get("n_transfer_rows"), "n_warned_rows": cell.get("n_warned_rows"),
        "full": full.get("core_to_transfer"), "full_ci": full.get("core_to_transfer_ci95"),
        "full_lodo": full.get("core_lodo"), "full_lofo": full.get("transfer_lofo"),
        "J_fill": jf, "J_fill_ci": Jf.get("core_to_transfer_ci95"), "J_fill_lodo": Jf.get("core_lodo"), "J_fill_lofo": Jf.get("transfer_lofo"),
        "J_pursuit": jp, "J_pursuit_ci": J.get("core_to_transfer_ci95"), "J_pursuit_lodo": J.get("core_lodo"), "J_pursuit_lofo": J.get("transfer_lofo"),
        "C_fill": Cf.get("core_to_transfer"), "C_fill_lodo": Cf.get("core_lodo"), "C_fill_lofo": Cf.get("transfer_lofo"),
        "pca": pca.get("core_to_transfer"), "pca_ci": pca.get("core_to_transfer_ci95"), "pca_lodo": pca.get("core_lodo"), "pca_lofo": pca.get("transfer_lofo"),
        "fixedframe": ff.get("core_to_transfer"), "shuffled": sh.get("core_to_transfer"),
        "randsub_n": rs.get("n"), "randsub_mean": rs.get("c2t_mean"), "randsub_p95": rs95,
        "randatoms_n": ra.get("n"), "randatoms_mean": ra.get("c2t_mean"), "randatoms_p95": ra95,
        "p_J_fill_vs_randsub": (rs.get("J_fill_percentile") or [None, None])[1],
        "p_J_fill_vs_randatoms": (ra.get("J_fill_percentile") or [None, None])[1],
        "p_J_pursuit_vs_randsub": (rs.get("J_pursuit_percentile") or [None, None])[1],
        "p_J_pursuit_vs_randatoms": (ra.get("J_pursuit_percentile") or [None, None])[1],
        "J_fill_minus_full_ci": kb.get("J_fill_minus_full_ci95"), "J_pursuit_minus_full_ci": kb.get("J_pursuit_minus_full_ci95"),
        "J_fill_minus_pca_ci": kb.get("J_fill_minus_pca_ci95"), "C_fill_minus_J_fill_ci": kb.get("C_fill_minus_J_fill_ci95"),
        "recovery_J_fill": kb.get("recovery_J_fill"), "recovery_J_pursuit": kb.get("recovery_J_pursuit"),
        "pursuit_steps_core_mean": cell.get("pursuit_steps_core_mean"),
        "mv": {"full": full.get("matched_vocab"), "J_fill": Jf.get("matched_vocab"), "J_pursuit": J.get("matched_vocab"),
               "pca": pca.get("matched_vocab"), "C_fill": Cf.get("matched_vocab"), "randsub": rs.get("matched_vocab")},
    }
    both = jf is not None and rs95 is not None and ra95 is not None
    r["J_fill_clears_both_nulls"] = bool(both and jf > rs95 and jf > ra95)
    r["J_pursuit_clears_both_nulls"] = bool(jp is not None and rs95 is not None and ra95 is not None and jp > rs95 and jp > ra95)
    pc = kb.get("J_fill_minus_pca_ci95")
    r["J_fill_beats_pca"] = bool(pc and pc[0] > 0)
    r["pca_beats_J_fill"] = bool(pc and pc[1] < 0)
    fc = kb.get("J_fill_minus_full_ci95")
    r["full_beats_J_fill"] = bool(fc and fc[1] < 0)
    return r


def band(rows, key):
    v = [r[key] for r in rows if r.get(key) is not None]
    return float(np.mean(v)) if v else None


# ----------------------------------------------------------------------------------------------------------------- tables
def tab_headline(rows, conds, positions, layers, k, out):
    """Per condition x position at one k: band mean over layers of full, J (fill, pursuit), complement, PCA, random nulls,
    and how many layers clear both nulls."""
    lines = [r"\begin{tabular}{llcccccccc}\toprule",
             r"Cond. & Position & Full & J fill & J pursuit & Compl. & PCA-$k$ & Rand.\ sub.\ mean (p95) & Rand.\ atoms p95 & Layers clearing both\\\midrule"]
    for cond in conds:
        first = True
        for pos in positions:
            rr = [r for r in rows if r["condition"] == cond and r["position"] == pos and r["k"] == k]
            if not rr:
                continue
            clears = sum(r["J_fill_clears_both_nulls"] for r in rr)
            lines.append(" & ".join([
                tex(cond) if first else "", tex(pos), f3(band(rr, "full")), f3(band(rr, "J_fill")), f3(band(rr, "J_pursuit")),
                f3(band(rr, "C_fill")), f3(band(rr, "pca")),
                f"{f3(band(rr, 'randsub_mean'))} ({f3(band(rr, 'randsub_p95'))})", f3(band(rr, "randatoms_p95")),
                f"{clears}/{len(rr)}"]) + r"\\")
            first = False
        lines.append(r"\midrule")
    if lines[-1] == r"\midrule":
        lines.pop()
    lines.append(r"\bottomrule\end{tabular}")
    (out / f"tab_headline_k{k}.tex").write_text("\n".join(lines) + "\n")


def tab_ksweep(rows, cond, pos, ks, out):
    lines = [r"\begin{tabular}{lcccccccc}\toprule",
             r"$k$ & Full & J fill & J pursuit & Compl. & PCA-$k$ & Rand.\ sub.\ p95 & Rand.\ atoms p95 & J fill $-$ PCA (95\% CI)\\\midrule"]
    for k in ks:
        rr = [r for r in rows if r["condition"] == cond and r["position"] == pos and r["k"] == k]
        if not rr:
            continue
        # band mean over layers; the paired CI is reported for the layer with the best J fill
        best = max(rr, key=lambda r: (r["J_fill"] if r["J_fill"] is not None else -1))
        lines.append(" & ".join([str(k), f3(band(rr, "full")), f3(band(rr, "J_fill")), f3(band(rr, "J_pursuit")), f3(band(rr, "C_fill")),
                                 f3(band(rr, "pca")), f3(band(rr, "randsub_p95")), f3(band(rr, "randatoms_p95")),
                                 f"{ci(best['J_fill_minus_pca_ci'])} (L{best['layer']})"]) + r"\\")
    lines.append(r"\bottomrule\end{tabular}")
    (out / f"tab_ksweep_{cond}_{pos}.tex").write_text("\n".join(lines) + "\n")


def tab_layers(rows, cond, pos, k, out):
    lines = [r"\begin{tabular}{lcccccccc}\toprule",
             r"Layer & Full & J fill & J pursuit & PCA-$k$ & Rand.\ sub.\ p95 & Rand.\ atoms p95 & $p$ vs rand.\ sub. & Clears both\\\midrule"]
    for r in sorted([r for r in rows if r["condition"] == cond and r["position"] == pos and r["k"] == k], key=lambda r: r["layer"]):
        lines.append(" & ".join([str(r["layer"]), f"{f3(r['full'])} {ci(r['full_ci'])}", f"{f3(r['J_fill'])} {ci(r['J_fill_ci'])}", f3(r["J_pursuit"]),
                                 f3(r["pca"]), f3(r["randsub_p95"]), f3(r["randatoms_p95"]), f3(r["p_J_fill_vs_randsub"]),
                                 mark(r["J_fill_clears_both_nulls"])]) + r"\\")
    lines.append(r"\bottomrule\end{tabular}")
    (out / f"tab_layers_{cond}_{pos}_k{k}.tex").write_text("\n".join(lines) + "\n")


def tab_controls(rows, layers, k, out):
    """prompt_only (prompt_end) and nosys (last) next to original/swap at the last token, per layer, at one k."""
    arms = [("original", "last"), ("swap", "last"), ("nosys", "last"), ("prompt_only", "prompt_end")]
    lines = [r"\begin{tabular}{l" + "cc" * len(arms) + r"}\toprule", "Layer & " + " & ".join(
        rf"\multicolumn{{2}}{{c}}{{{tex(c)}/{tex(p)}}}" for c, p in arms) + r"\\",
        " & " + " & ".join(["Full & J fill"] * len(arms)) + r"\\\midrule"]
    for L in layers:
        cells = []
        for c, p in arms:
            rr = [r for r in rows if r["condition"] == c and r["position"] == p and r["layer"] == L and r["k"] == k]
            cells += [f3(rr[0]["full"]), f3(rr[0]["J_fill"])] if rr else ["n/r", "n/r"]
        lines.append(f"{L} & " + " & ".join(cells) + r"\\")
    lines.append(r"\bottomrule\end{tabular}")
    (out / f"tab_controls_k{k}.tex").write_text("\n".join(lines) + "\n")


def tab_protocols(rows, k, out):
    """Per condition and read position (last, prevote, prompt_end): band mean over layers of Core LODO, Core->Transfer and Transfer LOFO
    for the full residual, the filled J-only arm and PCA-k. Within-split numbers show whether a probe still finds colluders when the
    instruction is removed (swap, nosys) even if nothing transfers."""
    arms = [("full", "Full"), ("J_fill", f"J fill $k{{=}}{k}$"), ("pca", f"PCA-{k}")]
    lines = [r"\begin{tabular}{ll" + "ccc" * len(arms) + r"}\toprule", "Cond. & Pos. & " + " & ".join(
        rf"\multicolumn{{3}}{{c}}{{{lab}}}" for _, lab in arms) + r"\\",
        " & & " + " & ".join(["LODO & C$\\to$T & LOFO"] * len(arms)) + r"\\\midrule"]
    for cond in ["original", "swap", "nosys", "prompt_only"]:
        first = True
        for pos in ["last", "prevote", "prompt_end"]:
            rr = [r for r in rows if r["condition"] == cond and r["position"] == pos and r["k"] == k]
            if not rr:
                continue
            cells = []
            for name, _ in arms:
                cells += [f3(band(rr, f"{name}_lodo")), f3(band(rr, name)), f3(band(rr, f"{name}_lofo"))]
            lines.append(" & ".join([tex(cond) if first else "", tex(pos)] + cells) + r"\\")
            first = False
    lines.append(r"\bottomrule\end{tabular}")
    (out / f"tab_protocols_k{k}.tex").write_text("\n".join(lines) + "\n")


def mv_fields(m):
    if not m:
        return None
    return {"colluder_vs_warned": m.get("auroc_colluder_vs_warned"), "colluder_vs_warned_p": m.get("auroc_colluder_vs_warned_perm_p"),
            "warned_vs_unwarned": m.get("auroc_warned_vs_unwarned"), "warned_vs_unwarned_p": m.get("p_warned_vs_unwarned_within_episode"),
            "warned_vs_control": m.get("auroc_warned_vs_control_honest"), "colluder_vs_control": m.get("auroc_colluder_vs_control_honest"),
            "echo_ratio": m.get("echo_ratio"),
            "ep_warned_vs_control": (m.get("episode") or {}).get("warned_vs_control_primary"),
            "ep_collusion_vs_warned": (m.get("episode") or {}).get("collusion_vs_warned_primary"),
            "ep_collusion_vs_control": (m.get("episode") or {}).get("collusion_vs_control_primary"),
            "n_warned_episodes": m.get("n_warned_episodes")}


def tab_matched_vocab(rows, cells_sel, k, out):
    lines = [r"\begin{tabular}{llcccccc}\toprule",
             r"Cell & Arm & Coll.\ vs warned & Warned vs unwarned ($p$) & Warned vs control & Echo & Ep.\ warned vs control & Ep.\ coll.\ vs warned\\\midrule"]
    for (cond, pos, L) in cells_sel:
        rr = [r for r in rows if (r["condition"], r["position"], r["layer"], r["k"]) == (cond, pos, L, k)]
        if not rr or not rr[0]["mv"].get("full"):
            continue
        r = rr[0]
        first = True
        for name, label in [("full", "Full"), ("J_fill", f"J fill $k{{=}}{k}$"), ("pca", f"PCA-{k}")]:
            m = mv_fields(r["mv"].get(name))
            if not m:
                continue
            lines.append(" & ".join([f"{tex(cond)}/{tex(pos)}/L{L}" if first else "", label, f3(m["colluder_vs_warned"]),
                                     f"{f3(m['warned_vs_unwarned'])} ({f3(m['warned_vs_unwarned_p'], 2)})", f3(m["warned_vs_control"]),
                                     f3(m["echo_ratio"], 2), f3(m["ep_warned_vs_control"]), f3(m["ep_collusion_vs_warned"])]) + r"\\")
            first = False
        rs = r["mv"].get("randsub")
        if rs:
            g = lambda key, stat: (rs.get(key) or {}).get(stat)  # noqa: E731
            echo_draws = (rs.get("echo_ratio") or {}).get("draws")
            echo_med = float(np.median(echo_draws)) if echo_draws else None  # the ratio blows up for weak probes, so the median
            lines.append(" & ".join(["", f"Random {k}-dim, mean of {rs.get('n')} (echo: median)", f3(g("auroc_colluder_vs_warned", "mean")),
                                     f3(g("auroc_warned_vs_unwarned", "mean")), f3(g("auroc_warned_vs_control_honest", "mean")),
                                     f3(echo_med, 2), f3(g("episode_warned_vs_control_primary", "mean")), "n/a"]) + r"\\")
        lines.append(r"\midrule")
    if lines[-1] == r"\midrule":
        lines.pop()
    lines.append(r"\bottomrule\end{tabular}")
    (out / f"tab_matched_vocab_k{k}.tex").write_text("\n".join(lines) + "\n")


def exp1_rows(exp1):
    """Concept-atom mass differences with BH over all (cell, split, group) tests."""
    recs = []
    for (cond, pos, L), e in sorted(exp1.items(), key=lambda kv: (kv[0][0], pos_key(kv[0][1]), kv[0][2])):
        for split in ("core", "transfer"):
            blk = e.get(split) or {}
            for g in GROUPS:
                a = blk.get(f"atoms:{g}")
                if not a:
                    continue
                recs.append({"condition": cond, "position": pos, "layer": L, "split": split, "group": g,
                             "colluder_minus_honest": a.get("colluder_minus_honest"), "p_within": a.get("p_within"),
                             "colluder_minus_control": a.get("colluder_minus_control"), "p_strat": a.get("p_strat"),
                             "mean_colluder": a.get("mean_colluder"), "mean_honest": a.get("mean_honest_in_collusion"), "mean_control": a.get("mean_control")})
    ps = [r["p_within"] if r["p_within"] is not None else 1.0 for r in recs]
    for r, q in zip(recs, bh(ps)):
        r["p_within_bh"] = q
    return recs


def tab_exp1(recs, exp1, cells_sel, out):
    lines = [r"\begin{tabular}{llcccc}\toprule", r"Cell & Group & Colluder & Honest (same ep.) & Control & BH $p$ (within ep.)\\\midrule"]
    for (cond, pos, L) in cells_sel:
        rr = [r for r in recs if (r["condition"], r["position"], r["layer"], r["split"]) == (cond, pos, L, "core")]
        if not rr:
            continue
        first = True
        for r in rr:
            lines.append(" & ".join([f"{tex(cond)}/{tex(pos)}/L{L}" if first else "", GROUP_SHORT[r["group"]], f3(r["mean_colluder"], 4),
                                     f3(r["mean_honest"], 4), f3(r["mean_control"], 4), f3(r["p_within_bh"], 3)]) + r"\\")
            first = False
        e = exp1.get((cond, pos, L), {}).get("core", {})
        mca = e.get("most_common_atoms") or []
        if mca:
            toks = ", ".join(f"\\texttt{{{tex_tok(t[0])}}} ({t[1]})" for t in mca[:6])
            lines.append(r"\multicolumn{6}{l}{\footnotesize most common atoms (Core): " + toks + r"}\\")
        lines.append(r"\midrule")
    if lines[-1] == r"\midrule":
        lines.pop()
    lines.append(r"\bottomrule\end{tabular}")
    (out / "tab_exp1_concepts.tex").write_text("\n".join(lines) + "\n")


def steer_summary(st):
    cells = (st.get("summary") or {}).get("cells", [])
    passes = (st.get("summary") or {}).get("passes_all_bars", {})
    alphas = st.get("alphas") or sorted({c["alpha"] for c in cells})
    dirs = [d for d in (st.get("directions") or []) if not str(d).startswith("random:")]
    rnd = [d for d in (st.get("directions") or []) if str(d).startswith("random:")]
    groups = sorted({c["group"] for c in cells})
    recs = st.get("records", [])
    # baseline forced-vote logit difference y0 = logit[target] - logit[fair] per group, and vote flips under the steered sign
    base = {}
    for g in groups:
        y0 = np.array([r["y0"] for r in recs if r["role"] == g], float)
        base[g] = {"n": int(len(y0)), "y0_mean": float(y0.mean()) if len(y0) else None, "frac_target": float(np.mean(y0 > 0)) if len(y0) else None}
    def flips(d, g, a):
        sign = "+1" if g == "honest" else "-1"
        pairs = [(r["y0"], r["y0"] + r["delta"][f"{d}|{a}|{sign}"]) for r in recs if r["role"] == g and f"{d}|{a}|{sign}" in r["delta"]]
        if not pairs:
            return None
        y0 = np.array([p[0] for p in pairs]); y1 = np.array([p[1] for p in pairs])
        return float(np.mean((y0 <= 0) & (y1 > 0))) if g == "honest" else float(np.mean((y0 > 0) & (y1 <= 0)))
    rand_by_alpha = {}
    for g in groups:
        sign = "+1" if g == "honest" else "-1"
        for a in alphas:
            means = [np.mean([r["delta"].get(f"{x}|{a}|{sign}", np.nan) for r in recs if r["role"] == g]) for x in rnd]
            means = [m for m in means if m == m]
            if means:
                rand_by_alpha[f"{g}|{a}"] = {"mean_delta_min": float(min(means)), "mean_delta_max": float(max(means)),
                                             "abs_p95": float(np.percentile(np.abs(means), 95))}
    rows = []
    for d in dirs:
        for g in groups:
            cc = {c["alpha"]: c for c in cells if c["direction"] == d and c["group"] == g}
            if not cc:
                continue
            deltas = [cc[a]["mean_delta"] if a in cc else None for a in alphas]
            mags = [abs(x) for x in deltas if x is not None]
            rows.append({"direction": d, "group": g, "n": next(iter(cc.values()))["n"],
                         "mean_delta_by_alpha": dict(zip([str(a) for a in alphas], deltas)),
                         "flip_by_alpha": {str(a): flips(d, g, a) for a in alphas},
                         "predicted_sign_ok_all": all(c["predicted_sign_ok"] for c in cc.values()),
                         "beats_random_any": any(c["beats_random"] for c in cc.values()),
                         "beats_random_all": all(c["beats_random"] for c in cc.values()),
                         "bh_reject_any": any(c["bh_reject"] for c in cc.values()),
                         "dose_monotone": bool(len(mags) >= 2 and all(b >= a for a, b in zip(mags, mags[1:]))),
                         "passes_all_bars": passes.get(f"{g}:{d}")})
    return {"layer": st.get("layer"), "scale": st.get("scale"), "scale_source": st.get("scale_source"), "alphas": alphas,
            "signs": st.get("signs"), "prefix_cache": st.get("prefix_cache"), "cache_check": st.get("cache_check"),
            "n_records": len(st.get("records", [])), "n_random": sum(str(d).startswith("random:") for d in (st.get("directions") or [])),
            "baseline": base, "random_by_group_alpha": rand_by_alpha, "rows": rows, "file": st.get("file")}


def tab_steer(steers, out):
    lines = [r"\begin{tabular}{lllccccccccc}\toprule",
             r"Run & Direction & Group ($n$) & $\Delta y$ at $\alpha{=}1$ & $\alpha{=}2$ & $\alpha{=}4$ & Flips $\alpha{=}2$ & $\alpha{=}4$ & Sign & $>$ rand.\ p95 & BH & All bars\\\midrule"]
    for key, s in steers.items():
        split, cond, L = key
        first = True
        for r in s["rows"]:
            d = r["mean_delta_by_alpha"]; f = r["flip_by_alpha"]
            lines.append(" & ".join([f"{tex(cond)} L{L}" if first else "", tex(r["direction"].replace("concept:", "").replace("jspace_role_probe", "J role probe")),
                                     f"{r['group']} ({r['n']})", f3(d.get("1.0"), 2), f3(d.get("2.0"), 2), f3(d.get("4.0"), 2),
                                     f3(f.get("2.0"), 2), f3(f.get("4.0"), 2), mark(r["predicted_sign_ok_all"]),
                                     mark(r["beats_random_all"]), mark(r["bh_reject_any"]), mark(r["passes_all_bars"])]) + r"\\")
            first = False
        for g, b in s["baseline"].items():
            ra = s["random_by_group_alpha"].get(f"{g}|4.0", {})
            lines.append(r"\multicolumn{12}{l}{\footnotesize " + f"{g}: baseline $y_0$ mean {f3(b['y0_mean'], 2)}, share with $y_0>0$ {f3(b['frac_target'], 2)}; "
                         f"20 random directions at $\\alpha{{=}}4$: mean $\\Delta y$ from {f3(ra.get('mean_delta_min'), 2)} to {f3(ra.get('mean_delta_max'), 2)}" + r"}\\")
        lines.append(r"\midrule")
    if lines[-1] == r"\midrule":
        lines.pop()
    lines.append(r"\bottomrule\end{tabular}")
    (out / "tab_steer.tex").write_text("\n".join(lines) + "\n")


# ----------------------------------------------------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=ROOT / "results/jspace_interp")
    ap.add_argument("--files", nargs="*", default=None, help="sweep_results.json files (default: <root>/gemma2_9b/*/sweep_results.json)")
    ap.add_argument("--steer", nargs="*", default=None, help="steering result files (default: <root>/steer/*.json)")
    ap.add_argument("--pod-json", type=Path, default=None, help="pod.json with the planned settings (default: newest under <root>/pod_runs)")
    ap.add_argument("--out", type=Path, default=ROOT / "docs/jspace_interp/tables")
    ap.add_argument("--summary-dir", type=Path, default=None, help="where SUMMARY.json and RESULTS.md go (default: <root>/gemma2_9b)")
    ap.add_argument("--k", type=int, default=25, help="k for the headline tables")
    ap.add_argument("--plan-conditions", default="original,swap")
    ap.add_argument("--plan-positions", default=None, help="override the planned positions (default: pod.json or the 7 standard ones)")
    ap.add_argument("--plan-layers", default=None)
    ap.add_argument("--plan-ks", default="8,25,50,100")
    a = ap.parse_args()

    # the pod writes gemma2_9b/<cond>_<pos>/sweep_results.json; cells finished later on another machine go in subdirectories
    # (gemma2_9b/<cond>_<pos>/local_L22/sweep_results.json) and override the pod's cell of the same key
    files = a.files if a.files is not None else sorted(glob.glob(str(a.root / "gemma2_9b" / "**" / "sweep_results.json"), recursive=True))
    steer_files = a.steer if a.steer is not None else sorted(glob.glob(str(a.root / "steer" / "*.json")))
    pod_json = a.pod_json
    if pod_json is None:
        cands = sorted(glob.glob(str(a.root / "pod_runs" / "*" / "pod.json")))
        pod_json = Path(cands[-1]) if cands else None
    plan = {"positions": "last,prevote,eot,reply_mean,frac50,frac25,frac75", "layers": "19-23", "steer_runs": "swap:22,original:22,swap:23,original:23"}
    if pod_json and pod_json.exists():
        plan.update({k: v for k, v in (json.load(open(pod_json)).get("settings") or {}).items() if k in plan})
    if a.plan_positions:
        plan["positions"] = a.plan_positions
    if a.plan_layers:
        plan["layers"] = a.plan_layers
    lo, hi = plan["layers"].split("-") if "-" in plan["layers"] else (None, None)
    plan_layers = list(range(int(lo), int(hi) + 1)) if lo else [int(x) for x in plan["layers"].split(",")]
    plan_positions = plan["positions"].split(",")
    plan_ks = [int(x) for x in a.plan_ks.split(",")]
    plan_conds = a.plan_conditions.split(",")

    cells, exp1, role_dirs, settings, ks, positions = load_sweeps(files)
    steers_raw = load_steer(steer_files)
    rows = [r for c in cells.values() for k in ks for r in [cell_row(c, k)] if r]
    layers = sorted({c["layer"] for c in cells.values()})
    conds = [c for c in ["original", "swap", "nosys", "prompt_only"] if any(r["condition"] == c for r in rows)]

    # planned grid vs what ran
    planned = [(c, p, L, k) for c in plan_conds for p in plan_positions for L in plan_layers for k in plan_ks]
    planned += [("prompt_only", "prompt_end", L, k) for L in plan_layers for k in plan_ks]
    planned += [("nosys", "last", L, k) for L in plan_layers for k in plan_ks]
    have = {(r["condition"], r["position"], r["layer"], r["k"]) for r in rows}
    not_run_cells = [x for x in planned if x not in have]
    planned_steer = [tuple(x.split(":")) for x in plan["steer_runs"].split(",") if x]
    not_run_steer = [f"{c}:L{L}" for c, L in planned_steer if ("core", c, int(L)) not in steers_raw]
    by_cell_missing = sorted({(c, p) for c, p, L, k in not_run_cells})

    a.out.mkdir(parents=True, exist_ok=True)
    k0 = a.k if a.k in ks else (ks[0] if ks else None)
    summary = {"inputs": {"sweep_files": files, "steer_files": steer_files, "pod_json": str(pod_json) if pod_json else None},
               "plan": plan, "ks": ks, "layers": layers, "positions": positions, "conditions": conds, "settings": settings,
               "nov5_reference": NOV5, "rows": rows, "role_dirs": role_dirs,
               "not_run": {"cells": [{"condition": c, "position": p, "layer": L, "k": k} for c, p, L, k in not_run_cells],
                           "condition_positions": [{"condition": c, "position": p} for c, p in by_cell_missing],
                           "steer_runs": not_run_steer,
                           "n_planned_cells": len(planned), "n_ran_cells": len(planned) - len(not_run_cells)}}

    # headline aggregates per condition at k0
    verdict = {}
    if k0 is not None:
        for cond in conds:
            rr = [r for r in rows if r["condition"] == cond and r["k"] == k0]
            if not rr:
                continue
            best = max(rr, key=lambda r: r["J_fill"] if r["J_fill"] is not None else -1)
            verdict[cond] = {
                "k": k0, "n_cells": len(rr),
                "cells_J_fill_clears_both_nulls": sum(r["J_fill_clears_both_nulls"] for r in rr),
                "cells_J_pursuit_clears_both_nulls": sum(r["J_pursuit_clears_both_nulls"] for r in rr),
                "cells_J_fill_beats_pca_ci": sum(r["J_fill_beats_pca"] for r in rr),
                "cells_pca_beats_J_fill_ci": sum(r["pca_beats_J_fill"] for r in rr),
                "cells_full_beats_J_fill_ci": sum(r["full_beats_J_fill"] for r in rr),
                "best_J_fill": {k: best[k] for k in ("position", "layer", "J_fill", "J_fill_ci", "full", "pca", "randsub_p95", "randatoms_p95",
                                                     "J_fill_clears_both_nulls", "J_fill_minus_pca_ci", "recovery_J_fill")},
                "band_by_position": {p: {"full": band([r for r in rr if r["position"] == p], "full"),
                                         "full_lodo": band([r for r in rr if r["position"] == p], "full_lodo"),
                                         "full_lofo": band([r for r in rr if r["position"] == p], "full_lofo"),
                                         "J_fill_lodo": band([r for r in rr if r["position"] == p], "J_fill_lodo"),
                                         "J_fill_lofo": band([r for r in rr if r["position"] == p], "J_fill_lofo"),
                                         "pca_lodo": band([r for r in rr if r["position"] == p], "pca_lodo"),
                                         "J_fill": band([r for r in rr if r["position"] == p], "J_fill"),
                                         "J_pursuit": band([r for r in rr if r["position"] == p], "J_pursuit"),
                                         "pca": band([r for r in rr if r["position"] == p], "pca"),
                                         "randsub_p95": band([r for r in rr if r["position"] == p], "randsub_p95"),
                                         "randatoms_p95": band([r for r in rr if r["position"] == p], "randatoms_p95"),
                                         "clears": sum(r["J_fill_clears_both_nulls"] for r in rr if r["position"] == p),
                                         "n": sum(1 for r in rr if r["position"] == p)} for p in positions if any(r["position"] == p for r in rr)},
            }
        # prompt-only comparison: J fill (original/swap) vs the prompt_only probe at the same layer
        po = {r["layer"]: r for r in rows if r["condition"] == "prompt_only" and r["position"] == "prompt_end" and r["k"] == k0}
        if po:
            verdict["prompt_only_probe"] = {str(L): {"full": po[L]["full"], "J_fill": po[L]["J_fill"], "pca": po[L]["pca"], "randsub_p95": po[L]["randsub_p95"]} for L in sorted(po)}
            for cond in ("original", "swap"):
                for r in rows:
                    if r["condition"] == cond and r["k"] == k0 and r["layer"] in po and r["J_fill"] is not None and po[r["layer"]]["full"] is not None:
                        r["J_fill_minus_prompt_only_full"] = r["J_fill"] - po[r["layer"]]["full"]
                        r["J_fill_minus_prompt_only_J_fill"] = r["J_fill"] - po[r["layer"]]["J_fill"] if po[r["layer"]]["J_fill"] is not None else None
    summary["verdict"] = verdict

    # matched vocabulary (k0) for every cell that has it
    mv_rows = []
    for r in rows:
        if r["k"] == k0 and r["mv"].get("full"):
            mv_rows.append({"condition": r["condition"], "position": r["position"], "layer": r["layer"],
                            **{name: mv_fields(r["mv"].get(name)) for name in ("full", "J_fill", "pca", "C_fill")},
                            "randsub": r["mv"].get("randsub")})
    summary["matched_vocab"] = mv_rows
    recs = exp1_rows(exp1)
    summary["exp1_concepts"] = recs
    summary["exp1_top_tokens"] = {f"{c}/{p}/L{L}": {s: [t for t in (e.get(s) or {}).get("top_tokens", []) if t.get("significant")][:15] for s in ("core", "transfer")}
                                  for (c, p, L), e in exp1.items()}
    summary["exp1_most_common_atoms"] = {f"{c}/{p}/L{L}": {s: (e.get(s) or {}).get("most_common_atoms", [])[:12] for s in ("core", "transfer")}
                                         for (c, p, L), e in exp1.items()}
    steers = {k: steer_summary(v) for k, v in steers_raw.items()}
    summary["steer"] = {f"{s}/{c}/L{L}": v for (s, c, L), v in steers.items()}

    # tables
    if k0 is not None:
        for k in ks:
            tab_headline(rows, [c for c in conds if c in ("original", "swap")], positions, layers, k, a.out)
        tab_controls(rows, layers, k0, a.out)
        tab_protocols(rows, k0, a.out)
        sel = []
        for cond in ("swap", "original"):
            if cond in verdict:
                bp = max(verdict[cond]["band_by_position"].items(), key=lambda kv: kv[1]["J_fill"] or -1)[0]
                for pos_ in {bp, "last"}:  # the best position by band mean, and always the last token for comparability
                    if any(r["condition"] == cond and r["position"] == pos_ for r in rows):
                        tab_ksweep(rows, cond, pos_, ks, a.out)
                        tab_layers(rows, cond, pos_, k0, a.out)
                # fixed-name copies for the paper's \input{}: the best position of each condition
                import shutil
                shutil.copy(a.out / f"tab_ksweep_{cond}_{bp}.tex", a.out / f"tab_ksweep_{cond}_best.tex")
                shutil.copy(a.out / f"tab_layers_{cond}_{bp}_k{k0}.tex", a.out / f"tab_layers_{cond}_best_k{k0}.tex")
                verdict[cond]["best_position_by_band"] = bp
                bl = verdict[cond]["best_J_fill"]["layer"]
                sel += [(cond, bp, bl)]
                for p in ("last", "prevote"):
                    if p != bp and (cond, p, bl) in cells:
                        sel.append((cond, p, bl))
        tab_matched_vocab(rows, sel, k0, a.out)
        tab_exp1(recs, exp1, sel, a.out)
    if steers:
        tab_steer(steers, a.out)

    sdir = a.summary_dir or (a.root / "gemma2_9b")
    sdir.mkdir(parents=True, exist_ok=True)
    (sdir / "SUMMARY.json").write_text(json.dumps(summary, indent=1, default=float) + "\n")

    # RESULTS.md
    md = ["# jspace-interp results summary", "",
          f"Inputs: {len(files)} sweep file(s), {len(steer_files)} steering file(s). Planned cells {len(planned)}, ran {len(planned) - len(not_run_cells)}.", ""]
    for cond, v in verdict.items():
        if cond == "prompt_only_probe":
            md += ["## prompt_only probe (prompt_end)", "", "| layer | full | J fill | PCA | rand p95 |", "|---|---|---|---|---|"]
            md += [f"| {L} | {f3(x['full'])} | {f3(x['J_fill'])} | {f3(x['pca'])} | {f3(x['randsub_p95'])} |" for L, x in v.items()]
            md.append("")
            continue
        b = v["best_J_fill"]
        md += [f"## {cond} (k={v['k']})", "",
               f"- cells: {v['n_cells']}; J fill clears both nulls in {v['cells_J_fill_clears_both_nulls']}, J pursuit in {v['cells_J_pursuit_clears_both_nulls']}; "
               f"J fill beats PCA (paired CI) in {v['cells_J_fill_beats_pca_ci']}, PCA beats J fill in {v['cells_pca_beats_J_fill_ci']}; full beats J fill in {v['cells_full_beats_J_fill_ci']}",
               f"- best J fill: {b['position']} L{b['layer']} = {f3(b['J_fill'])} {ci(b['J_fill_ci'])} (full {f3(b['full'])}, PCA {f3(b['pca'])}, rand sub p95 {f3(b['randsub_p95'])}, "
               f"rand atoms p95 {f3(b['randatoms_p95'])}, clears both: {b['J_fill_clears_both_nulls']}, J-PCA CI {ci(b['J_fill_minus_pca_ci'])}, recovery {f3(b['recovery_J_fill'], 2)})", "",
               "| position | full | J fill | J pursuit | PCA | rand sub p95 | rand atoms p95 | clears |", "|---|---|---|---|---|---|---|---|"]
        md += [f"| {p} | {f3(x['full'])} | {f3(x['J_fill'])} | {f3(x['J_pursuit'])} | {f3(x['pca'])} | {f3(x['randsub_p95'])} | {f3(x['randatoms_p95'])} | {x['clears']}/{x['n']} |"
               for p, x in v["band_by_position"].items()]
        md.append("")
    if mv_rows:
        md += ["## matched vocabulary (k=%s)" % k0, "", "| cell | arm | colluder vs warned | warned vs unwarned (p) | warned vs control | echo | ep warned vs control | ep collusion vs warned |", "|---|---|---|---|---|---|---|---|"]
        for m in mv_rows:
            for name in ("full", "J_fill", "pca"):
                x = m.get(name)
                if x:
                    md.append(f"| {m['condition']}/{m['position']}/L{m['layer']} | {name} | {f3(x['colluder_vs_warned'])} | {f3(x['warned_vs_unwarned'])} ({f3(x['warned_vs_unwarned_p'], 2)}) | "
                              f"{f3(x['warned_vs_control'])} | {f3(x['echo_ratio'], 2)} | {f3(x['ep_warned_vs_control'])} | {f3(x['ep_collusion_vs_warned'])} |")
        md.append("")
    if steers:
        md += ["## steering", ""]
        for key, s in steers.items():
            md.append(f"- {key[1]} L{key[2]}: scale {f3(s['scale'], 2)} ({s['scale_source']}), signs {s['signs']}, prefix cache {s['prefix_cache']} "
                      f"(check {json.dumps(s['cache_check'])}), {s['n_records']} prefixes, {s['n_random']} random directions")
            md.append(f"  baseline: {json.dumps(s['baseline'])}; random at alpha 4: {json.dumps({k: v for k, v in s['random_by_group_alpha'].items() if k.endswith('|4.0')})}")
            for r in s["rows"]:
                d = r["mean_delta_by_alpha"]; f = r["flip_by_alpha"]
                md.append(f"  - {r['direction']} / {r['group']} (n={r['n']}): dy at alpha 0.5/1/2/4 = " + "/".join(f3(d.get(str(float(x))), 3) for x in (0.5, 1, 2, 4)) +
                          "; flips at alpha 2/4 = " + "/".join(f3(f.get(str(float(x))), 2) for x in (2, 4)) +
                          f"; sign ok {r['predicted_sign_ok_all']}, beats random all {r['beats_random_all']}, BH any {r['bh_reject_any']}, dose monotone {r['dose_monotone']}, passes all bars {r['passes_all_bars']}")
        md.append("")
    md += ["## not run", "", f"- condition/position pairs with missing cells: {', '.join(f'{c}/{p}' for c, p in by_cell_missing) or 'none'}",
           f"- steering runs missing: {', '.join(not_run_steer) or 'none'}", ""]
    (sdir / "RESULTS.md").write_text("\n".join(md))
    print("\n".join(md))
    print(f"wrote {sdir / 'SUMMARY.json'}, {sdir / 'RESULTS.md'} and {len(list(a.out.glob('tab_*.tex')))} table fragments in {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
