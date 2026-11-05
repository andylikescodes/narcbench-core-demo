#!/usr/bin/env python3
"""Figures for the Gemma-2-9B-it J-space paper, drawn from the frozen record.

Reads results/jspace/gemma2_9b_frozen/frozen_stats.json (written by
scripts/jspace_offline_stats.py) and writes PDF + PNG figures to
results/jspace/gemma2_9b_frozen/figures/ and, if present, to
papers/jspace-gemma2-9b-2026-11/figures/. No number is typed here.

Since 5 November 2026 it also draws the paper's J-space arms (published Jacobian lens,
nonnegative pursuit) from the pre-registered pipeline's own summaries:
results/jspace/gemma2_9b_jpaper_nnomp_k25/{summary.json,arm_layer_cis.json} and
results/jspace/gemma2_9b_jpaper_variants_all/{summary.json,arm_pairs.json}, writing
fig_gemma_paper_arms, fig_gemma_paper_layers and fig_gemma_variants to
results/jspace/gemma2_9b_jpaper_nnomp_k25/figures/ and the paper's figures/.

  python3 scripts/jspace_gemma_figures.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "results/jspace/gemma2_9b_frozen/frozen_stats.json"
OUTS = [ROOT / "results/jspace/gemma2_9b_frozen/figures", ROOT / "papers/jspace-gemma2-9b-2026-11/figures"]

INK, J, C, MUTED, GRID = "#15202b", "#eb6834", "#2a78d6", "#898781", "#e1e0d9"
ARMS = [  # key in frozen_stats, label, colour, hollow marker?
    ("full", "full residual", INK, False),
    ("W_U k25 J-only", "J-only, $W_U$ frame", J, False),
    ("jac k25 J-only", "J-only, Jacobian frame", J, True),
    ("W_U k25 complement", "complement, $W_U$ frame", C, False),
    ("jac k25 complement", "complement, Jacobian frame", C, True),
]
CORE_LDO = {  # band-mean Core leave-one-domain-out primaries recorded in docs/NEXT-SCIENCE.md
    "full": 1.000, "W_U k25 J-only": None, "jac k25 J-only": None, "W_U k25 complement": None, "jac k25 complement": None,
}

plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": INK,
                     "ytick.color": INK, "axes.spines.top": False, "axes.spines.right": False})


def style(ax):
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def marker_kw(col, hollow):
    return dict(marker="o", ms=7, mfc=("white" if hollow else col), mec=col, mew=1.8, color=col, ls="none")


def fig_arms(d):
    core = d["core_battery_ldo"]
    ldo = {"full": 1.000, "W_U k25 J-only": core["wu_topk k25"]["J_only"], "jac k25 J-only": core["jac_topk k25"]["J_only"],
           "W_U k25 complement": core["wu_topk k25"]["complement"], "jac k25 complement": core["jac_topk k25"]["complement"]}
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6), sharey=True, gridspec_kw={"width_ratios": [1, 1.25]})
    ys = list(range(len(ARMS)))[::-1]
    for ax, title in zip(axes, ["Core, leave one domain out", "Core discussion → Transfer final (n = 72)"]):
        style(ax)
        ax.axvline(0.5, color=MUTED, lw=1, ls=(0, (3, 3)))
        ax.set_title(title, fontsize=9.5, loc="left", color=INK)
        ax.set_xlim(0.38, 1.03)
        ax.set_xticks([0.5, 0.75, 1.0])
        ax.set_xlabel("primary AUROC, band mean", fontsize=8.5)
    for y, (key, label, col, hollow) in zip(ys, ARMS):
        v = ldo[key]
        axes[0].plot([v], [y], **marker_kw(col, hollow))
        axes[0].annotate(f"{v:.3f}", (v, y), xytext=(-8 if v > 0.9 else 8, 0), textcoords="offset points",
                         ha="right" if v > 0.9 else "left", va="center", fontsize=8, color=INK)
        a = d["transfer_arms"][key]["primary"]
        lo, hi = a["ci95"]
        axes[1].plot([lo, hi], [y, y], color=col, lw=2, solid_capstyle="butt", alpha=0.9)
        axes[1].plot([a["auroc"]], [y], **marker_kw(col, hollow))
        axes[1].annotate(f"{a['auroc']:.3f}", (a["auroc"], y), xytext=(0, 7), textcoords="offset points",
                         ha="center", va="bottom", fontsize=8, color=INK)
    axes[0].set_yticks(ys)
    axes[0].set_yticklabels([lab for _, lab, _, _ in ARMS], fontsize=8.5)
    axes[0].set_ylim(-0.7, len(ARMS) - 0.3)
    fig.tight_layout()
    return fig


def fig_layers(d):
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    style(ax)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.axhline(0.5, color=MUTED, lw=1, ls=(0, (3, 3)))
    offsets = {"full": -0.16, "W_U k25 J-only": -0.08, "jac k25 J-only": 0.0, "W_U k25 complement": 0.08, "jac k25 complement": 0.16}
    for key, label, col, hollow in ARMS:
        per = d["transfer_arms"][key]["per_layer"]
        xs = [int(L) + offsets[key] for L in per]
        vals = [per[L]["primary"] for L in per]
        los = [per[L]["ci95"][0] for L in per]
        his = [per[L]["ci95"][1] for L in per]
        ax.vlines(xs, los, his, color=col, lw=1.6, alpha=0.9)
        ax.plot(xs, vals, label=label, **marker_kw(col, hollow))
    ax.set_xticks([19, 20, 21, 22, 23])
    ax.set_xlabel("layer (residual after block)", fontsize=8.5)
    ax.set_ylabel("primary AUROC", fontsize=8.5)
    ax.set_ylim(0.3, 1.02)
    ax.legend(ncol=3, fontsize=8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.22))
    fig.tight_layout()
    return fig


def fig_core_k(d):
    core = d["core_battery_ldo"]
    ks = [8, 16, 25, 50]
    fig, ax = plt.subplots(figsize=(4.6, 2.8))
    style(ax)
    ax.grid(axis="y", color=GRID, lw=0.8)
    wu = [core[f"wu_topk k{k}"]["J_only"] for k in ks]
    jac = [core[f"jac_topk k{k}"]["J_only"] for k in ks]
    ax.plot(ks, [1.0] * 4, color=C, lw=1.6, label="complement, both frames")
    ax.plot(ks, wu, color=J, lw=1.6, marker="o", ms=6, label="J-only, $W_U$ frame")
    ax.plot(ks, jac, color=J, lw=1.6, ls="--", marker="o", ms=6, mfc="white", mew=1.8, label="J-only, Jacobian frame")
    fx = core["wu_fixed k25"]["J_only"]
    ax.plot([25], [fx], marker="s", ms=6, color=J, mfc="white", mew=1.6, ls="none", label="fixed $W_U$ frame, k = 25")
    for k, v in zip(ks, wu):
        ax.annotate(f"{v:.3f}", (k, v), xytext=(0, -13), textcoords="offset points", ha="center", fontsize=7.5, color=INK)
    ax.set_xscale("log", base=2)
    ax.set_xticks(ks)
    ax.set_xticklabels([str(k) for k in ks])
    ax.set_xlabel("frame size k", fontsize=8.5)
    ax.set_ylabel("Core LDO primary AUROC", fontsize=8.5)
    ax.set_ylim(0.85, 1.02)
    ax.legend(fontsize=7.5, frameon=False, loc="lower right")
    fig.tight_layout()
    return fig


def fig_families(d):
    fams = [("cicd", "CI/CD pipelines"), ("code_review", "code review"), ("news", "news"),
            ("pump_dump", "pump-and-dump"), ("rag", "RAG corpora"), ("research", "research integrity")]
    arms = [("full", "full residual", INK), ("W_U k25 J-only", "J-only, $W_U$ frame", J), ("W_U k25 complement", "complement, $W_U$ frame", C)]
    fig, ax = plt.subplots(figsize=(4.6, 2.8))
    style(ax)
    ax.axvline(0.5, color=MUTED, lw=1, ls=(0, (3, 3)))
    ys = list(range(len(fams)))[::-1]
    for i, (key, label, col) in enumerate(arms):
        pf = d["transfer_arms"][key]["per_layer"]["21"]["per_family"]
        ax.plot([pf[f]["split"] for f, _ in fams], [y + (i - 1) * 0.22 for y in ys], label=label, **marker_kw(col, False))
    ax.set_yticks(ys)
    ax.set_yticklabels([n for _, n in fams], fontsize=8.5)
    ax.set_xlim(0.25, 1.05)
    ax.set_xlabel("Split AUROC at layer 21 (12 episodes per family)", fontsize=8.5)
    ax.legend(fontsize=7.5, frameon=False, loc="upper left", bbox_to_anchor=(0.0, 0.98))
    fig.tight_layout()
    return fig


# ---- 5 November 2026: the paper's J-space on the same caches, drawn from the pre-registered run summaries ----
SRC_PAPER = ROOT / "results/jspace/gemma2_9b_jpaper_nnomp_k25/summary.json"
SRC_PAPER_LAYERS = ROOT / "results/jspace/gemma2_9b_jpaper_nnomp_k25/arm_layer_cis.json"
SRC_VARIANTS = ROOT / "results/jspace/gemma2_9b_jpaper_variants_all/summary.json"
SRC_VARIANT_PAIRS = ROOT / "results/jspace/gemma2_9b_jpaper_variants_all/arm_pairs.json"
OUTS_PAPER = [ROOT / "results/jspace/gemma2_9b_jpaper_nnomp_k25/figures", ROOT / "papers/jspace-gemma2-9b-2026-11/figures"]

PAPER_ARMS = [  # arm key in summary.json, label, colour, hollow?, marker
    ("full", "full residual", INK, False, "o"),
    ("Jpaper", "J-only, paper's J-space", J, False, "o"),
    ("C_Jpaper", "complement, paper's J-space", C, False, "o"),
    ("shuffled_paper", "another turn's atoms (same dictionary)", J, True, "o"),
    ("randsub", "random fixed 25-dim subspace", MUTED, False, "o"),
    ("J", "J-only, $W_U$ frame", J, False, "s"),
    ("C_J", "complement, $W_U$ frame", C, False, "s"),
]
PAPER_LAYER_ARMS = [  # arm key in arm_layer_cis.json, label, colour, hollow?, x offset
    ("full", "full residual", INK, False, -0.16),
    ("Jpaper", "J-only, paper's J-space", J, False, -0.08),
    ("C_Jpaper", "complement, paper's J-space", C, False, 0.0),
    ("shuffled_paper", "another turn's atoms", J, True, 0.08),
    ("randsub", "random 25-dim subspace", MUTED, False, 0.16),
]
VARIANTS = [  # arm key in the combined run, its k block, label (fill counts from decomp_summary.md of the runs)
    ("Jpaper_k50", "50", "pursuit, k = 50 (≈33 fill atoms)"),
    ("Jpaper", "25", "primary: pursuit, k = 25 (≈8 fill atoms)"),
    ("Jpaper_gp", "25", "gradient pursuit, k = 25"),
    ("Jpaper_raw", "25", "raw-score selection, k = 25"),
    ("Jpaper_topk", "25", "top 25 by correlation, no pursuit"),
    ("Jpaper_k8", "8", "pursuit cut at k = 8"),
    ("Jpaper_k16", "16", "pursuit, k = 16 (≈1 fill atom)"),
    ("Jpaper_cone", "25", "cone fit (fill atoms weigh zero)"),
    ("Jpaper_nofill", "25", "pursuit, no fill"),
]


def _primary(x):
    return float(x["primary"]) if isinstance(x, dict) else float(x)


def fig_paper_arms(s):
    k = "25"
    a1 = s["A1_core_ldo"]["band_mean"][k]
    a2, a3 = s["A2_inference"][k], s["A3_inference"][k]
    fig, axes = plt.subplots(1, 3, figsize=(10.6, 2.9), sharey=True, gridspec_kw={"width_ratios": [1, 1.3, 1.3], "wspace": 0.12})
    ys = list(range(len(PAPER_ARMS)))[::-1]
    titles = ["Core, leave one domain out", f"Core → Transfer final (n = {a2['n_episodes']})",
              f"within Transfer, family held out (n = {a3['n_episodes']})"]
    for ax, title in zip(axes, titles):
        style(ax)
        ax.axvline(0.5, color=MUTED, lw=1, ls=(0, (3, 3)))
        ax.set_title(title, fontsize=9.5, loc="left", color=INK)
        ax.set_xlim(0.38, 1.03)
        ax.set_xticks([0.5, 0.75, 1.0])
        ax.set_xlabel("primary AUROC, band mean L19–23", fontsize=8.5)
    for y, (key, label, col, hollow, mk) in zip(ys, PAPER_ARMS):
        kw = marker_kw(col, hollow)
        kw["marker"] = mk
        v = _primary(a1[key])
        axes[0].plot([v], [y], **kw)
        axes[0].annotate(f"{v:.3f}", (v, y), xytext=(-8 if v > 0.9 else 8, 0), textcoords="offset points",
                         ha="right" if v > 0.9 else "left", va="center", fontsize=8, color=INK)
        for ax, inf in ((axes[1], a2), (axes[2], a3)):
            e = inf["arms"][key]
            lo, hi = e["ci"]
            ax.plot([lo, hi], [y, y], color=col, lw=2, solid_capstyle="butt", alpha=0.9)
            ax.plot([e["est"]], [y], **kw)
            ax.annotate(f"{e['est']:.3f}", (e["est"], y), xytext=(0, 7), textcoords="offset points", ha="center", va="bottom",
                        fontsize=8, color=INK)
    axes[0].set_yticks(ys)
    axes[0].set_yticklabels([lab for _, lab, _, _, _ in PAPER_ARMS], fontsize=8.5)
    axes[0].set_ylim(-0.7, len(PAPER_ARMS) - 0.3)
    fig.tight_layout()
    return fig


def fig_paper_layers(dl):
    a2 = dl["analyses"]["A2"]
    layers = a2["layers"]
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    style(ax)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.axhline(0.5, color=MUTED, lw=1, ls=(0, (3, 3)))
    for key, label, col, hollow, off in PAPER_LAYER_ARMS:
        per = a2["arms"][key]["per_layer"]
        xs = [int(L) + off for L in layers]
        ax.vlines(xs, [per[L]["ci"][0] for L in layers], [per[L]["ci"][1] for L in layers], color=col, lw=1.6, alpha=0.9)
        ax.plot(xs, [per[L]["est"] for L in layers], label=label, **marker_kw(col, hollow))
    ax.set_xticks([int(L) for L in layers])
    ax.set_xlabel("layer (residual after block)", fontsize=8.5)
    ax.set_ylabel("primary AUROC, Core → Transfer final", fontsize=8.5)
    ax.set_ylim(0.3, 1.02)
    ax.legend(ncol=3, fontsize=8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.22))
    fig.tight_layout()
    return fig


def fig_variants(sv, pairs):
    inf = sv["A2_inference"]
    pr = pairs["analyses"]["A2"]["pairs"]
    full = inf["25"]["arms"]["full"]["est"]
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.3), sharey=True, gridspec_kw={"width_ratios": [1.25, 1], "wspace": 0.1})
    ys = list(range(len(VARIANTS)))[::-1]
    for ax in axes:
        style(ax)
    axes[0].axvline(full, color=INK, lw=1, ls=(0, (3, 3)))
    axes[0].annotate(f"full {full:.3f}", (full, len(VARIANTS) - 0.55), xytext=(-4, 0), textcoords="offset points", ha="right",
                     va="center", fontsize=7.5, color=INK)
    axes[1].axvline(0.0, color=MUTED, lw=1, ls=(0, (3, 3)))
    for y, (key, k, label) in zip(ys, VARIANTS):
        e = inf[k]["arms"][key]
        r = inf[k]["arms"]["randsub"]
        axes[0].plot(e["ci"], [y, y], color=J, lw=2, solid_capstyle="butt", alpha=0.9)
        axes[0].plot([e["est"]], [y], **marker_kw(J, False))
        axes[0].plot([r["est"]], [y], marker="|", ms=9, mew=1.8, color=MUTED, ls="none")
        axes[0].annotate(f"{e['est']:.3f}", (e["est"], y), xytext=(0, 7), textcoords="offset points", ha="center", va="bottom",
                         fontsize=7.5, color=INK)
        if key == "Jpaper":
            axes[1].plot([0.0], [y], **marker_kw(J, True))
            continue
        d = pr[f"{key}@{k}-Jpaper@25"]["band"]
        axes[1].plot(d["ci"], [y, y], color=J, lw=2, solid_capstyle="butt", alpha=0.9)
        axes[1].plot([d["est"]], [y], **marker_kw(J, False))
        axes[1].annotate(f"{d['est']:+.3f}", (d["est"], y), xytext=(0, 7), textcoords="offset points", ha="center", va="bottom",
                         fontsize=7.5, color=INK)
    axes[0].set_yticks(ys)
    axes[0].set_yticklabels([lab for _, _, lab in VARIANTS], fontsize=8.5)
    axes[0].set_ylim(-0.7, len(VARIANTS) - 0.3)
    axes[0].set_xlim(0.5, 1.0)
    axes[0].set_xticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    axes[0].set_title("Core → Transfer final, the J-only arm of each decomposition", fontsize=9.5, loc="left", color=INK)
    axes[0].set_xlabel("primary AUROC, band mean (grey tick: k-matched random subspace)", fontsize=8.5)
    axes[1].set_xlim(-0.12, 0.12)
    axes[1].set_title("difference to the k = 25 primary", fontsize=9.5, loc="left", color=INK)
    axes[1].set_xlabel("ΔAUROC, paired, 95% bootstrap interval", fontsize=8.5)
    fig.tight_layout()
    return fig


def save_all(figs, outs):
    for out in outs:
        if not out.parent.exists():
            continue
        out.mkdir(parents=True, exist_ok=True)
        for name, fig in figs.items():
            fig.savefig(out / f"{name}.pdf", bbox_inches="tight")
            fig.savefig(out / f"{name}.png", dpi=200, bbox_inches="tight")
        print("wrote", sorted(figs), "to", out)


def main() -> int:
    d = json.loads(SRC.read_text())
    figs = {"fig_gemma_arms": fig_arms(d), "fig_gemma_layers": fig_layers(d), "fig_gemma_core_k": fig_core_k(d), "fig_gemma_families": fig_families(d)}
    save_all(figs, OUTS)
    if SRC_PAPER.exists():
        figs_paper = {"fig_gemma_paper_arms": fig_paper_arms(json.loads(SRC_PAPER.read_text()))}
        if SRC_PAPER_LAYERS.exists():
            figs_paper["fig_gemma_paper_layers"] = fig_paper_layers(json.loads(SRC_PAPER_LAYERS.read_text()))
        if SRC_VARIANTS.exists() and SRC_VARIANT_PAIRS.exists():
            figs_paper["fig_gemma_variants"] = fig_variants(json.loads(SRC_VARIANTS.read_text()), json.loads(SRC_VARIANT_PAIRS.read_text()))
        save_all(figs_paper, OUTS_PAPER)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
