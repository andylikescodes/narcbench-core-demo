#!/usr/bin/env python3
"""Figures for the Gemma-2-9B-it J-space paper, drawn from the frozen record.

Reads results/jspace/gemma2_9b_frozen/frozen_stats.json (written by
scripts/jspace_offline_stats.py) and writes PDF + PNG figures to
results/jspace/gemma2_9b_frozen/figures/ and, if present, to
papers/jspace-gemma2-9b-2026-11/figures/. No number is typed here.

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


def main() -> int:
    d = json.loads(SRC.read_text())
    figs = {"fig_gemma_arms": fig_arms(d), "fig_gemma_layers": fig_layers(d), "fig_gemma_core_k": fig_core_k(d), "fig_gemma_families": fig_families(d)}
    for out in OUTS:
        if not out.parent.exists():
            continue
        out.mkdir(parents=True, exist_ok=True)
        for name, fig in figs.items():
            fig.savefig(out / f"{name}.pdf", bbox_inches="tight")
            fig.savefig(out / f"{name}.png", dpi=200, bbox_inches="tight")
        print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
