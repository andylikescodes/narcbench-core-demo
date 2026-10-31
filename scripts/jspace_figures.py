#!/usr/bin/env python3
"""Figures for the J-space separability analysis, from results/jspace/<model>/summary.json.

  python3 scripts/jspace_figures.py --models qwen3_32b deepseek_r1_32b gpt_oss_20b --out results/jspace/figures

Writes fig_k_curves_<model>.png, fig_k25_cross_model.png, fig_three_settings.png (and .pdf twins).
Palette: the validated reference palette's first three categorical slots (blue, orange, aqua), ink and chrome
tokens from the same reference; marks 2px lines, >= 8px markers, hairline solid grid.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SLOT = {"1": "#2a78d6", "2": "#eb6834", "3": "#1baf7a"}  # categorical slots 1-3 (validated all-pairs, light surface)
INK = {"primary": "#0b0b0b", "secondary": "#52514e", "muted": "#898781", "grid": "#e1e0d9", "axis": "#c3c2b7", "surface": "#fcfcfb"}
LABEL = {"qwen3_32b": "Qwen3-32B", "deepseek_r1_32b": "DeepSeek-R1-32B", "gpt_oss_20b": "GPT-OSS-20B", "gemma2_9b": "Gemma-2-9B (frozen)",
         "qwen3_32b_gamma": "Qwen3-32B, γ-scaled frames", "deepseek_r1_32b_gamma": "DeepSeek-R1-32B, γ-scaled frames", "gpt_oss_20b_gamma": "GPT-OSS-20B, γ-scaled frames"}

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 9, "axes.edgecolor": INK["axis"], "axes.linewidth": 0.8,
    "axes.labelcolor": INK["secondary"], "xtick.color": INK["muted"], "ytick.color": INK["muted"],
    "axes.titlecolor": INK["primary"], "axes.titlesize": 10, "figure.facecolor": INK["surface"], "axes.facecolor": INK["surface"],
    "axes.grid": True, "grid.color": INK["grid"], "grid.linewidth": 0.8, "grid.linestyle": "-", "axes.spines.top": False,
    "axes.spines.right": False, "legend.frameon": False, "legend.fontsize": 8, "savefig.dpi": 160,
})


def load(models: list[str]) -> dict[str, dict]:
    out = {}
    for m in models:
        p = ROOT / "results/jspace" / m / "summary.json"
        if p.exists():
            out[m] = json.loads(p.read_text())
    return out


def series(ax, x, y, lo, hi, color, label, marker="o"):
    y, lo, hi = np.asarray(y, float), np.asarray(lo, float), np.asarray(hi, float)
    ax.plot(x, y, color=color, lw=2, solid_capstyle="round", solid_joinstyle="round", label=label, zorder=3)
    ax.scatter(x, y, s=48, color=color, edgecolors=INK["surface"], linewidths=1.5, zorder=4)
    ok = np.isfinite(lo) & np.isfinite(hi)
    ax.vlines(np.asarray(x)[ok], lo[ok], hi[ok], color=color, lw=1.2, alpha=0.9, zorder=2)
    return float(y[-1])


def end_labels(ax, x_end, ends: list[tuple[float, str]], min_gap: float = 0.035) -> None:
    """Direct end-labels only where they do not collide; the legend carries the rest (no stacking)."""
    placed: list[float] = []
    for y, text in sorted(ends, key=lambda t: -t[0]):
        if all(abs(y - p) >= min_gap for p in placed):
            ax.annotate(text, (x_end, y), xytext=(6, 0), textcoords="offset points", va="center", fontsize=8, color=INK["secondary"])
            placed.append(y)


def fig_k_curves(tag: str, s: dict, out: Path) -> None:
    ks = [int(k) for k in sorted(s["A2_inference"], key=int)]
    inf = s["A2_inference"]
    full = inf[str(ks[0])]["arms"]["full"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for ax, arms, title in (
        (axes[0], [("J", SLOT["1"], "J-only"), ("shuffled", SLOT["2"], "another turn's frame"), ("randsub", SLOT["3"], "random subspace")],
         "J-space against dimension-matched controls"),
        (axes[1], [("C_J", SLOT["1"], "complement of J"), ("pca", SLOT["2"], "PCA (unsupervised)"), ("contrast", SLOT["3"], "contrast (supervised)")],
         "the complement and data-aligned subspaces"),
    ):
        ax.axhline(full["est"], color=INK["muted"], lw=1.2, zorder=1)
        ax.annotate(f"full residual {full['est']:.2f}", (ks[0], full["est"]), xytext=(-6, -11), textcoords="offset points", fontsize=8, color=INK["secondary"])
        ax.axhline(0.5, color=INK["axis"], lw=0.8, zorder=1)
        ends = []
        for arm, color, label in arms:
            y = [inf[str(k)]["arms"][arm]["est"] for k in ks]
            lo = [inf[str(k)]["arms"][arm]["ci"][0] for k in ks]
            hi = [inf[str(k)]["arms"][arm]["ci"][1] for k in ks]
            ends.append((series(ax, ks, y, lo, hi, color, label), f"{y[-1]:.2f}"))
        end_labels(ax, ks[-1], ends)
        ax.set_xscale("log", base=2)
        ax.set_xticks(ks)
        ax.set_xticklabels([str(k) for k in ks])
        ax.set_xlabel("frame / subspace size k")
        ax.set_title(title)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=3, columnspacing=1.0, handlelength=1.6, fontsize=8)
        ax.set_xlim(ks[0] / 1.3, ks[-1] * 1.8)
    axes[0].set_ylabel("primary AUROC, Core → Transfer")
    axes[0].set_ylim(0.3, 1.02)
    fig.suptitle(f"{LABEL.get(tag, tag)}: does the J-space probe transfer?  (band mean over layers; bars: 95% bootstrap CI over episodes)",
                 x=0.01, ha="left", fontsize=10.5, color=INK["primary"])
    fig.set_size_inches(10, 4.4)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    for ext in ("png", "pdf"):
        fig.savefig(out / f"fig_k_curves_{tag}.{ext}")
    plt.close(fig)


def fig_cross_model(res: dict[str, dict], out: Path, frozen: dict | None) -> None:
    rows = []
    for tag, s in res.items():
        inf = s["A2_inference"]["25"]["arms"]
        rows.append((LABEL.get(tag, tag), {a: (inf[a]["est"], *inf[a]["ci"]) for a in ("full", "J", "C_J", "randsub")}))
    if frozen:
        arms = frozen["transfer_arms"]
        g = {"full": arms["full"]["primary"], "J": arms["W_U k25 J-only"]["primary"], "C_J": arms["W_U k25 complement"]["primary"]}
        rows.append((LABEL["gemma2_9b"], {a: (v["auroc"], *v["ci95"]) for a, v in g.items()}))
    fig, ax = plt.subplots(figsize=(8, 1.9 + 0.7 * len(rows)))
    ypos = np.arange(len(rows))[::-1]
    offs = {"J": 0.18, "C_J": -0.18, "randsub": 0.0}
    colors = {"J": SLOT["1"], "C_J": SLOT["2"], "randsub": SLOT["3"]}
    names = {"J": "J-only, k = 25", "C_J": "complement, k = 25", "randsub": "random 25-dim subspace"}
    ax.axvline(0.5, color=INK["axis"], lw=0.8, zorder=1)
    for y, (name, arms) in zip(ypos, rows):
        f = arms["full"]
        ax.plot([f[1], f[2]], [y, y], color=INK["muted"], lw=1.0, zorder=1)
        ax.scatter([f[0]], [y], marker="|", s=160, color=INK["primary"], zorder=5, label="full residual" if y == ypos[0] else None)
        for arm in ("J", "C_J", "randsub"):
            if arm not in arms:
                continue
            est, lo, hi = arms[arm]
            yy = y + offs[arm]
            ax.plot([lo, hi], [yy, yy], color=colors[arm], lw=1.4, zorder=2)
            ax.scatter([est], [yy], s=52, color=colors[arm], edgecolors=INK["surface"], linewidths=1.5, zorder=4, label=names[arm] if y == ypos[0] else None)
    ax.set_yticks(ypos)
    ax.set_yticklabels([r[0] for r in rows], color=INK["primary"])
    ax.set_xlim(0.3, 1.02)
    ax.set_xlabel("primary AUROC, Core discussion → Transfer final (band mean)")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, borderaxespad=0.4)
    fig.suptitle("Core → Transfer at k = 25: J-only next to its random-subspace control, the complement next to the full residual\n"
                 "95% CI: bootstrap over episodes; Hanley–McNeil for the frozen Gemma numbers", x=0.01, ha="left", fontsize=9.5, color=INK["primary"])
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    for ext in ("png", "pdf"):
        fig.savefig(out / f"fig_k25_cross_model.{ext}")
    plt.close(fig)


def fig_three_settings(res: dict[str, dict], out: Path) -> None:
    settings = [("A1_core_ldo", "Core\n(domain held out)"), ("A3_transfer_lofo", "Transfer\n(family held out)"), ("A2_transfer", "Core →\nTransfer")]
    fig, axes = plt.subplots(1, len(res), figsize=(3.6 * len(res), 3.8), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, (tag, s) in zip(axes, res.items()):
        x = np.arange(len(settings))
        for arm, color, off, label in (("J", SLOT["1"], -0.12, "J-only, k = 25"), ("C_J", SLOT["2"], 0.12, "complement, k = 25")):
            ys, los, his = [], [], []
            for key, _ in settings:
                bm = s[key]["band_mean"]["25"][arm]
                ys.append(bm["primary"])
                if key == "A2_transfer":
                    c = s["A2_inference"]["25"]["arms"][arm]["ci"]
                elif key == "A3_transfer_lofo" and s.get("A3_inference"):
                    c = s["A3_inference"]["25"]["arms"][arm]["ci"]
                else:
                    sd = np.mean([v.get("fold_sd_primary", np.nan) for L in s[key]["per_layer"] for kk, v in [( "25", s[key]["per_layer"][L].get("25", {}).get(arm))] if v])
                    c = [bm["primary"] - 1.96 * sd / np.sqrt(max(1, s[key]["per_layer"][list(s[key]["per_layer"])[0]]["25"][arm]["n_folds"])), bm["primary"] + 1.96 * sd / np.sqrt(max(1, s[key]["per_layer"][list(s[key]["per_layer"])[0]]["25"][arm]["n_folds"]))]
                los.append(c[0]); his.append(c[1])
            ax.vlines(x + off, los, his, color=color, lw=1.4, zorder=2)
            ax.scatter(x + off, ys, s=52, color=color, edgecolors=INK["surface"], linewidths=1.5, zorder=4, label=label)
        fulls = [s[key]["band_mean"]["25"]["full"]["primary"] for key, _ in settings]
        ax.scatter(x, fulls, marker="_", s=220, color=INK["primary"], zorder=5, label="full residual")
        ax.axhline(0.5, color=INK["axis"], lw=0.8, zorder=1)
        ax.set_xticks(x)
        ax.set_xticklabels([lab for _, lab in settings])
        ax.set_title(LABEL.get(tag, tag))
        ax.set_ylim(0.3, 1.02)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("primary AUROC (band mean, k = 25)")
    axes[0].legend(loc="lower left")
    fig.suptitle("Where J-space can and cannot tell colluders apart\nbars: 95% CI (bootstrap over episodes; Core: fold mean ± 1.96 SE)",
                 x=0.01, ha="left", fontsize=9.5, color=INK["primary"])
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    for ext in ("png", "pdf"):
        fig.savefig(out / f"fig_three_settings.{ext}")
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["qwen3_32b", "deepseek_r1_32b", "gpt_oss_20b"])
    ap.add_argument("--out", type=Path, default=ROOT / "results/jspace/figures")
    ap.add_argument("--frozen", type=Path, default=ROOT / "results/jspace/gemma2_9b_frozen/frozen_stats.json")
    args = ap.parse_args()
    out = args.out if args.out.is_absolute() else ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    res = load(args.models)
    if not res:
        raise SystemExit("no summary.json found")
    for tag, s in res.items():
        fig_k_curves(tag, s, out)
    frozen = json.loads(args.frozen.read_text()) if args.frozen.exists() else None
    fig_cross_model(res, out, frozen)
    if all("A3_transfer_lofo" in s and s["A3_transfer_lofo"].get("band_mean") and s["A1_core_ldo"].get("band_mean") for s in res.values()):
        fig_three_settings(res, out)
    print("wrote", sorted(p.name for p in out.glob("*.png")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
