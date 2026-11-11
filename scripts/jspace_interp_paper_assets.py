#!/usr/bin/env python3
"""Figures and compact tables for docs/jspace_interp/jspace_colluder_study.tex, built from the result files.

Nothing here is typed by hand: every plotted or tabulated number is read from
  results/jspace_interp/gemma2_9b/SUMMARY.json                  (written by jspace_interp_summarize.py)
  results/jspace_interp/gemma2_9b/original_last/sweep_results.json  (the 200 random draws per layer)
  results/jspace_interp/steer/steer_core_*.json                 (steering runs)

Outputs
  docs/jspace_interp/figs/fig_controls.pdf    Finding 1/3: the four context conditions, across splits and within Core
  docs/jspace_interp/figs/fig_layers.pdf      Finding 2: J-space against the two random nulls, per layer
  docs/jspace_interp/figs/fig_positions.pdf   Finding 4: read positions
  docs/jspace_interp/figs/fig_steering.pdf    Finding 7: steering against random lens directions
  docs/jspace_interp/tables/tabp_*.tex        compact tables used in the body of the paper
  <--numbers>                                 optional JSON with every number the paper's text quotes (for checking)

  python scripts/jspace_interp_paper_assets.py [--png-dir <dir>] [--numbers <file.json>]

Colours: the first three slots of a palette checked for colour-vision deficiency (blue = full residual,
orange = J-space, green = PCA); random nulls are grey. Values are also printed in the tables, so no reading
depends on colour alone.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results/jspace_interp"
DOC = ROOT / "docs/jspace_interp"
LAYERS = [19, 20, 21, 22, 23]
K = 25
SWEEP = "gemma2_9b"     # folder under results/jspace_interp; --sweep-dir gemma2_9b_gp reads the gradient-pursuit rerun

BLUE, ORANGE, GREEN = "#2a78d6", "#eb6834", "#1baf7a"   # full residual, J-space, PCA
NULL_A, NULL_B = "#c3c2b7", "#898781"                   # random subspaces, random lens-atom sets
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"

COND_LABEL = {"original": "Original", "swap": "Swap", "nosys": "No system prompt", "prompt_only": "Prompt only"}
COND_POS = {"original": "last", "swap": "last", "nosys": "last", "prompt_only": "prompt_end"}
# read positions in the order they occur in a turn; names say what the token is for the probe's rows
POSITIONS = [("frac25", "25% into the reply"), ("frac50", "50% into the reply"), ("frac75", "75% into the reply"),
             ("prevote", "last token of the reply text"), ("eot", "<end_of_turn> token"),
             ("last", "newline after <end_of_turn> (last token)"), ("reply_mean", "mean over all reply tokens")]
CONCEPTS = [("concept:deception_secrecy", "deception / secrecy"), ("concept:steering_persuasion", "steering / persuasion"),
            ("concept:coordination", "coordination"), ("concept:honesty_standards", "honesty / standards")]


# ------------------------------------------------------------------------------------------------------------ data
class Data:
    def __init__(self):
        self.S = json.loads((RES / SWEEP / "SUMMARY.json").read_text())
        self.recon = self.S.get("primary_j_arm") == "recon"   # J_fill* keys then hold the non-negative combination
        self.rows = {(r["condition"], r["position"], r["layer"], r["k"]): r for r in self.S["rows"]}
        self.sweep_ol = json.loads((RES / SWEEP / "original_last/sweep_results.json").read_text())
        self.steer = {p.stem.replace("steer_core_", ""): json.loads(p.read_text()) for p in sorted((RES / "steer").glob("steer_core_*.json"))}

    def cell(self, cond, pos, L, k=K):
        return self.rows[(cond, pos, L, k)]

    def role_tokens(self, cond, pos, L, split):
        """Tokens whose atoms are selected for a different share of colluders than of honest agents (BH q=0.05).
        The sweep file keeps the 30 tokens with the smallest p per cell; returns (token text, difference in share)."""
        sw = json.loads((RES / SWEEP / f"{cond}_{pos}/sweep_results.json").read_text())
        e = next(x for x in sw["exp1"] if x["layer"] == L and x["position"] == pos and x["condition"] == cond)
        return [(t["text"], t["colluder_minus_honest"]) for t in e[split]["top_tokens"] if t["significant"]]

    def band(self, cond, pos, key, k=K):
        return float(np.mean([self.rows[(cond, pos, L, k)][key] for L in LAYERS]))

    def draws(self, L, kind, k=K):
        """The 200 Core->Transfer scores of the random draws at one layer (original condition, last token)."""
        for c in self.sweep_ol["cells"]:
            if c["condition"] == "original" and c["position"] == "last" and c["layer"] == L:
                d = np.array(c[f"k{k}"][kind]["c2t_draws"], float)
                ref = self.cell("original", "last", L, k)[f"{kind}_p95"]
                assert abs(np.percentile(d, 95) - ref) < 1e-9, (L, kind)  # same cell as the summary
                return d
        raise KeyError((L, kind))

    def steer_means(self, run, alpha=4.0):
        """Mean change of the vote logit difference per direction and group at one strength (+u honest, -u colluders)."""
        s = self.steer[run]
        out = {}
        for d in s["directions"]:
            for group, sign in (("honest", "+1"), ("colluder", "-1")):
                key = f"{d}|{alpha}|{sign}"
                v = [r["delta"][key] for r in s["records"] if r["role"] == group and key in r["delta"]]
                y0 = [r["y0"] for r in s["records"] if r["role"] == group and key in r["delta"]]
                dv = np.array(v, float)
                y0 = np.array(y0, float)
                flip = np.mean((y0 < 0) & (y0 + dv > 0)) if group == "honest" else np.mean((y0 > 0) & (y0 + dv < 0))
                out[(d, group)] = {"mean": float(dv.mean()), "n": int(len(dv)), "flip": float(flip)}
        return out


# ----------------------------------------------------------------------------------------------------------- style
def setup():
    import matplotlib as mpl
    from matplotlib import font_manager

    have = {f.name for f in font_manager.fontManager.ttflist}
    fam = next((f for f in ("Arial", "Helvetica", "DejaVu Sans") if f in have), "sans-serif")
    mpl.rcParams.update({
        "font.family": fam, "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "xtick.labelsize": 8,
        "ytick.labelsize": 8, "legend.fontsize": 8, "axes.edgecolor": AXIS, "axes.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False, "axes.labelcolor": INK2, "xtick.color": INK2,
        "ytick.color": INK2, "text.color": INK, "axes.titlecolor": INK, "xtick.major.size": 0, "ytick.major.size": 0,
        "grid.color": GRID, "grid.linewidth": 0.6, "grid.linestyle": "-", "axes.axisbelow": True,
        "pdf.fonttype": 42, "ps.fonttype": 42, "figure.dpi": 150, "savefig.dpi": 200,
        "legend.frameon": False, "legend.handlelength": 1.6, "legend.handletextpad": 0.5, "legend.columnspacing": 1.6,
    })


def save(fig, name, png_dir):
    out = DOC / "figs"
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.pdf")
    if png_dir:
        Path(png_dir).mkdir(parents=True, exist_ok=True)
        fig.savefig(Path(png_dir) / f"{name}.png")
    import matplotlib.pyplot as plt
    plt.close(fig)


def rounded_bar(ax, x, y, base, w, color):
    """A bar from ``base`` to ``y`` with a rounded data end and a square foot (call after limits and size are set)."""
    from matplotlib.patches import FancyBboxPatch, Rectangle

    h = y - base
    if abs(h) < 1e-9:
        return
    bb = ax.get_window_extent()
    (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()
    aspect = ((y1 - y0) / bb.height) / ((x1 - x0) / bb.width)   # y data units per x data unit, at equal screen length
    r = min(w * 0.22, abs(h) / aspect * 0.9)
    lo = min(base, y)
    ax.add_patch(FancyBboxPatch((x - w / 2, lo), w, abs(h), boxstyle=f"round,pad=0,rounding_size={r}", mutation_aspect=aspect,
                                fc=color, ec="none", lw=0, zorder=3))
    foot = min(abs(h), r * aspect)
    ax.add_patch(Rectangle((x - w / 2, base if h > 0 else base - foot), w, foot, fc=color, ec="none", lw=0, zorder=3))


# --------------------------------------------------------------------------------------------------------- figures
def fig_controls(D, png_dir):
    """Core->Transfer and within-Core AUROC of three representations under the four context conditions."""
    import matplotlib.pyplot as plt

    conds = ["original", "swap", "nosys", "prompt_only"]
    xl = ["Original", "Swap", "No system\nprompt", "Prompt only"]
    arms = [("full", BLUE, "Full residual"), ("pca", GREEN, "PCA, 25 components"), ("J_fill", ORANGE, "J-space, 25 atoms")]
    panels = [("", "Train on Core, test on Transfer"), ("_lodo", "Within Core (leave one domain out)")]
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.3), sharey=True)
    fig.subplots_adjust(left=0.085, right=0.995, top=0.775, bottom=0.33, wspace=0.07)
    w, gap = 0.17, 0.035
    for ax, (suf, title) in zip(axes, panels):
        ax.set_xlim(-0.55, 3.55)
        ax.set_ylim(0.38, 1.06)
        ax.set_yticks([0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        ax.grid(axis="y")
        ax.axhline(0.5, color=INK2, lw=0.8, zorder=2)
        ax.set_title(title, loc="left", pad=6, fontweight="bold")
        fig.canvas.draw()
        for i, c in enumerate(conds):
            for j, (key, col, _) in enumerate(arms):
                v = D.band(c, COND_POS[c], key + suf)
                x = i + (j - 1) * (w + gap)
                rounded_bar(ax, x, v, 0.5, w, col)
                if key == "full":
                    # a bar below chance is labelled above the baseline, where nothing else is drawn
                    ax.text(x, max(v, 0.5) + 0.012, f"{v:.3f}", ha="center", va="bottom", fontsize=7.5, color=INK)
        ax.set_xticks(range(4))
        ax.set_xticklabels(xl, color=INK, linespacing=1.15)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)
        tr = ax.get_xaxis_transform()
        for x, txt in ((0, "instruction\npresent"), (1.5, "instruction removed,\nconversation kept"), (3, "instruction\nalone")):
            ax.text(x, -0.37, txt, transform=tr, ha="center", va="top", fontsize=7, color=INK2, linespacing=1.15)
        ax.plot([0.62, 2.38], [-0.33, -0.33], transform=tr, clip_on=False, color=AXIS, lw=0.8)
    axes[0].set_ylabel("AUROC (band mean, layers 19–23)")
    axes[0].text(2.36, 0.507, "chance", fontsize=7, color=INK2, va="bottom", ha="center")
    handles = [plt.Rectangle((0, 0), 1, 1, fc=c, ec="none") for _, c, _ in arms]
    fig.legend(handles, [n for _, _, n in arms], loc="upper left", bbox_to_anchor=(0.075, 1.0), ncol=3, handlelength=1.0, handleheight=0.9)
    save(fig, "fig_controls", png_dir)


def fig_layers(D, png_dir):
    """J-space against the two random nulls at each layer (original condition, last token, k=25)."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.25), sharey=True)
    fig.subplots_adjust(left=0.085, right=0.995, top=0.785, bottom=0.11, wspace=0.07)
    panels = [("randsub", NULL_A, "Against 200 random 25-dimensional subspaces"),
              ("randatoms", NULL_B, "Against 200 random sets of 25 lens atoms")]
    for ax, (kind, col, title) in zip(axes, panels):
        ax.set_xlim(18.5, 23.5)
        ax.set_ylim(0.30, 1.02)
        ax.set_yticks([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        ax.grid(axis="y")
        ax.axhline(0.5, color=INK2, lw=0.8, zorder=1)
        ax.set_title(title, loc="left", pad=6, fontweight="bold")
        for L in LAYERS:
            d = D.draws(L, kind)
            ax.scatter(L + rng.uniform(-0.2, 0.2, len(d)), d, s=5, color=col, alpha=0.75, lw=0, zorder=2)
            p95 = float(np.percentile(d, 95))
            ax.plot([L - 0.27, L + 0.27], [p95, p95], color=INK, lw=1.4, solid_capstyle="butt", zorder=4)
        for key, c, mk in (("full", BLUE, "D"), ("pca", GREEN, "s")):
            ys = [D.cell("original", "last", L)[key] for L in LAYERS]
            ax.plot(LAYERS, ys, color=c, lw=1.5, marker=mk, ms=5.5, mec="white", mew=1.0, zorder=5, solid_capstyle="round")
        ys = [D.cell("original", "last", L)["J_fill"] for L in LAYERS]
        ax.plot(LAYERS, ys, color=ORANGE, lw=1.5, marker="o", ms=7, mec="white", mew=1.2, zorder=6, solid_capstyle="round")
        for L, y in zip(LAYERS, ys):
            ax.annotate(f"{y:.3f}", (L, y), xytext=(0, -9), textcoords="offset points", ha="center", va="top", fontsize=7.5, color=INK,
                        zorder=7, bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.92))
        ax.set_xticks(LAYERS)
        ax.set_xticklabels([f"layer {L}" for L in LAYERS], color=INK)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)
    axes[0].set_ylabel("Core → Transfer AUROC")
    handles = [Line2D([], [], color=ORANGE, lw=1.5, marker="o", ms=7, mec="white", mew=1.2),
               Line2D([], [], color=BLUE, lw=1.5, marker="D", ms=5.5, mec="white", mew=1.0),
               Line2D([], [], color=GREEN, lw=1.5, marker="s", ms=5.5, mec="white", mew=1.0),
               Line2D([], [], color=NULL_B, lw=0, marker="o", ms=3.5, mec="none"),
               Line2D([], [], color=INK, lw=1.4)]
    fig.legend(handles, ["J-space (value printed)", "Full residual", "PCA-25", "one random draw", "95th percentile of the draws"],
               loc="upper left", bbox_to_anchor=(0.075, 1.0), ncol=5, handlelength=1.4, columnspacing=1.1)
    save(fig, "fig_layers", png_dir)


def fig_positions(D, png_dir):
    """Where in the turn the probe is read: across splits and within Core (original condition, k=25, band mean)."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    order = POSITIONS[:6]
    ypos = {p: 6.6 - i for i, (p, _) in enumerate(order)}
    ypos["reply_mean"] = 0.0
    labels = dict(POSITIONS)
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.6), sharey=True)
    fig.subplots_adjust(left=0.335, right=0.985, top=0.79, bottom=0.165, wspace=0.09)
    panels = [("", "Train on Core, test on Transfer"), ("_lodo", "Within Core (leave one domain out)")]
    labelled = ("prevote", "eot", "last", "reply_mean")   # the end of the turn is the story; the table has every value
    for ax, (suf, title) in zip(axes, panels):
        ax.set_xlim(0.38, 1.03)
        ax.set_ylim(-0.6, 7.45)
        ax.set_xticks([0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        ax.grid(axis="x")
        ax.axvline(0.5, color=INK2, lw=0.8, zorder=1)
        ax.axhline(0.8, color=AXIS, lw=0.6, zorder=1)
        ax.set_title(title, loc="left", pad=6, fontweight="bold")
        for p, y in ypos.items():
            vals = {k: D.band("original", p, k + suf) for k in ("full", "pca", "J_fill")}
            ax.plot([min(vals.values()), max(vals.values())], [y, y], color=AXIS, lw=1.0, zorder=2)
            if not suf:
                r = D.band("original", p, "randsub_p95")
                ax.plot([r, r], [y - 0.28, y + 0.28], color=INK, lw=1.4, zorder=3)
            for key, c, mk, ms in (("pca", GREEN, "s", 5.5), ("J_fill", ORANGE, "o", 6.5), ("full", BLUE, "D", 5.5)):
                ax.plot([vals[key]], [y], color=c, marker=mk, ms=ms, mec="white", mew=1.0, lw=0, zorder=5)
            if not suf and p in labelled:
                edge = max(max(vals.values()), D.band("original", p, "randsub_p95"))
                ax.annotate(f"{vals['full']:.3f}", (edge, y), xytext=(7, 0), textcoords="offset points", ha="left", va="center",
                            fontsize=7.5, color=INK, zorder=6)
        ax.set_yticks(list(ypos.values()))
        ax.set_yticklabels([labels[p] for p in ypos], color=INK)
        ax.spines["left"].set_visible(False)
        ax.set_xlabel("AUROC (band mean, layers 19–23)")
    axes[0].text(0.507, 7.4, "chance", fontsize=7, color=INK2, va="top", ha="left")
    handles = [Line2D([], [], color=BLUE, lw=0, marker="D", ms=5.5, mec="white", mew=1.0),
               Line2D([], [], color=GREEN, lw=0, marker="s", ms=5.5, mec="white", mew=1.0),
               Line2D([], [], color=ORANGE, lw=0, marker="o", ms=6.5, mec="white", mew=1.0),
               Line2D([], [], color=INK, lw=1.4)]
    fig.legend(handles, ["Full residual (value printed)", "PCA-25", "J-space",
                         "95th percentile of 200 random subspaces"], loc="upper left", bbox_to_anchor=(0.0, 1.0), ncol=4,
               columnspacing=1.2, handlelength=1.2)
    save(fig, "fig_positions", png_dir)


def fig_steering(D, png_dir):
    """Steering at swap, layer 23, strength 4: named directions against 20 random lens and 20 Gaussian directions."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    g = D.steer_means("swap_L23")
    ln = D.steer_means("swap_L23_lensnull")
    rows = [(d, lab) for d, lab in CONCEPTS] + [("jspace_role_probe", "J-space role direction")]
    ys = {d: 7.0 - i for i, (d, _) in enumerate(rows)}
    y_lens, y_gauss = 1.2, 0.2
    rng = np.random.default_rng(1)
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.25), sharey=True)
    fig.subplots_adjust(left=0.285, right=0.985, top=0.77, bottom=0.185, wspace=0.09)
    panels = [("honest", "Honest agents, +u  (300 prompts)", (-0.6, 4.3)), ("colluder", "Colluders, −u  (100 prompts)", (-2.6, 1.1))]
    for ax, (grp, title, xlim) in zip(axes, panels):
        ax.set_xlim(*xlim)
        ax.set_ylim(-0.5, 7.6)
        ax.grid(axis="x")
        ax.axvline(0, color=INK2, lw=0.8, zorder=1)
        ax.axhline(2.1, color=AXIS, lw=0.6, zorder=1)
        ax.set_title(title, loc="left", pad=6, fontweight="bold")
        lens = np.array([ln[(d, grp)]["mean"] for d in D.steer["swap_L23_lensnull"]["directions"] if d.startswith("random:")])
        gauss = np.array([g[(d, grp)]["mean"] for d in D.steer["swap_L23"]["directions"] if d.startswith("random:")])
        ax.axvspan(lens.min(), lens.max(), color=NULL_A, alpha=0.38, lw=0, zorder=0)
        ax.scatter(lens, y_lens + rng.uniform(-0.2, 0.2, len(lens)), s=13, color=NULL_B, lw=0, zorder=3)
        ax.scatter(gauss, y_gauss + rng.uniform(-0.2, 0.2, len(gauss)), s=13, color=NULL_A, lw=0, zorder=3)
        for d, _ in rows:
            v = g[(d, grp)]["mean"]
            role = d == "jspace_role_probe"
            ax.plot([0, v], [ys[d], ys[d]], color=AXIS, lw=1.0, zorder=2)
            ax.plot([v], [ys[d]], color=ORANGE, marker="D" if role else "o", ms=6 if role else 7, mec="white", mew=1.0, lw=0, zorder=5)
            away = 1 if v >= 0 else -1
            ax.annotate(f"{v:+.2f}".replace("-", "\u2212"), (v, ys[d]), xytext=(7 * away, 0), textcoords="offset points", ha="left" if away > 0 else "right",
                        va="center", fontsize=7.5, color=INK, zorder=6)
        ax.set_yticks([ys[d] for d, _ in rows] + [y_lens, y_gauss])
        ax.set_yticklabels([lab for _, lab in rows] + ["20 random lens directions", "20 random Gaussian directions"], color=INK)
        ax.spines["left"].set_visible(False)
        ax.set_xlabel("mean change of the vote logit difference")
    handles = [Line2D([], [], color=ORANGE, lw=0, marker="o", ms=7, mec="white", mew=1.0),
               Line2D([], [], color=ORANGE, lw=0, marker="D", ms=6, mec="white", mew=1.0),
               plt.Rectangle((0, 0), 1, 1, fc=NULL_A, ec="none", alpha=0.38)]
    fig.legend(handles, ["concept direction (mean lens atom of the word group)", "role direction", "range of the 20 random lens directions"],
               loc="upper left", bbox_to_anchor=(0.0, 1.0), ncol=3, columnspacing=1.2, handlelength=1.2)
    save(fig, "fig_steering", png_dir)


# ---------------------------------------------------------------------------------------------------------- tables
def f3(x):
    return f"{x:.3f}"


def rng_txt(vals, nd=2):
    lo, hi = min(vals), max(vals)
    return f"{lo:.{nd}f}" if f"{lo:.{nd}f}" == f"{hi:.{nd}f}" else f"{lo:.{nd}f}--{hi:.{nd}f}"


def write(name, lines):
    out = DOC / "tables"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{name}.tex").write_text("\n".join(lines) + "\n")


def layers_where(D, key_a, key_b):
    return [L for L in LAYERS if D.cell("original", "last", L)[key_a] > D.cell("original", "last", L)[key_b]]


def tables(D):
    b = lambda key, cond="original", pos="last", k=K: D.band(cond, pos, key, k)  # noqa: E731
    j = b("J_fill")
    # --- scoreboard: the three baselines (Finding 2)
    ls, la = layers_where(D, "J_fill", "randsub_p95"), layers_where(D, "J_fill", "randatoms_p95")
    lp = layers_where(D, "J_fill", "pca")
    how = lambda ls_: ", ".join(str(x) for x in ls_) if ls_ else "none"  # noqa: E731
    po = D.band("prompt_only", "prompt_end", "full")
    lpo = [L_ for L_ in LAYERS if D.cell("original", "last", L_)["J_fill"] > D.cell("prompt_only", "prompt_end", L_)["full"]]
    write("tabp_scoreboard", [
        r"\begin{tabularx}{\linewidth}{@{}Xcccc@{}}\toprule",
        r"Baseline J-space had to beat & Baseline & J-space & Beaten? & Above it at layers\\\midrule",
        rf"1a. Random 25-dim subspaces, 95th percentile & {f3(b('randsub_p95'))} & {f3(j)} & No & {how(ls)}\\",
        rf"1b. Random sets of 25 lens atoms, 95th percentile & {f3(b('randatoms_p95'))} & {f3(j)} & No & {how(la)}\\",
        rf"2. Prompt-only probe (system prompt alone) & {f3(po)} & {f3(j)} & No & {how(lpo)}\\",
        rf"3. PCA subspace, 25 components fit on Core & {f3(b('pca'))} & {f3(j)} & No & {how(lp)}\\",
        r"\bottomrule\end{tabularx}"])
    # --- controls: three protocols per condition (Findings 1 and 3)
    L = [r"\begin{tabular}{@{}lccccccccc@{}}\toprule",
         r" & \multicolumn{3}{c}{Core $\to$ Transfer} & \multicolumn{3}{c}{Within Core (LODO)} & \multicolumn{3}{c}{Within Transfer (LOFO)}\\",
         r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(l){8-10}",
         r"Condition & Full & PCA-25 & J-space & Full & PCA-25 & J-space & Full & PCA-25 & J-space\\\midrule"]
    for c in ("original", "swap", "nosys", "prompt_only"):
        p = COND_POS[c]
        L.append(" & ".join([COND_LABEL[c]] + [f3(D.band(c, p, a + s)) for s in ("", "_lodo", "_lofo") for a in ("full", "pca", "J_fill")]) + r"\\")
    L.append(r"\bottomrule\end{tabular}")
    write("tabp_controls", L)
    # --- k sweep (Finding 2)
    kcols = ("J_fill", "J_span", "pca", "randsub_p95", "randatoms_p95", "full") if D.recon else ("J_fill", "pca", "randsub_p95", "randatoms_p95", "full")
    L = [r"\begin{tabular}{@{}l" + "c" * len(kcols) + r"@{}}\toprule",
         r"$k$ & J-space & " + (r"Span of exactly $k$ & " if D.recon else "") + r"PCA-$k$ & Random subspaces (p95) & Random lens atoms (p95) & Full residual\\\midrule"]
    for k in (8, 25, 50, 100):
        L.append(" & ".join([str(k)] + [f3(b(a, k=k)) for a in kcols]) + r"\\")
    L.append(r"\bottomrule\end{tabular}")
    write("tabp_ksweep", L)
    # --- positions (Finding 4)
    L = [r"\begin{tabular}{@{}llcccccc@{}}\toprule",
         r" & & \multicolumn{4}{c}{Core $\to$ Transfer} & \multicolumn{2}{c}{Full residual, within a split}\\",
         r"\cmidrule(lr){3-6}\cmidrule(l){7-8}",
         r"Read position & Code name & Full & PCA-25 & J-space & Random p95 & Core LODO & Transfer LOFO\\\midrule"]
    for p, lab in POSITIONS:
        L.append(" & ".join([lab.replace("%", r"\%").replace("<end_of_turn>", r"\texttt{<end\_of\_turn>}"), r"\texttt{" + p.replace("_", r"\_") + "}"] +
                            [f3(b(a, pos=p)) for a in ("full", "pca", "J_fill", "randsub_p95", "full_lodo", "full_lofo")]) + r"\\")
    L.append(r"\bottomrule\end{tabular}")
    write("tabp_positions", L)
    # --- matched vocabulary, last token, ranges over layers (Finding 5)
    mv = {m["layer"]: m for m in D.S["matched_vocab"] if m["condition"] == "original" and m["position"] == "last"}
    assert sorted(mv) == LAYERS
    L = [r"\begin{tabularx}{\linewidth}{@{}Xcccc@{}}\toprule",
         r" & Colluders vs & Warned vs & Echo & Collusion vs\\",
         r"Representation & warned agents & unwarned agents & ratio & warned episodes\\\midrule"]
    for arm, lab in (("full", "Full residual"), ("J_fill", "J-space, 25 atoms"), ("pca", "PCA, 25 components")):
        L.append(" & ".join([lab] + [rng_txt([mv[x][arm][key] for x in LAYERS]) for key in
                                     ("colluder_vs_warned", "warned_vs_unwarned", "echo_ratio", "ep_collusion_vs_warned")]) + r"\\")
    rs = [mv[x]["randsub"] for x in LAYERS]
    L.append(" & ".join(["Random 25-dimensional subspaces",
                         rng_txt([r["auroc_colluder_vs_warned"]["mean"] for r in rs]),
                         rng_txt([r["auroc_warned_vs_unwarned"]["mean"] for r in rs]),
                         rng_txt([float(np.median(r["echo_ratio"]["draws"])) for r in rs]), "n/a"]) + r"\\")
    L.append(r"\bottomrule\end{tabularx}")
    write("tabp_matched_vocab", L)
    # --- atoms (Finding 6): most common atoms at layer 22, two positions
    mc = D.S["exp1_most_common_atoms"]

    def tok(t):
        t = t.replace("▁", r"\textvisiblespace{}")
        return r"\texttt{" + t.replace("_", r"\_") + "}"

    a_last = mc["original/last/L22"]["core"][:8]
    a_prev = mc["original/prevote/L22"]["core"][:8]
    L = [r"\begin{tabular}{@{}lc@{\hskip 2.2em}lc@{}}\toprule",
         r"\multicolumn{2}{@{}l}{Last token (newline after \texttt{<end\_of\_turn>})} & \multicolumn{2}{l}{Last token of the reply text}\\",
         r"Atom (vocabulary token) & Agents & Atom (vocabulary token) & Agents\\\midrule"]
    for (t1, c1), (t2, c2) in zip(a_last, a_prev):
        L.append(f"{tok(t1)} & {c1} & {tok(t2)} & {c2}" + r"\\")
    L.append(r"\bottomrule\end{tabular}")
    write("tabp_atoms", L)
    # --- role differences in the atoms, with and without the instruction (layer 22, last token)
    def plain(t):
        w = t.replace("▁", "")
        return w.isascii() and w.isalpha()

    def lst(cond, split, sign, n):
        tk = [(t, d) for t, d in D.role_tokens(cond, "last", 22, split) if plain(t) and d * sign > 0]
        tk = sorted(tk, key=lambda x: -abs(x[1]))[:n]
        if not tk:
            return "none"
        return ", ".join(f"{t.replace('▁', '')} (${100 * d:+.0f}$)" for t, d in tk)

    cols = [("original", "core"), ("original", "transfer"), ("swap", "core"), ("swap", "transfer")]
    nsig = {c: len(D.role_tokens(c[0], "last", 22, c[1])) for c in cols}
    L = [r"\begin{tabular}{@{}>{\raggedright\arraybackslash}p{0.105\linewidth}>{\raggedright\arraybackslash}p{0.2\linewidth}>{\raggedright\arraybackslash}p{0.225\linewidth}>{\raggedright\arraybackslash}p{0.2\linewidth}>{\raggedright\arraybackslash}p{0.13\linewidth}@{}}\toprule",
         r" & \multicolumn{2}{l}{Instruction in context (original)} & \multicolumn{2}{l}{Instruction removed (swap)}\\",
         r"\cmidrule(r){2-3}\cmidrule(l){4-5}",
         r" & Core & Transfer & Core & Transfer\\\midrule",
         "Selected for more colluders & " + " & ".join(lst(c, sp, +1, 5) for c, sp in cols) + r"\\[2pt]",
         "Selected for fewer colluders & " + " & ".join(lst(c, sp, -1, 3) for c, sp in cols) + r"\\[2pt]",
         "Tokens that differ & " + " & ".join(("30 or more" if nsig[c] >= 30 else str(nsig[c])) for c in cols) + r"\\",
         r"\bottomrule\end{tabular}"]
    write("tabp_atom_roles", L)
    # --- steering (Finding 7): swap, layer 23, strength 4
    g, ln = D.steer_means("swap_L23"), D.steer_means("swap_L23_lensnull")
    lens = {grp: np.array([ln[(d, grp)]["mean"] for d in D.steer["swap_L23_lensnull"]["directions"] if d.startswith("random:")])
            for grp in ("honest", "colluder")}
    gauss = {grp: np.array([g[(d, grp)]["mean"] for d in D.steer["swap_L23"]["directions"] if d.startswith("random:")])
             for grp in ("honest", "colluder")}
    gflip = {grp: max(g[(d, grp)]["flip"] for d in D.steer["swap_L23"]["directions"] if d.startswith("random:")) for grp in ("honest", "colluder")}
    lflip = {grp: max(ln[(d, grp)]["flip"] for d in D.steer["swap_L23_lensnull"]["directions"] if d.startswith("random:")) for grp in ("honest", "colluder")}
    pct = lambda x: f"{100 * x:.0f}\\%"  # noqa: E731
    sg = lambda x: f"${x:+.2f}$"  # noqa: E731  (math mode, so the minus sign is a minus sign)
    L = [r"\begin{tabular}{@{}lccccc@{}}\toprule",
         r" & \multicolumn{2}{c}{Honest agents, $+u$} & \multicolumn{2}{c}{Colluders, $-u$} & Beyond all 20 random\\",
         r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
         r"Direction $u$ & $\Delta y$ & votes flipped & $\Delta y$ & votes flipped & lens directions?\\\midrule"]
    for d, lab in CONCEPTS + [("jspace_role_probe", "J-space role direction")]:
        h, c = g[(d, "honest")], g[(d, "colluder")]
        if d.startswith("concept:"):
            bh, bc = h["mean"] > lens["honest"].max(), c["mean"] < lens["colluder"].min()
            verdict = "both groups" if bh and bc else ("honest agents only" if bh else ("colluders only" if bc else "neither group"))
        else:
            verdict = "not tested"
        L.append(f"{lab} & {sg(h['mean'])} & {pct(h['flip'])} & {sg(c['mean'])} & {pct(c['flip'])} & {verdict}" + r"\\")
    L.append(r"\midrule")
    L.append(rf"20 random lens directions (range) & {sg(lens['honest'].min())} to {sg(lens['honest'].max())} & up to {pct(lflip['honest'])} & "
             rf"{sg(lens['colluder'].min())} to {sg(lens['colluder'].max())} & up to {pct(lflip['colluder'])} & \\")
    L.append(rf"20 random Gaussian directions (range) & {sg(gauss['honest'].min())} to {sg(gauss['honest'].max())} & up to {pct(gflip['honest'])} & "
             rf"{sg(gauss['colluder'].min())} to {sg(gauss['colluder'].max())} & up to {pct(gflip['colluder'])} & \\")
    L.append(r"\bottomrule\end{tabular}")
    write("tabp_steer", L)


def solver_table():
    """Table of the readings of the paper's definition (scripts/jspace_interp_solver_check.py), if that check has been run
    with its ``extra`` block on all five layers.

    Row 1 is the reading the sweep takes as its J-space arm: gradient pursuit over the whole vocabulary, non-negative
    combination. Every other row is compared with it on the same episode resamples. The rows without the final-norm gain
    come from the sweeps results/jspace_interp/gemma2_9b_gpfull_nogain (whole vocabulary) and gemma2_9b_gp_nogain
    (shortlist). Returns the numbers written, for the text check."""
    p = RES / "solver_check/solver_check.json"
    if not p.exists():
        return None
    S = json.loads(p.read_text())
    layers = [str(x) for x in S["layers"]]
    ex = S.get("extra") if (S.get("extra") or {}).get("layers") == S["layers"] else None
    if ex is None or "band_minus_gpfull_recon_ci95" not in ex:
        return None
    DF = ex["band_minus_gpfull_recon_ci95"]
    out = {"layers": S["layers"], "rows": {}}

    def row(lab, per, band, diff, key, ci=None):
        out["rows"][key] = {"per_layer": [round(x, 4) for x in per], "band": round(band, 4), "diff_ci": [round(x, 3) for x in diff] if diff else None,
                            "ci": [round(x, 3) for x in ci] if ci else None}
        d = "--" if diff is None else f"$[{diff[0]:+.3f}, {diff[1]:+.3f}]$"
        return " & ".join([lab] + [f3(x) for x in per] + [f3(band), d]) + r"\\"

    def main_row(key, lab):
        return row(lab, [S["per_layer"][x][key]["core_to_transfer"] for x in layers], S["band_mean"][key], DF[key], key, S["band_ci95"][key])

    def extra_row(key, lab, ref=False):
        return row(lab, [ex["per_layer"][x][key]["core_to_transfer"] for x in layers], ex["band_mean"][key], None if ref else DF[key], key,
                   ex["band_ci95"][key])

    def sweep_cells(folder):
        f = RES / folder / "original_last/sweep_results.json"
        if not f.exists():
            return None
        cells = {str(c["layer"]): c for c in json.loads(f.read_text())["cells"]}
        return cells if all(x in cells and cells[x].get(f"k{K}") for x in layers) else None

    def band_boot(folder, arm):
        fs = [RES / folder / f"original_last/boot_original_last_L{x}_k{K}.npz" for x in layers]
        if not all(f.exists() for f in fs):
            return None
        return np.mean(np.stack([np.load(f)[arm] for f in fs]), axis=0)

    ncol = len(layers) + 3
    L = [r"\begin{tabularx}{\linewidth}{@{}X" + "c" * len(layers) + r"cc@{}}\toprule",
         "Reading of the definition & " + " & ".join(f"L{x}" for x in layers) + r" & Band & Difference to row 1\\\midrule",
         rf"\multicolumn{{{ncol}}}{{@{{}}l}}{{\emph{{The pursuit searches the whole vocabulary at every step}}}}\\",
         extra_row("gpfull_recon", r"1. Gradient pursuit, non-negative combination (\textbf{primary})", ref=True),
         extra_row("gpfull_span", "2. As 1, projected onto the span of the chosen vectors"),
         extra_row("gpfull_lit", "3. As 1, with gradient pursuit's own weights (no refit)"),
         r"\midrule",
         rf"\multicolumn{{{ncol}}}{{@{{}}l}}{{\emph{{The pursuit searches the 1{{,}}024 tokens best aligned with the activation}}}}\\",
         extra_row("gp_recon", "4. Gradient pursuit, non-negative combination"),
         main_row("gp_span", "5. Gradient pursuit, span of exactly 25"),
         main_row("nnomp_span", "6. Non-negative OMP, span of exactly 25 (November 9)"),
         main_row("nnomp_cone", "7. Non-negative OMP, combination of the 25 after the fill"),
         r"\midrule",
         rf"\multicolumn{{{ncol}}}{{@{{}}l}}{{\emph{{J-lens vectors without the gain of the final normalisation}}}}\\"]
    n = 8
    ngf = sweep_cells("gemma2_9b_gpfull_nogain")
    if ngf is not None:
        per = [ngf[x][f"k{K}"][f"J{K}_recon"]["core_to_transfer"] for x in layers]
        b_ng, b_pr = band_boot("gemma2_9b_gpfull_nogain", f"J{K}_recon"), band_boot("gemma2_9b_gpfull", f"J{K}_recon")
        diff = [float(v) for v in np.nanpercentile(b_ng - b_pr, [2.5, 97.5])] if b_ng is not None and b_pr is not None else None
        ci = [float(v) for v in np.nanpercentile(b_ng, [2.5, 97.5])] if b_ng is not None else None
        L.append(row(f"{n}. As 1 (whole vocabulary)", per, float(np.mean(per)), diff, "nogain_gpfull_recon", ci))
        n += 1
        out["nogain_nulls"] = {
            "randatoms_p95": [round(ngf[x][f"k{K}"]["randatoms"]["c2t_p95"], 4) for x in layers],
            "randatoms_mean": [round(ngf[x][f"k{K}"]["randatoms"]["c2t_mean"], 4) for x in layers],
            "randsub_p95": [round(ngf[x][f"k{K}"]["randsub"]["c2t_p95"], 4) for x in layers],
            "p_J_recon_vs_randatoms": [round(ngf[x][f"k{K}"]["randatoms"]["J_recon_percentile"][1], 4) for x in layers],
            "p_J_recon_vs_randsub": [round(ngf[x][f"k{K}"]["randsub"]["J_recon_percentile"][1], 4) for x in layers],
            "pca": [round(ngf[x][f"k{K}"][f"pca{K}"]["core_to_transfer"], 4) for x in layers],
            "C_recon": [round(ngf[x][f"k{K}"][f"C{K}_recon"]["core_to_transfer"], 4) for x in layers],
            "J_span": [round(ngf[x][f"k{K}"][f"J{K}_fill"]["core_to_transfer"], 4) for x in layers],
            "steps_core": [round(ngf[x]["pursuit_steps_core_mean"], 2) for x in layers],
            "J_recon_minus_pca_ci": [[round(v, 3) for v in ngf[x][f"k{K}"]["J_recon_minus_pca_ci95"]] for x in layers]}
    ngs = sweep_cells("gemma2_9b_gp_nogain")
    if ngs is not None:
        per = [ngs[x][f"k{K}"][f"J{K}_recon"]["core_to_transfer"] for x in layers]
        L.append(row(f"{n}. As 4 (shortlist, gradient pursuit)", per, float(np.mean(per)), None, "nogain_gp_recon"))
        n += 1
    L.append(main_row("nogain_nnomp_span", f"{n}. As 6 (shortlist, non-negative OMP, span)"))
    L.append(r"\midrule")
    L.append(main_row("full", "Full residual, for reference"))
    L.append(r"\bottomrule\end{tabularx}")
    write("tabp_solver", L)
    out["overlap"] = ex["overlap"]
    out["steps"] = {k: [round(ex["per_layer"][x][k]["steps_core_mean"], 2) for x in layers] for k in ("gp_recon", "gpfull_recon")}
    out["steps_main"] = {k: [round(S["per_layer"][x][k]["steps_core_mean"], 2) for x in layers] for k in ("nnomp_span", "gp_span", "nogain_nnomp_span")}
    out["fve"] = {k: [round(ex["per_layer"][x][k]["fve_core"], 4) for x in layers] for k in ("gp_recon", "gpfull_recon")}
    return out


# --------------------------------------------------------------------------------------------------------- numbers
def numbers(D):
    """Every number the paper's text quotes, recomputed from the result files (for checking the text)."""
    N = {}
    b = lambda key, cond="original", pos="last", k=K: round(D.band(cond, pos, key, k), 4)  # noqa: E731
    N["band_original_last"] = {a: b(a) for a in ("full", "C_fill", "pca", "J_fill", "J_pursuit", "randsub_mean", "randsub_p95",
                                                  "randatoms_mean", "randatoms_p95", "shuffled", "fixedframe", "full_lodo", "full_lofo",
                                                  "J_fill_lodo", "J_fill_lofo", "pca_lodo", "pca_lofo")}
    for c in ("swap", "nosys", "prompt_only"):
        p = COND_POS[c]
        N[f"band_{c}"] = {a: b(a, c, p) for a in ("full", "pca", "J_fill", "randsub_mean", "randsub_p95", "randatoms_p95", "full_lodo",
                                                   "full_lofo", "J_fill_lodo", "J_fill_lofo", "pca_lodo", "pca_lofo")}
    N["per_layer_original_last"] = {L: {a: round(D.cell("original", "last", L)[a], 4) for a in
                                         ("full", "pca", "J_fill", "J_pursuit", "C_fill", "randsub_mean", "randsub_p95", "randatoms_mean",
                                          "randatoms_p95", "p_J_fill_vs_randsub", "p_J_fill_vs_randatoms", "shuffled", "fixedframe",
                                          "pursuit_steps_core_mean")} for L in LAYERS}
    for L in LAYERS:
        c = D.cell("original", "last", L)
        N["per_layer_original_last"][L].update({"J_fill_ci": [round(x, 3) for x in c["J_fill_ci"]], "full_ci": [round(x, 3) for x in c["full_ci"]],
                                                "J_minus_pca_ci": [round(x, 3) for x in c["J_fill_minus_pca_ci"]],
                                                "C_minus_J_ci": [round(x, 3) for x in c["C_fill_minus_J_fill_ci"]]})
    N["per_layer_swap_last_full"] = {L: round(D.cell("swap", "last", L)["full"], 4) for L in LAYERS}
    N["per_layer_prompt_only"] = {L: {a: round(D.cell("prompt_only", "prompt_end", L)[a], 4) for a in ("full", "pca", "J_fill", "randsub_mean", "randsub_p95")} for L in LAYERS}
    N["positions_original"] = {p: {a: b(a, pos=p) for a in ("full", "pca", "J_fill", "randsub_p95", "full_lodo", "full_lofo")} for p, _ in POSITIONS}
    N["ksweep"] = {k: {a: b(a, k=k) for a in ("J_fill", "J_pursuit", "J_span", "pca", "randsub_p95", "randatoms_p95", "full")} for k in (8, 25, 50, 100)}
    N["primary_j_arm"] = D.S.get("primary_j_arm", "fill")
    N["band_original_last"].update({a: b(a) for a in ("J_span", "C_span", "J_fve", "J_span_fve") if D.cell("original", "last", 22).get(a) is not None})
    for L in LAYERS:
        c = D.cell("original", "last", L)
        N["per_layer_original_last"][L].update({a: (round(c[a], 4) if c.get(a) is not None else None) for a in ("J_span", "C_span", "J_fve", "J_atoms_used")})
        if c.get("J_recon_minus_J_span_ci"):
            N["per_layer_original_last"][L]["J_minus_J_span_ci"] = [round(x, 3) for x in c["J_recon_minus_J_span_ci"]]
    for cond in ("original", "swap"):
        cells = [r for r in D.S["rows"] if r["condition"] == cond and r["k"] == K]
        best = max(cells, key=lambda r: r["full"])
        N[f"cells_{cond}"] = {"n": len(cells), "best_full": round(best["full"], 3), "best_full_at": f"{best['position']}/L{best['layer']}",
                              "full_ci_excludes_half": sum(1 for r in cells if r["full_ci"][0] > 0.5 or r["full_ci"][1] < 0.5),
                              "J_clears_both": sum(1 for r in cells if r["J_fill_clears_both_nulls"]),
                              "pca_beats_J": sum(1 for r in cells if r["pca_beats_J_fill"]), "J_beats_pca": sum(1 for r in cells if r["J_fill_beats_pca"])}
    mv = {m["layer"]: m for m in D.S["matched_vocab"] if m["condition"] == "original" and m["position"] == "last"}
    N["matched_vocab_last"] = {arm: {key: [round(mv[L][arm][key], 3) for L in LAYERS] for key in
                                     ("colluder_vs_warned", "warned_vs_unwarned", "echo_ratio", "ep_warned_vs_control", "ep_collusion_vs_warned")}
                               for arm in ("full", "J_fill", "pca")}
    N["matched_vocab_last"]["echo_J_over_full"] = [round(mv[L]["J_fill"]["echo_ratio"] / mv[L]["full"]["echo_ratio"], 2) for L in LAYERS]
    # concept share of the atom selections: sum of the four groups' mean share, per cell and role
    conv, pe = [], []
    cells = {}
    for r in D.S["exp1_concepts"]:
        if r.get("source", "atoms") != "atoms":
            continue
        cells.setdefault((r["condition"], r["position"], r["layer"], r["split"]), []).append(r)
    for (cond, pos, L, split), rs in cells.items():
        for role in ("mean_colluder", "mean_honest", "mean_control"):
            tot = sum((x.get(role) or 0.0) for x in rs)
            (pe if pos == "prompt_end" else conv).append(tot)
    N["concept_share_max"] = {"conversation": round(max(conv), 4), "prompt_end": round(max(pe), 4), "n_cells": len(cells)}
    N["atoms_L22"] = {k: D.S["exp1_most_common_atoms"][k]["core"][:8] for k in ("original/last/L22", "original/prevote/L22")}
    N["atoms_L22_transfer"] = D.S["exp1_most_common_atoms"]["original/last/L22"]["transfer"][:6]
    tt = D.S["exp1_top_tokens"]
    N["token_tests"] = {"cells": len(tt), "core_cells_with_differences": sum(1 for v in tt.values() if v["core"]),
                        "transfer_cells_with_differences": sum(1 for v in tt.values() if v["transfer"]),
                        "transfer_cells_with_differences_list": sorted(k for k, v in tt.items() if v["transfer"])}
    N["role_tokens"] = {}
    for cond in ("original", "swap", "nosys"):
        for L_ in LAYERS:
            for split in ("core", "transfer"):
                tk = D.role_tokens(cond, "last", L_, split)
                N["role_tokens"][f"{cond}/last/L{L_}/{split}"] = {"n": len(tk), "top": sorted([(t, round(d, 2)) for t, d in tk], key=lambda x: -abs(x[1]))[:10]}
    N["positions_swap"] = {p: {a: b(a, cond="swap", pos=p) for a in ("full", "pca", "J_fill", "full_lodo", "full_lofo")} for p, _ in POSITIONS}
    rs = {}
    for cond, pos in (("original", "last"), ("swap", "last"), ("nosys", "last"), ("prompt_only", "prompt_end")):
        sw = json.loads((RES / SWEEP / f"{cond}_{pos}/sweep_results.json").read_text())
        jkey = f"J{K}_recon" if D.recon else f"J{K}_fill"
        rs[cond] = {c["layer"]: {"lodo_mean_20": round(float(np.mean(c[f"k{K}"]["randsub"]["lodo_draws"])), 3),
                                 "c2t_mean": round(c[f"k{K}"]["randsub"]["c2t_mean"], 3),
                                 "fve_J_fill": round(c[f"k{K}"][jkey]["fve_core"], 4),
                                 "steps_core": round(c["pursuit_steps_core_mean"], 2), "steps_transfer": round(c["pursuit_steps_transfer_mean"], 2)}
                    for c in sw["cells"]}
    N["randsub_and_size"] = rs
    cl = [r for r in D.S["rows"] if r["condition"] == "original" and r["k"] == K]
    N["cells_original"]["clears_both_at"] = [f"{r['position']}/L{r['layer']}" for r in cl if r["J_fill_clears_both_nulls"]]
    N["cells_original"]["J_beats_pca_at"] = [f"{r['position']}/L{r['layer']}" for r in cl if r["J_fill_beats_pca"]]
    st = {}
    for run in D.steer:
        s = D.steer[run]
        m4 = D.steer_means(run)
        rnd = {grp: [m4[(d, grp)]["mean"] for d in s["directions"] if d.startswith("random:")] for grp in ("honest", "colluder")}
        st[run] = {"scale": round(s["scale"], 2), "layer": s["layer"], "cache_check": s.get("cache_check"),
                   "baseline": {grp: {"y0_mean": round(float(np.mean([r["y0"] for r in s["records"] if r["role"] == grp])), 3),
                                      "frac_target": round(float(np.mean([r["y0"] > 0 for r in s["records"] if r["role"] == grp])), 3),
                                      "n": sum(1 for r in s["records"] if r["role"] == grp)} for grp in ("honest", "colluder")},
                   "named_alpha4": {f"{d}|{grp}": {"mean": round(m4[(d, grp)]["mean"], 3), "flip": round(m4[(d, grp)]["flip"], 3)}
                                    for d in s["directions"] if not d.startswith("random:") for grp in ("honest", "colluder")},
                   "random_alpha4": {grp: {"min": round(min(v), 3), "max": round(max(v), 3), "median": round(float(np.median(v)), 3),
                                           "max_flip": round(max(m4[(d, grp)]["flip"] for d in s["directions"] if d.startswith("random:")), 3)}
                                     for grp, v in rnd.items()}}
    N["steer"] = st
    N["not_run"] = D.S["not_run"]
    return N


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--png-dir", type=Path, default=None, help="also write PNG previews of the figures here")
    ap.add_argument("--numbers", type=Path, default=None, help="write the numbers the paper's text quotes to this JSON file")
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument("--sweep-dir", default=SWEEP, help="folder under results/jspace_interp with SUMMARY.json and the sweep files "
                                                         "(gemma2_9b: 9 November run; gemma2_9b_gp: gradient-pursuit rerun)")
    a = ap.parse_args()
    globals()["SWEEP"] = a.sweep_dir
    D = Data()
    if not a.no_figures:
        setup()
        for f in (fig_controls, fig_layers, fig_positions, fig_steering):
            f(D, a.png_dir)
    tables(D)
    solver = solver_table()
    if solver is None:
        print("note: results/jspace_interp/solver_check/solver_check.json not found; tabp_solver.tex not written")
    if a.numbers:
        a.numbers.parent.mkdir(parents=True, exist_ok=True)
        N = numbers(D)
        N["solver_check"] = solver
        a.numbers.write_text(json.dumps(N, indent=1, default=float) + "\n")
    print("figures:", sorted(p.name for p in (DOC / "figs").glob("fig_*.pdf")))
    print("tables:", sorted(p.name for p in (DOC / "tables").glob("tabp_*.tex")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
