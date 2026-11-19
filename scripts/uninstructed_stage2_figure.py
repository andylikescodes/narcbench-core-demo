#!/usr/bin/env python3
"""Stage 2: one figure per contrast, the probe's score along the conversation next to what the words give.

Read from results/uninstructed/stage2: probe/probe.json (the sweep), nulls*/seed*/probe.json (the passes with shuffled
labels, as a grey band where the contrast has at least 20 of them: fewer make no band worth reading) and
explore/readers.json (the exploratory reader check of scripts/uninstructed_stage2_readers.py, as a dotted line, when it
is there). The last token of the vote is not drawn: the vote is written there, and the note under the figure says so.

  python3 scripts/uninstructed_stage2_figure.py                 # vote and hidden, into results/uninstructed/stage2/figures
  python3 scripts/uninstructed_stage2_figure.py --contrasts vote --out /tmp/figs

Needs numpy and matplotlib. The colours are the first three of a palette checked for colour-blind separation; every
line also has its own marker and a label at its right end.
"""
from __future__ import annotations

import argparse
import json
import sys
import textwrap
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
COL = {"full": "#2a78d6", "bowwho": "#eb6834", "neutral": "#1baf7a"}
NAME = {"full": "The model's state (probe on its activation)", "bowwho": "The words so far, counted (bag of words, marked by speaker)",
        "neutral": "The same model reading the words alone, without the prompt"}
SHORT = {"full": "model's state", "bowwho": "words, counted", "neutral": "model reading\nthe words alone"}
TITLE = {"vote": "When can a member's later vote be read off?", "hidden": "Who votes for the stake option after arguing against it in public?"}
SUB = {"vote": "Members who vote for the stake option against members who do not; each pair read the same prompt word for word.",
       "hidden": "Only members who argue for the other option in public: those who vote for the stake option all the same against those who do not."}
PUB0, PUB_STEP, PUB_END = 25.6, 4.5, 1.6            # where the positions after the room sit on the x axis
MIN_PASSES = 20                                     # shuffled-label passes a contrast needs before its band is drawn


def draw(s2: Path, contrast: str, out: Path) -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tab = json.loads((s2 / "probe/probe.json").read_text())["contrasts"][contrast]
    pos = [p for p in tab if p != "v.end"]
    nulls = []
    for f in sorted(s2.glob("nulls*/seed*/probe.json")):
        d = json.loads(f.read_text())
        if d.get("labels_shuffled_inside_matched_sets") is not None and contrast in d["contrasts"]:
            nulls.append([d["contrasts"][contrast][p]["mean"]["full"]["auroc"] for p in pos])
    readers = json.loads((s2 / "explore/readers.json").read_text()) if (s2 / "explore/readers.json").exists() else {}

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.edgecolor": GRID, "axes.linewidth": 1.0})
    fig, ax = plt.subplots(figsize=(11.5, 6.6), dpi=200)
    fig.patch.set_facecolor(SURF)
    ax.set_facecolor(SURF)
    # room positions one apart; the positions after the room wider apart, so that their blocks can be labelled
    x = np.array([float(i) if p.startswith("m") else PUB0 + PUB_STEP * ["p1", "p2", "p3", "v"].index(p.split(".")[0]) + (PUB_END if p.endswith(".end") else 0.0)
                  for i, p in enumerate(pos)])
    get = lambda k, f: np.array([tab[p]["mean"][k][f] if tab[p]["mean"].get(k) else np.nan for p in pos])   # noqa: E731
    pub = [i for i, p in enumerate(pos) if not p.startswith("m")]
    xr = x[-1] + 0.9
    if len(nulls) < MIN_PASSES:                     # a band from a handful of passes would look like a result
        nulls = []
    if nulls:
        N = np.array(nulls)
        ax.fill_between(x, np.percentile(N, 2.5, axis=0), np.percentile(N, 97.5, axis=0), color="#b9b8b2", alpha=0.45, lw=0, zorder=1,
                        label=f"Labels shuffled (nothing to find): where 95% of {len(N)} passes fall")
    ax.plot([-0.6, xr], [0.5, 0.5], color=INK2, lw=1.0, ls=(0, (4, 3)), zorder=2)
    ax.text(xr + 0.25, 0.5, "chance", color=INK2, fontsize=9.5, va="center", ha="left")
    ax.fill_between(x, get("full", "lo"), get("full", "hi"), color=COL["full"], alpha=0.16, lw=0, zorder=2)
    ends = {}
    for k, ms, mk in (("neutral", 4.5, "s"), ("bowwho", 4.5, "D"), ("full", 6, "o")):
        y = get(k, "auroc")
        ax.plot(x, y, color=COL[k], lw=2.0, marker=mk, ms=ms, mec=SURF, mew=1.0, zorder=4 if k == "full" else 3, label=NAME[k])
        ends[k] = y[-1]
    rd = None
    if contrast in readers:
        r = readers[contrast]["positions"]["v.pre"]
        key = "room+pub" if "room+pub" in r else "room"
        rd = r[key][0]
        ax.plot([x[pub[0]] - 0.5, x[-1] + 0.5], [rd, rd], color=INK, lw=1.8, ls=(0, (1, 1.5)), zorder=3,
                label=("Claude Fable 5.2 reading the room and the public position" if key == "room+pub"
                       else "Claude Fable 5.2 reading the transcript of the room") + " (added after the results)")
    # labels at the right end, pushed apart where they would touch
    lab = sorted([(v, SHORT[k], COL[k]) for k, v in ends.items()] + ([(rd, "Claude reading\nthe transcript", INK)] if rd else []))
    prev = -1.0
    for v, s, c in lab:
        yy = max(v, prev + 0.058)
        prev = yy
        ax.plot([x[-1] + 0.55, xr + 0.15], [v, yy], color=c, lw=1.0, zorder=3, solid_capstyle="round")
        ax.text(xr + 0.3, yy, f"{s} {v:.2f}", color=INK, fontsize=9.3, va="center", ha="left", linespacing=1.05)
    for k in range(4):
        ax.text((x[6 * k] + x[6 * k + 5]) / 2, 0.352, f"Room\nmessage {k + 1}", ha="center", va="top", fontsize=10, color=INK, linespacing=1.15)
        if k:
            ax.plot([x[6 * k] - 0.5] * 2, [0.38, 1.0], color=GRID, lw=1.0, zorder=0)
    for k, name in enumerate(("Public\nstatement 1", "Public\nstatement 2", "Public\nstatement 3", "Vote")):
        ax.text(PUB0 + PUB_STEP * k + (PUB_END / 2 if k < 3 else 0.0), 0.352, name, ha="center", va="top", fontsize=9.6, color=INK, linespacing=1.15)
        if k:
            ax.plot([PUB0 + PUB_STEP * k - 1.45] * 2, [0.38, 1.0], color=GRID, lw=1.0, zorder=0)
    ax.plot([24.3] * 2, [0.38, 1.0], color=INK2, lw=1.2, zorder=0)
    ax.text(11.5, 1.012, "in the private room (the member who is writing)", ha="center", va="bottom", fontsize=10.5, color=INK2)
    ax.text((24.3 + xr) / 2, 1.012, "after the room (every member)", ha="center", va="bottom", fontsize=10.5, color=INK2)
    ax.set_xlim(-0.9, xr + 6.3)
    ax.set_ylim(0.38, 1.0)
    ax.set_xticks(x)
    ax.set_xticklabels([""] * len(pos))
    ax.tick_params(axis="x", length=3, color="#b9b8b2")
    ax.set_yticks([0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax.tick_params(axis="y", length=0, labelcolor=INK2)
    for yv in (0.4, 0.6, 0.7, 0.8, 0.9, 1.0):
        ax.plot([-0.6, xr], [yv, yv], color=GRID, lw=0.8, zorder=0)
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(False)
    ax.set_ylabel("Share of pairs ranked the right way round\n(0.5 = chance, 1 = always)", color=INK2, fontsize=10)
    fig.text(0.012, 0.972, TITLE[contrast], ha="left", va="top", fontsize=15, color=INK, fontweight="bold")
    fig.text(0.012, 0.925, SUB[contrast], ha="left", va="top", fontsize=10.5, color=INK2)
    h, l = ax.get_legend_handles_labels()
    rank = lambda t: 0 if t == NAME["full"] else 1 if t == NAME["bowwho"] else 2 if t == NAME["neutral"] else 3 if t.startswith("Claude") else 4   # noqa: E731
    order = sorted(range(len(l)), key=lambda i: rank(l[i]))
    leg = ax.legend([h[i] for i in order], [l[i] for i in order], loc="upper left", bbox_to_anchor=(0.005, 0.985), frameon=True, fontsize=9.3,
                    labelcolor=INK, borderpad=0.6, handlelength=2.2, labelspacing=0.45)
    leg.get_frame().set_facecolor(SURF)
    leg.get_frame().set_edgecolor(GRID)
    leg.get_frame().set_alpha(0.94)
    e = tab["v.end"]
    note = (f"Gemma-2-9B, mean of layers 19 to 23. After the room: {e['n1']} members against {e['n0']}, {e['pairs']} pairs; in the room only the member who "
            "is writing counts, about half as many. Marks within a message, left to right: just before it, its first token, 25%, 50%, 75%, its last token; "
            "for a public statement: just before it and its last token; for the vote: just before it. Not drawn: the last token of the vote, where the vote "
            f"is already written ({e['mean']['full']['auroc']:.2f}). Blue band: 95% interval of the probe.")
    if rd:
        note += " Dotted line: a model fitted on Claude's codes of the transcript, which were made without seeing any vote" + (
            "; its reading of the public position covers all three statements, so before the third it has read ahead." if contrast == "vote" else ".")
    fig.text(0.012, 0.012, "\n".join(textwrap.wrap(note, 168)), fontsize=8.6, color=INK2, ha="left", va="bottom", linespacing=1.3)
    fig.subplots_adjust(left=0.075, right=0.995, top=0.86, bottom=0.245)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor=SURF)
    plt.close(fig)
    return len(nulls)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage2", type=Path, default=ROOT / "results/uninstructed/stage2")
    ap.add_argument("--contrasts", default="vote,hidden")
    ap.add_argument("--out", type=Path, default=None, help="folder for the figures (default: <stage2>/figures)")
    a = ap.parse_args()
    for c in [c for c in a.contrasts.split(",") if c]:
        f = (a.out or a.stage2 / "figures") / f"{c}_timeline.png"
        n = draw(a.stage2, c, f)
        print(f"wrote {f} ({n} shuffled-label passes in the band)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
