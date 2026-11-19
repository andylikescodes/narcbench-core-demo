#!/usr/bin/env python3
"""Stage 2: checks of a probe pod's results on the tables it served. No GPU; nothing from uninstructed/ is imported,
so the probe fit and the pair count here are a second, separate implementation.

  1. Identical prompts. Members who open a room in runs of the same matched set have read the same text before the
     first message, so their activations there should be the same. On the GPU they differ by rounding (bfloat16, rows
     padded next to different rows); the check reports how much, next to the difference where the texts differ.
  2. A few cells of the sweep, computed again: standardise, logistic regression fit on the other domains, matched-prompt
     AUROC counted pair by pair. Compared with the same cells of probe.json.
  3. How large the J-space part of an activation is (norms and the share of the squared norm).

  python scripts/uninstructed_stage2_recheck.py                       # tables and probe.json at their usual places
  python scripts/uninstructed_stage2_recheck.py --acts <capture folder> --probe <probe.json> --out <RECHECK.md>

Needs numpy and scikit-learn. Exit code 1 if a recomputed cell is more than --tolerance away from the sweep.
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CELLS = [("vote", "v.pre", 22), ("vote", "p3.pre", 22), ("vote", "m4.end", 22), ("vote", "m2.end", 22), ("vote", "m1.q3", 19),
         ("vote", "m4.pre", 21), ("vote", "m1.pre", 22), ("hidden", "v.pre", 22), ("hidden", "p3.pre", 22), ("hidden", "m4.end", 22)]


def pair_auroc(score, y, sets) -> tuple[float, int]:
    by = collections.defaultdict(list)
    for i, s in enumerate(sets):
        by[s].append(i)
    num = den = 0.0
    for idx in by.values():
        one = [i for i in idx if y[i] == 1]
        zero = [i for i in idx if y[i] == 0]
        for i in one:
            for j in zero:
                den += 1
                num += 1.0 if score[i] > score[j] else (0.5 if score[i] == score[j] else 0.0)
    return (num / den if den else float("nan")), int(den)


def lodo(X, y, dom) -> np.ndarray:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    out = np.full(len(y), np.nan)
    for d in sorted(set(dom)):
        te = np.array([x == d for x in dom])
        tr = ~te
        if len(set(y[tr])) < 2:
            continue
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000).fit(sc.transform(X[tr]), y[tr])
        out[te] = clf.decision_function(sc.transform(X[te]))
    return out


def member_point(point: str) -> tuple[str | None, str]:
    """A room point belongs to the member who writes there: messages 1 and 3 the opener, 2 and 4 the one who answers."""
    if not point.startswith("m"):
        return None, point
    k = int(point[1])
    return ("opener" if k in (1, 3) else "answerer"), ("o1" if k in (1, 2) else "o2") + point[2:]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--acts", type=Path, default=ROOT / "results/uninstructed/acts/stage2")
    ap.add_argument("--probe", type=Path, default=ROOT / "results/uninstructed/stage2/probe/probe.json")
    ap.add_argument("--out", type=Path, default=ROOT / "results/uninstructed/stage2/explore/RECHECK.md")
    ap.add_argument("--tolerance", type=float, default=0.02)
    a = ap.parse_args()
    T = a.acts / "tables"
    ix = json.loads((T / "index.json").read_text())
    M, pts = ix["members"], ix["member_points"]
    probe = json.loads(a.probe.read_text())
    L = ["# Stage 2: checks of the probe results on the served tables", "",
         "Written by `scripts/uninstructed_stage2_recheck.py`, which shares no code with the sweep.", ""]

    # 1. identical prompts
    layer = 22 if 22 in ix["layers"] else ix["layers"][0]
    A = np.load(T / f"member_L{layer}.npy", mmap_mode="r")
    p0 = pts.index("o1.pre")
    rel = {}
    for role in ("opener", "answerer"):
        sets = collections.defaultdict(list)
        for i, m in enumerate(M):
            if (m["role"] == "opener") == (role == "opener"):
                sets[m["set"]].append(i)
        d = []
        for idx in sets.values():
            if len(idx) > 1:
                X = np.asarray(A[idx, p0, :], dtype=np.float32)
                d += [float(np.linalg.norm(x - X[0]) / np.linalg.norm(X[0])) for x in X[1:]]
        rel[role] = np.array(d)
    L += ["## 1. Identical prompts", "",
          f"Layer {layer}, the token before a member's first room message. Relative difference between two members of a matched set:", "",
          f"- members who open the room (same text up to that token): {len(rel['opener'])} comparisons, median {np.median(rel['opener']):.2%}, "
          f"largest {rel['opener'].max():.2%}, exactly equal in {int((rel['opener'] == 0).sum())};",
          f"- members who answer (the first message differs): median {np.median(rel['answerer']):.1%}.", "",
          "So identical prompts give the same activation up to the rounding of the GPU pass, and a score at `m1.pre` scatters around 0.5 "
          "instead of being exactly 0.5 as on the CPU test.", ""]

    # 2. cells of the sweep
    L += ["## 2. Cells of the sweep, computed again", "", "| Contrast | Position | Layer | Units 1 / 0 | Pairs | Here | The sweep | Difference |", "|---|---|---|---|---|---|---|---|"]
    worst = 0.0
    for con, point, layer in CELLS:
        role, mp = member_point(point)
        rows = [i for i, m in enumerate(M) if not m.get("apply_only") and m["vote"] in (0, 1)
                and (role is None or (m["role"] == "opener") == (role == "opener")) and (con != "hidden" or m.get("public") == "other option")]
        X = np.asarray(np.load(T / f"member_L{layer}.npy", mmap_mode="r")[rows, pts.index(mp), :], dtype=np.float32)
        y = np.array([M[i]["vote"] for i in rows])
        mine, n = pair_auroc(lodo(X, y, [M[i]["domain"] for i in rows]), y, [M[i]["set"] for i in rows])
        e = probe["contrasts"][con][point]
        ref = e["layers"][str(layer)]["full"]["auroc"]
        assert n == e["pairs"] and int(y.sum()) == e["n1"], "the units or pairs differ from the sweep's"
        worst = max(worst, abs(mine - ref))
        L.append(f"| {con} | `{point}` | {layer} | {int(y.sum())} / {int((1 - y).sum())} | {n} | {mine:.3f} | {ref:.3f} | {mine - ref:+.3f} |")
        print(L[-1], flush=True)
    L += ["", f"Largest difference: {worst:.3f}. The units and the pairs are the same in every cell.", ""]

    # 3. the size of the J-space part
    if (T / f"memberJ_L{layer}.npy").exists():
        J = np.load(T / f"memberJ_L{layer}.npy", mmap_mode="r")
        L += ["## 3. The size of the J-space part", "", f"Layer {layer}.", "", "| Position | Norm of the activation (median) | Norm of its J-space part (median) | Share of the squared norm (mean) |", "|---|---|---|---|"]
        for name in ("o1.pre", "o1.q2", "o2.end", "p1.pre", "p3.end", "v.pre", "v.end"):
            p = pts.index(name)
            na = np.linalg.norm(np.asarray(A[:, p, :], dtype=np.float32), axis=1)
            nj = np.linalg.norm(np.asarray(J[:, p, :], dtype=np.float32), axis=1)
            L.append(f"| `{name}` | {np.median(na):.0f} | {np.median(nj):.0f} | {np.mean(nj ** 2 / na ** 2):.1%} |")
        L.append("")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text("\n".join(L) + "\n")
    print(f"wrote {a.out}; largest difference {worst:.3f}")
    return 0 if worst <= a.tolerance else 1


if __name__ == "__main__":
    sys.exit(main())
