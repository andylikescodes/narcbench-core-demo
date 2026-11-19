#!/usr/bin/env python3
"""Stage 2, exploratory, added after the probe results were seen (not among the rules fixed in advance):
how well do stronger readers of the WORDS predict a member's vote at the positions after the room, on the same pairs
as the probes? The rule of Q4 compares the probe with a bag of words and with the same model reading the words without
the prompt. Neither of those understands what was agreed in the room, so this adds readers that do:

  room       Claude Fable 5.2's reading of the private room: the room code and the counts of the three marks (stake
             named, proposal, acceptance). Coded earlier, without the votes (uninstructed/coding.py).
  room+pub   the same plus Claude's reading of the member's public position, coded without the vote, the stake or the
             room (scripts/uninstructed_public_stance.py). It covers all three public statements, so at a position
             before the third statement it has read ahead; at the token before the vote it has not.
  agree      one feature and no fit: the room was read as an agreement on the stake option.
  neutral+   the neutral reader's activation at the end of the room joined to its activation at the public position.

Each reader is a logistic regression on its few features, fit on the other domains, scored by the matched-prompt AUROC
with the same resampling of scenarios as the sweep (2,000 resamples, seed 20261118). The probe is fit here again in
the same way at each of the five layers and averaged, so every difference is paired. Positions: the seven after the
room, without the last token of the vote, where the vote is written. Nothing from uninstructed/ is imported.

  python scripts/uninstructed_stage2_readers.py                                  # about 15 minutes on a laptop
  python scripts/uninstructed_stage2_readers.py --report-only                    # READERS.md again from readers.json

Needs numpy and scikit-learn and the tables a probe pod served (results/uninstructed/acts/stage2, never committed).
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BOOT, SEED = 2000, 20261118
CODES = ["AGREE_TARGET", "AGREE_MERITS", "SPLIT", "NO_PACT"]
PUB = ["stake option", "other option", "neither"]
NAMES = {"room": "Claude's reading of the room", "room+pub": "Claude's reading of the room and of the public position",
         "agree_target_only": "One feature: the room read as an agreement on the stake option", "neutral+": "Neutral reader, room and discussion joined"}
MINUS = {"room": "Activation minus Claude's reading of the room", "room+pub": "Activation minus Claude's reading of the room and of the public position",
         "agree_target_only": "Activation minus the one feature", "neutral+": "Activation minus the neutral reader, joined"}
ABOUT = {"vote": "members who vote for the stake option against members who do not",
         "hidden": "members who argue for the other option in public: those who vote for the stake option all the same against those who do not"}


def lodo(X, y, dom) -> np.ndarray:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    out = np.full(len(y), np.nan)
    for d in sorted(set(dom)):
        te = np.array([x == d for x in dom])
        tr = ~te
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=3000).fit(sc.transform(X[tr]), y[tr])
        out[te] = clf.decision_function(sc.transform(X[te]))
    return out


def per_scenario(score, y, sets, scen):
    """Wins and pairs of each scenario; pairs only inside matched sets."""
    by = collections.defaultdict(list)
    for i, s in enumerate(sets):
        by[s].append(i)
    w, n = collections.defaultdict(float), collections.defaultdict(float)
    for idx in by.values():
        one = [i for i in idx if y[i] == 1]
        zero = [i for i in idx if y[i] == 0]
        for i in one:
            for j in zero:
                n[scen[i]] += 1
                w[scen[i]] += 1.0 if score[i] > score[j] else (0.5 if score[i] == score[j] else 0.0)
    return w, n


def summarize(parts, scens, C):
    """parts: (wins, pairs) by scenario, one per layer; their AUROCs are averaged. Point value and resampled values."""
    pt = np.mean([sum(w.values()) / sum(n.values()) for w, n in parts])
    bs = np.zeros(BOOT)
    for w, n in parts:
        wv = np.array([w.get(s, 0.0) for s in scens])
        nv = np.array([n.get(s, 0.0) for s in scens])
        den = C @ nv
        bs += np.where(den > 0, (C @ wv) / np.where(den > 0, den, 1), np.nan)
    return float(pt), bs / len(parts)


def iv(b):
    b = b[np.isfinite(b)]
    return float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def compute(T: Path, log) -> dict:
    ix = json.loads((T / "index.json").read_text())
    M, R = ix["members"], {r["uid"]: (k, r) for k, r in enumerate(ix["rooms"])}
    pts, rpts, ppts, layers = ix["member_points"], ix["room_points"], ix["public_points"], ix["layers"]
    out = {}
    for contrast in ("vote", "hidden"):
        rows = [i for i, m in enumerate(M) if not m.get("apply_only") and m["vote"] in (0, 1) and (contrast == "vote" or m.get("public") == "other option")]
        y = np.array([M[i]["vote"] for i in rows])
        dom, sets, scen = [M[i]["domain"] for i in rows], [M[i]["set"] for i in rows], [M[i]["scenario"] for i in rows]
        _, n0 = per_scenario(np.zeros(len(y)), y, sets, scen)
        scens = sorted(n0)
        rng = np.random.default_rng(SEED)
        C = np.stack([np.bincount(rng.integers(0, len(scens), len(scens)), minlength=len(scens)) for _ in range(BOOT)]).astype(float)
        rooms = [R[M[i]["room_uid"]] for i in rows]
        F_room = np.array([[float(r["code"] == c) for c in CODES] + [r["marks"]["stake"], r["marks"]["proposal"], r["marks"]["acceptance"]] for _, r in rooms], dtype=float)
        F_pub = np.array([[float(M[i].get("public") == c) for c in PUB] for i in rows], dtype=float)
        readers = {"room": F_room}
        if contrast == "vote":                     # in "hidden" every member has the same public position
            readers["room+pub"] = np.hstack([F_room, F_pub])
        res = {"units": [int(y.sum()), int((1 - y).sum())], "pairs": int(sum(n0.values())), "scenarios": len(scens), "positions": {}}
        fixed = {k: summarize([per_scenario(lodo(F, y, dom), y, sets, scen)], scens, C) for k, F in readers.items()}
        fixed["agree_target_only"] = summarize([per_scenario(F_room[:, 0], y, sets, scen)], scens, C)
        ridx = [k for k, _ in rooms]
        allb = collections.defaultdict(list)
        for point in ppts[:-1]:                    # not the last token of the vote
            p, q = pts.index(point), ppts.index(point)
            act, neu = [], []
            for L in layers:
                A = np.load(T / f"member_L{L}.npy", mmap_mode="r")
                NR = np.load(T / f"neutral_room_L{L}.npy", mmap_mode="r")
                NP = np.load(T / f"neutral_pub_L{L}.npy", mmap_mode="r")
                act.append(per_scenario(lodo(np.asarray(A[rows, p, :], dtype=np.float32), y, dom), y, sets, scen))
                Xn = np.hstack([np.asarray(NR[ridx, rpts.index("m4.end"), :], dtype=np.float32), np.asarray(NP[rows, q, :], dtype=np.float32)])
                neu.append(per_scenario(lodo(Xn, y, dom), y, sets, scen))
            a_pt, a_bs = summarize(act, scens, C)
            n_pt, n_bs = summarize(neu, scens, C)
            rec = {"activation": [a_pt, *iv(a_bs)], "neutral+": [n_pt, *iv(n_bs)], "activation_minus_neutral+": [a_pt - n_pt, *iv(a_bs - n_bs)]}
            allb["activation"].append(a_bs)
            allb["neutral+"].append(n_bs)
            for k, (f_pt, f_bs) in fixed.items():
                rec[k] = [f_pt, *iv(f_bs)]
                rec[f"activation_minus_{k}"] = [a_pt - f_pt, *iv(a_bs - f_bs)]
                allb[k].append(f_bs)
            rec["_pt"] = {"activation": a_pt, "neutral+": n_pt, **{k: v[0] for k, v in fixed.items()}}
            res["positions"][point] = rec
            log(f"{contrast} {point}: activation {a_pt:.3f}, " + ", ".join(f"{k} {v[0]:.3f}" for k, v in fixed.items()) + f", neutral+ {n_pt:.3f}")
        a_mean_bs = np.mean(allb["activation"], axis=0)
        a_mean_pt = float(np.mean([res["positions"][p]["_pt"]["activation"] for p in res["positions"]]))
        res["mean_over_positions"] = {"activation": [a_mean_pt, *iv(a_mean_bs)]}
        for k in [k for k in allb if k != "activation"]:
            k_pt = float(np.mean([res["positions"][p]["_pt"][k] for p in res["positions"]]))
            k_bs = np.mean(allb[k], axis=0)
            res["mean_over_positions"][k] = [k_pt, *iv(k_bs)]
            res["mean_over_positions"][f"activation_minus_{k}"] = [a_mean_pt - k_pt, *iv(a_mean_bs - k_bs)]
        out[contrast] = res
    return out


def write_md(res: dict, path: Path) -> None:
    f = lambda v: f"{v[0]:.3f} [{v[1]:.3f}, {v[2]:.3f}]"        # noqa: E731
    g = lambda v: f"{v[0]:+.3f} [{v[1]:+.3f}, {v[2]:+.3f}]"     # noqa: E731
    L = ["# Stage 2, exploratory: the probe against stronger readers of the words", "",
         "Added after the probe results were seen; it is not among the rules fixed in advance. Written by "
         "`scripts/uninstructed_stage2_readers.py`; every number is in `readers.json`.", "",
         "Every number is a matched-prompt AUROC (0.5 = chance) on the pairs of the sweep, with a 95% interval from 2,000 resamples of the "
         "scenarios; a difference is paired. *Activation*: the probe on the member's activation, mean of the five layers. The readers see words "
         "only: Claude Fable 5.2's codes were made without the votes, and a small logistic regression is fit on them on the other domains.", ""]
    for contrast, r in res.items():
        keys = [k for k in ("room", "room+pub", "agree_target_only", "neutral+") if k in r["mean_over_positions"]]
        L += [f"## {contrast}: {ABOUT[contrast]}", "", f"{r['units'][0]} against {r['units'][1]} members, {r['pairs']} pairs in {r['scenarios']} scenarios.", ""]
        L += [f"- {NAMES[k]}: {f(r['mean_over_positions'][k])}" for k in keys if k != "neutral+"]
        L += ["", "| Position | Activation | " + " | ".join(MINUS[k] for k in keys) + " | Neutral reader, joined |", "|---|---|" + "---|" * (len(keys) + 1)]
        for p, row in list(r["positions"].items()) + [("mean of the seven", r["mean_over_positions"])]:
            L.append(f"| {'`' + p + '`' if '.' in p else p} | {f(row['activation'])} | " + " | ".join(g(row[f'activation_minus_{k}']) for k in keys) + f" | {f(row['neutral+'])} |")
        L.append("")
    L += ["Reading: where a difference's interval includes zero, the probe is not told apart from that reader of the words. Claude's reading of "
          "the public position covers all three statements, so only at `v.pre` has it read no further than the activation.", ""]
    path.write_text("\n".join(L) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--acts", type=Path, default=ROOT / "results/uninstructed/acts/stage2")
    ap.add_argument("--out", type=Path, default=ROOT / "results/uninstructed/stage2/explore")
    ap.add_argument("--report-only", action="store_true", help="write READERS.md again from readers.json")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    if not a.report_only:
        res = compute(a.acts / "tables", lambda m: print(m, flush=True))
        (a.out / "readers.json").write_text(json.dumps(res, indent=1) + "\n")
    res = json.loads((a.out / "readers.json").read_text())
    write_md(res, a.out / "READERS.md")
    print(f"wrote {a.out / 'READERS.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
