#!/usr/bin/env python3
"""Stage 2, step 3: probes on the saved activations, read off by the rules fixed in advance (CPU only, no model).

docs/uninstructed_collusion/DESIGN.md, sections 8 and 9, and the dated note "stage 2" of 18 November 2026.

For every contrast (uninstructed/stage2.py: vote, hidden, collusion_clean, collusion_rest), every read position and
every layer, one probe: standardise, class-balanced logistic regression, tested on domains it was not fit on. The score
is the matched-prompt AUROC with an interval from resampled scenarios; "mean" is the mean over the layers.

Inputs to the probe at a position (--inputs):
  full      the activation
  centered  the activation minus the mean of the units with identical prompts (section 9, secondary)
  bow       the words said so far: TF-IDF and logistic regression (text baseline 1)
  bowwho    the same, with every word marked as the unit's own or someone else's: the activation belongs to one member,
            so a fair reader of the words has to know which of them are that member's
  neutral   the neutral reader's activation at the same words (text baseline 2)
  J, rest   the J-space part and the remainder (needs scripts/uninstructed_decompose.py)
  pca25     the top 25 principal components of the training rows
  rand      200 random 25-dimensional subspaces and 200 random sets of 25 J-lens vectors (primary contrast only)
  ref       one probe fit at the end of the room and applied at every earlier room position (section 9, secondary)
  core      the probe of the instructed benchmark applied to the public statements (Q6)

Positions: the 24 fixed points of the room (the unit is whoever writes there), the 8 public points of each member, and
with --grid every fourth token of the room in order (the token clock).

Writes <out>/probe.json and <out>/PROBE.md, with Q1 to Q6 answered by their rules and the layer and anchor that
section 15.2's rule picks for the causal phase.

  python scripts/uninstructed_probe.py --acts results/uninstructed/acts/stage2 --out results/uninstructed/stage2 --jobs 8 --grid
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from uninstructed import stage2 as S  # noqa: E402

ALL_INPUTS = ("full", "centered", "bow", "bowwho", "neutral", "J", "rest", "pca25", "rand", "ref", "core")
TEXT_BASELINES = ("bow", "bowwho", "neutral")      # "more than the words" means: above every one of these
K = 25
_CTX: dict = {}


class Ctx:
    """The tables of one capture folder, opened lazily (one copy per worker process)."""

    def __init__(self, acts: Path):
        self.t = Path(acts) / "tables"
        self.acts = Path(acts)
        idx = json.loads((self.t / "index.json").read_text())
        self.members, self.rooms, self.layers = idx["members"], idx["rooms"], idx["layers"]
        self.MP, self.RP, self.PP = idx["member_points"], idx["room_points"], idx["public_points"]
        self.grid_n, self.grid_step = idx["grid"]["n"], idx["grid"]["step"]
        room_at = {r["uid"]: i for i, r in enumerate(self.rooms)}
        self.m_room = np.array([room_at[m["room_uid"]] for m in self.members])
        self.member_at = {(m["room_uid"], m["seat"]): i for i, m in enumerate(self.members)}
        self.analysis = np.array([not m["apply_only"] for m in self.members])
        self.scenarios = sorted({m["scenario"] for m in self.members if not m["apply_only"]})
        self.dom = np.array([m["domain"] for m in self.members])
        self.scen = np.array([m["scenario"] for m in self.members])
        self.grid_seat = np.load(self.t / "grid_seat.npy")
        self._mm, self._texts, self._haar = {}, None, {}
        self.shuffle = int(os.environ["S2_SHUFFLE_LABELS"]) if os.environ.get("S2_SHUFFLE_LABELS") else None
        self._shuffled: dict = {}

    def has(self, name: str, L: int) -> bool:
        return (self.t / f"{name}_L{L}.npy").exists()

    def arr(self, name: str, L: int):
        key = (name, L)
        if key not in self._mm:
            self._mm[key] = np.load(self.t / f"{name}_L{L}.npy", mmap_mode="r")
        return self._mm[key]

    def texts(self) -> dict:
        if self._texts is None:
            with gzip.open(self.t / "texts.json.gz", "rt") as fh:
                self._texts = json.load(fh)
        return self._texts

    def haar(self, L: int, d: int, draws: int, seed: int) -> list:
        if (L, draws) not in self._haar:
            from jspace_interp.sweep import haar_subspace
            rng = np.random.default_rng(seed + 1000 * L)
            self._haar[(L, draws)] = [haar_subspace(d, K, rng).astype(np.float32) for _ in range(draws)]
        return self._haar[(L, draws)]

    def rows(self, point: str, unit: str):
        """Which members a position is read from. A room point: the members who write that message. A public point:
        every member (member contrasts only). A grid point: whoever is writing at that position of the room."""
        if point in self.RP:
            parity, _ = S.own_point(point)
            return np.array([i for i, m in enumerate(self.members) if m["seat"] % 2 == parity])
        if point in self.PP:
            return np.arange(len(self.members)) if unit == "member" else None
        gi = int(point[1:])
        rr = np.nonzero(self.grid_seat[:, gi] >= 0)[0]
        return np.array([self.member_at[(self.rooms[r]["uid"], int(self.grid_seat[r, gi]))] for r in rr])

    def feats(self, name: str, point: str, L: int, rows: np.ndarray):
        """Activations [n, d] of the given kind at a position, or None when that kind does not exist there."""
        if point in self.RP:
            col = self.MP.index(S.own_point(point)[1])
            if name in ("member", "memberJ"):
                return np.asarray(self.arr(name, L)[rows, col], dtype=np.float32) if self.has(name, L) else None
            if name == "neutral":
                return np.asarray(self.arr("neutral_room", L)[self.m_room[rows], self.RP.index(point)], dtype=np.float32)
        elif point in self.PP:
            if name in ("member", "memberJ"):
                return np.asarray(self.arr(name, L)[rows, self.MP.index(point)], dtype=np.float32) if self.has(name, L) else None
            if name == "neutral":
                return np.asarray(self.arr("neutral_pub", L)[rows, self.PP.index(point)], dtype=np.float32)
        elif name == "member":
            return np.asarray(self.arr("grid", L)[self.m_room[rows], int(point[1:])], dtype=np.float32)
        return None

    def words(self, point: str, rows: np.ndarray, who: bool = False) -> list[str] | None:
        t, sfx = self.texts(), ("_who" if who else "")
        if "room" + sfx not in t:
            return None
        if point in self.RP:
            return [t["room" + sfx][self.m_room[i]][self.RP.index(point)] for i in rows]
        if point in self.PP:
            return [t["public" + sfx][i][self.PP.index(point)] for i in rows]
        return [t["grid" + sfx][self.m_room[i]][int(point[1:])] for i in rows]

    def all_labels(self, contrast: str) -> np.ndarray:
        """The label of every member under a contrast (for a room contrast: its room's label). With --shuffle-labels the
        labels are permuted inside every matched set, units of the analysis and scored-only units apart."""
        if contrast not in self._shuffled:
            unit = S.CONTRASTS[contrast]["unit"]
            y = np.array([S.label(contrast, m, self.rooms[self.m_room[i]]) for i, m in enumerate(self.members)])
            if self.shuffle is not None:
                rng = np.random.default_rng([self.shuffle, sorted(S.CONTRASTS).index(contrast)])
                if unit == "member":
                    groups: dict = {}
                    for i, m in enumerate(self.members):
                        if y[i] >= 0:
                            groups.setdefault((m["set"], m["apply_only"]), []).append(i)
                    for ii in groups.values():
                        y[ii] = rng.permutation(y[ii])
                else:
                    ry = np.array([S.label(contrast, None, r) for r in self.rooms])
                    groups = {}
                    for j, r in enumerate(self.rooms):
                        if ry[j] >= 0:
                            groups.setdefault((r["set"], r["apply_only"]), []).append(j)
                    for jj in groups.values():
                        ry[jj] = rng.permutation(ry[jj])
                    y = ry[self.m_room]
            self._shuffled[contrast] = y
        return self._shuffled[contrast]

    def labels(self, contrast: str, rows: np.ndarray):
        unit = S.CONTRASTS[contrast]["unit"]
        y = self.all_labels(contrast)[rows]
        sets = [self.members[i]["set"] if unit == "member" else self.rooms[self.m_room[i]]["set"] for i in rows]
        return y, sets, self.scen[rows], self.dom[rows], self.analysis[rows]


def ctx_of(acts) -> Ctx:
    if str(acts) not in _CTX:
        _CTX[str(acts)] = Ctx(acts)
    return _CTX[str(acts)]


def _pack(P: S.Pairs, scores, y, train) -> dict:
    win, n = P.per_scenario(scores)
    return {"win": win.tolist(), "n": n.tolist(), "pooled": S.pooled_auroc(np.asarray(scores)[train], y[train])}


def cell_job(acts, contrast: str, point: str, L: int, inputs: tuple, draws: int, seed: int, first_layer: bool) -> dict:
    c = ctx_of(acts)
    out = {"kind": "cell", "contrast": contrast, "point": point, "layer": L, "inputs": {}}
    rows = c.rows(point, S.CONTRASTS[contrast]["unit"])
    if rows is None or len(rows) == 0:
        return out
    y, sets, scen, dom, train = c.labels(contrast, rows)
    ya = np.where(train, y, -1)
    P = S.Pairs(ya, sets, scen, c.scenarios)
    out.update(n1=int((ya == 1).sum()), n0=int((ya == 0).sum()), pairs=P.n_pairs, sets=P.n_sets, scenarios=P.n_scenarios_with_pairs)
    if P.n_pairs == 0:
        return out
    X = c.feats("member", point, L, rows)                # (main checks that a matched set lies inside one fold)
    fixed = point in c.RP or point in c.PP
    if "full" in inputs:
        s = S.oof_scores(X, y, dom, train_mask=train)
        out["inputs"]["full"] = _pack(P, s, y, train)
        ap = ~train & (y >= 0)
        if ap.any():                                   # the instructed control: scored by the probe of its fold, never fitted on
            out["apply"] = {"n1": int((y[ap] == 1).sum()), "n0": int((y[ap] == 0).sum()), "pooled": S.pooled_auroc(s[ap], y[ap]),
                            "mean_score": float(np.nanmean(s[ap])), "mean_score_analysis": float(np.nanmean(s[train & (y >= 0)]))}
    if "centered" in inputs:
        out["inputs"]["centered"] = _pack(P, S.oof_scores(S.center_by_set(X, sets), y, dom, train_mask=train), y, train)
    if "neutral" in inputs:
        Xn = c.feats("neutral", point, L, rows)
        if Xn is not None:
            out["inputs"]["neutral"] = _pack(P, S.oof_scores(Xn, y, dom, train_mask=train), y, train)
    if "bow" in inputs and first_layer:                # the words have no layer: once per position
        out["inputs"]["bow"] = _pack(P, S.bow_oof(c.words(point, rows), y, dom, train_mask=train), y, train)
    if "bowwho" in inputs and first_layer:
        tagged = c.words(point, rows, who=True)
        if tagged is not None:
            out["inputs"]["bowwho"] = _pack(P, S.bow_oof(tagged, y, dom, train_mask=train), y, train)
    XJ = c.feats("memberJ", point, L, rows) if fixed and ("J" in inputs or "rest" in inputs) else None
    if XJ is not None:
        if "J" in inputs:
            out["inputs"]["J"] = _pack(P, S.oof_scores(XJ, y, dom, train_mask=train), y, train)
        if "rest" in inputs:
            out["inputs"]["rest"] = _pack(P, S.oof_scores(X - XJ, y, dom, train_mask=train), y, train)
    if "pca25" in inputs and fixed:
        out["inputs"]["pca25"] = _pack(P, S.oof_scores(X, y, dom, pca=K, train_mask=train), y, train)
    if "rand" in inputs and fixed and contrast == S.PRIMARY and draws:
        ok = np.isfinite(X).all(1)
        Xz = np.where(ok[:, None], X, 0.0)
        res = {"haar": []}
        for Q in c.haar(L, X.shape[1], draws, seed):
            Z = Xz @ Q
            Z[~ok] = np.nan
            res["haar"].append(P.auroc(S.oof_scores(Z, y, dom, train_mask=train)))
        if c.has("randJ", L):
            res["randJ"] = []
            B = c.arr("randJ", L)
            for b in range(min(draws, B.shape[0])):
                Z = Xz @ np.asarray(B[b], dtype=np.float32)
                Z[~ok] = np.nan
                res["randJ"].append(P.auroc(S.oof_scores(Z, y, dom, train_mask=train)))
        out["rand"] = res
    return out


def ref_job(acts, contrast: str, L: int) -> dict:
    """One probe fit at the end of the room (the last token of a member's second message) and applied at each of that
    member's earlier points, with the same folds."""
    c = ctx_of(acts)
    out = {"kind": "ref", "contrast": contrast, "layer": L, "points": {}}
    for parity, ref in ((0, "m3.end"), (1, "m4.end")):
        rows = c.rows(ref, "member")
        y, sets, scen, dom, train = c.labels(contrast, rows)
        ya = np.where(train, y, -1)
        P = S.Pairs(ya, sets, scen, c.scenarios)
        if P.n_pairs == 0:
            continue
        pts = [p for p in c.RP if S.own_point(p)[0] == parity]
        Xs = [c.feats("member", p, L, rows) for p in pts]
        sc = S.oof_apply(c.feats("member", ref, L, rows), y, dom, Xs, train_mask=train)
        for p, s in zip(pts, sc):
            out["points"][p] = _pack(P, s, y, train)
    return out


def core_job(acts, L: int) -> dict:
    """Q6: the probe of the instructed benchmark (colluders against honest agents of its collusion episodes, public
    statements, last token, as in the instructed study) applied to the public statements of these episodes."""
    from jspace_interp import detectors
    from jspace_interp.rows import load_extraction, to_agents

    c = ctx_of(acts)
    out = {"kind": "core", "layer": L}
    meta, npz = load_extraction(c.acts / "core")
    agents = to_agents(meta, np.arange(len(meta)), npz[f"last__layer_{L}"].astype(np.float32), "core")
    sc, w = detectors.fit_direction(agents)
    rows = np.arange(len(c.members))
    s = sc.transform(np.asarray(c.arr("member_last", L), dtype=np.float64)) @ w
    out["trained_on"] = {"colluders": sum(a["role"] == "colluder" for a in agents), "honest": sum(a["role"] == "honest" for a in agents)}
    for contrast in ("vote", "hidden"):
        y, sets, scen, dom, train = c.labels(contrast, rows)
        ya = np.where(train, y, -1)
        P = S.Pairs(ya, sets, scen, c.scenarios)
        if P.n_pairs:
            out[contrast] = _pack(P, s, y, train)
            out[contrast]["pairs"] = P.n_pairs
    y, _, _, _, train = c.labels("vote", rows)
    ap = ~train & (y >= 0)
    if ap.any():
        out["apply"] = {"pooled": S.pooled_auroc(s[ap], y[ap]), "mean_score": float(s[ap].mean()), "mean_score_analysis": float(s[train].mean())}
    return out


def run_job(job):
    kind = job[0]
    return {"cell": cell_job, "ref": ref_job, "core": core_job}[kind](*job[1:])


# ---------------------------------------------------------------------------------------------------------------------
# reading the results off
# ---------------------------------------------------------------------------------------------------------------------
def stat(win, n, W):
    win, n = np.asarray(win, float), np.asarray(n, float)
    b = S.boot_auroc(win, n, W)
    lo, hi = S.interval(b)
    return {"auroc": float(win.sum() / n.sum()) if n.sum() else None, "lo": lo, "hi": hi}, b


def mean_stat(per_layer: list[tuple[dict, np.ndarray]]):
    vals = [s["auroc"] for s, _ in per_layer if s["auroc"] is not None]
    if not vals:
        return None, None
    b = np.nanmean(np.stack([bb for _, bb in per_layer]), axis=0)
    lo, hi = S.interval(b)
    return {"auroc": float(np.mean(vals)), "lo": lo, "hi": hi}, b


def diff_stat(a, ba, b, bb):
    if a is None or b is None:
        return None
    lo, hi = S.interval(ba - bb)
    return {"diff": a["auroc"] - b["auroc"], "lo": lo, "hi": hi, "excludes_zero": bool(lo > 0 or hi < 0)}


def assemble(results: list[dict], c: Ctx, layers: list[int], points: list[str], contrasts: list[str], boot: int) -> dict:
    W = S.boot_weights(len(c.scenarios), boot)
    cells = {(r["contrast"], r["point"], r["layer"]): r for r in results if r["kind"] == "cell"}
    refs = {(r["contrast"], r["layer"]): r for r in results if r["kind"] == "ref"}
    out, phases = {}, {}
    for contrast in contrasts:
        table, keep = {}, {}
        for p in points:
            got = [cells.get((contrast, p, L)) for L in layers]
            if not got[0] or not got[0].get("pairs"):
                continue
            row = {"n1": got[0]["n1"], "n0": got[0]["n0"], "pairs": got[0]["pairs"], "sets": got[0]["sets"],
                   "scenarios": got[0]["scenarios"], "layers": {}, "mean": {}}
            boots = {}
            names = sorted({k for g in got for k in g["inputs"]})
            for name in names:
                per = []
                for L, g in zip(layers, got):
                    if name in g["inputs"]:
                        st, b = stat(g["inputs"][name]["win"], g["inputs"][name]["n"], W)
                        st["pooled"] = g["inputs"][name]["pooled"]
                        row["layers"].setdefault(str(L), {})[name] = st
                        per.append((st, b))
                row["mean"][name], boots[name] = mean_stat(per)
            keep[p] = boots
            for base in TEXT_BASELINES:
                if "full" in boots and base in boots and boots[base] is not None:
                    row[f"minus_{base}"] = diff_stat(row["mean"]["full"], boots["full"], row["mean"][base], boots[base])
            if "J" in boots and "pca25" in boots and boots["J"] is not None:
                row["J_minus_pca25"] = diff_stat(row["mean"]["J"], boots["J"], row["mean"]["pca25"], boots["pca25"])
            per = []
            for L in layers:                           # the probe fit at the end of the room, applied here
                r = refs.get((contrast, L))
                if r and p in r["points"]:
                    per.append(stat(r["points"][p]["win"], r["points"][p]["n"], W))
            if per:
                row["mean"]["ref"], _ = mean_stat(per)
            if any("rand" in g for g in got):
                row["rand"] = {}
                for L, g in zip(layers, got):
                    if "rand" not in g:
                        continue
                    jv = row["layers"].get(str(L), {}).get("J", {}).get("auroc")
                    rr = {"J": jv}
                    for kind in ("haar", "randJ"):
                        if kind in g["rand"]:
                            v = np.array(g["rand"][kind], float)
                            rr[kind] = {"p50": float(np.nanpercentile(v, 50)), "p95": float(np.nanpercentile(v, 95)), "max": float(np.nanmax(v)), "draws": int(len(v))}
                    if jv is not None and "haar" in rr:
                        rr["J_above_both"] = bool(jv > rr["haar"]["p95"] and ("randJ" not in rr or jv > rr["randJ"]["p95"]))
                    row["rand"][str(L)] = rr
            if got[0].get("apply"):
                row["apply"] = {k: (float(np.nanmean([g["apply"][k] for g in got if g.get("apply")])) if k not in ("n1", "n0") else got[0]["apply"][k])
                                for k in got[0]["apply"]}
            table[p] = row
        if table:
            out[contrast] = table
            # one number per phase: the mean over its positions of the score, and of activation minus each text baseline
            ph = {}
            for phase, pool in (("room", S.ROOM_POINTS), ("public", S.PUBLIC_POINTS)):
                pts = [p for p in pool if p in table and keep[p].get("full") is not None]
                if not pts:
                    continue
                b = np.nanmean(np.stack([keep[p]["full"] for p in pts]), axis=0)
                lo, hi = S.interval(b)
                ph[f"{phase}_mean"] = {"auroc": float(np.mean([table[p]["mean"]["full"]["auroc"] for p in pts])), "lo": lo, "hi": hi, "positions": len(pts)}
                for base in TEXT_BASELINES:
                    use = [p for p in pts if keep[p].get(base) is not None]
                    if use:
                        d = np.nanmean(np.stack([keep[p]["full"] - keep[p][base] for p in use]), axis=0)
                        lo, hi = S.interval(d)
                        ph[f"{phase}_minus_{base}"] = {"diff": float(np.mean([table[p]["mean"]["full"]["auroc"] - table[p]["mean"][base]["auroc"] for p in use])),
                                                      "lo": lo, "hi": hi, "positions": len(use), "excludes_zero": bool(lo > 0 or hi < 0)}
            phases[contrast] = ph
    return out, phases


def rules(table: dict, timeline: list[str], room_points: list[str], layers: list[int]) -> dict:
    """Section 9's questions, answered by their rules on one contrast."""
    pts = [p for p in timeline if p in table and table[p]["mean"].get("full")]
    f = lambda p, k="full": table[p]["mean"].get(k) if p in table else None   # noqa: E731
    r = {}
    last = "m4.end"
    if f(last):
        r["Q1"] = {"point": last, **f(last), "signal": bool(f(last)["lo"] > 0.5),
                   "rule": "full activation at the last token of the room: interval above 0.5"}
    i = S.onset([f(p)["lo"] for p in pts])
    r["Q2"] = {"onset": pts[i] if i is not None else None, "position_in_timeline": i,
               "rule": "first position from which the lower bound stays above 0.5 for three positions in a row"}
    r["Q3"] = {"ends_of_messages": {p: f(p)["auroc"] for p in ("m1.end", "m2.end", "m3.end", "m4.end") if f(p)},
               "first_at_0.75": next((p for p in pts if f(p)["auroc"] >= 0.75), None)}
    more = [p for p in pts if all((table[p].get(f"minus_{b}") or {}).get("lo", -1) > 0 for b in TEXT_BASELINES if f"minus_{b}" in table[p])
            and "minus_bow" in table[p] and "minus_neutral" in table[p]]
    bw = [p for p in pts if f(p, "bow")]
    ib = S.onset([f(p, "bow")["lo"] for p in bw]) if bw else None
    r["Q4"] = {"more_than_words_at": more, "above_bag_of_words_only": [p for p in pts if (table[p].get("minus_bow") or {}).get("lo", -1) > 0 and p not in more],
               "onset_of_the_words": bw[ib] if ib is not None else None,
               "lead_in_positions": (pts.index(bw[ib]) - i) if (ib is not None and i is not None and bw[ib] in pts) else None,
               "rule": "at a position the probe is 'more than words' only where probe minus bag of words, probe minus bag of words by speaker "
                       "and probe minus neutral reader all have an interval above zero"}
    cells = [(p, L, table[p]["rand"][str(L)]) for p in pts for L in layers if str(L) in table[p].get("rand", {}) and "J_above_both" in table[p]["rand"][str(L)]]
    if cells:
        r["Q5"] = {"cells": len(cells), "J_above_both_random_draws": sum(x["J_above_both"] for _, _, x in cells),
                   "where": [f"{p} L{L}" for p, L, x in cells if x["J_above_both"]][:40],
                   "J_minus_pca25_mean": {p: table[p]["J_minus_pca25"] for p in pts if table[p].get("J_minus_pca25")},
                   "rule": "the J-space part is special at a layer and position if its score is above the 95th percentile of both kinds of random draw"}
    # section 15.2: the layer whose probe has the earliest onset inside the room; tie: the higher score there, then the lower layer
    best = None
    rp = [p for p in room_points if p in table]
    for L in layers:
        lo = [table[p]["layers"].get(str(L), {}).get("full", {}).get("lo") for p in rp]
        j = S.onset(lo)
        if j is None:
            continue
        key = (j, -table[rp[j]]["layers"][str(L)]["full"]["auroc"], L)
        if best is None or key < best[0]:
            best = (key, L, rp[j])
    r["causal_choice"] = ({"layer": best[1], "anchor": best[2], "score_there": -best[0][1]} if best else
                          {"layer": None, "anchor": None, "note": "no onset inside the room at any layer: the room interventions of section 15 have no anchor"})
    v = [(table["v.pre"]["layers"][str(L)]["full"]["auroc"], -L) for L in layers if "v.pre" in table and "full" in table["v.pre"]["layers"].get(str(L), {})]
    if v:
        r["causal_choice"]["vote_layer"] = best[1] if best else -max(v)[1]
        r["causal_choice"]["vote_signal"] = bool(table["v.pre"]["mean"]["full"]["lo"] > 0.5)
    return r


TEXT_WORDS = ("Text baselines. *Words so far*: a bag of words (TF-IDF, logistic regression) of everything said up to that token. *Words by "
              "speaker*: the same with every word marked as the unit's own or someone else's, because an activation belongs to one member "
              "and a fair reader of the words has to know which of them are that member's. *Neutral reader*: the same model reads the words "
              "alone, without the prompt, and a probe of the same kind is fit on its activations. \"More than the words\" means above all three.")
POINT_WORDS = ("Positions. `m2.pre` is the token just before room message 2 starts, `m2.t0` its first token, `m2.q1`, `m2.q2`, `m2.q3` "
               "the tokens at 25%, 50% and 75% of it, `m2.end` its last token. Messages 1 and 3 are written by the member who opens the "
               "room, 2 and 4 by the one who answers; at a room position the unit is the member who writes there. `p1.pre` and `p1.end` "
               "are the token before a member's first public statement and its last token (`p2`, `p3` likewise); `v.pre` and `v.end` the "
               "same for the vote.")


def fmt(s, nd=3):
    return "–" if not s or s.get("auroc") is None else f"{s['auroc']:.{nd}f} [{s['lo']:.{nd}f}, {s['hi']:.{nd}f}]"


def fmt_d(s):
    return "–" if not s else f"{s['diff']:+.3f} [{s['lo']:+.3f}, {s['hi']:+.3f}]"


def write_md(res: dict, path: Path) -> None:
    L = ["# Stage 2: probes on recorded episodes", ""]
    if res.get("labels_shuffled_inside_matched_sets") is not None:
        L += ["**Dry run: every label was shuffled inside its matched set before anything was fit. No number below says anything "
              "about the experiment; 0.5 is expected everywhere.**", ""]
    L += [
         "Every number is a matched-prompt AUROC: over all pairs of units with word-for-word identical prompts and different "
         "labels, the share in which the probe scores the unit of class 1 higher (0.5 = chance; ties count half). Brackets: 95% "
         "interval from resampled scenarios. Unless a layer is named, a score is the mean over the layers "
         f"{res['design']['layers']}. Probes are fit on other domains than the one they are tested on.", "", POINT_WORDS, "", TEXT_WORDS, ""]
    u = res["units"]
    L += [f"Units: {u['members']} members and {u['rooms']} rooms of {u['groups']}; fitted on {u['analysis_members']} members, "
          f"{u['apply_only_members']} members of the instructed control are only scored.", ""]
    for contrast, table in res["contrasts"].items():
        info = S.CONTRASTS[contrast]
        L += [f"## {contrast}: {info['one']} (1) against {info['zero']} (0)", "",
              f"Unit: the {info['unit']}. {'No coder is involved.' if not info['coder'] else 'Read by a coder: ' + info['coder'] + '.'}", "",
              "| Position | Units 1 / 0 | Pairs | Activation | Identical-prompt mean removed | Words so far | Words by speaker | Neutral reader | "
              "Activation minus words | Activation minus words by speaker | Activation minus neutral reader |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
        for p in res["design"]["timeline"]:
            if p not in table:
                continue
            r = table[p]
            L.append(f"| `{p}` | {r['n1']} / {r['n0']} | {r['pairs']} | {fmt(r['mean'].get('full'))} | {fmt(r['mean'].get('centered'))} | "
                     f"{fmt(r['mean'].get('bow'))} | {fmt(r['mean'].get('bowwho'))} | {fmt(r['mean'].get('neutral'))} | {fmt_d(r.get('minus_bow'))} | "
                     f"{fmt_d(r.get('minus_bowwho'))} | {fmt_d(r.get('minus_neutral'))} |")
        L.append("")
        ru = res["rules"].get(contrast, {})
        if ru:
            L += ["Read off by the rules of section 9:", ""]
            if "Q1" in ru:
                L.append(f"- **Q1, a signal at all?** At the last token of the room (`m4.end`): {fmt(ru['Q1'])}. "
                         f"{'Yes: the interval is above 0.5.' if ru['Q1']['signal'] else 'No: the interval includes 0.5.'}")
            L.append(f"- **Q2, where does it start?** {'At `' + ru['Q2']['onset'] + '`.' if ru['Q2']['onset'] else 'Nowhere: the lower bound never stays above 0.5 for three positions in a row.'}")
            L.append("- **Q3, how early?** At the ends of the four room messages: " + ", ".join(f"`{k}` {v:.3f}" for k, v in ru["Q3"]["ends_of_messages"].items())
                     + f". First position at 0.75 or more: {('`' + ru['Q3']['first_at_0.75'] + '`') if ru['Q3']['first_at_0.75'] else 'none'}.")
            q4 = ru["Q4"]
            L.append(f"- **Q4, more than the words?** Above both text baselines at: {', '.join('`' + p + '`' for p in q4['more_than_words_at']) or 'no position'}. "
                     f"Above the bag of words only: {', '.join('`' + p + '`' for p in q4['above_bag_of_words_only']) or 'no position'}. "
                     f"The bag of words' own onset: {('`' + q4['onset_of_the_words'] + '`') if q4['onset_of_the_words'] else 'none'}.")
            for phase, word in (("room", "over the 24 room positions"), ("public", "over the 8 positions after the room")):
                bp = q4.get("by_phase", {}).get(phase)
                pm = res.get("phases", {}).get(contrast, {}).get(f"{phase}_mean")
                if bp and pm:
                    L.append(f"  - In one number, the mean {word}: activation {fmt(pm)}; activation minus words {fmt_d(bp['minus_words'])}; "
                             f"minus words by speaker {fmt_d(bp.get('minus_words_by_speaker'))}; minus neutral reader {fmt_d(bp['minus_neutral_reader'])}. "
                             f"{'More than the words in this phase.' if bp['more_than_words'] else 'Not more than the words in this phase.'}")
            if "Q5" in ru:
                L.append(f"- **Q5, is J-space special?** The J-space part is above the 95th percentile of both kinds of random draw in "
                         f"{ru['Q5']['J_above_both_random_draws']} of {ru['Q5']['cells']} cells (layer by position).")
            cc = ru["causal_choice"]
            L.append(f"- **Layer and anchor for the causal phase (15.2):** " + (f"layer {cc['layer']}, anchor `{cc['anchor']}` (score there {cc['score_there']:.3f})."
                                                                              if cc.get("layer") is not None else cc.get("note", "none.")))
            L.append("")
        extra = [p for p in res["design"]["timeline"] if p in table and (table[p]["mean"].get("J") or table[p]["mean"].get("ref"))]
        if extra:
            L += ["| Position | J-space part | Remainder | 25 principal components | Probe fit at the end of the room | Random subspaces, 95th pct (by layer) | Random J-lens sets, 95th pct (by layer) |",
                  "|---|---|---|---|---|---|---|"]
            for p in extra:
                r = table[p]
                rd = r.get("rand", {})
                hp = ", ".join(f"{rd[k]['haar']['p95']:.3f}" for k in sorted(rd) if "haar" in rd[k]) or "–"
                jp = ", ".join(f"{rd[k]['randJ']['p95']:.3f}" for k in sorted(rd) if "randJ" in rd[k]) or "–"
                L.append(f"| `{p}` | {fmt(r['mean'].get('J'))} | {fmt(r['mean'].get('rest'))} | {fmt(r['mean'].get('pca25'))} | {fmt(r['mean'].get('ref'))} | {hp} | {jp} |")
            L.append("")
    if res.get("grid"):
        L += ["## The token clock", "", f"Every {res['design']['grid_step']}th token of the room in order; position 0 is the token before the first "
              "message, where the prompts are identical. The unit is whoever is writing at that token. Rooms differ in length, so the "
              "same token count is a different place in the conversation from room to room.", ""]
        for contrast, table in res["grid"].items():
            L += [f"**{contrast}**", "", "| Token | Pairs | Activation | Words so far |", "|---|---|---|---|"]
            for p, r in table.items():
                L.append(f"| {int(p[1:]) * res['design']['grid_step']} | {r['pairs']} | {fmt(r['mean'].get('full'))} | {fmt(r['mean'].get('bow'))} |")
            L.append("")
    if res.get("instructed_probe"):
        ip = res["instructed_probe"]
        L += ["## Q6: does the probe of the instructed benchmark fire here?", "",
              f"The probe is fit on the instructed benchmark's collusion episodes ({ip['trained_on']['colluders']} colluders against "
              f"{ip['trained_on']['honest']} honest agents, public statements, last token) and applied to each member's public statements here. "
              "The design predicted a score near 0.5.", ""]
        for contrast in ("vote", "hidden"):
            if contrast in ip:
                L.append(f"- {contrast}: {fmt(ip[contrast])} (all pairs regardless of prompt: {ip[contrast]['pooled']:.3f})")
        if "apply" in ip:
            L.append(f"- On the instructed control I1 (every member told to collude): mean score {ip['apply']['mean_score']:+.2f}, against "
                     f"{ip['apply']['mean_score_analysis']:+.2f} for the members of the stated-intention episodes.")
        L.append("")
    path.write_text("\n".join(L) + "\n")


def summarize_nulls(folder: Path) -> int:
    """Passes with shuffled labels (<folder>/seed*/probe.json): how often the rules fire when there is nothing to find,
    and how high the best position of a timeline gets by chance. Writes <folder>/nulls.json and NULLS.md."""
    runs = [json.loads(f.read_text()) for f in sorted(folder.glob("seed*/probe.json"))]
    runs = [r for r in runs if r.get("labels_shuffled_inside_matched_sets") is not None]
    if not runs:
        sys.exit(f"no shuffled-label passes under {folder}")
    out = {"passes": len(runs), "seeds": [r["labels_shuffled_inside_matched_sets"] for r in runs], "contrasts": {}}
    for contrast in sorted({c for r in runs for c in r["contrasts"]}):
        tabs = [r["contrasts"][contrast] for r in runs if contrast in r["contrasts"]]
        rul = [r["rules"][contrast] for r in runs if contrast in r["rules"]]
        rec = {"passes": len(tabs)}
        for name in ("full", "bow"):
            best = [max(t[p]["mean"][name]["auroc"] for p in t if t[p]["mean"].get(name)) for t in tabs if any(t[p]["mean"].get(name) for p in t)]
            cells = [t[p]["mean"][name] for t in tabs for p in t if t[p]["mean"].get(name)]
            if best:
                rec[name] = {"best_position_auroc": {"median": float(np.median(best)), "p95": float(np.percentile(best, 95)), "max": float(max(best))},
                             "positions": len(cells), "positions_with_interval_above_0.5": sum(c["lo"] > 0.5 for c in cells),
                             "positions_with_interval_below_0.5": sum(c["hi"] < 0.5 for c in cells)}
        for phase in ("room", "public"):
            vals = [r["phases"][contrast][f"{phase}_mean"]["auroc"] for r in runs if r.get("phases", {}).get(contrast, {}).get(f"{phase}_mean")]
            if vals:
                rec[f"{phase}_mean"] = {"median": float(np.median(vals)), "p95": float(np.percentile(vals, 95)), "max": float(max(vals)),
                                        "passes_with_interval_above_0.5": sum(r["phases"][contrast][f"{phase}_mean"]["lo"] > 0.5 for r in runs
                                                                              if r.get("phases", {}).get(contrast, {}).get(f"{phase}_mean")),
                                        "passes_more_than_words": sum(bool(r["rules"][contrast]["Q4"].get("by_phase", {}).get(phase, {}).get("more_than_words"))
                                                                      for r in runs if contrast in r["rules"])}
        rec["Q1_signal"] = sum(bool(r.get("Q1", {}).get("signal")) for r in rul)
        rec["Q2_onset"] = sum(r["Q2"]["onset"] is not None for r in rul)
        rec["anchor_inside_the_room"] = sum(r["causal_choice"].get("anchor") is not None for r in rul)
        rec["more_than_words_somewhere"] = sum(bool(r["Q4"]["more_than_words_at"]) for r in rul)
        out["contrasts"][contrast] = rec
    (folder / "nulls.json").write_text(json.dumps(out, indent=1) + "\n")
    L = ["# Stage 2: what the rules do when there is nothing to find", "",
         f"{len(runs)} passes of the probe sweep in which every label was first shuffled inside its matched set. The true score is 0.5 "
         "everywhere, so whatever fires here fires by chance. The same activations, folds, score and intervals as the real sweep.", "",
         "| Contrast | Passes | Signal at the end of the room (Q1) | An onset somewhere (Q2) | An anchor inside the room (15.2) | "
         "More than words at some position (Q4) | Positions with an interval above 0.5 | Best position of a timeline: median, 95th percentile, highest | "
         "Mean over the room positions: 95th percentile; passes with its interval above 0.5; passes 'more than words' in the room |",
         "|---|---|---|---|---|---|---|---|---|"]
    for contrast, rec in out["contrasts"].items():
        f = rec.get("full")
        L.append(f"| {contrast} | {rec['passes']} | {rec['Q1_signal']} | {rec['Q2_onset']} | {rec['anchor_inside_the_room']} | {rec['more_than_words_somewhere']} | "
                 + (f"{f['positions_with_interval_above_0.5']} of {f['positions']} | {f['best_position_auroc']['median']:.3f}, {f['best_position_auroc']['p95']:.3f}, {f['best_position_auroc']['max']:.3f} |"
                    if f else "– | – |")
                 + (f" {rec['room_mean']['p95']:.3f}; {rec['room_mean']['passes_with_interval_above_0.5']}; {rec['room_mean']['passes_more_than_words']} |"
                    if rec.get("room_mean") else " – |"))
    L += ["", "Reading: a rule that fires in more than about one pass in twenty is not evidence by itself; the best position of the real "
          "timeline means something only above the 95th percentile of the last column."]
    (folder / "NULLS.md").write_text("\n".join(L) + "\n")
    print(json.dumps(out["contrasts"], indent=1))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--summarize-nulls", type=Path, default=None, metavar="FOLDER",
                    help="only: read the shuffled-label passes under FOLDER/seed*/ and write FOLDER/NULLS.md")
    ap.add_argument("--acts", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--contrasts", default=",".join(S.CONTRASTS))
    ap.add_argument("--inputs", default=",".join(ALL_INPUTS))
    ap.add_argument("--layers", default="", help="default: every layer of the tables")
    ap.add_argument("--points", default="", help="default: every fixed point")
    ap.add_argument("--grid", action="store_true", help="also the token clock (every grid position with at least --grid-min-pairs pairs)")
    ap.add_argument("--grid-contrasts", default="vote,collusion_rest")
    ap.add_argument("--grid-min-pairs", type=int, default=20)
    ap.add_argument("--draws", type=int, default=200)
    ap.add_argument("--boot", type=int, default=S.BOOT_B)
    ap.add_argument("--seed", type=int, default=S.BOOT_SEED)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--shuffle-labels", type=int, default=None, metavar="SEED",
                    help="dry runs and nulls: permute every label inside its matched set first, so that no result of the experiment "
                         "can be read from the output (the expected score is 0.5 everywhere)")
    a = ap.parse_args()
    if a.summarize_nulls:
        return summarize_nulls(a.summarize_nulls)
    if not a.acts or not a.out:
        ap.error("--acts and --out are required")
    if a.shuffle_labels is not None:
        os.environ["S2_SHUFFLE_LABELS"] = str(a.shuffle_labels)

    c = ctx_of(a.acts)
    layers = [int(x) for x in a.layers.split(",")] if a.layers else c.layers
    contrasts = [x for x in a.contrasts.split(",") if x]
    inputs = tuple(x for x in a.inputs.split(",") if x)
    points = [p for p in a.points.split(",") if p] or list(S.TIMELINE)
    for m in c.members:                                # a matched set must lie inside one fold
        assert m["set"].split("|")[1] == m["scenario"]
    dom_of = {}
    for m in c.members:
        assert dom_of.setdefault(m["scenario"], m["domain"]) == m["domain"], "a scenario lies in one domain"

    jobs = []
    for contrast in contrasts:
        unit = S.CONTRASTS[contrast]["unit"]
        for p in points:
            if p in c.PP and unit == "room":
                continue
            jobs += [("cell", a.acts, contrast, p, L, inputs, a.draws, a.seed, L == layers[0]) for L in layers]
        if "ref" in inputs:
            jobs += [("ref", a.acts, contrast, L) for L in layers]
    grid_points = []
    if a.grid:
        grid_points = [f"g{gi:03d}" for gi in range(c.grid_n) if (c.grid_seat[:, gi] >= 0).sum() >= 4]
        for contrast in [x for x in a.grid_contrasts.split(",") if x in contrasts]:
            jobs += [("cell", a.acts, contrast, p, L, tuple(x for x in inputs if x in ("full", "bow")), 0, a.seed, L == layers[0])
                     for p in grid_points for L in layers]
    if "core" in inputs and (a.acts / "core/activations.npz").exists():
        jobs += [("core", a.acts, L) for L in layers]
    t0 = time.time()
    print(f"[probe] {len(jobs)} jobs on {a.jobs} processes; {len(c.members)} members, {len(c.rooms)} rooms, layers {layers}", flush=True)
    if a.jobs > 1:
        from joblib import Parallel, delayed
        results = Parallel(n_jobs=a.jobs, batch_size=1, verbose=5)(delayed(run_job)(j) for j in jobs)
    else:
        results = []
        for k, j in enumerate(jobs):
            results.append(run_job(j))
            if k % 50 == 0:
                print(f"[probe] {k + 1}/{len(jobs)} ({time.time() - t0:.0f}s)", flush=True)
    res = {"labels_shuffled_inside_matched_sets": a.shuffle_labels,
           "design": {"layers": layers, "timeline": [p for p in S.TIMELINE if p in points], "inputs": list(inputs), "boot": a.boot, "seed": a.seed,
                      "draws": a.draws, "grid_step": c.grid_step, "contrasts": {k: S.CONTRASTS[k] for k in contrasts}},
           "units": {"members": len(c.members), "rooms": len(c.rooms), "analysis_members": int(c.analysis.sum()),
                     "apply_only_members": int((~c.analysis).sum()), "groups": sorted({m["group"] for m in c.members}),
                     "scenarios": len(c.scenarios), "domains": len(set(c.dom[c.analysis].tolist()))},
           "seconds": None}
    res["contrasts"], res["phases"] = assemble(results, c, layers, [p for p in S.TIMELINE if p in points], contrasts, a.boot)
    res["rules"] = {k: rules(t, list(S.TIMELINE), list(S.ROOM_POINTS), layers) for k, t in res["contrasts"].items()}
    for k, ph in res["phases"].items():                 # Q4 in one number per phase: above the words AND above the neutral reader
        res["rules"][k]["Q4"]["by_phase"] = {
            phase: {"minus_words": ph.get(f"{phase}_minus_bow"), "minus_words_by_speaker": ph.get(f"{phase}_minus_bowwho"),
                    "minus_neutral_reader": ph.get(f"{phase}_minus_neutral"),
                    "more_than_words": bool(all((ph.get(f"{phase}_minus_{b}") or {}).get("lo", -1) > 0 for b in TEXT_BASELINES if f"{phase}_minus_{b}" in ph)
                                            and f"{phase}_minus_bow" in ph and f"{phase}_minus_neutral" in ph)}
            for phase in ("room", "public") if f"{phase}_mean" in ph}
    if grid_points:
        g, _ = assemble(results, c, layers, grid_points, contrasts, a.boot)
        res["grid"] = {k: {p: r for p, r in t.items() if r["pairs"] >= a.grid_min_pairs} for k, t in g.items()}
    core = [r for r in results if r["kind"] == "core"]
    if core:
        W = S.boot_weights(len(c.scenarios), a.boot)
        ip = {"trained_on": core[0]["trained_on"]}
        for contrast in ("vote", "hidden"):
            per = [stat(r[contrast]["win"], r[contrast]["n"], W) for r in core if contrast in r]
            if per:
                ip[contrast], _ = mean_stat(per)
                ip[contrast]["pooled"] = float(np.mean([r[contrast]["pooled"] for r in core if contrast in r]))
        if "apply" in core[0]:
            ip["apply"] = {k: float(np.mean([r["apply"][k] for r in core])) for k in core[0]["apply"]}
        res["instructed_probe"] = ip
    res["seconds"] = round(time.time() - t0, 1)
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "probe.json").write_text(json.dumps(res, indent=1) + "\n")
    write_md(res, a.out / "PROBE.md")
    cc = res["rules"].get(S.PRIMARY, {}).get("causal_choice")
    print(json.dumps({"seconds": res["seconds"], "contrasts": {k: len(v) for k, v in res["contrasts"].items()}, "causal_choice": cc}))
    return 0


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    sys.exit(main())
