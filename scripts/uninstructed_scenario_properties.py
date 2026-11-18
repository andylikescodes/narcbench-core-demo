#!/usr/bin/env python3
"""What separates the scenarios: one version of one round read scenario by scenario, against where in the episode
the scenarios part and against properties of the scenario texts coded by two blind passes.

    python scripts/uninstructed_scenario_properties.py --round results/uninstructed/pilot4 --version T7 \
        --properties results/uninstructed/pilot4/scenario_properties_blind.json --out results/uninstructed/pilot4

Writes <out>/SCENARIO_PROPERTIES.md and <out>/scenario_properties.json. Reads recorded files only.

EXPLORATORY. The properties in --properties were chosen after the results of the round were known (the coders saw
only the scenario texts). A relation found here describes; it does not test. Intervals are 95% intervals from
resampling scenarios 10,000 times.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from uninstructed import protocol as P  # noqa: E402

SEED, DRAWS = 20261117, 10000
GROUPS = (("zero", "with no vote for the stake option"), ("mixed", "in between"), ("all", "with every vote for the stake option"))
# word groups counted in the replies of the private rooms (messages 2 and 4)
WORDS = {"ethics, integrity, conscience": r"\b(ethic|integrity|conscience|moral|principle)\w*",
         "law, legal, regulation, liability": r"\b(law|laws|legal|legally|illegal|lawsuit|lawsuits|regulation|regulations|regulatory|compliance|liabilit)\w*",
         "align, united, same page, together": r"\b(align\w*|united|same page|on board|together)\b"}
# the properties of the blind passes: field, the values in the order shown, how each value is worded
PROPS = [("setting", [("PUBLIC", "a public body"), ("PROFESSIONAL", "a professional body"), ("COMPANY", "inside a company")], "The committee is"),
         ("controller", [("SUPERIOR", "an office above the member"), ("OUTSIDE", "an outside sponsor"), ("AUTOMATIC", "nobody: it pays out automatically")],
          "The member's own stake is controlled by"),
         ("own_stake_direction", [("LOSS", "lose something it holds"), ("GAIN", "gain something new")], "The member would"),
         ("wish_stated", [(1, "yes"), (0, "no")], "The interest sentence says that someone wants the outcome"),
         ("rule_stated", [(1, "yes"), (0, "no")], "The case text cites a law, rule or standard against the stake option"),
         ("argument_on_file", [(1, "yes"), (0, "no")], "The case text offers a fact in favour of the stake option"),
         ("beneficiaries", [("SELF", "only the member"), ("OTHERS", "also others")], "The member's stake covers")]


def spearman(a: list[float], b: list[float]) -> float:
    def rank(x):
        order = sorted(range(len(x)), key=lambda i: x[i])
        r, i = [0.0] * len(x), 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and x[order[j + 1]] == x[order[i]]:
                j += 1
            for k in range(i, j + 1):
                r[order[k]] = (i + j) / 2
            i = j + 1
        return r
    ra, rb = rank(a), rank(b)
    ma, mb = sum(ra) / len(ra), sum(rb) / len(rb)
    den = (sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb)) ** 0.5
    return sum((x - ma) * (y - mb) for x, y in zip(ra, rb)) / den if den else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--round", type=Path, required=True)
    ap.add_argument("--version", default="T7")
    ap.add_argument("--properties", type=Path, required=True)
    ap.add_argument("--earlier", action="append", default=[], help="FOLDER:VERSION of an earlier round to rank the shared scenarios against; "
                    "default: T7, T8, T6 of pilot3 and T3 of pilot2")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    rooms = [r for r in json.loads((a.round / "rooms.json").read_text()) if r["version"] == a.version]
    if not rooms:
        sys.exit(f"no rooms of {a.version} in {a.round}")
    tv, nv, col = defaultdict(int), defaultdict(int), defaultdict(int)
    for r in rooms:
        s = r["scenario_id"]
        tv[s] += sum(v == r["target"] for v in r["votes"]); nv[s] += len(r["votes"]); col[s] += r["label"] == "collusion"
    sids = sorted(tv)
    per = nv[sids[0]]
    assert all(nv[s] == per for s in sids), "every scenario needs the same number of votes"

    def grp(s): return "zero" if tv[s] == 0 else "all" if tv[s] == nv[s] else "mixed"
    def rate(ss): return sum(tv[s] for s in ss) / sum(nv[s] for s in ss) if ss else float("nan")
    def crate(ss): return sum(col[s] for s in ss) / (len(ss) * per / 2) if ss else float("nan")

    def interval(x: list[str], y: list[str]) -> tuple[float, float]:
        rng = random.Random(SEED)
        d = sorted(rate([rng.choice(x) for _ in x]) - rate([rng.choice(y) for _ in y]) for _ in range(DRAWS))
        return d[int(0.025 * DRAWS)], d[int(0.975 * DRAWS)]

    L = [f"# What separates the scenarios: {a.version}", "",
         f"Round `{a.round.name}`, version {a.version}: {len(sids)} scenarios, {len(rooms)} rooms, {sum(nv.values())} votes "
         f"({per} votes per scenario). Written by `scripts/uninstructed_scenario_properties.py`.", "",
         "**Exploratory.** The properties of the last section were chosen after these results were known; the two coders "
         "saw only the scenario texts. A relation found here describes and does not test.", ""]
    out: dict = {"round": a.round.name, "version": a.version, "scenarios": len(sids), "votes_per_scenario": per,
                 "groups": {g: sorted(s for s in sids if grp(s) == g) for g, _ in GROUPS}}

    # 1. where in the episode the scenarios part
    L += ["## Where in the episode the scenarios part", "",
          "Rooms in which, by each coder's marks, a member proposes backing the stake option and the other member goes along; "
          "rooms in which both coders read a pact; rooms in which both members vote for the stake option. Ranges are the two "
          "coders (Gemma, Claude).", "",
          "| Scenarios | Rooms | A member proposes it | The other goes along | Pact, both coders | Both vote for it | Labelled collusion |", "|---|---|---|---|---|---|---|"]
    out["funnel"] = {}
    for g, words in GROUPS:
        rs = [r for r in rooms if grp(r["scenario_id"]) == g]
        prop = [sum(r["marks"][c]["proposal"] > 0 for r in rs) for c in (0, 1)]
        acc = [sum(r["marks"][c]["acceptance"] > 0 for r in rs) for c in (0, 1)]
        pact = sum(r["codes"] == ["AGREE_TARGET", "AGREE_TARGET"] for r in rs)
        both = sum(all(v == r["target"] for v in r["votes"]) for r in rs)
        n_col = sum(r["label"] == "collusion" for r in rs)
        n_sc = len(out["groups"][g])
        L.append(f"| {n_sc} {words} | {len(rs)} | {min(prop)} to {max(prop)} | {min(acc)} to {max(acc)} | {pact} | {both} | {n_col} |")
        out["funnel"][g] = {"scenarios": n_sc, "rooms": len(rs), "proposal": prop, "acceptance": acc, "pact_both": pact, "both_vote": both, "collusion": n_col}
    L.append("")

    # 2. what the replies say
    replies = defaultdict(list)
    for r in rooms:
        res = json.loads((a.round / "runs" / a.version / r["run_id"] / "results.json").read_text())
        for i, m in enumerate(res["room_messages"][r["room"]]):
            if i in (1, 3):
                replies[grp(r["scenario_id"])].append(m.split("]:", 1)[1].lower())
    L += ["## What the replies in the private rooms say", "",
          "Share of the replies (messages 2 and 4 of a room) that contain a word of the group. A word count, not a reading.", "",
          "| Words | " + " | ".join(f"{w} ({len(replies[g])} replies)" for g, w in GROUPS) + " |", "|---|---|---|---|"]
    out["reply_words"] = {}
    for name, pat in WORDS.items():
        row = [sum(bool(re.search(pat, t)) for t in replies[g]) / len(replies[g]) for g, _ in GROUPS]
        L.append(f"| {name} | " + " | ".join(f"{100 * x:.0f}%" for x in row) + " |")
        out["reply_words"][name] = dict(zip([g for g, _ in GROUPS], [round(x, 3) for x in row]))
    L.append("")

    # 3. is it the scenario: the same scenarios in other rounds and versions; the option layout
    earlier = a.earlier or ["pilot3:T7", "pilot3:T8", "pilot3:T6", "pilot2:T3"]
    L += ["## Is it the scenario", "",
          "Rank correlation (Spearman) of the votes for the stake option, over the scenarios that this round shares with an "
          "earlier round and version.", "", "| Earlier round and version | Scenarios shared | Rank correlation |", "|---|---|---|"]
    out["rank_correlation"] = {}
    for item in earlier:
        folder, version = item.split(":")
        path = a.round.parent / folder / "rooms.json"
        if not path.exists():
            continue
        e = defaultdict(int)
        for r in json.loads(path.read_text()):
            if r["version"] == version:
                e[r["scenario_id"]] += sum(v == r["target"] for v in r["votes"])
        shared = sorted(set(e) & set(sids))
        if len(shared) >= 3:
            rho = spearman([e[s] for s in shared], [tv[s] for s in shared])
            L.append(f"| `{folder}`, {version} ({P.ABOUT[version]}) | {len(shared)} | {rho:.2f} |")
            out["rank_correlation"][item] = {"scenarios": len(shared), "rho": round(rho, 4)}
    _, scenarios, idx = P.load_scenarios()
    lay = {0: "letters, the other option first", 2: "numbers, the other option first", 1: "numbers, the stake option first", 3: "numbers, the stake option first"}
    L += ["", "Option layout (set by the scenario's place in the bank):", "", "| Layout | Scenarios | Votes for the stake option |", "|---|---|---|"]
    out["layout"] = {}
    for v in (0, 2, 1, 3):
        ss = [s for s in sids if idx[s] % 4 == v]
        L.append(f"| {v}: {lay[v]} | {len(ss)} | {100 * rate(ss):.0f}% |")
        out["layout"][str(v)] = {"scenarios": len(ss), "rate": round(rate(ss), 3)}
    L.append("")

    # 4. properties of the scenario texts, two blind passes
    props = json.loads(a.properties.read_text())
    agreed = props["agreed"]
    L += ["## Properties of the scenario texts (two blind passes)", "",
          f"`{a.properties.name}`: seven properties coded from the case, the two option texts, the interest sentence and the "
          "own-stake sentence. A scenario enters a row only where the two passes agree; agreement per property: "
          + ", ".join(f"{k} {v} of {len(agreed)}" for k, v in props["agreement"].items()) + ".", "",
          "| Property | Value | Scenarios | Votes for the stake option | Rooms labelled collusion | With no vote / in between / every vote |", "|---|---|---|---|---|---|"]
    out["properties"] = {}
    for field, values, lead in PROPS:
        out["properties"][field] = {}
        for val, words in values:
            ss = [s for s in sids if agreed[s][field] == val]
            g = Counter(grp(s) for s in ss)
            L.append(f"| {lead} | {words} | {len(ss)} | {100 * rate(ss):.0f}% | {100 * crate(ss):.0f}% | {g['zero']} / {g['mixed']} / {g['all']} |")
            out["properties"][field][str(val)] = {"scenarios": len(ss), "rate": round(rate(ss), 3), "collusion": round(crate(ss), 3),
                                                  "zero": g["zero"], "mixed": g["mixed"], "all": g["all"]}
    L += ["", "Differences, with intervals:", "", "| Contrast | Difference in votes for the stake option | 95% interval |", "|---|---|---|"]
    out["contrasts"] = {}

    def contrast(name, x, y):
        lo, hi = interval(x, y)
        L.append(f"| {name} ({len(x)} against {len(y)} scenarios) | {100 * (rate(x) - rate(y)):+.0f} points | {100 * lo:+.0f} to {100 * hi:+.0f} |")
        out["contrasts"][name] = {"n": [len(x), len(y)], "difference": round(rate(x) - rate(y), 3), "interval": [round(lo, 3), round(hi, 3)]}

    has = lambda f: [s for s in sids if agreed[s][f] is not None]  # noqa: E731
    contrast("a public body against the rest", [s for s in has("setting") if agreed[s]["setting"] == "PUBLIC"], [s for s in has("setting") if agreed[s]["setting"] != "PUBLIC"])
    contrast("inside a company against the rest", [s for s in has("setting") if agreed[s]["setting"] == "COMPANY"], [s for s in has("setting") if agreed[s]["setting"] != "COMPANY"])
    contrast("an office above controls the stake, against the rest", [s for s in has("controller") if agreed[s]["controller"] == "SUPERIOR"], [s for s in has("controller") if agreed[s]["controller"] != "SUPERIOR"])
    contrast("nobody controls the stake, against the rest", [s for s in has("controller") if agreed[s]["controller"] == "AUTOMATIC"], [s for s in has("controller") if agreed[s]["controller"] != "AUTOMATIC"])
    contrast("a loss against a gain", [s for s in sids if agreed[s]["own_stake_direction"] == "LOSS"], [s for s in sids if agreed[s]["own_stake_direction"] == "GAIN"])
    for f, name in (("wish_stated", "a wish is stated, against not"), ("rule_stated", "a rule is cited in the case, against not"),
                    ("argument_on_file", "the case offers a fact for the stake option, against not")):
        contrast(name, [s for s in sids if agreed[s][f] == 1], [s for s in sids if agreed[s][f] == 0])
    contrast("the stake covers only the member, against also others", [s for s in sids if agreed[s]["beneficiaries"] == "SELF"], [s for s in sids if agreed[s]["beneficiaries"] == "OTHERS"])
    L += ["", "Two properties together:", "", "| Scenarios | Number | Votes for the stake option | Rooms labelled collusion | The scenarios, with their votes |", "|---|---|---|---|---|"]
    out["pairs"] = {}
    for name, setting, direction in (("A public body and a loss", "PUBLIC", "LOSS"), ("A public body and a gain", "PUBLIC", "GAIN"),
                                     ("A professional body and a loss", "PROFESSIONAL", "LOSS"), ("Inside a company and a gain", "COMPANY", "GAIN")):
        ss = sorted((s for s in sids if agreed[s]["setting"] == setting and agreed[s]["own_stake_direction"] == direction), key=lambda s: -tv[s])
        L.append(f"| {name} | {len(ss)} | {100 * rate(ss):.0f}% | {100 * crate(ss):.0f}% | " + ", ".join(f"`{s}` {tv[s]}" for s in ss) + " |")
        out["pairs"][name] = {"scenarios": ss, "rate": round(rate(ss), 3), "collusion": round(crate(ss), 3)}
    L.append("")

    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "SCENARIO_PROPERTIES.md").write_text("\n".join(L) + "\n")
    (a.out / "scenario_properties.json").write_text(json.dumps(out, indent=1) + "\n")
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    sys.exit(main())
