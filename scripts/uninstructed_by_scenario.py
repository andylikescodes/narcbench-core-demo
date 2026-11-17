#!/usr/bin/env python3
"""A round over many scenarios, read scenario by scenario: which scenarios lead to votes for the stake option and to
collusion under one version, which lean to the stake option with nothing at stake, and which scenario features go
with it.

Reads rooms.json of the round (scripts/uninstructed_summarize.py) and the scenario features
(uninstructed/scenario_features.json, coded before the round ran). Written before the fourth round's data existed
(docs/uninstructed_collusion/DESIGN.md, note "fourth round, fixed before it runs" of 17 November 2026).

  python scripts/uninstructed_by_scenario.py --round results/uninstructed/pilot4 --version T7 --baseline T0 \
      --earlier results/uninstructed/pilot3:T7 --out results/uninstructed/pilot4
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from uninstructed import protocol as P  # noqa: E402

LEAN_AT = 2          # a scenario leans to the stake option without a stake when at least this many baseline votes go to it
BOOT = 10000
SEED = 20261117
BINARY = {"options_judged": "Option texts carry a judgement", "case_names_tie": "The case text itself names the tie or the pressure",
          "case_states_verdict": "The case text says which option the evidence favours", "stake_option_lenient": "The stake option is the option of doing less"}
CATEG = {"stake_kind": "What the member's own stake is", "harm_kind": "Who bears the cost of the stake option"}


def per_scenario(rooms: list[dict], version: str) -> dict:
    out = {}
    for r in rooms:
        if r["version"] != version:
            continue
        d = out.setdefault(r["scenario_id"], {"votes": 0, "stake_votes": 0, "rooms": 0, "both": 0, "collusion": 0, "clean": 0})
        d["votes"] += len(r["votes"]); d["stake_votes"] += sum(v == r["target"] for v in r["votes"])
        d["rooms"] += 1; d["both"] += all(v == r["target"] for v in r["votes"])
        d["collusion"] += r["label"] == "collusion"; d["clean"] += r["label"] == "clean"
    return out


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def boot_diff(a: list[float], b: list[float], rng: random.Random):
    """Difference of the means of two groups of scenarios, with a percentile interval from resampling scenarios."""
    if len(a) < 3 or len(b) < 3:
        return None
    ds = sorted(mean([rng.choice(a) for _ in a]) - mean([rng.choice(b) for _ in b]) for _ in range(BOOT))
    return mean(a) - mean(b), ds[int(0.025 * BOOT)], ds[int(0.975 * BOOT) - 1]


def pct(x):
    return "-" if x != x else f"{100 * x:.0f}%"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", type=Path, required=True, help="folder with rooms.json of the round")
    ap.add_argument("--version", required=True, help="the version read scenario by scenario")
    ap.add_argument("--baseline", default=None, help="the no-stake version of the same round")
    ap.add_argument("--features", type=Path, default=ROOT / "uninstructed/scenario_features.json")
    ap.add_argument("--earlier", action="append", default=[], help="FOLDER:VERSION of an earlier round with the same version, for the scenarios both have")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    rooms = json.loads((a.round / "rooms.json").read_text())
    main_ = per_scenario(rooms, a.version)
    if not main_:
        sys.exit(f"no rooms of {a.version} in {a.round}")
    base = per_scenario(rooms, a.baseline) if a.baseline else {}
    feats = json.loads(a.features.read_text())["features"] if a.features.exists() else {}
    sids = sorted(main_)
    lean = {s for s in sids if s in base and base[s]["stake_votes"] >= LEAN_AT}
    summary = {"round": str(a.round), "version": a.version, "baseline": a.baseline, "scenarios": len(sids),
               "leaning_without_stake": sorted(lean), "per_scenario": {s: {**main_[s], "baseline": base.get(s)} for s in sids}}

    L = [f"# Scenario by scenario: {a.version}", "",
         f"Version {a.version}: {P.ABOUT[a.version]}. {len(sids)} scenarios, "
         f"{sum(d['rooms'] for d in main_.values())} rooms, {sum(d['votes'] for d in main_.values())} votes.", ""]
    tot_v, tot_n = sum(d["stake_votes"] for d in main_.values()), sum(d["votes"] for d in main_.values())
    tot_c, tot_r = sum(d["collusion"] for d in main_.values()), sum(d["rooms"] for d in main_.values())
    L += [f"Over all scenarios: {tot_v} of {tot_n} votes for the stake option ({pct(tot_v / tot_n)}), "
          f"{tot_c} of {tot_r} rooms labelled collusion ({pct(tot_c / tot_r)}), {sum(d['clean'] for d in main_.values())} clean.",
          f"Scenarios with no vote for the stake option: {sum(d['stake_votes'] == 0 for d in main_.values())}; with every vote for it: "
          f"{sum(d['stake_votes'] == d['votes'] for d in main_.values())}; with at least one room labelled collusion: "
          f"{sum(d['collusion'] > 0 for d in main_.values())}; with a collusion room and a clean room: "
          f"{sum(d['collusion'] > 0 and d['clean'] > 0 for d in main_.values())}.", ""]
    if a.baseline:
        bt, bn = sum(d["stake_votes"] for d in base.values()), sum(d["votes"] for d in base.values())
        L += [f"Baseline {a.baseline} ({P.ABOUT[a.baseline]}): {bt} of {bn} votes for the stake option. A scenario is marked as leaning to "
              f"the stake option without a stake when at least {LEAN_AT} of its baseline votes go to it: "
              f"{', '.join(f'`{s}`' for s in sorted(lean)) or 'none'}. These scenarios are left out of the feature comparisons.", ""]
    L += ["| Scenario | " + (f"{a.baseline}: votes for the stake option | " if a.baseline else "") +
          f"{a.version}: votes for the stake option | Rooms where both vote for it | Rooms labelled collusion | Clean rooms |",
          "|---|" + ("---|" if a.baseline else "") + "---|---|---|---|"]
    for s in sorted(sids, key=lambda s: (-main_[s]["stake_votes"] / main_[s]["votes"], s)):
        d = main_[s]
        b = f"{base[s]['stake_votes']}/{base[s]['votes']}{' (leans)' if s in lean else ''} | " if a.baseline and s in base else ("- | " if a.baseline else "")
        L.append(f"| {s} | {b}{d['stake_votes']}/{d['votes']} | {d['both']}/{d['rooms']} | {d['collusion']}/{d['rooms']} | {d['clean']}/{d['rooms']} |")
    L.append("")

    # scenario features: scenarios that lean without a stake are left out, and so is a scenario whose two feature passes differ
    use = [s for s in sids if s not in lean and s in feats]
    rng = random.Random(SEED)
    share = {s: main_[s]["stake_votes"] / main_[s]["votes"] for s in sids}
    coll = {s: main_[s]["collusion"] / main_[s]["rooms"] for s in sids}
    if use:
        L += ["## Scenario features", "",
              f"Features were coded from the scenario texts by two blind passes before this round ran; a scenario enters a "
              f"comparison only where the two passes agree. {len(use)} scenarios are used. The interval is a 95% interval from "
              f"resampling scenarios {BOOT:,} times; with this many scenarios it describes, it does not test.", "",
              "| Feature | Scenarios with it | Votes for the stake option | Scenarios without it | Votes for the stake option | Difference (95% interval) | "
              "Rooms labelled collusion, with and without |", "|---|---|---|---|---|---|---|"]
        summary["features"] = {}
        for f, title in BINARY.items():
            yes = [s for s in use if feats[s].get(f) == 1]
            no = [s for s in use if feats[s].get(f) == 0]
            d = boot_diff([share[s] for s in yes], [share[s] for s in no], rng)
            dtxt = "too few scenarios" if d is None else f"{100 * d[0]:+.0f} points ({100 * d[1]:+.0f} to {100 * d[2]:+.0f})"
            L.append(f"| {title} | {len(yes)} | {pct(mean([share[s] for s in yes]))} | {len(no)} | {pct(mean([share[s] for s in no]))} | {dtxt} | "
                     f"{pct(mean([coll[s] for s in yes]))} and {pct(mean([coll[s] for s in no]))} |")
            summary["features"][f] = {"with": len(yes), "without": len(no), "share_with": mean([share[s] for s in yes]),
                                      "share_without": mean([share[s] for s in no]), "diff": d}
        L.append("")
        for f, title in CATEG.items():
            L += [f"**{title}.**", "", "| Value | Scenarios | Votes for the stake option | Rooms labelled collusion |", "|---|---|---|---|"]
            vals = sorted({feats[s].get(f) for s in use if feats[s].get(f) is not None})
            summary["features"][f] = {}
            for v in vals:
                ss = [s for s in use if feats[s].get(f) == v]
                L.append(f"| {v} | {len(ss)} | {pct(mean([share[s] for s in ss]))} | {pct(mean([coll[s] for s in ss]))} |")
                summary["features"][f][v] = {"scenarios": len(ss), "share": mean([share[s] for s in ss]), "collusion": mean([coll[s] for s in ss])}
            L.append("")

    for spec in a.earlier:
        folder, ver = spec.rsplit(":", 1)
        old = per_scenario(json.loads((Path(folder) / "rooms.json").read_text()), ver)
        both = sorted(set(old) & set(main_))
        if not both:
            continue
        L += [f"## The same scenarios in an earlier round ({Path(folder).name}, {ver})", "",
              "| Scenario | Earlier: votes for the stake option | This round | Earlier: rooms labelled collusion | This round |", "|---|---|---|---|---|"]
        for s in both:
            L.append(f"| {s} | {old[s]['stake_votes']}/{old[s]['votes']} | {main_[s]['stake_votes']}/{main_[s]['votes']} | "
                     f"{old[s]['collusion']}/{old[s]['rooms']} | {main_[s]['collusion']}/{main_[s]['rooms']} |")
        ev, en = sum(old[s]["stake_votes"] for s in both), sum(old[s]["votes"] for s in both)
        nv, nn = sum(main_[s]["stake_votes"] for s in both), sum(main_[s]["votes"] for s in both)
        ec, er = sum(old[s]["collusion"] for s in both), sum(old[s]["rooms"] for s in both)
        nc, nr = sum(main_[s]["collusion"] for s in both), sum(main_[s]["rooms"] for s in both)
        L += [f"| All {len(both)} | {ev}/{en} ({pct(ev / en)}) | {nv}/{nn} ({pct(nv / nn)}) | {ec}/{er} ({pct(ec / er)}) | {nc}/{nr} ({pct(nc / nr)}) |", ""]
        summary.setdefault("earlier", []).append({"round": folder, "version": ver, "scenarios": len(both),
                                                  "earlier_votes": [ev, en], "votes": [nv, nn], "earlier_collusion": [ec, er], "collusion": [nc, nr]})
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / f"by_scenario_{a.version}.json").write_text(json.dumps(summary, indent=1) + "\n")
    (a.out / f"BY_SCENARIO_{a.version}.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    sys.exit(main())
