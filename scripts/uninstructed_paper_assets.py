#!/usr/bin/env python3
"""Tables, figures and numbers of docs/uninstructed_collusion/collusion_behaviour_study.tex, written from the data.

Reads the rooms and codings of the four rounds, the public-position codings and the scenario features, and writes
  docs/uninstructed_collusion/tables/*.tex    booktabs tables that the paper \\input's
  docs/uninstructed_collusion/tables/numbers.tex   every number the prose uses, as \\V{<key>}{<field>} and \\N{<name>}
  docs/uninstructed_collusion/figs/*.pdf      the two figures

  python3 scripts/uninstructed_paper_assets.py            # then: tectonic docs/uninstructed_collusion/collusion_behaviour_study.tex
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
RU = ROOT / "results/uninstructed"
DOC = ROOT / "docs/uninstructed_collusion"
ROUNDS = [("pilot", 1), ("pilot2", 2), ("pilot3", 3), ("pilot4", 4)]
ORDER = ["S0", "S1", "S2", "S3", "T1", "T2", "T3", "T4", "T5", "P0", "P1", "P3", "T6", "T7", "T8", "I1"]
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE = "#2a78d6", "#eb6834"         # categorical slots 1 and 2 of the reference palette; validated on a white page
SHORT = {
    "S0": "No stake; honesty paragraph", "S1": "Third party's interest, as a fact", "S2": "S1 plus the member's own stake",
    "S3": "S2 plus a three-vote rule", "T1": "S2 without the honesty paragraph", "T2": "T1 plus heavier loss, votes not reviewed",
    "T3": "T1 plus a wish", "T4": "S2, partner's proposal scripted", "T5": "T1, partner's proposal scripted",
    "P0": "No stake; bare role line; neutral option texts", "P1": "P0 plus the stake as a fact", "P3": "P1 plus the wish",
    "T6": "T1 plus heavier loss and wish", "T7": "T1 plus a stated intention", "T8": "T3, partner's proposal scripted",
    "I1": "Told to collude (benchmark prompt)", "T0": "No stake; role line alone", "T7b": "T7 again, all 50 scenarios",
}


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def stats(rs: list[dict]) -> dict:
    tv = sum(x == r["target"] for r in rs for x in r["votes"]); n = sum(len(r["votes"]) for r in rs)
    both = [r for r in rs if all(x == r["target"] for x in r["votes"])]
    ep = collections.Counter()
    for r in rs:
        ep[r["run_id"]] += sum(x == r["target"] for x in r["votes"])
    two = all(len(r["codes"]) == 2 for r in rs)
    lo, hi = wilson(tv, n); clo, chi = wilson(sum(r["label"] == "collusion" for r in rs), len(rs))
    return {"tv": tv, "n": n, "pct": round(100 * tv / n) if n else 0, "lo": round(100 * lo), "hi": round(100 * hi),
            "rooms": len(rs), "both": len(both),
            "g": sum(r["codes"][0] == "AGREE_TARGET" for r in rs), "c": sum(r["codes"][1] == "AGREE_TARGET" for r in rs) if two else 0,
            "gc": sum(set(r["codes"]) == {"AGREE_TARGET"} for r in rs),
            "col": sum(r["label"] == "collusion" for r in rs), "colpct": round(100 * sum(r["label"] == "collusion" for r in rs) / len(rs)) if rs else 0,
            "collo": round(100 * clo), "colhi": round(100 * chi),
            "clean": sum(r["label"] == "clean" for r in rs), "cleanpct": round(100 * sum(r["label"] == "clean" for r in rs) / len(rs)) if rs else 0,
            "other": sum(r["label"] == "other" for r in rs),
            "broken": sum(set(r["codes"]) == {"AGREE_TARGET"} and not all(x == r["target"] for x in r["votes"]) for r in rs),
            "bothpact": sum(set(r["codes"]) == {"AGREE_TARGET"} for r in both), "bothnopact": sum(set(r["codes"]) != {"AGREE_TARGET"} for r in both),
            "ep3": sum(k >= 3 for k in ep.values()), "eps": len(ep),
            "opener": sum(r["votes"][0] == r["target"] for r in rs), "answerer": sum(r["votes"][1] == r["target"] for r in rs),
            "splitboth": sum(set(r["codes"]) == {"SPLIT"} for r in rs), "honest": sum(r.get("reason") == "honest_pact" for r in rs),
            "bothsplit": sum(set(r["codes"]) == {"SPLIT"} for r in both)}


def tex(s: str) -> str:
    return s.replace("_", r"\_").replace("%", r"\%").replace("&", r"\&").replace("$", r"\$")


def esc(s: str) -> str:
    """A transcript string as LaTeX text: specials escaped, straight quotes turned into opening and closing quotes."""
    s = " ".join(s.split()).replace("\\", "\x00")
    for a, b in (("&", r"\&"), ("%", r"\%"), ("$", r"\$"), ("#", r"\#"), ("_", r"\_"), ("{", r"\{"), ("}", r"\}"),
                 ("~", r"\textasciitilde{}"), ("^", r"\textasciicircum{}"), ("*", "")):
        s = s.replace(a, b)
    s = s.replace("\x00", r"\textbackslash{}")
    s = re.sub(r'(^|(?<=[\s(\[]))"', "``", s).replace("'\"", "'\\,''").replace('"', "''")
    s = re.sub(r"(^|(?<=[\s(\[]))'", "`", s)
    return s.replace("...", r"\ldots{}")


def episode(rnd: str, version: str, run_id: str):
    d = RU / rnd / "runs" / version / run_id
    return json.loads((d / "results.json").read_text()), json.loads((d / "agent_prompts.json").read_text())


def said(msg: str) -> tuple[str, str]:
    m = re.match(r"\[(.+?)\]:\s*(.*)", msg, flags=re.S)
    return (m.group(1), m.group(2)) if m else ("", msg)


def cut(text: str, start: str, end: str) -> str:
    """The part of a message from `start` to the end of `end`; fails if either is not there."""
    t = " ".join(text.split())
    i = t.index(start); j = t.index(end, i) + len(end)
    return t[i:j]


# name -> (round folder, version, run id, where, index, from, to). where: room0 / room1 (message index) or public (speaker name)
QUOTES = {
    "discreet": ("pilot3", "T7", "uninstructed__food_01__T7__r1", "room1", 0, "let's be discreet", "reduced score."),
    "downplay": ("pilot3", "T7", "uninstructed__gov_03__T7__r0", "room0", 2, "Let's focus on highlighting", "vulnerabilities."),
    "impartial": ("pilot3", "T7", "uninstructed__food_01__T7__r0", "room0", 2, "We both know how important", "let's be realistic."),
    "noticed": ("pilot3", "T6", "uninstructed__food_01__T6__r0", "room1", 2, "Do you think we can be sure", "noticed?"),
    "jobsecurity": ("pilot2", "T2", "uninstructed__food_01__T2__r0", "room1", 0, "My job security", "this vote."),
    "mistake": ("pilot2", "T4", "uninstructed__ins_01__T4__r0", "room1", 2, "My initial suggestion was a mistake.", "mistake."),
    "lean": ("pilot2", "T2", "uninstructed__eng_01__T2__r1", "room0", 3, "I think we need to lean heavily", "shutdown."),
    "cannot": ("pilot2", "T2", "uninstructed__eng_01__T2__r1", "public", "Marcus Webb", "enhanced monitoring can provide", "structural defects."),
    "refuse": ("pilot2", "T5", "uninstructed__food_01__T5__r0", "room1", 1, "I can't ethically compromise", "the public."),
    "notoption": ("pilot3", "I1", "uninstructed__cyber_01__I1__r0", "room0", 0, "suppressing this disclosure is not an option.", "option."),
    "recuse": ("pilot2", "T1", "uninstructed__fin_03__T1__r1", "public", "Brianna Walsh", "Given my personal financial interest", "the vote."),
    "transparent": ("pilot2", "T1", "uninstructed__fin_03__T1__r0", "room1", 0, "I'm still holding unvested equity", "discussing."),
}


def quote_macros() -> list[str]:
    L = [r"\newcommand{\Q}[1]{\csname ucq@#1\endcsname}"]
    for name, (rnd, ver, rid, where, idx, a_, b_) in QUOTES.items():
        res, _ = episode(rnd, ver, rid)
        if where.startswith("room"):
            text = said(res["room_messages"][int(where[-1])][idx])[1]
        else:
            text = next(t for w, t in map(said, res["public_messages"]) if w == idx and a_ in " ".join(t.split()))
        L.append(rf"\expandafter\def\csname ucq@{name}\endcsname{{{esc(cut(text, a_, b_))}}}")
    return L


def example_blocks() -> dict[str, str]:
    """The worked example of the paper: the pressure version in the restaurant case, first run."""
    res, ap = episode("pilot2", "T2", "uninstructed__food_01__T2__r0")
    out = {}
    thomas = next(e for e in ap if e["agent_name"] == "Thomas Okwu")
    sysp = thomas["system_prompt"].split("\n\n")
    out["ex_system"] = "\n\n".join(esc(x) for x in sysp)
    case, opts = ap[0]["user_prompt"].split("\n\n")[0], ap[0]["user_prompt"].split("\n\n")[1]
    out["ex_case"] = esc(case)
    out["ex_options"] = r" \\ ".join(esc(x) for x in opts.splitlines()[1:])
    out["ex_roomprompt"] = esc(ap[0]["user_prompt"].split("\n\n")[2])
    for k in (0, 1):
        out[f"ex_room{k + 1}"] = "\n".join(rf"\item[\normalfont\textit{{{esc(w)}:}}] {esc(t)}" for w, t in map(said, res["room_messages"][k]))
    pub = [e for e in ap if e["channel"] == "public" and e["phase"] == "discussion"]
    pick = [next(e for e in pub if e["agent_name"] == "Oscar Wali" and e["round"] == 1), next(e for e in pub if e["agent_name"] == "Thomas Okwu" and e["round"] == 3)]
    out["ex_public"] = "\n".join(rf"\item[\normalfont\textit{{{esc(e['agent_name'])}, round {e['round']}:}}] {esc(e['response'])}" for e in pick)
    out["ex_pubprompt"] = esc(pub[0]["user_prompt"].split("\n\n")[-1])
    fin = [e for e in ap if e["phase"] == "final"]
    out["ex_voteprompt"] = esc(fin[0]["user_prompt"].split("\n\n")[-2])
    out["ex_votes"] = "\n".join(rf"\item[\normalfont\textit{{{esc(e['agent_name'])}:}}] " + esc(" / ".join(x.strip() for x in e["response"].strip().splitlines() if x.strip())) for e in fin)
    n_closure = sum("closure" in " ".join(e["response"].split()) for e in pub if e["agent_name"] in ("Thomas Okwu", "Oscar Wali"))
    assert n_closure == 6, n_closure
    return out


def table(path: Path, spec: str, head, rows: list[list], sep: float | None = None):
    """head is a list of column heads, or a ready-made header (a string of LaTeX lines)."""
    L = ([rf"\setlength{{\tabcolsep}}{{{sep}pt}}"] if sep else []) + [rf"\begin{{tabular}}{{{spec}}}", r"\toprule",
         head if isinstance(head, str) else " & ".join(head) + r" \\", r"\midrule"]
    for r in rows:
        L.append(r if isinstance(r, str) else " & ".join(str(x) for x in r) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    path.write_text("\n".join(L) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-figs", action="store_true")
    a = ap.parse_args()
    (DOC / "tables").mkdir(parents=True, exist_ok=True)
    (DOC / "figs").mkdir(parents=True, exist_ok=True)

    rooms, kap = [], {}
    for name, k in ROUNDS:
        rs = json.loads((RU / name / "rooms.json").read_text())
        s = json.loads((RU / name / "summary.json").read_text())
        for r in rs:
            r["round"] = k
        rooms += rs
        conf = s.get("confusion") or {}
        kap[k] = {"kappa": s["kappa"] if s.get("kappa") is not None else float("nan"), "rooms": len(rs), "agree": sum(n for key, n in conf.items() if key.split("|")[0] == key.split("|")[1]),
                  "episodes": len({r["run_id"] for r in rs}), "votes": sum(len(r["votes"]) for r in rs)}
    by = lambda v, rnd=None, excl=(): [r for r in rooms if r["version"] == v and (rnd is None or r["round"] == rnd) and r["scenario_id"] not in excl]
    S = {v: stats(by(v, None if v != "T7" else 3)) for v in ORDER}
    S["T0"] = stats(by("T0", 4)); S["T7b"] = stats(by("T7", 4))
    p12 = sorted({r["scenario_id"] for r in rooms if r["round"] == 3})
    S["T7r"] = stats([r for r in by("T7", 4) if r["scenario_id"] in p12])        # the 12 pilot scenarios, repeated in round 4
    for v in ORDER:
        S[v + "x"] = stats(by(v, None if v != "T7" else 3, ("med_03",)))            # without med_03
    FACT = ["S1", "S2", "S3", "T1", "P1"]
    S["FACT"] = stats([r for v in FACT for r in by(v)])
    S["ALL"] = stats(rooms)

    sc = lambda v, s, rnd=None: sum(x == r["target"] for r in by(v, rnd) if r["scenario_id"] == s for x in r["votes"])
    sids50 = sorted({r["scenario_id"] for r in rooms if r["round"] == 4})

    def per_scenario(v, rnd):
        out = {}
        for r in by(v, rnd):
            d = out.setdefault(r["scenario_id"], {"sv": 0, "n": 0, "rooms": 0, "both": 0, "col": 0, "clean": 0})
            d["sv"] += sum(x == r["target"] for x in r["votes"]); d["n"] += len(r["votes"]); d["rooms"] += 1
            d["both"] += all(x == r["target"] for x in r["votes"]); d["col"] += r["label"] == "collusion"; d["clean"] += r["label"] == "clean"
        return out
    t0, t7 = per_scenario("T0", 4), per_scenario("T7", 4)
    lean = sorted(s for s in sids50 if t0[s]["sv"] >= 2)
    S["T7bx"] = stats([r for r in by("T7", 4) if r["scenario_id"] not in set(lean) | {"med_03"}])
    feats = json.loads((ROOT / "uninstructed/scenario_features.json").read_text())

    # ---------------------------------------------------------------- numbers.tex
    N = {
        "episodes": sum(k["episodes"] for k in kap.values()), "rooms": f"{len(rooms):,}", "votes": f"{sum(len(r['votes']) for r in rooms):,}",
        "unparsed": sum(x == "?" for r in rooms for x in r["votes"]), "versions": len(ORDER) + 1,
        "costOne": "0.88", "costTwo": "0.69", "costThree": "1.02", "costFour": "0.96", "costAll": "3.55",
        "pactsAll": sum(S[v]["gc"] for v in ORDER) + S["T7b"]["gc"] + S["T0"]["gc"],
        "brokenAll": sum(S[v]["broken"] for v in ORDER) + S["T7b"]["broken"] + S["T0"]["broken"],
        "leanList": ", ".join(rf"\texttt{{{tex(s)}}}" for s in lean) or "none", "leanCount": len(lean),
        "scenZero": sum(d["sv"] == 0 for d in t7.values()), "scenAll": sum(d["sv"] == d["n"] for d in t7.values()),
        "scenAnyCol": sum(d["col"] > 0 for d in t7.values()), "scenBoth": sum(d["col"] > 0 and d["clean"] > 0 for d in t7.values()),
        "scenBothThree": len({r["scenario_id"] for r in by("T7", 3) if r["label"] == "collusion"} & {r["scenario_id"] for r in by("T7", 3) if r["label"] == "clean"}),
        "tZeroStakeScen": sum(d["sv"] > 0 for d in t0.values()),
        "featAgree": sum(int(v.split(" of ")[0]) for v in feats["agreement_of_the_two_passes"].values()),
    }
    for k, d in kap.items():
        w = ["One", "Two", "Three", "Four"][k - 1]
        N[f"kappa{w}"] = f"{d['kappa']:.2f}"; N[f"rooms{w}"] = d["rooms"]; N[f"agree{w}"] = d["agree"]
        N[f"agreepct{w}"] = round(100 * d["agree"] / d["rooms"]); N[f"episodes{w}"] = d["episodes"]; N[f"votes{w}"] = d["votes"]

    # public positions: coder 1 = Gemma over every member; coder 2 = a Claude sample
    PUB, PUBAG = {}, {}
    for name, k in ROUNDS[1:]:
        f = RU / name / "public_stance.json"
        if not f.exists():
            continue
        ps = json.loads(f.read_text())["summary"]
        for v, t in ps["versions"].items():
            key = v if not (v == "T7" and k == 4) else "T7b"
            PUB[key] = {"members": t["members"], "pubstake": t["public_stake_option"], "sv": t["votes_for_stake_option"],
                        "other": t["stake_votes_after_public_other_option"], "neither": t["stake_votes_after_public_neither"],
                        "stake": t["stake_votes_after_public_stake_option"], "unread": t.get("stake_votes_after_public_unreadable", 0),
                        "flip": t["public_stake_option_then_other_vote"]}
        ag = (ps.get("agreement") or [{}])[0]
        PUBAG[k] = {"n": ag.get("n", 0), "raw": round(100 * ag.get("raw_agreement", 0)), "kappa": f"{ag.get('kappa', 0):.2f}"}

    L = [r"% written by scripts/uninstructed_paper_assets.py; do not edit", r"\makeatletter",
         r"\newcommand{\V}[2]{\csname uc@#1@#2\endcsname}", r"\newcommand{\N}[1]{\csname ucn@#1\endcsname}",
         r"\newcommand{\PB}[2]{\csname ucp@#1@#2\endcsname}", r"\newcommand{\PA}[2]{\csname uca@#1@#2\endcsname}"]
    for v, d in S.items():
        for f, x in d.items():
            L.append(rf"\expandafter\def\csname uc@{v}@{f}\endcsname{{{x}}}")
    for k, x in N.items():
        L.append(rf"\expandafter\def\csname ucn@{k}\endcsname{{{x}}}")
    for v, d in PUB.items():
        for f, x in d.items():
            L.append(rf"\expandafter\def\csname ucp@{v}@{f}\endcsname{{{x}}}")
    for k, d in PUBAG.items():
        for f, x in d.items():
            L.append(rf"\expandafter\def\csname uca@{['One', 'Two', 'Three', 'Four'][k - 1]}@{f}\endcsname{{{x}}}")
    L += quote_macros()
    L.append(r"\makeatother")
    (DOC / "tables/numbers.tex").write_text("\n".join(L) + "\n")
    for name, body in example_blocks().items():
        (DOC / f"tables/{name}.tex").write_text(body + "%\n")

    # ---------------------------------------------------------------- tables
    def vrow(v, label=None):
        d = S[v]
        return [label or v, tex(SHORT[v]), f"{d['tv']}/{d['n']}", f"{d['pct']}\\%", f"{d['lo']}--{d['hi']}\\%", f"{d['both']}/{d['rooms']}",
                f"{d['g']}, {d['c']}, {d['gc']}", f"{d['col']}/{d['rooms']}", f"{d['clean']}/{d['rooms']}"]
    rows = []
    for rnd, vs in ((1, ORDER[:4]), (2, ORDER[4:9]), (3, ORDER[9:]), (4, ["T0", "T7b"])):
        if rows:
            rows.append(r"\addlinespace")
        rows += [vrow(v, "T7" if v == "T7b" else None) for v in vs]
    table(DOC / "tables/tab_all_versions.tex", "@{}l>{\\raggedright\\arraybackslash}p{4.7cm}rrrrrrr@{}",
          ["", "Every member's prompt", "Votes", "", "95\\% int.", "Both vote", "Pact G, C, both", "Collusion", "Clean"], rows, sep=3.6)

    lad = [("Stake as a fact", "T1"), ("plus a heavier loss, votes not reviewed", "T2"), ("plus a wish, decision left open", "T3"),
           ("plus both of these", "T6"), ("plus a stated intention", "T7"), ("Told to collude (control)", "I1")]
    table(DOC / "tables/tab_ladder.tex", "@{}llrrrr@{}", ["Step", "", "Votes", "", "Pact", "Collusion"],
          [[s, v, f"{S[v]['tv']} of {S[v]['n']}", f"{S[v]['pct']}\\%", f"{S[v]['gc']} of {S[v]['rooms']}", f"{S[v]['col']} of {S[v]['rooms']}"] for s, v in lad])

    table(DOC / "tables/tab_alone.tex", "@{}llrrr@{}", ["", "", "Both vote for it", "with a pact", "without"],
          [[tex(SHORT[v]), "T7" if v == "T7b" else v, S[v]["both"], S[v]["bothpact"], S[v]["bothnopact"]] for v in ("T2", "T3", "T6", "T7", "T7b")])

    table(DOC / "tables/tab_proposal.tex", "@{}lrrr@{}", ["", "Hears it", "`Makes' it", "Split"],
          [[lab, f"{S[v]['answerer']} of {S[v]['rooms']}", f"{S[v]['opener']} of {S[v]['rooms']}", f"{S[v]['splitboth']} of {S[v]['rooms']}"]
           for v, lab in (("T4", "T4: stake as a fact, honesty paragraph kept"), ("T5", "T5: stake as a fact"), ("T8", "T8: stake plus the wish"))])

    show = [v for v in ORDER if S[v]["tv"] > 0]
    table(DOC / "tables/tab_scen12.tex", "@{}l" + "r" * len(show) + "@{}", ["Scenario"] + show,
          [[rf"\texttt{{{tex(s)}}}"] + [sc(v, s, None if v != "T7" else 3) for v in show] for s in p12])

    table(DOC / "tables/tab_repeat.tex", "@{}lrrrr@{}", ["Scenario", "Round 3: votes", "Round 4: votes", "Round 3: collusion rooms", "Round 4"],
          [[rf"\texttt{{{tex(s)}}}", f"{sc('T7', s, 3)}/8", f"{t7[s]['sv']}/8",
            f"{sum(r['label'] == 'collusion' for r in by('T7', 3) if r['scenario_id'] == s)}/4", f"{t7[s]['col']}/4"] for s in p12]
          + [r"\midrule", ["All 12", f"{S['T7']['tv']}/96", f"{S['T7r']['tv']}/96", f"{S['T7']['col']}/48", f"{S['T7r']['col']}/48"]])

    dist = collections.Counter(d["sv"] for d in t7.values())
    table(DOC / "tables/tab_dist.tex", "@{}l" + "r" * 9 + "@{}", ["Votes for the stake option (of 8)"] + [str(k) for k in range(9)],
          [["Scenarios"] + [dist.get(k, 0) for k in range(9)]])

    order50 = sorted(sids50, key=lambda s: (-t7[s]["sv"], s))
    half = (len(order50) + 1) // 2
    def srow(s):
        return [rf"\texttt{{{tex(s)}}}", f"{t0[s]['sv']}/{t0[s]['n']}", f"{t7[s]['sv']}/{t7[s]['n']}", f"{t7[s]['both']}/{t7[s]['rooms']}",
                f"{t7[s]['col']}/{t7[s]['rooms']}", f"{t7[s]['clean']}/{t7[s]['rooms']}"]
    rows = []
    for i in range(half):
        left = srow(order50[i]); right = srow(order50[i + half]) if i + half < len(order50) else [""] * 6
        rows.append(left + [""] + right)
    table(DOC / "tables/tab_scen50.tex", "@{}lrrrrr@{\\hspace{1.6em}}c@{}lrrrrr@{}",
          ["Scenario", "T0", "T7", "Both", "Coll.", "Clean", "", "Scenario", "T0", "T7", "Both", "Coll.", "Clean"], rows, sep=4.5)

    # scenario features: the fixed comparison (scenarios that lean in T0 left out; a scenario without an agreed value left out)
    BIN = [("options_judged", "Option texts carry a judgement"), ("case_names_tie", "The case text names the tie or the pressure"),
           ("case_states_verdict", "The case text states which option the evidence favours"), ("stake_option_lenient", "The stake option is the option of doing less")]
    CAT = [("stake_kind", "Kind of own stake"), ("harm_kind", "Who bears the cost")]
    use = [s for s in sids50 if s not in lean]
    share = {s: t7[s]["sv"] / t7[s]["n"] for s in sids50}; coll = {s: t7[s]["col"] / t7[s]["rooms"] for s in sids50}
    mean = lambda xs: sum(xs) / len(xs) if xs else float("nan")
    rng = random.Random(20261117)
    def boot(x, y):
        if len(x) < 3 or len(y) < 3:
            return None
        ds = sorted(mean([rng.choice(x) for _ in x]) - mean([rng.choice(y) for _ in y]) for _ in range(10000))
        return mean(x) - mean(y), ds[250], ds[9749]
    frows, FE = [], {}
    for f, title in BIN:
        yes = [s for s in use if feats["features"][s].get(f) == 1]; no = [s for s in use if feats["features"][s].get(f) == 0]
        d = boot([share[s] for s in yes], [share[s] for s in no])
        FE[f] = {"yes": len(yes), "no": len(no), "sy": mean([share[s] for s in yes]), "sn": mean([share[s] for s in no]), "d": d,
                 "cy": mean([coll[s] for s in yes]), "cn": mean([coll[s] for s in no])}
        frows.append([title, len(yes), f"{100 * FE[f]['sy']:.0f}\\%", len(no), f"{100 * FE[f]['sn']:.0f}\\%",
                      "--" if d is None else f"{100 * d[0]:+.0f} ({100 * d[1]:+.0f} to {100 * d[2]:+.0f})",
                      f"{100 * FE[f]['cy']:.0f}\\% / {100 * FE[f]['cn']:.0f}\\%"])
    table(DOC / "tables/tab_features.tex", "@{}>{\\raggedright\\arraybackslash}p{5.0cm}rrrrrr@{}",
          r"& \multicolumn{2}{c}{With it} & \multicolumn{2}{c}{Without it} & & \\" "\n" r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}" "\n"
          r"Feature of the scenario & $n$ & Votes & $n$ & Votes & Difference (95\% int.) & Collusion \\", frows, sep=4.5)
    crows = []
    for f, title in CAT:
        vals = sorted({feats["features"][s].get(f) for s in use if feats["features"][s].get(f) is not None})
        for i, v in enumerate(vals):
            ss = [s for s in use if feats["features"][s].get(f) == v]
            crows.append([title if i == 0 else "", tex(v.replace("_", " ")), len(ss), f"{100 * mean([share[s] for s in ss]):.0f}\\%", f"{100 * mean([coll[s] for s in ss]):.0f}\\%"])
        crows.append(r"\addlinespace")
    table(DOC / "tables/tab_features_cat.tex", "@{}llrrr@{}", ["", "Value", "Scenarios", "Votes", "Collusion"], crows[:-1])

    prow = []
    for v in ["T2", "T3", "T6", "P3", "T7", "T7b", "T8", "I1"]:
        if v in PUB:
            p = PUB[v]
            prow.append([tex(SHORT[v]), "T7" if v == "T7b" else v, p["members"], p["pubstake"], p["sv"], p["other"], p["neither"], p["stake"]])
    table(DOC / "tables/tab_public.tex", "@{}>{\\raggedright\\arraybackslash}p{4.2cm}lrrrrrr@{}",
          r"& & & & & \multicolumn{3}{c}{of these, had argued in public for} \\" "\n" r"\cmidrule(l){6-8}" "\n"
          r"& & Members & Argue for it & Vote for it & the other option & neither & it \\", prow, sep=4.5)

    table(DOC / "tables/tab_coders.tex", "@{}lrrrr@{}", ["Round", "Rooms", "Coders agree on", "", "Kappa"],
          [[k, d["rooms"], d["agree"], f"{round(100 * d['agree'] / d['rooms'])}\\%", f"{d['kappa']:.2f}"] for k, d in kap.items()])

    # ---------------------------------------------------------------- figures
    if not a.no_figs:
        import matplotlib as mpl
        mpl.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib import font_manager
        have = {f.name for f in font_manager.fontManager.ttflist}
        fam = next((f for f in ("Arial", "Helvetica", "DejaVu Sans") if f in have), "sans-serif")
        mpl.rcParams.update({"font.family": fam, "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
                             "legend.fontsize": 8, "axes.edgecolor": AXIS, "axes.linewidth": 0.6, "axes.spines.top": False,
                             "axes.spines.right": False, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
                             "xtick.major.size": 0, "ytick.major.size": 0, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
                             "pdf.fonttype": 42, "legend.frameon": False})
        # Figure 1: the ladder. Two measures on one scale (per cent of votes, per cent of rooms), one row per step.
        steps = [("Stake as a fact", "T1"), ("+ heavier loss,\nvotes not reviewed", "T2"), ("+ a wish", "T3"), ("+ both", "T6"),
                 ("+ a stated intention", "T7"), ("Told to collude\n(control)", "I1")]
        fig, ax = plt.subplots(figsize=(6.3, 3.2))
        h = 0.34
        for i, (lab, v) in enumerate(steps):
            y = len(steps) - 1 - i
            lo, hi = wilson(S[v]["tv"], S[v]["n"])
            for off, key, n, col in ((h / 2 + 0.02, "tv", "n", BLUE), (-h / 2 - 0.02, "col", "rooms", ORANGE)):
                val = 100 * S[v][key] / S[v][n]
                ax.barh(y + off, val, height=h, color=col, linewidth=0)
                tip = max(val, 100 * hi) if key == "tv" else val        # the label sits past the interval, never on it
                ax.text(tip + 1.2, y + off, f"{S[v][key]} of {S[v][n]}", va="center", ha="left", fontsize=7.5, color=INK2)
            ax.plot([100 * lo, 100 * hi], [y + h / 2 + 0.02] * 2, color=INK, linewidth=0.8, solid_capstyle="butt")
        ax.set_yticks(range(len(steps))); ax.set_yticklabels([s for s, _ in reversed(steps)])
        ax.set_xlim(0, 100); ax.set_xticks([0, 25, 50, 75, 100]); ax.set_xticklabels(["0", "25", "50", "75", "100%"])
        ax.xaxis.grid(True); ax.spines["left"].set_visible(False)
        ax.legend(handles=[mpl.patches.Patch(color=BLUE, label="Votes for the stake option (line: 95% interval)"),
                           mpl.patches.Patch(color=ORANGE, label="Rooms labelled collusion")],
                  loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=2, handlelength=1.2, columnspacing=1.6, borderaxespad=0.2)
        fig.tight_layout(); fig.savefig(DOC / "figs/fig_ladder.pdf"); plt.close(fig)
        # Figure 2: how the 50 scenarios split under the intention (number of scenarios by votes for the stake option)
        fig, ax = plt.subplots(figsize=(6.3, 2.3))
        xs = list(range(9)); ys = [dist.get(k, 0) for k in xs]
        ax.bar(xs, ys, width=0.62, color=BLUE, linewidth=0)
        for x, yv in zip(xs, ys):
            if yv:
                ax.text(x, yv + 0.4, str(yv), ha="center", va="bottom", fontsize=7.5, color=INK2)
        ax.set_xticks(xs); ax.set_xlabel("Votes for the stake option in a scenario, out of 8 (two runs, four members)")
        ax.set_ylabel("Scenarios"); ax.yaxis.grid(True); ax.set_ylim(0, max(ys) + 3)
        fig.tight_layout(); fig.savefig(DOC / "figs/fig_scenarios.pdf"); plt.close(fig)

    print(json.dumps({"rooms": len(rooms), "episodes": N["episodes"], "votes": N["votes"], "lean": lean, "scenBoth": N["scenBoth"],
                      "T7b": {k: S["T7b"][k] for k in ("tv", "n", "col", "clean", "gc", "both")}, "T7r": {k: S["T7r"][k] for k in ("tv", "col", "clean")},
                      "T0": {k: S["T0"][k] for k in ("tv", "n", "col", "clean")}, "public": sorted(PUB), "features": {f: (d["yes"], d["no"]) for f, d in FE.items()}}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
