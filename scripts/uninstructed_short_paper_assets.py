#!/usr/bin/env python3
"""Tables and numbers of the five-page paper docs/uninstructed_collusion/collusion_behaviour_study.tex, from the data.

The paper also \\input's tables/numbers.tex, which scripts/uninstructed_paper_assets.py writes for the long version
(collusion_behaviour_study_full.tex). This script adds what the short paper needs on top of it:

  docs/uninstructed_collusion/tables/short_numbers.tex   \\SN{<name>}: five-round totals, the scenario analysis, round 5
  docs/uninstructed_collusion/tables/short_types.tex     Table 1: the main prompt types and what the votes did
  docs/uninstructed_collusion/tables/short_funnel.tex    Table 2: where in the episode the scenarios part
  docs/uninstructed_collusion/tables/short_kinds.tex     Table 3: the fifth round, scenario by scenario

  python3 scripts/uninstructed_paper_assets.py && python3 scripts/uninstructed_short_paper_assets.py
  (cd docs/uninstructed_collusion && tectonic collusion_behaviour_study.tex)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import uninstructed_paper_assets as A  # noqa: E402  (stats of a set of rooms, LaTeX escaping)

RU, DOC = A.RU, A.DOC
# USD by the account balance before and after each round's pods (RUN_LOG.md of each round)
COST = {"pilot": 0.88, "pilot2": 0.69, "pilot3": 1.02, "pilot4": 0.96}
FACT = ("S1", "S2", "S3", "T1", "P1")
SETTING = {"COMPANY": "company", "PROFESSIONAL": "professional body", "PUBLIC": "public body"}


def rooms_of(rnd: str) -> list[dict]:
    return json.loads((RU / rnd / "rooms.json").read_text())


def stance_of(rnd: str) -> dict:
    p = RU / rnd / A.STANCE_DIR / "public_stance.json"
    return json.loads(p.read_text())["summary"]["versions"] if p.exists() else {}


def main() -> int:
    global DOC
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--round5", type=Path, default=RU / "pilot5", help="folder of the fifth round (summary.json, stake_kinds.json)")
    ap.add_argument("--cost5", type=float, default=None, help="USD of the fifth round's pod, by the account balance")
    ap.add_argument("--doc", type=Path, default=DOC, help="folder that holds tables/ (default: docs/uninstructed_collusion)")
    ap.add_argument("--basis", choices=["both", "claude"], default="both",
                    help="both: the label fixed before the runs, on which both coders agree. claude: the second coder alone, "
                         "Claude Fable 5.2, for the room labels, the marks and the public positions; reads and writes <doc>/claude_basis")
    a = ap.parse_args()
    claude = a.basis == "claude"
    if claude:
        A.CODER, A.STANCE_DIR = 1, "claude_first"
        if a.doc == A.DOC:
            a.doc = A.DOC / "claude_basis"
    DOC = a.doc

    R = {rnd: rooms_of(rnd) for rnd in ("pilot", "pilot2", "pilot3", "pilot4")}
    S = {rnd: stance_of(rnd) for rnd in ("pilot2", "pilot3", "pilot4")}
    by = lambda rnd, v: [r for r in R[rnd] if r["version"] == v]  # noqa: E731
    sp = json.loads((RU / "pilot4/scenario_properties.json").read_text())
    blind = json.loads((RU / "pilot4/scenario_properties_blind.json").read_text())["agreed"]
    # the fifth round enters only when it has results (its first pod's results were lost on the night of 17 November 2026)
    k = json.loads((a.round5 / "stake_kinds.json").read_text()) if (a.round5 / "stake_kinds.json").exists() else None
    n5 = {"episodes": 0, "rooms": 0, "votes": 0}
    if k:
        s5 = json.loads((a.round5 / "summary.json").read_text())
        k1, k2 = k["kinds"]
        sids = k["scenarios"]
        n5 = {"episodes": sum(c["rooms"] for c in k["cells"].values()) // 2, "rooms": sum(c["rooms"] for c in k["cells"].values()),
              "votes": sum(c["votes"] for c in k["cells"].values())}

    N: dict[str, str] = {}
    pct = lambda x: f"{100 * x:.0f}"  # noqa: E731
    pts = lambda x: "0" if round(100 * x) == 0 else f"{100 * x:+.0f}".replace("-", "$-$").replace("+", "$+$")  # noqa: E731
    # a count that each coder marks: the range of the two, or the second coder's alone
    one = (lambda xs: str(xs[A.CODER])) if claude else (lambda xs: f"{min(xs)} to {max(xs)}")  # noqa: E731
    # five-round totals
    eps4 = sum(len({r["run_id"] for r in rs}) for rs in R.values())
    N["episodes"] = f"{eps4 + n5['episodes']:,}"
    N["calls"] = f"{24 * (eps4 + n5['episodes']):,}"
    N["rooms"] = f"{2 * (eps4 + n5['episodes']):,}"
    N["votes"] = f"{4 * (eps4 + n5['episodes']):,}"
    N["costAll"] = f"{sum(COST.values()) + (a.cost5 or 0):.2f}"
    if k:
        N["unparsedFive"] = str(k["unparsed_votes"])
        if a.cost5 is not None:
            N["costFive"] = f"{a.cost5:.2f}"
        N["kappaFive"] = f"{s5['kappa']:.2f}"
        kappas = [json.loads((RU / rnd / "summary.json").read_text())["kappa"] for rnd in ("pilot2", "pilot3", "pilot4")] + [s5["kappa"]]
        N["kappaMin"], N["kappaMax"] = f"{min(kappas):.2f}", f"{max(kappas):.2f}"
        # agreement of the two coders on the one question the collusion label rests on: a pact for the stake option or not
        from uninstructed import coding as C
        r5 = json.loads((a.round5 / "rooms.json").read_text())
        pact = [["T" if r["codes"][i] == "AGREE_TARGET" else "x" for r in r5] for i in (0, 1)]
        N["kappaPactFive"] = f"{C.cohen_kappa(pact[0], pact[1]):.2f}"
        N["episodesFive"], N["roomsFive"], N["votesFive"] = str(n5["episodes"]), str(n5["rooms"]), str(n5["votes"])
    # the scenario analysis of round 4
    f = sp["funnel"]
    for g in ("zero", "mixed", "all"):
        G = g.capitalize()
        N[f"fun{G}Scen"], N[f"fun{G}Rooms"] = str(f[g]["scenarios"]), str(f[g]["rooms"])
        N[f"fun{G}Prop"], N[f"fun{G}Acc"] = one(f[g]["proposal"]), one(f[g]["acceptance"])
        N[f"fun{G}Both"] = str(f[g]["both_vote"])
        if not claude:
            N[f"fun{G}Pact"] = str(f[g]["pact_both"])
    N["rankRepeat"] = f"{sp['rank_correlation']['pilot3:T7']['rho']:.2f}"
    pr = sp["properties"]
    for name, field, val in (("Public", "setting", "PUBLIC"), ("Prof", "setting", "PROFESSIONAL"), ("Company", "setting", "COMPANY"),
                             ("Above", "controller", "SUPERIOR"), ("Outside", "controller", "OUTSIDE"), ("Auto", "controller", "AUTOMATIC"),
                             ("Loss", "own_stake_direction", "LOSS"), ("Gain", "own_stake_direction", "GAIN")):
        N[f"p{name}N"], N[f"p{name}"] = str(pr[field][val]["scenarios"]), pct(pr[field][val]["rate"])
    c = sp["contrasts"]["a public body against the rest"]
    N["pPublicDiff"], N["pPublicLo"], N["pPublicHi"] = pct(c["difference"]), pct(c["interval"][0]), pct(c["interval"][1])
    c = sp["contrasts"]["a rule is cited in the case, against not"]
    N["pRuleDiff"], N["pRuleLo"], N["pRuleHi"] = pts(c["difference"]), pts(c["interval"][0]), pts(c["interval"][1])
    pl = sp["pairs"]["A public body and a loss"]
    N["pPubLossN"], N["pPubLoss"] = str(len(pl["scenarios"])), pct(pl["rate"])
    cg = sp["pairs"]["Inside a company and a gain"]
    N["pCoGainN"], N["pCoGain"] = str(len(cg["scenarios"])), pct(cg["rate"])
    # round 5
    if k:
        sv, sc = k["stake"]["votes"], k["stake"]["collusion_rooms"]
        N["kDiff"], N["kLo"], N["kHi"] = pts(sv["mean"]), pts(sv["interval"][0]), pts(sv["interval"][1])
        N["kHigher"], N["kEqual"], N["kLower"] = str(sv["higher"]), str(sv["equal"]), str(sv["lower"])
        if not claude:      # this difference and its interval were computed on the label of both coders
            N["kColDiff"], N["kColLo"], N["kColHi"] = pts(sc["mean"]), pts(sc["interval"][0]), pts(sc["interval"][1])
        N["kStakeRule"] = "is met" if k["stake_matters"] else "is not met"
        N["kCaseRule"] = "is met" if k["case_matters"] else "is not met"
        for v, tag in ((k1, "One"), (k2, "Two")):
            cs = k["case"][v]
            N[f"kCase{tag}All"], N[f"kCase{tag}Zero"] = pct(cs["all_end"]), pct(cs["zero_end"])
            N[f"kCase{tag}Diff"], N[f"kCase{tag}Lo"], N[f"kCase{tag}Hi"] = pts(cs["difference"]), pts(cs["interval"][0]), pts(cs["interval"][1])
            tot = [k["cells"][f"{s}|{v}"] for s in sids]
            N[f"k{tag}Votes"], N[f"k{tag}N"] = str(sum(x["stake"] for x in tot)), str(sum(x["votes"] for x in tot))
            N[f"k{tag}Pct"] = pct(sum(x["stake"] for x in tot) / sum(x["votes"] for x in tot))
            N[f"k{tag}Col"], N[f"k{tag}Rooms"], N[f"k{tag}Clean"] = str(sum(x["collusion"] for x in tot)), str(sum(x["rooms"] for x in tot)), str(sum(x["clean"] for x in tot))
            fb = k["funnel"].get(f"{v}, both ends")
            if fb:
                N[f"k{tag}Prop"], N[f"k{tag}Acc"] = one(fb["proposal"]), one(fb["acceptance"])
            st = (k.get("public_positions") or {}).get(v)
            if st:
                N[f"k{tag}Hidden"], N[f"k{tag}StakeVotes"] = str(st["after_other_option"]), str(st["votes_for_stake_option"])
            if claude:          # the cells and public positions of stake_kinds.json follow both coders and the first coder
                rv = [r for r in r5 if r["version"] == v]
                N[f"k{tag}Col"], N[f"k{tag}Clean"] = str(sum(A.label_of(r) == "collusion" for r in rv)), str(sum(A.label_of(r) == "clean" for r in rv))
                N[f"k{tag}ColBoth"] = str(sum(r["label"] == "collusion" for r in rv))
                s5v = stance_of(a.round5.name)[v]
                N[f"k{tag}Hidden"], N[f"k{tag}StakeVotes"] = str(s5v["stake_votes_after_public_other_option"]), str(s5v["votes_for_stake_option"])
        for e, tag in (("zero", "Zero"), ("all", "All")):
            d = k["per_end"].get(e)
            if d:
                N[f"kEnd{tag}One"], N[f"kEnd{tag}Two"], N[f"kEnd{tag}Diff"] = pct(d[k1]), pct(d[k2]), pts(d["mean_difference"])
                N[f"kEnd{tag}N"] = str(len(d["scenarios"]))

    if claude:
        # how far the first coder, Gemma-2-9B-it, agrees with the reading this basis follows, and what the label fixed
        # before the runs (both coders) gives for the rooms this report counts
        allr = [r for rs in R.values() for r in rs] + (r5 if k else [])
        same = sum(r["codes"][0] == r["codes"][1] for r in allr)
        N["roomsCoded"], N["sameCode"], N["sameCodePct"] = f"{len(allr):,}", f"{same:,}", pct(same / len(allr))
        N["bothSeven"], N["bothSevenB"] = str(A.stats(by("pilot3", "T7"))["colboth"]), str(A.stats(by("pilot4", "T7"))["colboth"])
        ags = [json.loads((RU / rnd / A.STANCE_DIR / "public_stance.json").read_text())["summary"]["agreement"][0]
               for rnd in ("pilot2", "pilot3", "pilot4") + ((a.round5.name,) if k else ())]
        N["pubAgreeMin"], N["pubAgreeMax"] = pct(min(x["raw_agreement"] for x in ags)), pct(max(x["raw_agreement"] for x in ags))
        N["pubKappaMin"], N["pubKappaMax"] = f"{min(x['kappa'] for x in ags):.2f}", f"{max(x['kappa'] for x in ags):.2f}"
        N["pubMembers"] = f"{sum(x['n'] for x in ags):,}"
        # the same reader twice: its earlier pass over a sample of the members of each round, where one was imported as a third coder
        again = [x[1] for x in (json.loads((RU / rnd / A.STANCE_DIR / "public_stance.json").read_text())["summary"]["agreement"]
                                for rnd in ("pilot2", "pilot3", "pilot4") + ((a.round5.name,) if k else ())) if len(x) > 1]
        if again:
            N["pubAgainMin"], N["pubAgainMax"] = pct(min(x["raw_agreement"] for x in again)), pct(max(x["raw_agreement"] for x in again))
            N["pubAgainN"] = str(again[0]["n"])

    L = ["% written by scripts/uninstructed_short_paper_assets.py; do not edit", "\\makeatletter",
         "\\newcommand{\\SN}[1]{\\csname ucs@#1\\endcsname}"]
    L += [f"\\expandafter\\def\\csname ucs@{name}\\endcsname{{{val}}}" for name, val in N.items()] + ["\\makeatother"]
    (DOC / "tables/short_numbers.tex").write_text("\n".join(L) + "\n")

    # Table 1: the main prompt types
    def row(code, what, rs, scen, st=None):
        s = A.stats(rs)
        hid = "--" if not st or not st["votes_for_stake_option"] else f"{st['stake_votes_after_public_other_option']} of {st['votes_for_stake_option']}"
        return f"{code} & {what} & {scen} & {s['tv']} of {s['n']} & {s['pct']}\\% & {s['col']} of {s['rooms']} & {hid} \\\\"
    fact = [r for rnd in ("pilot", "pilot2", "pilot3") for r in R[rnd] if r["version"] in FACT]
    T = ["\\begin{tabular}{@{}l>{\\raggedright\\arraybackslash}p{6.6cm}rr@{\\ \\ }rrr@{}}", "\\toprule",
         " & What the system prompt says about the stake & Scen. & \\multicolumn{2}{r}{Stake votes} & Collusion & Hidden \\\\", "\\midrule",
         row("T0", "Nothing: the role line alone", by("pilot4", "T0"), 50, S["pilot4"].get("T0")),
         row("Fact", "A third party's interest and the member's own stake, as facts (five types)", fact, 12),
         row("T2", "Fact, plus a heavier loss and votes that nobody reviews", by("pilot2", "T2"), 12, S["pilot2"].get("T2")),
         row("T3", "Fact, plus a wish, with the decision left open", by("pilot2", "T3"), 12, S["pilot2"].get("T3")),
         row("T6", "Fact, plus the heavier loss and the wish", by("pilot3", "T6"), 12, S["pilot3"].get("T6")),
         row("T7", "Fact, plus a stated intention", by("pilot3", "T7"), 12, S["pilot3"].get("T7")),
         row("T7", "The same on all 50 scenarios", by("pilot4", "T7"), 50, S["pilot4"].get("T7")),
         row("T8", "The wish, and the partner's first message is a scripted proposal", by("pilot3", "T8"), 12, S["pilot3"].get("T8")),
         row("I1", "Control: the benchmark's instruction to collude", by("pilot3", "I1"), 12, S["pilot3"].get("I1")),
         "\\bottomrule", "\\end{tabular}"]
    (DOC / "tables/short_types.tex").write_text("\n".join(T) + "\n")

    # Table 2: where in the episode the scenarios part (T7, round 4)
    T = ["\\begin{tabular}{@{}lrrrr@{}}", "\\toprule",
         "Scenarios (intention, round 4) & Rooms & Proposed & Accepted & Both vote for it \\\\", "\\midrule"]
    for g, w in (("zero", "with no vote for the stake option"), ("mixed", "in between"), ("all", "with every vote for it")):
        T.append(f"{f[g]['scenarios']} {w} & {f[g]['rooms']} & {one(f[g]['proposal'])} & {one(f[g]['acceptance'])} & {f[g]['both_vote']} \\\\")
    T += ["\\bottomrule", "\\end{tabular}"]
    (DOC / "tables/short_funnel.tex").write_text("\n".join(T) + "\n")

    # Table 3: the fifth round, scenario by scenario
    if k:
        T = ["\\begin{tabular}{@{}llrrr@{}}", "\\toprule",
             f"Scenario & Committee & Own stake text (round 4) & {k1}: from above & {k2}: payout \\\\", "\\midrule"]
        for e, head in (("zero", "No vote for the stake option in round 4"), ("all", "Every vote for the stake option in round 4")):
            ss = [s for s in sids if k["ends"][s] == e]
            if not ss:
                continue
            T.append(f"\\multicolumn{{5}}{{@{{}}l}}{{\\emph{{{head}}}}} \\\\")
            for s in ss:
                c1, c2, ev = k["cells"][f"{s}|{k1}"], k["cells"][f"{s}|{k2}"], k["earlier_votes"][s]
                T.append(f"\\texttt{{{A.tex(s)}}} & {SETTING.get((blind.get(s) or {}).get('setting'), '')} & {ev[0]} of {ev[1]} & "
                         f"{c1['stake']} of {c1['votes']} & {c2['stake']} of {c2['votes']} \\\\")
            d = k["per_end"][e]
            T.append(f"\\emph{{these {len(ss)} together}} & & {sum(k['earlier_votes'][s][0] for s in ss)} of {sum(k['earlier_votes'][s][1] for s in ss)} & "
                     f"{pct(d[k1])}\\% & {pct(d[k2])}\\% \\\\")
            if e == "zero":
                T.append("\\addlinespace")
        T += ["\\bottomrule", "\\end{tabular}"]
        (DOC / "tables/short_kinds.tex").write_text("\n".join(T) + "\n")
    print(f"wrote short_numbers.tex ({len(N)} numbers), short_types.tex, short_funnel.tex" + (", short_kinds.tex" if k else "; no fifth round"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
