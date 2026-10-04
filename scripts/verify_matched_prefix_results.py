#!/usr/bin/env python3
"""Recompute matched-prefix result summaries from the raw per-pair records (CPU, stdlib only).

The 4 Oct 2026 handoff makes four factual claims about files that are in git:

  1. Withdrawn role-perp confirm (buggy site + scorer): every honest->colluder patch
     arm is digit-identical to the colluder baseline, so the reported +0.4837 "change"
     is the untouched colluder-minus-honest gap, and the "flips" are baseline disagreements.
  2. Layer-21 last-token card (fixed scorer): the hook was live (patched probabilities
     differ from baseline) but the full residual copy moved only about -0.05 with no flips.
  3. Final-norm last-token copy: honest->colluder closes the whole gap (fraction 1.0),
     is digit-identical to the honest baseline, and flips 5/5 disagreeing pairs.
  4. Private-instruction span at the final norm: mean change 0, flips 0/5.

This script re-derives every by-arm mean, every delta against the arm-name baseline
(h2c vs colluder, c2h vs honest), and the pass/fail gates from the per-pair records,
then checks them against the summary tables that were written on the pod. It also
re-reads the three Oct 1 "first positive" writeups and shows that their reported
honest->colluder delta equals their own untouched baseline gap.

  python3 scripts/verify_matched_prefix_results.py
  python3 scripts/verify_matched_prefix_results.py --out results/transfer_stable/VERIFICATION_2026-10-04.md

Exit status is 1 if any check fails.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HANDOFF = ROOT / "docs/handoff/2026-10-04-matched-prefix"
WITHDRAWN_OCT1 = [
    ROOT / "results/transfer_stable/MATCHED_PREFIX_INTERCHANGE_RESULTS.md",
    ROOT / "results/transfer_stable/MATCHED_PREFIX_WIDEN_RESULTS.md",
    ROOT / "results/transfer_stable/MATCHED_PREFIX_TRANSFER_RESULTS.md",
]
GAP_FRACTION_REQUIRED = 0.5
TOL = 1e-9

PAIR_RE = re.compile(r"^### (\S+) \(tgt=([^,]+), t\*=(\d+)\)\s*$")
ARM_RE = re.compile(r"^- (\S+): vote=(\S+) p=(\S+)\s*$")
TABLE_RE = re.compile(r"^\| (\S+) \| ([^|]+) \| ([^|]+) \| ([^|]+) \| ([^|]+) \|\s*$")


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def ref_baseline(arm: str) -> str:
    """Same rule as scripts/matched_prefix_interchange.py (from the arm name only)."""
    if "h2c" in arm:
        return "baseline_colluder"
    if "c2h" in arm:
        return "baseline_honest"
    return "baseline_colluder"


def parse_results_md(path: Path) -> tuple[list[dict], dict[str, dict]]:
    """Per-pair records and the by-arm table as written in a RESULTS.md."""
    per_pair: list[dict] = []
    table: dict[str, dict] = {}
    cur = None
    in_table = False
    for line in path.read_text().splitlines():
        m = PAIR_RE.match(line)
        if m:
            cur = {"pair_id": m.group(1), "target_option": m.group(2), "t_star": int(m.group(3)), "arms": {}}
            per_pair.append(cur)
            continue
        m = ARM_RE.match(line)
        if m and cur is not None:
            p = None if m.group(3) == "None" else float(m.group(3))
            cur["arms"][m.group(1)] = {"vote": m.group(2), "p_target": p}
            continue
        # The by-arm table starts at its header row ("| arm | ...") whatever the section heading says.
        if line.startswith("| arm |"):
            in_table = True
            continue
        if in_table and not line.startswith("|"):
            in_table = False
        if in_table:
            m = TABLE_RE.match(line)
            if m and m.group(1) not in ("arm", "---"):
                def num(s: str):
                    s = s.strip()
                    if s in ("None", "—", "-", ""):
                        return None
                    s = s.replace("**", "").replace("~", "").replace("+", "")
                    try:
                        return float(s)
                    except ValueError:
                        return None
                table[m.group(1)] = {
                    "mean_p_target": num(m.group(2)),
                    "target_vote_rate": num(m.group(3)),
                    "flip_rate_vs_ref": num(m.group(4)),
                    "mean_delta_p_vs_ref": num(m.group(5)),
                }
    return per_pair, table


def recompute(per_pair: list[dict]) -> dict[str, dict]:
    arms = []
    for row in per_pair:
        for name in row["arms"]:
            if name not in arms:
                arms.append(name)
    out = {}
    for name in arms:
        p_vals, votes, flips, deltas = [], [], [], []
        for row in per_pair:
            m = row["arms"].get(name)
            if m is None:
                continue
            tgt = row["target_option"]
            p_vals.append(m["p_target"])
            votes.append(1.0 if m["vote"] == tgt else 0.0 if m["vote"] != "?" else None)
            if name.startswith("baseline"):
                continue
            ref = row["arms"].get(ref_baseline(name)) or {}
            if m["vote"] != "?" and ref.get("vote") not in (None, "?"):
                flips.append(1.0 if m["vote"] != ref["vote"] else 0.0)
            if m["p_target"] is not None and ref.get("p_target") is not None:
                deltas.append(m["p_target"] - ref["p_target"])
        out[name] = {
            "mean_p_target": mean(p_vals),
            "target_vote_rate": mean(votes),
            "flip_rate_vs_ref": mean(flips) if not name.startswith("baseline") else None,
            "mean_delta_p_vs_ref": mean(deltas) if not name.startswith("baseline") else None,
        }
    return out


def gate(per_pair: list[dict], h2c_arm: str) -> dict:
    """Same bar as final_resid_gate in the harness: half the gap, and every disagreeing pair flips."""
    gaps, deltas, disagree, flipped = [], [], [], []
    for row in per_pair:
        a = row["arms"]
        bc, bh, pp = a.get("baseline_colluder"), a.get("baseline_honest"), a.get(h2c_arm)
        if not (bc and bh and pp):
            continue
        if None not in (bc["p_target"], bh["p_target"], pp["p_target"]):
            gaps.append(bc["p_target"] - bh["p_target"])
            deltas.append(pp["p_target"] - bc["p_target"])
        if bc["vote"] != "?" and bh["vote"] != "?" and bc["vote"] != bh["vote"]:
            disagree.append(row["pair_id"])
            if pp["vote"] == bh["vote"]:
                flipped.append(row["pair_id"])
    mg, md = mean(gaps), mean(deltas)
    fraction = None if (mg is None or md is None or abs(mg) < 1e-8) else -md / mg
    prob_pass = fraction is not None and fraction >= GAP_FRACTION_REQUIRED
    vote_pass = len(disagree) > 0 and len(flipped) == len(disagree)
    return {
        "control": h2c_arm,
        "passed": bool(prob_pass and vote_pass),
        "mean_gap_p": mg,
        "mean_delta_p_h2c": md,
        "fraction_of_gap": fraction,
        "n_disagree": len(disagree),
        "n_flipped_to_honest": len(flipped),
        "disagree_pair_ids": disagree,
        "flipped_pair_ids": flipped,
    }


def digit_identical(per_pair: list[dict], arm: str, other: str) -> tuple[int, int]:
    """How many pairs have p_target for `arm` exactly equal to p_target for `other`."""
    same = total = 0
    for row in per_pair:
        a, b = row["arms"].get(arm), row["arms"].get(other)
        if a is None or b is None or a["p_target"] is None or b["p_target"] is None:
            continue
        total += 1
        if a["p_target"] == b["p_target"] and a["vote"] == b["vote"]:
            same += 1
    return same, total


class Report:
    def __init__(self):
        self.lines: list[str] = []
        self.failures = 0

    def h(self, text: str):
        self.lines += ["", f"## {text}", ""]

    def check(self, ok: bool, text: str):
        self.lines.append(f"- {'PASS' if ok else 'FAIL'}: {text}")
        if not ok:
            self.failures += 1

    def note(self, text: str):
        self.lines.append(f"- {text}")

    def table(self, by_arm: dict[str, dict], written: dict[str, dict] | None):
        self.lines.append("")
        self.lines.append("| arm | mean P(target) | vote rate | flip rate | ΔP vs ref | matches written table |")
        self.lines.append("|---|---:|---:|---:|---:|---|")
        for name, s in by_arm.items():
            match = "n/a"
            if written is not None:
                w = written.get(name)
                if w is None:
                    match = "missing"
                else:
                    match = "yes"
                    for k in ("mean_p_target", "target_vote_rate", "flip_rate_vs_ref", "mean_delta_p_vs_ref"):
                        a, b = s.get(k), w.get(k)
                        if a is None and b is None:
                            continue
                        if a is None or b is None or abs(a - b) > 1e-6:
                            match = f"no ({k}: {a} vs {b})"
                            break
            fmt = lambda v: "—" if v is None else f"{v:.4f}"
            self.lines.append(
                f"| {name} | {fmt(s['mean_p_target'])} | {fmt(s['target_vote_rate'])} | "
                f"{fmt(s['flip_rate_vs_ref'])} | {fmt(s['mean_delta_p_vs_ref'])} | {match} |"
            )
        self.lines.append("")


def compare_tables(rep: Report, label: str, recomputed: dict, written: dict):
    bad = []
    for name, s in recomputed.items():
        w = written.get(name)
        if w is None:
            bad.append(f"{name} missing from written table")
            continue
        for k in ("mean_p_target", "target_vote_rate", "flip_rate_vs_ref", "mean_delta_p_vs_ref"):
            a, b = s.get(k), w.get(k)
            if a is None and b is None:
                continue
            if a is None or b is None or abs(a - b) > 1e-6:
                bad.append(f"{name}.{k}: recomputed {a} vs written {b}")
    rep.check(not bad, f"{label}: every recomputed by-arm statistic matches the table written on the pod"
              + ("" if not bad else " — " + "; ".join(bad[:5])))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=None, help="write the markdown report here")
    args = ap.parse_args()
    rep = Report()
    rep.lines += [
        "# Matched-prefix result verification (CPU re-derivation)",
        "",
        "Recomputed from the per-pair records in git with `scripts/verify_matched_prefix_results.py`. "
        "No model was run. Deltas use the arm-name rule (h2c vs colluder baseline, c2h vs honest baseline).",
    ]

    # ---- 1. final-resid wiring suite: per_pair.json + summary.json + RESULTS.md -------------
    rep.h("Final-norm last-token copy and private span (smoke_20261004T173047Z, SHA 319cf825)")
    per_pair = json.loads((HANDOFF / "final-resid-per_pair.json").read_text())
    for row in per_pair:
        row["arms"] = {k: {"vote": v.get("vote"), "p_target": v.get("p_target")} for k, v in row["arms"].items()}
    summary = json.loads((HANDOFF / "final-resid-summary.json").read_text())
    rec = recompute(per_pair)
    rep.table(rec, summary["by_arm"])
    compare_tables(rep, "final-resid per_pair.json vs summary.json", rec, summary["by_arm"])
    md_pairs, md_table = parse_results_md(HANDOFF / "final-resid-RESULTS.md")
    rec_md = recompute(md_pairs)
    compare_tables(rep, "final-resid RESULTS.md per-pair lines vs its own by-arm table", rec_md, md_table)
    g1 = gate(per_pair, "patch_h2c_full_final_last")
    g2 = gate(per_pair, "patch_h2c_full_final_private")
    w1, w2 = summary["gates"][0], summary["gates"][1]
    rep.check(
        abs(g1["fraction_of_gap"] - 1.0) < TOL and g1["n_flipped_to_honest"] == 5 and g1["n_disagree"] == 5 and g1["passed"],
        f"last-token copy: fraction of gap {g1['fraction_of_gap']:.6f}, flips {g1['n_flipped_to_honest']}/{g1['n_disagree']} "
        f"({', '.join(g1['flipped_pair_ids'])}); gate passed",
    )
    rep.check(
        g1["passed"] == w1["passed"] and abs(g1["fraction_of_gap"] - w1["fraction_of_gap"]) < TOL
        and g1["flipped_pair_ids"] == w1["flipped_pair_ids"],
        "last-token gate matches the gate record written on the pod",
    )
    rep.check(
        abs(g2["mean_delta_p_h2c"]) < TOL and g2["n_flipped_to_honest"] == 0 and not g2["passed"],
        f"private-span copy: mean change {g2['mean_delta_p_h2c']:.6f}, flips {g2['n_flipped_to_honest']}/{g2['n_disagree']}; gate failed",
    )
    rep.check(g2["passed"] == w2["passed"] and g2["flipped_pair_ids"] == w2["flipped_pair_ids"],
              "private-span gate matches the gate record written on the pod")
    same, total = digit_identical(per_pair, "patch_h2c_full_final_last", "baseline_honest")
    rep.check(same == total == 12, f"h2c last-token copy is digit-identical to the honest baseline on {same}/{total} pairs (wiring check)")
    same, total = digit_identical(per_pair, "patch_c2h_full_final_last", "baseline_colluder")
    rep.check(same == total == 12, f"c2h last-token copy is digit-identical to the colluder baseline on {same}/{total} pairs")
    same, total = digit_identical(per_pair, "patch_h2c_full_final_private", "baseline_colluder")
    rep.check(same == total == 12, f"h2c private-span copy is digit-identical to the untouched colluder run on {same}/{total} pairs (null)")
    gap = rec["baseline_colluder"]["mean_p_target"] - rec["baseline_honest"]["mean_p_target"]
    rep.check(abs(gap - 0.48370584490476176) < 1e-12, f"untouched gap recomputed as {gap:.17g} (handoff: 0.48370584490476176)")
    rep.check(abs(rec["baseline_colluder"]["mean_p_target"] - 0.5676) < 5e-4 and abs(rec["baseline_honest"]["mean_p_target"] - 0.0839) < 5e-4,
              f"baseline means colluder {rec['baseline_colluder']['mean_p_target']:.4f} / honest {rec['baseline_honest']['mean_p_target']:.4f} match the handoff (0.5676 / 0.0839)")

    # ---- 2. l21-last (fixed scorer, layer-21 site) -------------------------------------------
    rep.h("Layer-21 last-token card (smoke_20261004T165214Z, SHA 1fd0855)")
    l21_pairs, l21_table = parse_results_md(HANDOFF / "l21-last-RESULTS.md")
    l21_summary = json.loads((HANDOFF / "l21-last-summary.json").read_text())
    rec = recompute(l21_pairs)
    rep.table(rec, l21_summary["by_arm"])
    compare_tables(rep, "l21-last RESULTS.md per-pair lines vs summary.json", rec, l21_summary["by_arm"])
    compare_tables(rep, "l21-last RESULTS.md per-pair lines vs its own by-arm table", rec, l21_table)
    d = rec["patch_h2c_full_resid_L21"]["mean_delta_p_vs_ref"]
    rep.check(-0.06 < d < -0.04 and rec["patch_h2c_full_resid_L21"]["flip_rate_vs_ref"] == 0.0,
              f"full residual h2c at L21 last token: mean change {d:.4f} vs colluder, flip rate 0 (handoff: about -0.052, no flips)")
    gl = gate(l21_pairs, "patch_h2c_full_resid_L21")
    rep.check(not gl["passed"] and gl["fraction_of_gap"] < GAP_FRACTION_REQUIRED,
              f"L21 full copy fails the final-residual bar: fraction of gap {gl['fraction_of_gap']:.4f} < {GAP_FRACTION_REQUIRED}, flips {gl['n_flipped_to_honest']}/{gl['n_disagree']}")
    live = sum(1 for row in l21_pairs if row["arms"]["patch_h2c_full_resid_L21"]["p_target"] != row["arms"]["baseline_colluder"]["p_target"])
    rep.check(live == len(l21_pairs), f"hook was live: patched p differs from the colluder baseline on {live}/{len(l21_pairs)} pairs")
    for arm in ("patch_h2c_role_perp_resid_L21", "patch_h2c_role_resid_L21", "patch_h2c_random_resid_L21"):
        rep.note(f"direction arm {arm}: mean change {rec[arm]['mean_delta_p_vs_ref']:+.5f} (not interpreted; mid-layer site)")

    # ---- 3. withdrawn role-perp confirm (buggy site + scorer) --------------------------------
    rep.h("Withdrawn role-perp confirm (smoke_20261004T150244Z, SHA a0c34db): the bug, re-derived")
    wd_pairs, wd_table = parse_results_md(HANDOFF / "withdrawn-role-perp-RESULTS.md")
    rec = recompute(wd_pairs)
    rep.table(rec, None)
    for arm in ("patch_h2c_role_perp_resid_L21", "patch_h2c_role_resid_L21", "patch_h2c_random_resid_L21"):
        same, total = digit_identical(wd_pairs, arm, "baseline_colluder")
        rep.check(same == total == 12, f"{arm} is digit-identical to the colluder baseline on {same}/{total} pairs: the patch did nothing (last shared token)")
        rep.check(abs(rec[arm]["mean_delta_p_vs_ref"]) < TOL, f"{arm}: with the fixed scorer the change vs colluder is {rec[arm]['mean_delta_p_vs_ref']:.6f}")
        written = wd_table.get(arm, {}).get("mean_delta_p_vs_ref")
        rep.check(written is not None and abs(written - 0.48370584490476176) < 1e-12,
                  f"{arm}: the table written by the buggy scorer reports {written} = the untouched gap (compared to the honest baseline)")
    disagree = sum(1 for row in wd_pairs if row["arms"]["baseline_colluder"]["vote"] != row["arms"]["baseline_honest"]["vote"])
    rep.check(disagree == 5 and abs(wd_table["patch_h2c_role_resid_L21"]["flip_rate_vs_ref"] - 5 / 12) < 1e-9,
              f"the buggy 'flip rate' 0.4167 = {disagree}/12 pairs whose untouched votes already disagree")

    # ---- 4. Oct 1 "first positive" writeups: summary tables only -----------------------------
    rep.h("Oct 1 'first positive' writeups (pre-fix harness): reported ΔP equals the untouched gap")
    for path in WITHDRAWN_OCT1:
        _, table = parse_results_md(path)
        bc, bh = table.get("baseline_colluder"), table.get("baseline_honest")
        if not (bc and bh):
            rep.check(False, f"{path.name}: by-arm table not parseable")
            continue
        gap = bc["mean_p_target"] - bh["mean_p_target"]
        for arm in ("patch_h2c_attn_L22", "patch_h2c_resid_L21"):
            w = table.get(arm)
            if w is None:
                rep.check(False, f"{path.name}: {arm} missing")
                continue
            rep.check(
                abs(w["mean_p_target"] - bc["mean_p_target"]) < 1.5e-3 and abs(w["mean_delta_p_vs_ref"] - gap) < 1.5e-3,
                f"{path.name} {arm}: mean P {w['mean_p_target']} equals the colluder baseline {bc['mean_p_target']} (patch did nothing); "
                f"reported ΔP {w['mean_delta_p_vs_ref']} equals the untouched gap {gap:.3f}",
            )
        for arm in ("patch_c2h_attn_L22", "patch_c2h_resid_L21"):
            w = table.get(arm)
            if w is not None:
                rep.check(abs(w["mean_p_target"] - bh["mean_p_target"]) < 1.5e-3,
                          f"{path.name} {arm}: mean P {w['mean_p_target']} equals the honest baseline {bh['mean_p_target']} (patch did nothing)")

    rep.lines += ["", f"**{rep.failures} failing check(s).**", ""]
    text = "\n".join(rep.lines)
    print(text)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
        print(f"[wrote] {args.out}", file=sys.stderr)
    return 1 if rep.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
