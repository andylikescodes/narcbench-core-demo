#!/usr/bin/env python3
"""Cross-check the frozen claims sheet against the tables and metric files in git (CPU, stdlib only).

Every number in results/transfer_stable/WEEKEND_REVIEW/CLAIMS_NUMBER_SHEET.json is compared,
at three decimals, with the publication tables under WEEKEND_REVIEW/tables/ and with the raw
probe metrics under results/transfer/. The HARD markdown table is parsed too. As a soft check,
each frozen headline number is looked up in the paper source (papers/.../main.tex).

Sources that the claims sheet names but that are not in git (results/transfer_stable/layer_sweep.json,
MLP_ATTN_PROBES.md, CONTROLS_OFFLINE.md, RESIDUALIZE_MODE.md, the causal smoke writeups) cannot
be re-derived here; their table copies under WEEKEND_REVIEW/tables/ are the citable record.

  python3 scripts/verify_claims_sheet.py

Exit status is 1 if any hard check fails.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WR = ROOT / "results/transfer_stable/WEEKEND_REVIEW"
TABLES = WR / "tables"
TRANSFER = ROOT / "results/transfer"
TEX = ROOT / "papers/transfer-stable-collusion-2026-10/main.tex"

HARD_DIRS = {
    "full_acts_band_L19-23": "FULL_core_to_transfer_full",
    "W_U_k25_J_only": "FULL_core_to_transfer_wu_k25_j_only",
    "jac_k25_J_only": "FULL_core_to_transfer_jac_k25_j_only",
    "W_U_k25_complement": "FULL_core_to_transfer_wu_k25_complement",
    "jac_k25_complement": "FULL_core_to_transfer_jac_k25_complement",
}
HARD_MD_LABELS = {
    "full acts": "full_acts_band_L19-23",
    "W_U k25 J-only": "W_U_k25_J_only",
    "W_U k25 complement": "W_U_k25_complement",
    "jac k25 J-only": "jac_k25_J_only",
    "jac k25 complement": "jac_k25_complement",
}
METRIC_KEYS = {"primary": "auroc_primary", "peak": "auroc_peak_suspicion", "split": "auroc_deception_split", "asym": "auroc_asymmetry_probe"}

failures = 0
soft = 0


def r3(x):
    return None if x is None else round(float(x) + 1e-12, 3)


def check(ok: bool, text: str, hard: bool = True):
    global failures, soft
    tag = "PASS" if ok else ("FAIL" if hard else "WARN")
    print(f"- {tag}: {text}")
    if not ok:
        if hard:
            failures += 1
        else:
            soft += 1


def same(a, b, tol=5e-4):
    return a is not None and b is not None and abs(float(a) - float(b)) <= tol


def main() -> int:
    sheet = json.loads((WR / "CLAIMS_NUMBER_SHEET.json").read_text())
    print("# Claims sheet verification\n")

    # A. HARD table: sheet vs tables/hard_metrics.json vs raw metrics.json vs markdown table
    print("## A. HARD J vs full vs complement (n=72)")
    hard_tbl = {r["arm"]: r for r in json.loads((TABLES / "hard_metrics.json").read_text())["rows"]}
    summary = json.loads((TRANSFER / "FULL_HARD_SUMMARY.json").read_text())
    md_rows = {}
    for line in (TRANSFER / "TRANSFER_FULL_HARD_METRICS.md").read_text().splitlines():
        m = re.match(r"^\| ([^|]+) \| ([\d.]+) \| \*\*([\d.]+)\*\* \| ([\d.]+) \| ([\d.]+) \| ([\d.]+) \|", line)
        if m:
            md_rows[m.group(1).strip()] = dict(smoke=float(m.group(2)), primary=float(m.group(3)), peak=float(m.group(4)), split=float(m.group(5)), asym=float(m.group(6)))
    for row in sheet["hard_j_vs_full_vs_complement"]["rows"]:
        arm = row["arm"]
        raw = json.loads((TRANSFER / HARD_DIRS[arm] / "metrics.json").read_text())
        raw_sum = summary["metrics"][HARD_DIRS[arm]]
        for k, mk in METRIC_KEYS.items():
            check(same(row[k], r3(raw[mk])), f"{arm}.{k}: sheet {row[k]} = metrics.json {r3(raw[mk])}")
            check(same(row[k], r3(raw_sum[mk])), f"{arm}.{k}: sheet {row[k]} = FULL_HARD_SUMMARY.json {r3(raw_sum[mk])}")
            check(same(row[k], hard_tbl[arm][k]), f"{arm}.{k}: sheet {row[k]} = tables/hard_metrics.json {hard_tbl[arm][k]}")
        md_label = next(lbl for lbl, a in HARD_MD_LABELS.items() if a == arm)
        md = md_rows.get(md_label)
        check(md is not None and all(same(row[k], md[k]) for k in METRIC_KEYS), f"{arm}: markdown HARD table row matches ({md})")
    check(summary["n_transfer_runs"] == 72 and sheet["setup"]["n_transfer_runs_hard"] == 72, "n_transfer_runs = 72 in summary and sheet")

    # B. residual layer sweep: sheet vs tables/layer_sweep_residual.json vs per_layer of the full metrics
    print("\n## B. Residual layer sweep")
    sweep = {r["arm"]: r for r in json.loads((TABLES / "layer_sweep_residual.json").read_text())["rows"]}
    full_raw = json.loads((TRANSFER / "FULL_core_to_transfer_full/metrics.json").read_text())
    per_layer = {int(r["layer"]): r for r in full_raw["per_layer"]}
    for row in sheet["residual_layer_sweep"]["rows"]:
        arm = row["arm"]
        key = {"band_mean_L19-23": "band_mean_L19-23", "mean_pool_L19-23": "mean_pool_L19-23"}.get(arm, f"residual_{arm}")
        t = sweep[key]
        for k in METRIC_KEYS:
            check(same(row[k], r3(t[k])), f"{arm}.{k}: sheet {row[k]} = tables/layer_sweep_residual.json {r3(t[k])}")
        if arm.startswith("L") and arm[1:].isdigit():
            pl = per_layer[int(arm[1:])]
            for k, mk in METRIC_KEYS.items():
                check(same(row[k], r3(pl[mk])), f"{arm}.{k}: sheet {row[k]} = FULL metrics per_layer {r3(pl[mk])} (independent source)")
        if arm == "band_mean_L19-23":
            for k, mk in METRIC_KEYS.items():
                check(same(row[k], r3(full_raw[mk])), f"band mean.{k}: sheet {row[k]} = FULL metrics band {r3(full_raw[mk])}")

    # D. mlp / attn per site
    print("\n## D. MLP / attn per-site table")
    sites = {(r["site"], int(r["layer"])): r for r in json.loads((TABLES / "mlp_attn_site_aurocs.json").read_text())["rows"]}
    for row in sheet["mlp_attn_per_site"]["rows"]:
        t = sites[(row["site"], int(row["layer"]))]
        for k in ("primary", "peak", "split", "asym", "role_sample"):
            check(same(row[k], t[k]), f"{row['site']}_L{row['layer']}.{k}: sheet {row[k]} = table {t[k]}")
    attn = json.loads((TABLES / "cross_phase_attn.json").read_text())
    attn22 = next(r for r in attn["attn_cross_phase"] if r["layer"] == 22 and r["transfer_phase"] == "final")
    check(same(sheet["mlp_attn_per_site"]["json_raw_attn_L22_primary"], attn22["auroc_primary"], 1e-9),
          f"attn_L22 final raw primary {attn22['auroc_primary']} = sheet json_raw value")
    for fam, v in attn22["per_family"].items():
        check(v["peak"] == 1.0 and v["split"] == 1.0, f"attn_L22 final per-family {fam}: peak {v['peak']} split {v['split']}")

    # E. controls + residualize
    print("\n## E. Controls and residualize")
    ctl = {(int(r["layer"]), r["metric"]): r["auroc"] for r in json.loads((TABLES / "controls_residualize.json").read_text())["rows"]}
    for row in sheet["controls_role_mode"]["cross_tests"]:
        v = row.get("auroc", row.get("value"))
        check(same(v, ctl.get((row["layer"], row["metric"]))), f"L{row['layer']} {row['metric']}: sheet {v} = table {ctl.get((row['layer'], row['metric']))}")
    for row in sheet["residualize_mode"]["rows"]:
        L = row["layer"]
        check(same(row["role_perp_to_role"], ctl[(L, "role_perp→role")]) and same(row["role_perp_to_mode"], ctl[(L, "role_perp→mode")])
              and same(row["role_to_role"], ctl[(L, "role→role")]),
              f"L{L} residualize row matches controls_residualize.json")

    # F. causal nulls
    print("\n## F. Causal nulls")
    nulls = {r["battery"]: r for r in json.loads((TABLES / "causal_nulls.json").read_text())["rows"]}
    for row in sheet["causal_nulls"]["rows"]:
        t = nulls[row["battery"]]
        check(same(row["collusion_success_all_arms"], t["collusion_success_all_arms"]) and row["flips"] == t["flips_vs_baseline"] == 0,
              f"battery {row['battery']} ({row['name']}): success {row['collusion_success_all_arms']}, flips 0")

    # G. cross-phase, domain holdout, family
    print("\n## G. Science-freeze offline rows")
    cp = sheet["cross_phase_new_offline"]
    disc = json.loads((TABLES / "cross_phase_discussion_residual/metrics.json").read_text())
    pl = {int(r["layer"]): r for r in disc["per_layer"]}
    check(same(cp["residual_band_primary"], r3(disc["auroc_primary"])), f"cross-phase residual band {cp['residual_band_primary']} = {r3(disc['auroc_primary'])}")
    check(same(cp["residual_L21_primary"], r3(pl[21]["auroc_primary"])), f"cross-phase residual L21 {cp['residual_L21_primary']} = {r3(pl[21]['auroc_primary'])}")
    check(same(cp["residual_L22_primary"], r3(pl[22]["auroc_primary"])), f"cross-phase residual L22 {cp['residual_L22_primary']} = {r3(pl[22]['auroc_primary'])}")
    a21d = next(r for r in attn["attn_cross_phase"] if r["layer"] == 21 and r["transfer_phase"] == "discussion")
    a22d = next(r for r in attn["attn_cross_phase"] if r["layer"] == 22 and r["transfer_phase"] == "discussion")
    check(same(cp["attn_L22_discussion_primary"], r3(a22d["auroc_primary"])), f"cross-phase attn L22 discussion {cp['attn_L22_discussion_primary']} = {r3(a22d['auroc_primary'])}")
    check(same(cp["attn_L21_discussion_primary"], r3(a21d["auroc_primary"])), f"cross-phase attn L21 discussion {cp['attn_L21_discussion_primary']} = {r3(a21d['auroc_primary'])}")
    dh = sheet["domain_holdout_new_offline"]
    hold = json.loads((TABLES / "domain_holdout_L21.json").read_text())
    held = [r["primary"] for r in hold["rows"] if r["held_domain"] is not None]
    check(len(held) == 17 == dh["n_core_domains"], f"17 held-out Core domains ({len(held)} rows)")
    check(same(dh["holdout_mean"], r3(sum(held) / len(held))), f"holdout mean {dh['holdout_mean']} = recomputed {r3(sum(held) / len(held))}")
    check(same(dh["holdout_min"], r3(min(held))) and same(dh["holdout_max"], r3(max(held))), f"holdout min/max {dh['holdout_min']}/{dh['holdout_max']} = recomputed {r3(min(held))}/{r3(max(held))}")
    base = next(r for r in hold["rows"] if r["held_domain"] is None)
    check(same(dh["baseline_primary"], r3(base["primary"])) and same(dh["baseline_primary"], r3(per_layer[21]["auroc_primary"])),
          f"holdout baseline L21 {dh['baseline_primary']} = full-fit {r3(base['primary'])} = FULL metrics L21 {r3(per_layer[21]['auroc_primary'])}")
    fam = json.loads((TABLES / "family_auroc_L21.json").read_text())["per_family"]
    for row in sheet["family_auroc_L21"]["rows"]:
        f = fam[row["family"]]
        check(same(row["peak"], f["peak"]) and same(row["split"], f["split"]) and row["n"] == f["n"] == 12,
              f"family {row['family']}: peak {row['peak']} split {row['split']} n {row['n']}")
    fam_full = per_layer[21]["per_family"]
    check(all(same(fam[k]["peak"], fam_full[k]["peak"]) and same(fam[k]["split"], fam_full[k]["split"]) for k in fam),
          "family table at L21 equals the per-family block of the FULL metrics (independent source)")

    # H. paper source carries the headline numbers (soft)
    print("\n## H. Paper source (soft presence check)")
    tex = TEX.read_text()
    for label, val in [("HARD full", "0.902"), ("W_U J-only", "0.602"), ("jac J-only", "0.549"), ("W_U complement", "0.898"),
                       ("jac complement", "0.891"), ("residual L21", "0.941"), ("residual L22", "0.930"), ("attn L22", "0.999"),
                       ("role→role L21", "0.859"), ("role_perp→role L21", "0.796"), ("cross-phase band", "0.959"),
                       ("attn L22 discussion", "0.997"), ("holdout mean", "0.939"), ("PCA contrast k8", "0.964")]:
        check(val in tex, f"{label} {val} appears in main.tex", hard=False)
    check("matched_prefix" not in tex.lower().replace("\\_", "_").replace("-", "_") or "MATCHED_PREFIX" not in tex,
          "main.tex does not cite the withdrawn Oct 1 matched-prefix tables", hard=False)

    print(f"\n**{failures} hard failure(s), {soft} soft warning(s).**")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
