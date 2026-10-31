#!/usr/bin/env python3
"""Uncertainty for the frozen Gemma-2-9B J-space numbers, from the committed metrics alone.

No per-episode scores were kept for the HARD Transfer run (n = 36 collusion + 36 control episodes), so the
only honest thing to attach to each recorded AUROC is a Hanley-McNeil standard error, and to the J-only
versus complement contrast a normal-approximation test that treats the two arms as independent
(conservative for a paired contrast with positive correlation). Section 5 of
docs/jspace/JSPACE_SEPARABILITY_DESIGN.md.

  python3 scripts/jspace_offline_stats.py --out results/jspace/gemma2_9b_frozen
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARMS = {
    "full": "FULL_core_to_transfer_full",
    "W_U k25 J-only": "FULL_core_to_transfer_wu_k25_j_only",
    "W_U k25 complement": "FULL_core_to_transfer_wu_k25_complement",
    "jac k25 J-only": "FULL_core_to_transfer_jac_k25_j_only",
    "jac k25 complement": "FULL_core_to_transfer_jac_k25_complement",
}
METRICS = [("auroc_peak_suspicion", "Peak"), ("auroc_deception_split", "Split"), ("auroc_asymmetry_probe", "Asym")]
CORE_BATTERY = {  # docs/NEXT-SCIENCE.md "Core (discussion) headline", leave-one-domain-out, 50 + 50 episodes
    "wu_topk k8": (0.894, 1.000), "wu_topk k16": (0.950, 1.000), "wu_topk k25": (0.989, 1.000), "wu_topk k50": (1.000, 1.000),
    "wu_fixed k25": (0.990, 1.000), "jac_topk k8": (0.900, 1.000), "jac_topk k16": (0.919, 1.000), "jac_topk k25": (0.961, 1.000),
    "jac_topk k50": (0.997, 1.000),
}


def hanley_mcneil_se(theta: float, n1: int, n2: int) -> float:
    theta = min(max(theta, 1e-6), 1 - 1e-6)
    q1 = theta / (2 - theta)
    q2 = 2 * theta * theta / (1 + theta)
    var = (theta * (1 - theta) + (n1 - 1) * (q1 - theta**2) + (n2 - 1) * (q2 - theta**2)) / (n1 * n2)
    return math.sqrt(max(var, 0.0))


def norm_sf(z: float) -> float:
    return 0.5 * math.erfc(z / math.sqrt(2))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "results/jspace/gemma2_9b_frozen")
    ap.add_argument("--n1", type=int, default=36)
    ap.add_argument("--n2", type=int, default=36)
    args = ap.parse_args()
    out = args.out if args.out.is_absolute() else ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    n1, n2 = args.n1, args.n2

    rows = {}
    for arm, d in ARMS.items():
        m = json.loads((ROOT / "results/transfer" / d / "metrics.json").read_text())
        entry = {"layers": m["layers"], "per_metric": {}, "per_layer": {}}
        ses = []
        for key, label in METRICS:
            theta = m[key]
            se = hanley_mcneil_se(theta, n1, n2)
            z = (theta - 0.5) / se
            entry["per_metric"][label] = {"auroc": theta, "se": se, "ci95": [theta - 1.96 * se, min(1.0, theta + 1.96 * se)], "z_vs_chance": z, "p_two_sided": 2 * norm_sf(abs(z))}
            ses.append(se)
        prim = m["auroc_primary"]
        se_p = float(sum(ses) / len(ses))  # conservative: the three AUROCs treated as perfectly correlated
        z = (prim - 0.5) / se_p
        entry["primary"] = {"auroc": prim, "se_conservative": se_p, "ci95": [prim - 1.96 * se_p, min(1.0, prim + 1.96 * se_p)], "z_vs_chance": z, "p_two_sided": 2 * norm_sf(abs(z))}
        for pl in m["per_layer"]:
            L = pl["layer"]
            p = pl["auroc_primary"]
            se = sum(hanley_mcneil_se(pl[k], n1, n2) for k, _ in METRICS) / 3
            entry["per_layer"][str(L)] = {"primary": p, "se_conservative": se, "ci95": [p - 1.96 * se, min(1.0, p + 1.96 * se)], "per_family": pl.get("per_family", {})}
        rows[arm] = entry

    contrasts = {}
    for j, c in (("W_U k25 J-only", "W_U k25 complement"), ("jac k25 J-only", "jac k25 complement")):
        dj, dc = rows[j]["primary"], rows[c]["primary"]
        diff = dc["auroc"] - dj["auroc"]
        se = math.sqrt(dj["se_conservative"] ** 2 + dc["se_conservative"] ** 2)
        z = diff / se
        contrasts[f"{c} minus {j}"] = {"diff": diff, "se_indep": se, "ci95": [diff - 1.96 * se, diff + 1.96 * se], "z": z, "p_two_sided": 2 * norm_sf(abs(z))}
    for c in ("W_U k25 complement", "jac k25 complement"):
        dc, df = rows[c]["primary"], rows["full"]["primary"]
        diff = df["auroc"] - dc["auroc"]
        se = math.sqrt(dc["se_conservative"] ** 2 + df["se_conservative"] ** 2)
        contrasts[f"full minus {c}"] = {"diff": diff, "se_indep": se, "ci95": [diff - 1.96 * se, diff + 1.96 * se], "z": diff / se, "p_two_sided": 2 * norm_sf(abs(diff / se))}

    core = {}
    for arm, (j, c) in CORE_BATTERY.items():
        se_j = hanley_mcneil_se(j, 50, 50)
        core[arm] = {"J_only": j, "complement": c, "se_J_only_if_single_auroc": se_j, "ci95_J_only": [j - 1.96 * se_j, min(1.0, j + 1.96 * se_j)]}

    payload = {"n_collusion": n1, "n_control": n2, "method": "Hanley-McNeil (1982) SE per AUROC; primary SE = mean of the three metric SEs (conservative); contrasts assume independent arms (conservative when paired arms correlate positively)",
               "transfer_arms": rows, "contrasts": contrasts, "core_battery_ldo": core,
               "sources": ["results/transfer/FULL_core_to_transfer_*/metrics.json", "docs/NEXT-SCIENCE.md (Core battery)"]}
    (out / "frozen_stats.json").write_text(json.dumps(payload, indent=1) + "\n")

    L = ["# Frozen Gemma-2-9B J-space numbers with uncertainty", "",
         f"Source: `results/transfer/FULL_core_to_transfer_*/metrics.json` (HARD Transfer, {n1} collusion + {n2} control episodes; Core discussion → Transfer final). "
         "No per-episode scores were kept, so standard errors are Hanley–McNeil approximations; the primary's SE is the mean of the three metric SEs (treats them as perfectly correlated, which is conservative). "
         "Contrasts treat arms as independent. Method: `scripts/jspace_offline_stats.py`.", "",
         "## Band means (L19–23, or L19/21/23 for the Jacobian arms)", "",
         "| arm | primary | SE | 95% CI | z vs 0.5 | p | Peak [CI] | Split [CI] | Asym [CI] |", "|---|---:|---:|---|---:|---:|---|---|---|"]
    for arm, e in rows.items():
        p = e["primary"]
        pm = e["per_metric"]
        L.append(f"| {arm} | {p['auroc']:.3f} | {p['se_conservative']:.3f} | [{p['ci95'][0]:.3f}, {p['ci95'][1]:.3f}] | {p['z_vs_chance']:.2f} | {p['p_two_sided']:.3g} | "
                 + " | ".join(f"{pm[m]['auroc']:.3f} [{pm[m]['ci95'][0]:.2f}, {pm[m]['ci95'][1]:.2f}]" for m in ("Peak", "Split", "Asym")) + " |")
    L += ["", "## Contrasts", "", "| contrast | difference | SE | 95% CI | z | p |", "|---|---:|---:|---|---:|---:|"]
    for name, c in contrasts.items():
        L.append(f"| {name} | {c['diff']:+.3f} | {c['se_indep']:.3f} | [{c['ci95'][0]:+.3f}, {c['ci95'][1]:+.3f}] | {c['z']:.2f} | {c['p_two_sided']:.3g} |")
    L += ["", "## Per layer, primary with 95% CI", "", "| layer | " + " | ".join(rows) + " |", "|---:|" + "---:|" * len(rows)]
    layers = sorted({int(l) for e in rows.values() for l in e["per_layer"]})
    for Ly in layers:
        cells = []
        for arm, e in rows.items():
            pl = e["per_layer"].get(str(Ly))
            cells.append(f"{pl['primary']:.3f} [{pl['ci95'][0]:.2f}, {pl['ci95'][1]:.2f}]" if pl else "—")
        L.append(f"| {Ly} | " + " | ".join(cells) + " |")
    L += ["", "## Per family (12 episodes each; Split AUROC) for the W_U k25 arms, by layer", "",
          "| layer | family | full | J-only | complement |", "|---:|---|---:|---:|---:|"]
    for Ly in layers:
        fams = rows["full"]["per_layer"].get(str(Ly), {}).get("per_family", {})
        for fam in sorted(fams):
            def g(arm):
                return rows[arm]["per_layer"].get(str(Ly), {}).get("per_family", {}).get(fam, {}).get("split")
            vals = [g("full"), g("W_U k25 J-only"), g("W_U k25 complement")]
            L.append(f"| {Ly} | {fam} | " + " | ".join(f"{v:.3f}" if v is not None else "—" for v in vals) + " |")
    L += ["", "## Core leave-one-domain-out battery (50 + 50 episodes; `docs/NEXT-SCIENCE.md`)", "",
          "Fold-mean AUROCs; the CI below treats each as a single AUROC over 100 episodes, which ignores the fold structure and is only indicative.", "",
          "| variant | J-only | 95% CI (indicative) | complement |", "|---|---:|---|---:|"]
    for arm, c in core.items():
        L.append(f"| {arm} | {c['J_only']:.3f} | [{c['ci95_J_only'][0]:.3f}, {c['ci95_J_only'][1]:.3f}] | {c['complement']:.3f} |")
    L += ["", "## What these numbers can and cannot say", "",
          "- Every J-only Transfer arm is compatible with chance at this sample size (p above 0.05 against 0.5), and every complement arm is far above chance.",
          "- The complement-minus-J-only contrast is large and significant even under the conservative independence approximation.",
          "- What the frozen numbers lack: a dimension-matched control (a random 25-dimensional subspace and its complement), the shuffled-frame control, a within-Transfer J probe, and the phase-versus-family split. `scripts/jspace_separability.py` adds all of these; it needs the activation caches, which are not in git.", ""]
    (out / "FROZEN_STATS.md").write_text("\n".join(L))
    print(f"wrote {out/'FROZEN_STATS.md'}")
    for arm, e in rows.items():
        p = e["primary"]
        print(f"  {arm:22s} primary {p['auroc']:.3f} ± {p['se_conservative']:.3f}  z={p['z_vs_chance']:.2f} p={p['p_two_sided']:.3g}")
    for name, c in contrasts.items():
        print(f"  {name}: {c['diff']:+.3f} [{c['ci95'][0]:+.3f}, {c['ci95'][1]:+.3f}] p={c['p_two_sided']:.3g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
