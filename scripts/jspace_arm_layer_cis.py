#!/usr/bin/env python3
"""Per-layer and band-level bootstrap intervals for chosen arms of a jspace_separability.py run, from its episode_scores.json.

Post-hoc companion to the pre-registered band-level inference (design section 5): the same episode bootstrap, stratified by
family and label, B resamples with seed = run seed + 1 (so the band-level numbers coincide with summary.json), applied per
layer and to extra paired differences. Transfer analyses only (A2 Core -> Transfer finals, A3 within-Transfer LOFO, A4b
Core -> Transfer discussion), where the episode is the unit.

    python3 scripts/jspace_arm_layer_cis.py --run results/jspace/gemma2_9b_jpaper_nnomp_k25 --k 25 \
        --arms full,Jpaper,C_Jpaper,shuffled_paper,J,randsub \
        --diffs Jpaper-randsub,Jpaper-shuffled_paper,C_Jpaper-Jpaper,C_Jpaper-full,Jpaper-J

Writes <run>/arm_layer_cis.json and <run>/arm_layer_cis.md (another stem with --out-stem, e.g. arm_layer_cis_k16).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import jspace_separability as js  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, required=True, help="a jspace_separability.py --out directory")
    ap.add_argument("--k", type=int, default=25)
    ap.add_argument("--arms", default="full,Jpaper,C_Jpaper,shuffled_paper,J,C_J,randsub,shuffled,pca")
    ap.add_argument("--diffs", default="Jpaper-randsub,Jpaper-shuffled_paper,Jpaper-J,C_Jpaper-Jpaper,C_Jpaper-full,J-randsub")
    ap.add_argument("--analyses", default="A2,A3,A4b")
    ap.add_argument("--n-boot", type=int, default=None, help="default: the run's n_boot")
    ap.add_argument("--out-stem", default="arm_layer_cis", help="output file stem under --run (e.g. arm_layer_cis_k16)")
    a = ap.parse_args()
    run = a.run if a.run.is_absolute() else ROOT / a.run
    summary = json.loads((run / "summary.json").read_text())
    episodes = json.loads((run / "episode_scores.json").read_text())
    seed, B = int(summary["seed"]), int(a.n_boot or summary["n_boot"])
    k = str(a.k)
    arms = [x for x in a.arms.split(",") if x]
    diffs = [tuple(x.split("-", 1)) for x in a.diffs.split(",") if "-" in x]
    out = {"run": str(run.relative_to(ROOT)) if run.is_relative_to(ROOT) else str(run), "k": a.k, "n_boot": B, "seed_boot": seed + 1,
           "written": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()), "analyses": {}}
    md = [f"# Per-layer bootstrap intervals — {summary['tag']} (k = {a.k})", "",
          f"Episode bootstrap stratified by family and label, B = {B}, seed {seed + 1} (as the run's band-level inference); "
          f"post-hoc per-layer view of the pre-registered analyses. Written {out['written']} by `scripts/jspace_arm_layer_cis.py`.", ""]
    for an in [x for x in a.analyses.split(",") if x in episodes and episodes[x]]:
        block = episodes[an]
        layers = [L for L in block if k in block[L]]
        if not layers:
            continue
        ep0 = block[layers[0]][k]["full"]
        y = np.asarray(ep0["y"]); fams = ep0["family"]
        boots = js.bootstrap_indices(y, fams, B, np.random.RandomState(seed + 1))
        full_idx = np.arange(len(y))
        res = {"n_episodes": int(len(y)), "layers": layers, "arms": {}, "differences": {}}
        dist_layer: dict[str, dict[str, np.ndarray]] = {}
        dist_band: dict[str, np.ndarray] = {}
        for arm in arms:
            if any(arm not in block[L][k] for L in layers):
                continue
            per = {}
            dist_layer[arm] = {}
            for L in layers:
                ep = block[L][k][arm]
                vals = np.array([js.primary_on(ep, idx) for idx in boots])
                dist_layer[arm][L] = vals
                per[L] = {"est": js.primary_on(ep, full_idx), "ci": js.ci(vals)}
            band = js.bootstrap_band({L: block[L][k][arm] for L in layers}, boots)
            dist_band[arm] = band
            res["arms"][arm] = {"per_layer": per, "band": {"est": float(np.mean([per[L]["est"] for L in layers])), "ci": js.ci(band)}}
        for x, z in diffs:
            if x in dist_band and z in dist_band:
                d = {"band": {"est": res["arms"][x]["band"]["est"] - res["arms"][z]["band"]["est"], "ci": js.ci(dist_band[x] - dist_band[z])},
                     "per_layer": {L: {"est": res["arms"][x]["per_layer"][L]["est"] - res["arms"][z]["per_layer"][L]["est"],
                                       "ci": js.ci(dist_layer[x][L] - dist_layer[z][L])} for L in layers}}
                res["differences"][f"{x}-{z}"] = d
        out["analyses"][an] = res
        title = {"A2": "A2 Core → Transfer finals", "A3": "A3 within-Transfer leave-one-family-out", "A4b": "A4b Core → Transfer discussion"}.get(an, an)
        md += [f"## {title} ({len(y)} episodes)", "", "| arm | " + " | ".join(f"L{L}" for L in layers) + " | band |", "|---|" + "---|" * (len(layers) + 1)]
        for arm, r in res["arms"].items():
            cells = [f"{r['per_layer'][L]['est']:.3f} [{r['per_layer'][L]['ci'][0]:.3f}, {r['per_layer'][L]['ci'][1]:.3f}]" for L in layers]
            md.append(f"| {arm} | " + " | ".join(cells) + f" | **{r['band']['est']:.3f}** [{r['band']['ci'][0]:.3f}, {r['band']['ci'][1]:.3f}] |")
        md += ["", "| difference | " + " | ".join(f"L{L}" for L in layers) + " | band |", "|---|" + "---|" * (len(layers) + 1)]
        for name, d in res["differences"].items():
            cells = [f"{d['per_layer'][L]['est']:+.3f} [{d['per_layer'][L]['ci'][0]:+.3f}, {d['per_layer'][L]['ci'][1]:+.3f}]" for L in layers]
            md.append(f"| {name} | " + " | ".join(cells) + f" | **{d['band']['est']:+.3f}** [{d['band']['ci'][0]:+.3f}, {d['band']['ci'][1]:+.3f}] |")
        md.append("")
    (run / f"{a.out_stem}.json").write_text(json.dumps(out, indent=1) + "\n")
    (run / f"{a.out_stem}.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    sys.exit(main())
