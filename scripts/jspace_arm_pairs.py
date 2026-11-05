#!/usr/bin/env python3
"""Paired bootstrap differences between any two arms of a jspace_separability.py run, across frame sizes.

The run's own inference pairs arms within one k (design section 5). This post-hoc companion pairs arms across k blocks,
for example the paper's J-space at k = 16 against k = 25, with the same stratified episode bootstrap (B resamples,
seed = run seed + 1, so single-arm numbers coincide with summary.json), as a band mean over the run's layers and per
layer. Transfer analyses only (A2 Core -> Transfer finals, A3 within-Transfer LOFO, A4b Core -> Transfer discussion).

    python3 scripts/jspace_arm_pairs.py --run results/jspace/gemma2_9b_jpaper_variants \
        --pairs Jpaper_k16@16-Jpaper@25,Jpaper_k50@50-Jpaper@25,Jpaper_raw@25-Jpaper@25

Writes <run>/<out-stem>.json and <run>/<out-stem>.md (default stem arm_pairs).
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

TITLES = {"A2": "A2 Core → Transfer finals", "A3": "A3 within-Transfer leave-one-family-out", "A4b": "A4b Core → Transfer discussion"}


def parse_pair(spec: str) -> tuple[tuple[str, str], tuple[str, str]]:
    """'X@k1-Z@k2' -> (('X', 'k1'), ('Z', 'k2')); the arm name may itself contain '-' only before the '@'."""
    left, right = spec.split("-", 1)
    if "@" not in left or "@" not in right:
        raise SystemExit(f"pair {spec!r} must be ARM@k-ARM@k")
    (x, kx), (z, kz) = left.rsplit("@", 1), right.rsplit("@", 1)
    return (x, kx), (z, kz)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, required=True, help="a jspace_separability.py --out directory")
    ap.add_argument("--pairs", required=True, help="comma-separated ARM@k-ARM@k pairs (first minus second)")
    ap.add_argument("--analyses", default="A2,A3,A4b")
    ap.add_argument("--n-boot", type=int, default=None, help="default: the run's n_boot")
    ap.add_argument("--out-stem", default="arm_pairs")
    a = ap.parse_args()
    run = a.run if a.run.is_absolute() else ROOT / a.run
    summary = json.loads((run / "summary.json").read_text())
    episodes = json.loads((run / "episode_scores.json").read_text())
    seed, B = int(summary["seed"]), int(a.n_boot or summary["n_boot"])
    pairs = [parse_pair(p) for p in a.pairs.split(",") if p]
    out = {"run": str(run.relative_to(ROOT)) if run.is_relative_to(ROOT) else str(run), "n_boot": B, "seed_boot": seed + 1,
           "written": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()), "analyses": {}}
    md = [f"# Paired differences across frame sizes — {summary['tag']}", "",
          f"Episode bootstrap stratified by family and label, B = {B}, seed {seed + 1} (as the run's band-level inference); "
          f"post-hoc pairing of arms across k blocks. Written {out['written']} by `scripts/jspace_arm_pairs.py`.", ""]
    for an in [x for x in a.analyses.split(",") if x in episodes and episodes[x]]:
        block = episodes[an]
        layers = list(block)
        ref = block[layers[0]][next(iter(block[layers[0]]))]["full"]
        y, fams, rids = np.asarray(ref["y"]), ref["family"], ref["rids"]
        boots = js.bootstrap_indices(y, fams, B, np.random.RandomState(seed + 1))
        full_idx = np.arange(len(y))
        cache: dict[tuple[str, str], tuple[dict, dict, np.ndarray]] = {}

        def stats(arm: str, k: str):
            if (arm, k) in cache:
                return cache[(arm, k)]
            if any(k not in block[L] or arm not in block[L][k] for L in layers):
                return None
            eps = {L: block[L][k][arm] for L in layers}
            for L, ep in eps.items():
                if ep["rids"] != rids:
                    raise SystemExit(f"{an} L{L} k={k} {arm}: episode order differs from the reference arm")
            per_est = {L: float(js.primary_on(eps[L], full_idx)) for L in layers}
            per_dist = {L: np.array([js.primary_on(eps[L], idx) for idx in boots]) for L in layers}
            cache[(arm, k)] = (per_est, per_dist, js.bootstrap_band(eps, boots))
            return cache[(arm, k)]

        res = {"n_episodes": int(len(y)), "layers": layers, "pairs": {}}
        for (x, kx), (z, kz) in pairs:
            sx, sz = stats(x, kx), stats(z, kz)
            if sx is None or sz is None:
                print(f"[{an}] skip {x}@{kx}-{z}@{kz}: arm missing", file=sys.stderr)
                continue
            bx, bz = float(np.mean(list(sx[0].values()))), float(np.mean(list(sz[0].values())))
            res["pairs"][f"{x}@{kx}-{z}@{kz}"] = {
                "band": {"est_first": bx, "est_second": bz, "est": bx - bz, "ci": js.ci(sx[2] - sz[2])},
                "per_layer": {L: {"est": sx[0][L] - sz[0][L], "ci": js.ci(sx[1][L] - sz[1][L])} for L in layers}}
        out["analyses"][an] = res
        md += [f"## {TITLES.get(an, an)} ({len(y)} episodes)", "",
               "| difference | first | second | " + " | ".join(f"L{L}" for L in layers) + " | band |", "|---|---|---|" + "---|" * (len(layers) + 1)]
        for name, d in res["pairs"].items():
            cells = [f"{d['per_layer'][L]['est']:+.3f} [{d['per_layer'][L]['ci'][0]:+.3f}, {d['per_layer'][L]['ci'][1]:+.3f}]" for L in layers]
            md.append(f"| {name} | {d['band']['est_first']:.3f} | {d['band']['est_second']:.3f} | " + " | ".join(cells)
                      + f" | **{d['band']['est']:+.3f}** [{d['band']['ci'][0]:+.3f}, {d['band']['ci'][1]:+.3f}] |")
        md.append("")
    (run / f"{a.out_stem}.json").write_text(json.dumps(out, indent=1) + "\n")
    (run / f"{a.out_stem}.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    sys.exit(main())
