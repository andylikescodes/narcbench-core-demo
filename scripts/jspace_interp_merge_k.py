#!/usr/bin/env python3
"""Merge two runs of scripts/jspace_interp_analyze.py on the same cell that differ only in --ks.

The pursuit is nested in k (one selection order serves every k), so the k = 25 arms and the other sizes can be scored
by separate processes that read the same saved decomposition. This puts their per-layer results into one
sweep_results.json and checks that everything the two runs share is identical.

    python scripts/jspace_interp_merge_k.py --parts A/sweep_results.json B/sweep_results.json --out DIR
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

SAME = ("condition", "position", "layer", "n_core_rows", "n_transfer_rows", "n_warned_rows", "pursuit_steps_core_mean",
        "pursuit_steps_core_max", "pursuit_steps_transfer_mean", "full")
SETTINGS_SAME = ("root", "lens", "basis_dir", "layers", "positions", "conditions", "candidates", "norm_kind", "seed", "method",
                 "k_max", "boot", "n_random", "reuse_random_from", "redraw_random_atoms", "no_lodo", "no_warned")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", type=Path, nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    parts = [json.loads(p.read_text()) for p in a.parts]
    base = parts[0]
    for other in parts[1:]:
        for key in SETTINGS_SAME:
            if base["settings"].get(key) != other["settings"].get(key):
                raise SystemExit(f"settings differ in {key}: {base['settings'].get(key)} vs {other['settings'].get(key)}")
    cells = {}
    for part in parts:
        for c in part["cells"]:
            key = (c["condition"], c["position"], c["layer"])
            if key not in cells:
                cells[key] = dict(c)
                continue
            m = cells[key]
            for f in SAME:
                if m.get(f) != c.get(f):
                    raise SystemExit(f"{key}: the runs disagree on {f}")
            for f, v in c.items():
                if f.startswith("k") and f[1:].isdigit():
                    if f in m:
                        raise SystemExit(f"{key}: {f} is in more than one run")
                    m[f] = v
            m["seconds"] = round(m.get("seconds", 0) + c.get("seconds", 0), 1)
    out = dict(base)
    out["cells"] = [cells[k] for k in cells]
    ks = sorted({int(f[1:]) for c in out["cells"] for f in c if f.startswith("k") and f[1:].isdigit()})
    for c in out["cells"]:
        missing = [k for k in ks if f"k{k}" not in c]
        if missing:
            raise SystemExit(f"layer {c['layer']}: no results for k = {missing}")
    out["ks"] = ks
    out["settings"] = dict(base["settings"], ks=",".join(str(k) for k in ks), out=str(a.out),
                           merged_from=["/".join(p.parts[-2:]) for p in a.parts])    # run folder and file, not a local path
    out["exp1"] = [e for part in parts for e in part.get("exp1", [])]
    out["role_dirs"] = [e for part in parts for e in part.get("role_dirs", [])]
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "role_dirs").mkdir(exist_ok=True)
    for p in a.parts:
        for f in (p.parent / "role_dirs").glob("*.npy"):
            shutil.copy2(f, a.out / "role_dirs" / f.name)
        for f in p.parent.glob("boot_*.npz"):          # the saved bootstrap draws of each (layer, k)
            shutil.copy2(f, a.out / f.name)
    (a.out / "sweep_results.json").write_text(json.dumps(out, indent=1, default=float) + "\n")
    print("wrote", a.out / "sweep_results.json", "layers", [c["layer"] for c in out["cells"]], "ks", ks)


if __name__ == "__main__":
    main()
