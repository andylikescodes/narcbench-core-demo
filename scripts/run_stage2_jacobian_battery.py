#!/usr/bin/env python3
"""Exhaustive-ish Core J-lens probe battery (CPU once projections exist).

Runs project_j_variants + train_collusion_probe for each (variant, k, layers) cell.
True jac_topk cells are skipped until Jacobians land under --jac-dir.

  .venv-probe/bin/python scripts/run_stage2_jacobian_battery.py --wu-only
  .venv-probe/bin/python scripts/run_stage2_jacobian_battery.py --jac-only \\
      --jac-dir data/j_basis/gemma2_9b/jacobians
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = ROOT / ".venv-probe/bin/python"
ACTS = ROOT / "data/activations/gemma2_9b/core/20261001T012639Z"
OUT_ROOT = ROOT / "results/stage2_jacobian"
VAR_ROOT = ACTS / "variants"

# Scientifically meaningful grid
WU_CELLS = [
    # (variant, k, layers, tag)
    ("wu_topk", 8, "19-23", "wu_topk_k8_L19-23"),
    ("wu_topk", 16, "19-23", "wu_topk_k16_L19-23"),
    ("wu_topk", 25, "19-23", "wu_topk_k25_L19-23"),  # Stage2 baseline replica
    ("wu_topk", 50, "19-23", "wu_topk_k50_L19-23"),
    ("wu_topk", 25, "21", "wu_topk_k25_L21"),
    ("wu_fixed", 25, "19-23", "wu_fixed_k25_L19-23"),
]
JAC_CELLS = [
    ("jac_topk", 8, "19,21,23", "jac_topk_k8_L19-21-23"),
    ("jac_topk", 16, "19,21,23", "jac_topk_k16_L19-21-23"),
    ("jac_topk", 25, "19,21,23", "jac_topk_k25_L19-21-23"),
    ("jac_topk", 50, "19,21,23", "jac_topk_k50_L19-21-23"),
    ("jac_topk", 25, "21", "jac_topk_k25_L21"),
]


def run(cmd: list[str]) -> int:
    print("+", " ".join(cmd), flush=True)
    return subprocess.call(cmd, cwd=str(ROOT))


def probe_layers_for(layer_spec: str) -> str:
    # train script wants contiguous or list; for "19,21,23" pass as-is
    return layer_spec


def one_cell(variant: str, k: int, layers: str, tag: str, jac_dir: Path | None) -> dict:
    var_dir = VAR_ROOT / tag
    res_dir = OUT_ROOT / tag
    meta = {"tag": tag, "variant": variant, "k": k, "layers": layers, "t0": time.time()}
    prior = res_dir / "cell_summary.json"
    if prior.exists():
        try:
            prev = json.loads(prior.read_text())
            if prev.get("status") == "ok" and prev.get("metrics"):
                print(f"[skip-cell] {tag} already ok", flush=True)
                return prev
        except Exception:
            pass
    proj_cmd = [
        str(PY), "scripts/project_j_variants.py",
        "--acts-dir", str(ACTS),
        "--variant", variant,
        "--k", str(k),
        "--layers", layers,
        "--out-root", str(var_dir),
        "--batch", "96",
        "--vocab-chunk", "16384",
    ]
    if variant == "jac_topk":
        if jac_dir is None or not jac_dir.exists():
            meta["status"] = "skipped_no_jacobian"
            return meta
        proj_cmd += ["--jac-dir", str(jac_dir)]
    # skip project if already done
    if not (var_dir / "j_only" / "activations_gen.npz").exists():
        rc = run(proj_cmd)
        if rc != 0:
            meta["status"] = f"project_failed_{rc}"
            return meta
    else:
        print(f"[skip-project] {tag}", flush=True)

    for arm in ("j_only", "complement"):
        arm_out = res_dir / arm
        if (arm_out / "metrics.json").exists():
            print(f"[skip-probe] {tag}/{arm}", flush=True)
            continue
        rc = run([
            str(PY), "scripts/train_collusion_probe.py",
            "--acts-dir", str(var_dir / arm),
            "--out", str(arm_out),
            "--layers", probe_layers_for(layers),
        ])
        if rc != 0:
            meta["status"] = f"probe_failed_{arm}_{rc}"
            return meta
    # also full baseline pointer
    meta["status"] = "ok"
    meta["seconds"] = time.time() - meta["t0"]
    # collect metrics
    summary = {}
    for arm in ("j_only", "complement"):
        mp = res_dir / arm / "metrics.json"
        if mp.exists():
            m = json.loads(mp.read_text())
            summary[arm] = {
                "auroc_primary": m.get("auroc_primary"),
                "auroc_peak_suspicion": m.get("auroc_peak_suspicion"),
                "auroc_deception_split": m.get("auroc_deception_split"),
                "auroc_asymmetry_probe": m.get("auroc_asymmetry_probe"),
            }
    meta["metrics"] = summary
    (res_dir / "cell_summary.json").write_text(json.dumps(meta, indent=2) + "\n")
    return meta


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wu-only", action="store_true")
    ap.add_argument("--jac-only", action="store_true")
    ap.add_argument("--jac-dir", type=Path, default=ROOT / "data/j_basis/gemma2_9b/jacobians")
    args = ap.parse_args()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    cells = []
    if args.jac_only:
        cells = JAC_CELLS
    elif args.wu_only:
        cells = WU_CELLS
    else:
        cells = WU_CELLS + JAC_CELLS

    results = []
    for variant, k, layers, tag in cells:
        print(f"\n===== CELL {tag} =====", flush=True)
        results.append(one_cell(variant, k, layers, tag, args.jac_dir))
        (OUT_ROOT / "battery_progress.json").write_text(json.dumps(results, indent=2) + "\n")

    # write rollup table
    lines = [
        "# Stage2 Jacobian battery — Core",
        "",
        "| Tag | Arm | Peak | Split | Asym | Primary |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    # include Stage1 full reference
    full = json.loads((ROOT / "results/stage1/full/metrics.json").read_text())
    lines.append(
        f"| full_baseline | full | {full.get('auroc_peak_suspicion'):.3f} | "
        f"{full.get('auroc_deception_split'):.3f} | {full.get('auroc_asymmetry_probe'):.3f} | "
        f"{full.get('auroc_primary'):.3f} |"
    )
    for r in results:
        if r.get("status") != "ok":
            lines.append(f"| {r['tag']} | — | — | — | — | {r.get('status')} |")
            continue
        for arm, m in (r.get("metrics") or {}).items():
            lines.append(
                f"| {r['tag']} | {arm} | {m['auroc_peak_suspicion']:.3f} | "
                f"{m['auroc_deception_split']:.3f} | {m['auroc_asymmetry_probe']:.3f} | "
                f"{m['auroc_primary']:.3f} |"
            )
    (OUT_ROOT / "BATTERY_TABLE.md").write_text("\n".join(lines) + "\n")
    (OUT_ROOT / "battery_results.json").write_text(json.dumps(results, indent=2) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
