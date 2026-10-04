#!/usr/bin/env python3
"""Train collusion direction on Core; evaluate Peak/Split/Asym zero-shot on Transfer.

Paper protocol (Rose et al.): Core-trained deception direction → Transfer AUROC.

  .venv-probe/bin/python scripts/eval_transfer_from_core.py \\
    --core-acts data/activations/gemma2_9b/core/20261001T012639Z \\
    --transfer-acts data/activations/gemma2_9b/transfer/<run>/ \\
    --out results/transfer/core_to_transfer_full --layers 19-23 \\
    --core-phase discussion --transfer-phase final
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from train_collusion_probe import (  # noqa: E402
    build_groups,
    compute_scores,
    get_layer,
    load_acts,
    parse_layers,
    train_direction,
)


def family_of(run_id: str, scenario_domain: str = "", family: str = "") -> str:
    if family:
        return family
    if scenario_domain:
        return scenario_domain
    # transfer run_id: code_review__cr_01__collusion
    parts = run_id.split("__")
    return parts[0] if parts else run_id


def enrich_transfer_meta(meta: list[dict]) -> list[dict]:
    out = []
    for m in meta:
        m = dict(m)
        fam = m.get("family") or family_of(
            m.get("run_id", ""), m.get("scenario_domain", "") or ""
        )
        m["family"] = fam
        m["scenario_domain"] = m.get("scenario_domain") or fam
        out.append(m)
    return out


def build_transfer_groups(meta, acts, *, phase: str):
    """Transfer HF layout is often final-only; also accept discussion."""
    if phase == "discussion":
        return build_groups(meta, acts, phase="discussion", avg_rounds=True)
    # final: one vector per agent (paper Transfer)
    groups = build_groups(meta, acts, phase="final", avg_rounds=False)
    if groups:
        return groups
    # fallback: any public channel last round per agent
    accum: dict[tuple, dict] = defaultdict(lambda: {"vec": None, "meta": None, "round": -1})
    for i, m in enumerate(meta):
        if m.get("channel") != "public":
            continue
        key = (m["run_id"], m["agent_name"])
        rnd = int(m.get("round") or 0)
        if rnd >= accum[key]["round"]:
            accum[key] = {"vec": acts[i], "meta": m, "round": rnd}
    groups = {}
    for (rid, agent), data in accum.items():
        m = data["meta"]
        if rid not in groups:
            groups[rid] = {
                "agents": {},
                "mode": m["mode"],
                "roles": {},
                "domain": family_of(rid, m.get("scenario_domain", ""), m.get("family", "")),
                "scenario_id": m.get("scenario_id", ""),
            }
        groups[rid]["agents"][agent] = data["vec"]
        groups[rid]["roles"][agent] = m["role"]
    return {k: v for k, v in groups.items() if len(v["agents"]) >= 2}


def evaluate_transfer(core_meta, core_npz, xfer_meta, xfer_npz, layer: int,
                      *, core_phase: str, transfer_phase: str, seed: int) -> dict[str, Any]:
    core_acts = get_layer(core_npz, len(core_meta), layer)
    xfer_acts = get_layer(xfer_npz, len(xfer_meta), layer)
    scaler, w = train_direction(
        core_meta, core_acts, exclude_domain=None, phase=core_phase, seed=seed
    )
    # Fit asymmetry probe on Core scenario scores
    core_groups = build_groups(
        core_meta, core_acts, phase=core_phase, avg_rounds=(core_phase == "discussion")
    )
    core_rids = sorted(core_groups.keys())
    core_y = np.array([1 if core_groups[r]["mode"] == "collusion" else 0 for r in core_rids])
    core_scores = compute_scores(core_groups, core_rids, scaler, w)

    xfer_groups = build_transfer_groups(xfer_meta, xfer_acts, phase=transfer_phase)
    xfer_rids = sorted(xfer_groups.keys())
    if len(xfer_rids) < 4:
        return {"error": "too_few_transfer_runs", "n_runs": len(xfer_rids), "layer": layer}
    xfer_y = np.array([1 if xfer_groups[r]["mode"] == "collusion" else 0 for r in xfer_rids])
    xfer_scores = compute_scores(xfer_groups, xfer_rids, scaler, w)

    out: dict[str, Any] = {"layer": layer, "n_core_runs": len(core_rids), "n_transfer_runs": len(xfer_rids)}
    if len(np.unique(xfer_y)) < 2:
        out["error"] = "transfer_single_class"
        return out

    out["auroc_peak_suspicion"] = float(roc_auc_score(xfer_y, xfer_scores["peak_suspicion"]))
    out["auroc_deception_split"] = float(roc_auc_score(xfer_y, xfer_scores["deception_split"]))
    fs = StandardScaler()
    clf = LogisticRegression(max_iter=2000, C=0.01, random_state=seed)
    clf.fit(fs.fit_transform(core_scores["asymmetry_probe"]), core_y)
    out["auroc_asymmetry_probe"] = float(
        roc_auc_score(xfer_y, clf.predict_proba(fs.transform(xfer_scores["asymmetry_probe"]))[:, 1])
    )
    out["auroc_primary"] = float(
        np.mean(
            [
                out["auroc_peak_suspicion"],
                out["auroc_deception_split"],
                out["auroc_asymmetry_probe"],
            ]
        )
    )
    # per-family
    fam_auc = {}
    for fam in sorted({xfer_groups[r]["domain"] for r in xfer_rids}):
        idx = [i for i, r in enumerate(xfer_rids) if xfer_groups[r]["domain"] == fam]
        yf = xfer_y[idx]
        if len(np.unique(yf)) < 2 or len(idx) < 4:
            continue
        fam_auc[fam] = {
            "peak": float(roc_auc_score(yf, xfer_scores["peak_suspicion"][idx])),
            "split": float(roc_auc_score(yf, xfer_scores["deception_split"][idx])),
            "n": len(idx),
        }
    out["per_family"] = fam_auc
    out["mode_counts"] = dict(Counter(xfer_y.tolist()))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--core-acts", type=Path, required=True)
    ap.add_argument("--transfer-acts", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--layers", default="19-23")
    ap.add_argument("--core-phase", default="discussion")
    ap.add_argument("--transfer-phase", default="final")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    core_dir = args.core_acts if args.core_acts.is_absolute() else ROOT / args.core_acts
    xfer_dir = args.transfer_acts if args.transfer_acts.is_absolute() else ROOT / args.transfer_acts
    out = args.out if args.out.is_absolute() else ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)

    core_meta, core_npz = load_acts(core_dir)
    xfer_meta, xfer_npz = load_acts(xfer_dir)
    xfer_meta = enrich_transfer_meta(xfer_meta)

    layers = parse_layers(args.layers)
    avail_c = sorted(int(k.split("_")[1]) for k in core_npz.files if k.startswith("layer_"))
    avail_t = sorted(int(k.split("_")[1]) for k in xfer_npz.files if k.startswith("layer_"))
    layers = [L for L in layers if L in avail_c and L in avail_t]
    if not layers:
        print("BLOCKED: no overlapping layers", file=sys.stderr)
        return 2

    per = []
    for L in layers:
        print(f"[xfer-eval] layer {L}", flush=True)
        m = evaluate_transfer(
            core_meta, core_npz, xfer_meta, xfer_npz, L,
            core_phase=args.core_phase, transfer_phase=args.transfer_phase, seed=args.seed,
        )
        per.append(m)
        if "auroc_primary" in m:
            print(
                f"  Peak={m['auroc_peak_suspicion']:.3f} Split={m['auroc_deception_split']:.3f} "
                f"Asym={m['auroc_asymmetry_probe']:.3f} primary={m['auroc_primary']:.3f}",
                flush=True,
            )
        else:
            print(f"  {m}", flush=True)

    def nanmean(key):
        vs = [m[key] for m in per if m.get(key) is not None]
        return float(np.mean(vs)) if vs else float("nan")

    summary = {
        "protocol": "train_core_eval_transfer_zeroshot",
        "core_acts": str(core_dir),
        "transfer_acts": str(xfer_dir),
        "core_phase": args.core_phase,
        "transfer_phase": args.transfer_phase,
        "layers": layers,
        "auroc_peak_suspicion": nanmean("auroc_peak_suspicion"),
        "auroc_deception_split": nanmean("auroc_deception_split"),
        "auroc_asymmetry_probe": nanmean("auroc_asymmetry_probe"),
        "auroc_primary": nanmean("auroc_primary"),
        "per_layer": per,
    }
    (out / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: summary[k] for k in summary if k != "per_layer"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
