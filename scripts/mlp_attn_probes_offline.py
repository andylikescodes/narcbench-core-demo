#!/usr/bin/env python3
"""Core→Transfer probes on MLP / attn (and residual) activation sites.

Protocol: train Core discussion public role/mode; eval Transfer final.
Compares residual vs attn vs mlp at L19–23; picks best Transfer primary AUROC site.

  .venv-probe/bin/python scripts/mlp_attn_probes_offline.py \
    --core-acts data/activations/gemma2_9b/core/20261001T012639Z_mlp_attn \
    --transfer-acts data/activations/gemma2_9b/transfer/RUNPOD_mlp_attn \
    --residual-core data/activations/gemma2_9b/core/20261001T012639Z \
    --residual-transfer data/activations/gemma2_9b/transfer/RUNPOD \
    --out results/transfer_stable
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from eval_transfer_from_core import (  # noqa: E402
    build_transfer_groups,
    enrich_transfer_meta,
    evaluate_transfer,
)
from train_collusion_probe import (  # noqa: E402
    build_groups,
    compute_scores,
    get_layer,
    parse_layers,
    train_direction,
)
from transfer_stable_offline import (  # noqa: E402
    collusion_sample_indices,
    diff_in_means,
    identity_scaler,
    scenario_aurocs,
)

PT = ZoneInfo("America/Los_Angeles")


def load_site(acts_dir: Path, site: str) -> tuple[list[dict], Any]:
    """Load metadata + npz for residual|attn|mlp under acts_dir."""
    meta_p = acts_dir / "metadata_gen.json"
    if site == "residual":
        npz_p = acts_dir / "activations_gen.npz"
    elif site == "attn":
        npz_p = acts_dir / "activations_attn_gen.npz"
    elif site == "mlp":
        npz_p = acts_dir / "activations_mlp_gen.npz"
    else:
        raise ValueError(site)
    if not meta_p.exists() or not npz_p.exists():
        raise FileNotFoundError(f"Need {meta_p.name} and {npz_p.name} under {acts_dir}")
    meta = json.loads(meta_p.read_text())
    npz = np.load(npz_p)
    return meta, npz


def ambient_unit_from_lr(scaler: StandardScaler, w_scaled: np.ndarray) -> np.ndarray:
    """Convert scaled LR weight to ambient-space unit vector: (w/scale) L2-normed."""
    scale = np.asarray(scaler.scale_, dtype=np.float64)
    w = (w_scaled / (scale + 1e-12)).astype(np.float64)
    return w / (np.linalg.norm(w) + 1e-12)


def run_site_layer(
    site: str,
    L: int,
    core_meta,
    core_npz,
    xfer_meta,
    xfer_npz,
    *,
    seed: int,
) -> dict[str, Any]:
    m = evaluate_transfer(
        core_meta,
        core_npz,
        xfer_meta,
        xfer_npz,
        L,
        core_phase="discussion",
        transfer_phase="final",
        seed=seed,
    )
    m["site"] = site
    m["layer"] = L
    m["key"] = f"{site}_L{L}"

    # Also fit role LR + diff-in-means and export ambient unit
    core_acts = get_layer(core_npz, len(core_meta), L)
    xfer_acts = get_layer(xfer_npz, len(xfer_meta), L)
    scaler, w = train_direction(core_meta, core_acts, phase="discussion", seed=seed)
    u_lr = ambient_unit_from_lr(scaler, w)
    col_idx, hon_idx = collusion_sample_indices(core_meta, phase="discussion")
    u_dm = diff_in_means(core_acts, col_idx, hon_idx)

    # sample-level role AUROC on Transfer final (colluder vs honest, collusion mode)
    xcol, xhon = [], []
    for i, row in enumerate(xfer_meta):
        if row.get("mode") != "collusion" or row.get("phase") != "final" or row.get("channel") != "public":
            continue
        if row.get("role") == "colluder":
            xcol.append(i)
        elif row.get("role") == "honest":
            xhon.append(i)
    role_auroc = None
    if xcol and xhon:
        scores = np.concatenate([xfer_acts[xcol] @ u_lr, xfer_acts[xhon] @ u_lr])
        y = np.array([1] * len(xcol) + [0] * len(xhon))
        if len(np.unique(y)) >= 2:
            role_auroc = float(roc_auc_score(y, scores))
    m["auroc_role_xfer_sample"] = role_auroc
    m["direction_lr_ambient_unit"] = u_lr
    m["direction_diff_means"] = u_dm
    m["n_core_col"] = len(col_idx)
    m["n_core_hon"] = len(hon_idx)
    return m


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--core-acts", type=Path, required=True, help="mlp_attn extract dir (has attn+mlp npz)")
    ap.add_argument("--transfer-acts", type=Path, required=True)
    ap.add_argument(
        "--residual-core",
        type=Path,
        default=None,
        help="optional prior residual cache for residual baseline (else use residual in core-acts)",
    )
    ap.add_argument("--residual-transfer", type=Path, default=None)
    ap.add_argument("--layers", default="19-23")
    ap.add_argument("--sites", default="attn,mlp,residual")
    ap.add_argument("--out", type=Path, default=ROOT / "results/transfer_stable")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    core_dir = args.core_acts if args.core_acts.is_absolute() else ROOT / args.core_acts
    xfer_dir = args.transfer_acts if args.transfer_acts.is_absolute() else ROOT / args.transfer_acts
    out = args.out if args.out.is_absolute() else ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    dirs_dir = out / "directions"
    dirs_dir.mkdir(parents=True, exist_ok=True)

    layers = parse_layers(args.layers)
    sites = [s.strip() for s in args.sites.split(",") if s.strip()]

    # Load per site
    loaded: dict[str, tuple] = {}
    for site in sites:
        if site == "residual" and args.residual_core and args.residual_transfer:
            cdir = args.residual_core if args.residual_core.is_absolute() else ROOT / args.residual_core
            tdir = args.residual_transfer if args.residual_transfer.is_absolute() else ROOT / args.residual_transfer
            c_meta, c_npz = load_site(cdir, "residual")
            t_meta, t_npz = load_site(tdir, "residual")
        else:
            c_meta, c_npz = load_site(core_dir, site)
            t_meta, t_npz = load_site(xfer_dir, site)
        t_meta = enrich_transfer_meta(t_meta)
        loaded[site] = (c_meta, c_npz, t_meta, t_npz)
        print(f"[load] site={site} core_n={len(c_meta)} xfer_n={len(t_meta)} keys={list(c_npz.files)[:8]}", flush=True)

    rows: list[dict] = []
    for site in sites:
        c_meta, c_npz, t_meta, t_npz = loaded[site]
        for L in layers:
            print(f"[probe] {site} L{L}", flush=True)
            m = run_site_layer(site, L, c_meta, c_npz, t_meta, t_npz, seed=args.seed)
            # strip arrays for JSON row
            row = {k: v for k, v in m.items() if k not in ("direction_lr_ambient_unit", "direction_diff_means")}
            rows.append(row)
            if "auroc_primary" in m:
                print(
                    f"  primary={m['auroc_primary']:.3f} peak={m['auroc_peak_suspicion']:.3f} "
                    f"split={m['auroc_deception_split']:.3f} asym={m['auroc_asymmetry_probe']:.3f} "
                    f"role_sample={m.get('auroc_role_xfer_sample')}",
                    flush=True,
                )
            # keep full with dirs temporarily
            rows[-1]["_u_lr"] = m["direction_lr_ambient_unit"]
            rows[-1]["_u_dm"] = m["direction_diff_means"]

    # Best among mlp/attn only (primary Transfer AUROC)
    cand = [r for r in rows if r["site"] in ("attn", "mlp") and r.get("auroc_primary") is not None]
    if not cand:
        print("ERROR: no mlp/attn results", file=sys.stderr)
        return 1
    cand_sorted = sorted(cand, key=lambda r: -float(r["auroc_primary"]))
    best = cand_sorted[0]
    runner = cand_sorted[1] if len(cand_sorted) > 1 else None

    # Export best direction
    best_key = best["key"]
    u = best["_u_lr"]
    npy_name = f"lr_role_{best_key}.npy"
    np.save(dirs_dir / npy_name, u.astype(np.float32))
    dm_name = f"diff_means_{best_key}.npy"
    np.save(dirs_dir / dm_name, best["_u_dm"].astype(np.float32))

    # Also export all mlp/attn LR dirs for causal flexibility
    exported = {npy_name: {"site": best["site"], "layer": best["layer"], "kind": "lr_role_ambient_unit", "auroc_primary": best["auroc_primary"]}}
    for r in cand:
        name = f"lr_role_{r['key']}.npy"
        if name == npy_name:
            continue
        np.save(dirs_dir / name, r["_u_lr"].astype(np.float32))
        exported[name] = {
            "site": r["site"],
            "layer": r["layer"],
            "kind": "lr_role_ambient_unit",
            "auroc_primary": r.get("auroc_primary"),
        }

    # Clean rows for JSON
    clean_rows = []
    for r in rows:
        clean = {k: v for k, v in r.items() if not k.startswith("_")}
        clean_rows.append(clean)

    summary = {
        "protocol": "core_discussion_public → transfer_final; LR role direction (train_direction)",
        "sites": sites,
        "layers": layers,
        "best_site": {
            "key": best_key,
            "site": best["site"],
            "layer": best["layer"],
            "auroc_primary": best["auroc_primary"],
            "auroc_peak": best.get("auroc_peak_suspicion"),
            "auroc_split": best.get("auroc_deception_split"),
            "auroc_asym": best.get("auroc_asymmetry_probe"),
            "auroc_role_xfer_sample": best.get("auroc_role_xfer_sample"),
            "direction_npy": npy_name,
            "diff_means_npy": dm_name,
        },
        "runner_up": (
            {
                "key": runner["key"],
                "site": runner["site"],
                "layer": runner["layer"],
                "auroc_primary": runner["auroc_primary"],
            }
            if runner
            else None
        ),
        "per_site_layer": clean_rows,
        "exported_directions": exported,
        "core_acts": str(core_dir),
        "transfer_acts": str(xfer_dir),
        "residual_baseline_core": str(args.residual_core) if args.residual_core else None,
        "residual_baseline_transfer": str(args.residual_transfer) if args.residual_transfer else None,
        "seed": args.seed,
        "written_pt": datetime.now(PT).strftime("%Y-%m-%d %H:%M:%S %Z"),
        "hook_doc": "attn=self_attn out (residual-write); mlp=mlp out (residual-write); residual=DecoderLayer out",
    }
    (out / "MLP_ATTN_PROBES.json").write_text(json.dumps(summary, indent=2) + "\n")

    # Markdown
    lines = [
        "# MLP / Attn Core→Transfer probes",
        "",
        f"**Written:** {summary['written_pt']}",
        "",
        f"**Protocol:** {summary['protocol']}",
        "",
        f"**Hooks:** {summary['hook_doc']}",
        "",
        "## Best site (among mlp/attn)",
        "",
        f"- **{best_key}** primary AUROC = **{best['auroc_primary']:.3f}** "
        f"(peak={best.get('auroc_peak_suspicion', float('nan')):.3f} "
        f"split={best.get('auroc_deception_split', float('nan')):.3f} "
        f"asym={best.get('auroc_asymmetry_probe', float('nan')):.3f})",
        f"- Direction: `directions/{npy_name}`",
    ]
    if runner:
        lines.append(
            f"- Runner-up: **{runner['key']}** primary={runner['auroc_primary']:.3f}"
        )
    lines += ["", "## Per site × layer (Transfer primary AUROC)", "", "| site | L | primary | peak | split | asym | role_sample |", "|---|---:|---:|---:|---:|---:|---:|"]
    for r in sorted(clean_rows, key=lambda x: (x["site"], x["layer"])):
        if "auroc_primary" not in r:
            lines.append(f"| {r['site']} | {r['layer']} | ERR | | | | |")
            continue
        rs = r.get("auroc_role_xfer_sample")
        rs_s = f"{rs:.3f}" if rs is not None else "—"
        lines.append(
            f"| {r['site']} | {r['layer']} | {r['auroc_primary']:.3f} | "
            f"{r['auroc_peak_suspicion']:.3f} | {r['auroc_deception_split']:.3f} | "
            f"{r['auroc_asymmetry_probe']:.3f} | {rs_s} |"
        )
    lines += ["", f"JSON: `MLP_ATTN_PROBES.json`", ""]
    (out / "MLP_ATTN_PROBES.md").write_text("\n".join(lines))
    print(json.dumps({k: summary[k] for k in ("best_site", "runner_up")}, indent=2))
    print(f"[wrote] {out / 'MLP_ATTN_PROBES.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
