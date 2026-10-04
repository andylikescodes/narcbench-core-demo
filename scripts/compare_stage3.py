#!/usr/bin/env python3
"""Stage 3: compare full vs J-only vs complement probe AUROCs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KEYS = [
    "auroc_peak_suspicion",
    "auroc_deception_split",
    "auroc_asymmetry_probe",
    "auroc_primary",
]


def load(p: Path) -> dict:
    return json.loads(p.read_text())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", type=Path, default=ROOT / "results/stage1/full/metrics.json")
    ap.add_argument("--j", type=Path, default=ROOT / "results/stage2/j_only/metrics.json")
    ap.add_argument("--c", type=Path, default=ROOT / "results/stage2/complement/metrics.json")
    ap.add_argument("--out", type=Path, default=ROOT / "results/stage3")
    args = ap.parse_args()
    out = args.out if args.out.is_absolute() else ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)

    arms = {
        "full": load(args.full if args.full.is_absolute() else ROOT / args.full),
        "j_only": load(args.j if args.j.is_absolute() else ROOT / args.j),
        "complement": load(args.c if args.c.is_absolute() else ROOT / args.c),
    }

    table_rows = []
    for method, label in [
        ("auroc_peak_suspicion", "Peak Suspicion"),
        ("auroc_deception_split", "Deception Split"),
        ("auroc_asymmetry_probe", "Asymmetry Probe"),
        ("auroc_primary", "Primary (mean)"),
    ]:
        row = {"method": label, "key": method}
        for arm, m in arms.items():
            row[arm] = m.get(method)
        row["delta_j_minus_full"] = (row["j_only"] or 0) - (row["full"] or 0)
        row["delta_c_minus_full"] = (row["complement"] or 0) - (row["full"] or 0)
        row["delta_j_minus_c"] = (row["j_only"] or 0) - (row["complement"] or 0)
        table_rows.append(row)

    # Containment reading
    j_p = arms["j_only"].get("auroc_primary")
    c_p = arms["complement"].get("auroc_primary")
    f_p = arms["full"].get("auroc_primary")
    if j_p is not None and c_p is not None and f_p is not None:
        if j_p >= 0.90 and c_p < 0.70:
            reading = (
                "Containment-leaning: J-only recovers near-full collusion AUROC while "
                "complement collapses — collusion-relevant signal appears concentrated "
                "in the verbalizable / J frame (Gurnee privilege + NARCBench probes)."
            )
        elif c_p >= 0.90 and j_p < 0.70:
            reading = (
                "Anti-containment: complement retains near-full AUROC while J-only "
                "collapses — collusion signal lives largely outside the verbalizable frame."
            )
        elif j_p >= 0.90 and c_p >= 0.90:
            reading = (
                "Distributed: both J-only and complement retain high AUROC (near full). "
                "Collusion-discriminative information is not confined to J-space; "
                "linear probes can recover it from either arm (cf. skepticism in "
                "Kumar / Boxo about clean low-dim containment)."
            )
        else:
            reading = (
                f"Mixed/attenuated: full={f_p:.3f}, J={j_p:.3f}, complement={c_p:.3f}. "
                "Neither arm fully matches the full-activation baseline; signal may be "
                "shared across subspaces or damaged by the sparse projection."
            )
    else:
        reading = "Incomplete metrics."

    payload = {
        "protocol": "narcbench_core_stage3",
        "baseline_stage1": str(args.full),
        "j_only": str(args.j),
        "complement": str(args.c),
        "table": table_rows,
        "reading": reading,
        "j_space_definition_doc": "results/stage2/J_SPACE_DEFINITION.md",
        "cites": [
            "Rose et al. arXiv:2604.01151 (full-activation collusion probes)",
            "Gurnee/Lindsey arXiv:2607.15495 (J vs non-J)",
            "Yoo & Skapars arXiv:2609.02893 (subspace vs complement)",
        ],
    }
    (out / "comparison.json").write_text(json.dumps(payload, indent=2) + "\n")

    lines = [
        "# Stage 3 — Full vs J-only vs Complement",
        "",
        "Gemma-2-9B NARCBench Core · leave-domain-out · layers 19–23 band mean",
        "",
        "| Method | Full | J-only | Complement | Δ(J−full) | Δ(C−full) | Δ(J−C) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in table_rows:
        lines.append(
            f"| {r['method']} | {r['full']:.3f} | {r['j_only']:.3f} | {r['complement']:.3f} | "
            f"{r['delta_j_minus_full']:+.3f} | {r['delta_c_minus_full']:+.3f} | {r['delta_j_minus_c']:+.3f} |"
        )
    lines += ["", "## Reading", "", reading, "", "## Artifact paths", ""]
    lines += [
        f"- Full: `{args.full}`",
        f"- J-only: `{args.j}`",
        f"- Complement: `{args.c}`",
        f"- Definition: `results/stage2/J_SPACE_DEFINITION.md`",
        "",
    ]
    (out / "COMPARISON.md").write_text("\n".join(lines))
    print(json.dumps({"primary": {a: arms[a].get("auroc_primary") for a in arms}, "reading": reading}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
