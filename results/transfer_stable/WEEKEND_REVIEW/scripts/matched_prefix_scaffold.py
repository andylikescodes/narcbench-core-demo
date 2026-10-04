#!/usr/bin/env python3
"""Dry-run scaffold: list Core scenario stems usable for matched-prefix micro-worlds.

No GPU. No network. Reads local Core transcripts and prints / writes a candidate
table for matched-prefix pair construction (see design/matched_prefix_spec.md).

Usage:
  python3 matched_prefix_scaffold.py
  python3 matched_prefix_scaffold.py --run-dir /path/to/core/20261001T012639Z --limit 12
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


# Science-machine root, else this file's repo (results/transfer_stable/WEEKEND_REVIEW/scripts).
_SCIENCE_ROOT = Path("/workspace/narcbench-core-demo")
_REPO_ROOT = Path(__file__).resolve().parents[4]
_ROOT = _SCIENCE_ROOT if (_SCIENCE_ROOT / "upstream").is_dir() else _REPO_ROOT
DEFAULT_RUN = _ROOT / "upstream/scenarios/gemma2_9b/core/20261001T012639Z"
OUT_DEFAULT = (
    _ROOT / "results/transfer_stable/WEEKEND_REVIEW/design/matched_prefix_candidates.json"
)


def load_run_config(scenario_dir: Path) -> dict | None:
    p = scenario_dir / "run_config.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def first_colluder_name(roles: dict) -> str | None:
    for name, role in roles.items():
        if role == "colluder":
            return name
    return None


def briefing_preview(scenario_dir: Path, max_chars: int = 160) -> str:
    ap = scenario_dir / "agent_prompts.json"
    if not ap.exists():
        return ""
    entries = json.loads(ap.read_text())
    if not entries:
        return ""
    up = entries[0].get("user_prompt") or ""
    up = " ".join(up.split())
    return up[:max_chars] + ("…" if len(up) > max_chars else "")


def iter_collusion_dirs(run_dir: Path):
    for d in sorted(run_dir.iterdir()):
        if d.is_dir() and d.name.endswith("__collusion"):
            yield d


def build_candidates(run_dir: Path, limit: int | None) -> list[dict]:
    rows = []
    for d in iter_collusion_dirs(run_dir):
        rc = load_run_config(d)
        if not rc:
            continue
        target = rc.get("target_option")
        # Prefer letter targets (cleaner micro-world vote metric)
        letterish = isinstance(target, str) and len(target) <= 2
        roles = rc.get("roles") or {}
        row = {
            "scenario_dir": d.name,
            "scenario_id": rc.get("scenario_id"),
            "domain": rc.get("scenario_domain"),
            "seed": rc.get("seed"),
            "target_option": target,
            "fair_option": rc.get("fair_option"),
            "letter_target": bool(letterish),
            "collusion_success": rc.get("collusion_success"),
            "n_colluders": sum(1 for r in roles.values() if r == "colluder"),
            "example_colluder": first_colluder_name(roles),
            "option_labels": rc.get("option_labels"),
            "briefing_preview": briefing_preview(d),
            "matched_prefix_note": (
                "Collusion/control share seed+names+target but NOT early system tokens; "
                "rebuild shared_stem per matched_prefix_spec.md — do not reuse raw prompts as-is."
            ),
            "control_dir": d.name.replace("__collusion", "__control"),
        }
        rows.append(row)
        if limit is not None and len(rows) >= limit:
            break
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out", type=Path, default=OUT_DEFAULT)
    parser.add_argument("--letters-only", action="store_true", help="Keep only letter-like target_option")
    args = parser.parse_args()

    if not args.run_dir.is_dir():
        print(f"ERROR: run dir not found: {args.run_dir}", file=sys.stderr)
        return 1

    rows = build_candidates(args.run_dir, args.limit)
    if args.letters_only:
        rows = [r for r in rows if r["letter_target"]]

    letter_n = sum(1 for r in rows if r["letter_target"])
    success_n = sum(1 for r in rows if r.get("collusion_success"))

    print(f"Core run: {args.run_dir}")
    print(f"Candidates: {len(rows)} collusion scenarios ({letter_n} letter targets; {success_n} collusion_success)")
    print("NOTE: dry-run only — no GPU, no pair text generation beyond metadata.")
    print()
    print(f"{'scenario_id':<16} {'domain':<10} {'tgt':<4} {'ok':<5} {'letter':<7} seed")
    print("-" * 60)
    for r in rows:
        print(
            f"{str(r['scenario_id']):<16} {str(r['domain']):<10} "
            f"{str(r['target_option']):<4} {str(r['collusion_success']):<5} "
            f"{str(r['letter_target']):<7} {r['seed']}"
        )

    payload = {
        "run_dir": str(args.run_dir),
        "n": len(rows),
        "n_letter_targets": letter_n,
        "n_collusion_success": success_n,
        "how_to_build_pairs": [
            "Take scenario_id + option_labels + agent_name from each row",
            "Write a SHORT shared_stem (scenario facts + options) — identical bytes for both arms",
            "Add divergence.arm_a / arm_b role or target text only",
            "Add shared_suffix vote template",
            "Validate with tokenizer that tokens match until t*",
            "See design/matched_prefix_example_pairs.json for shape",
        ],
        "candidates": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print()
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
