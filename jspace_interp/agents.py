"""Build per-agent records from the transcripts in upstream/scenarios (no activations needed)."""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

CORE_RUNS = ROOT / "upstream/scenarios/gemma2_9b/core/20261001T012639Z"
XFER_RUNS = ROOT / "upstream/scenarios/gemma2_9b/transfer/RUNPOD"


def load_rows(split: str):
    from extract_activations_from_transcripts import load_samples  # noqa: E402

    return load_samples(CORE_RUNS if split == "core" else XFER_RUNS)


def domain_of(row: dict, split: str) -> str:
    if split == "core":
        sid = row.get("scenario_id", "")
        parts = sid.split("_")
        return "_".join(parts[:-1]) if len(parts) >= 2 else sid
    return row.get("scenario_domain") or row["run_id"].split("__")[0]


def agent_texts(rows: list[dict], split: str, phase: str, field: str = "response") -> list[dict]:
    """One record per (run, agent) with the concatenated ``field`` text of its turns in ``phase``.

    phase="discussion" uses public discussion rounds; "final" the closing turn.
    """
    acc: dict[tuple, dict] = {}
    for r in rows:
        if r["channel"] != "public" or r["phase"] != phase:
            continue
        k = (r["run_id"], r["agent_name"])
        a = acc.setdefault(k, {"run_id": r["run_id"], "agent_name": r["agent_name"], "role": r["role"],
                               "mode": r["mode"], "domain": domain_of(r, split),
                               "scenario_id": r.get("scenario_id", ""),
                               "vote": r.get("agent_vote"), "target": r.get("target_option"),
                               "texts": []})
        a["texts"].append(r.get(field) or "")
    return list(acc.values())
