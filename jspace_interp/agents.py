"""Build per-agent records from the transcripts in upstream/scenarios (no activations needed).

Splits: ``core`` and ``transfer`` are the recorded Gemma-2-9B-it episodes; ``warned`` are the matched-vocabulary
control episodes written by scripts/jspace_interp_gen_matched_vocab.py (control-mode Core episodes whose seats 0-1
carry the chair's warning in the colluders' vocabulary). Warned rows get ``warned`` (the warning is in the agent's
system prompt) and ``seat`` (the agent's index in the drawn order) so the scoring can split warned from unwarned
honest agents inside each episode.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

CORE_RUNS = ROOT / "upstream/scenarios/gemma2_9b/core/20261001T012639Z"
XFER_RUNS = ROOT / "upstream/scenarios/gemma2_9b/transfer/RUNPOD"
WARNED_RUNS = ROOT / "results/jspace_interp/matched_vocab_runs"
WARNING_MARKER = "Integrity notice from the committee chair"
SPLITS = ("core", "transfer", "warned")


def runs_dir_for(split: str, warned_runs: Path | None = None) -> Path:
    if split == "core":
        return CORE_RUNS
    if split == "transfer":
        return XFER_RUNS
    if split == "warned":
        return Path(warned_runs) if warned_runs else WARNED_RUNS
    raise ValueError(split)


def core_like(split: str) -> bool:
    """Warned episodes reuse the Core scenarios, so domains and swaps follow the Core rules."""
    return split in ("core", "warned")


def annotate_warned(rows: list[dict], runs_dir: Path) -> list[dict]:
    """Add ``warned`` and ``seat`` to each row (in place) from the system prompt and run_config.json."""
    seats: dict[str, list[str]] = {}
    for r in rows:
        rid = r["run_id"]
        if rid not in seats:
            cfg_p = runs_dir / rid / "run_config.json"
            cfg = json.loads(cfg_p.read_text()) if cfg_p.exists() else {}
            seats[rid] = list(cfg.get("agents") or (cfg.get("roles") or {}).keys())
        r["warned"] = WARNING_MARKER in (r.get("system_prompt") or "")
        r["seat"] = seats[rid].index(r["agent_name"]) if r["agent_name"] in seats[rid] else -1
    return rows


def load_rows(split: str, warned_runs: Path | None = None):
    from extract_activations_from_transcripts import load_samples  # noqa: E402

    runs_dir = runs_dir_for(split, warned_runs)
    rows = load_samples(runs_dir)
    if split == "warned":
        annotate_warned(rows, runs_dir)
    return rows


def domain_of(row: dict, split: str) -> str:
    if core_like(split):
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
                               "warned": bool(r.get("warned", False)), "seat": r.get("seat", -1),
                               "texts": []})
        a["texts"].append(r.get(field) or "")
    return list(acc.values())
