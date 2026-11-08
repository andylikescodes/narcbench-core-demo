"""Load extraction folders and turn rows into per-agent feature records for the detectors."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def load_extraction(folder: Path):
    meta = json.loads((Path(folder) / "metadata.json").read_text())
    npz = np.load(Path(folder) / "activations.npz")
    return meta, npz


def domain(m: dict, split: str) -> str:
    if split == "core":
        p = (m.get("scenario_id") or "").split("_")
        return "_".join(p[:-1]) if len(p) >= 2 else m.get("scenario_id", "")
    return m.get("scenario_domain") or m["run_id"].split("__")[0]


def select_rows(meta, phase: str, prompt_only: bool = False) -> np.ndarray:
    """Row indices used by the probe protocols: public turns of ``phase`` (prompt_only: one row per agent)."""
    idx = []
    seen = set()
    for i, m in enumerate(meta):
        if prompt_only:
            k = (m["run_id"], m["agent_name"])
            if k in seen or m.get("channel") != "public":
                continue
            seen.add(k)
            idx.append(i)
        elif m.get("channel") == "public" and m.get("phase") == phase:
            idx.append(i)
    return np.array(idx, dtype=int)


def to_agents(meta, idx, X, split: str) -> list[dict]:
    """Average a row-feature matrix X (aligned with idx) per (run, agent)."""
    acc: dict[tuple, dict] = {}
    for j, i in enumerate(idx):
        m = meta[i]
        k = (m["run_id"], m["agent_name"])
        a = acc.setdefault(k, {"run_id": m["run_id"], "agent_name": m["agent_name"], "role": m["role"],
                               "mode": m["mode"], "domain": domain(m, split), "vote": m.get("agent_vote"),
                               "target": m.get("target_option"), "scenario_id": m.get("scenario_id"),
                               "_x": []})
        a["_x"].append(X[j])
    out = []
    for a in acc.values():
        a["x"] = np.mean(a.pop("_x"), axis=0)
        out.append(a)
    return out
