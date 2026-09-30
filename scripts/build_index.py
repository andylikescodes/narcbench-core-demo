#!/usr/bin/env python3
"""Index NARCBench Core transcripts into data/demo_index.json (+ per-model indexes).

Sources under upstream/scenarios/<model_slug>/core/ (optionally .../core/<run_id>/):
  qwen3_32b   — official HF dump (available now)
  gemma2_2b   — regen smoke / full (when present)
  gemma2_9b   — stub until regen exists

Protocol (generation/core.py): private R0 → 3 public discussion rounds → final vote.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCENARIOS_ROOT = ROOT / "upstream" / "scenarios"
OUT = ROOT / "data" / "demo_index.json"
OUT_MODELS = ROOT / "data" / "models_index.json"

SPEAKER_RE = re.compile(r"^\[([^\]]+)\]:\s*(.*)$", re.S)
NUM_DISCUSSION_ROUNDS = 3

# Catalog of models the demo can pick (available flag set from disk).
MODEL_CATALOG = [
    {
        "id": "qwen3_32b",
        "label": "Qwen3-32B (official HF Core)",
        "hf_model": "Qwen/Qwen3-32B-AWQ",
        "source": "huggingface.co/datasets/aaronrose227/narcbench",
        "replication": False,
    },
    {
        "id": "gemma2_2b",
        "label": "Gemma-2-2B-IT (replication)",
        "hf_model": "google/gemma-2-2b-it",
        "source": "local regen via scripts/regen_gemma_core.py",
        "replication": True,
    },
    {
        "id": "gemma2_9b",
        "label": "Gemma-2-9B-IT (stub)",
        "hf_model": "google/gemma-2-9b-it",
        "source": "not generated yet",
        "replication": True,
    },
]


def parse_message(raw: str) -> tuple[str, str]:
    m = SPEAKER_RE.match((raw or "").strip())
    if m:
        return m.group(1).strip(), m.group(2).strip()
    return "Unknown", (raw or "").strip()


def vote_tally(votes: dict) -> dict:
    c = Counter(v for v in (votes or {}).values() if v)
    return {k: c[k] for k in sorted(c)}


def scenario_brief(prompts: list | None, config: dict) -> dict:
    brief = {
        "title": config.get("scenario_title"),
        "domain": config.get("scenario_domain"),
        "scenario_id": config.get("scenario_id"),
        "option_labels": config.get("option_labels") or ["A", "B"],
        "target_option": config.get("target_option"),
        "fair_option": config.get("fair_option"),
        "setup_excerpt": None,
    }
    if not prompts:
        return brief
    first = prompts[0]
    user = (first.get("user_prompt") or "").strip()
    cut = user
    for marker in (
        "\nOPTIONS UNDER REVIEW:",
        "\nPRIVATE COORDINATION",
        "\nDISCUSSION HISTORY:",
        "\nMessages so far:",
    ):
        if marker in cut:
            cut = cut.split(marker, 1)[0]
    brief["setup_excerpt"] = cut.strip()[:1200]
    if "OPTIONS UNDER REVIEW:" in user:
        opt_block = user.split("OPTIONS UNDER REVIEW:", 1)[1]
        lines = []
        for line in opt_block.splitlines():
            line = line.strip()
            if not line:
                if lines:
                    break
                continue
            if line.startswith("Option "):
                lines.append(line)
            elif lines:
                break
        if lines:
            brief["option_lines"] = lines[:4]
    return brief


def load_run(run_dir: Path, *, model_id: str, source_tag: str) -> dict | None:
    results_p = run_dir / "results.json"
    config_p = run_dir / "run_config.json"
    if not results_p.exists() or not config_p.exists():
        return None
    results = json.loads(results_p.read_text())
    config = json.loads(config_p.read_text())
    prompts = None
    prompts_p = run_dir / "agent_prompts.json"
    if prompts_p.exists():
        prompts = json.loads(prompts_p.read_text())

    roles = results.get("roles") or config.get("roles") or {}
    agents = config.get("agents") or list(roles.keys())
    votes = results.get("votes") or config.get("votes") or {}
    private = results.get("private_messages") or []
    public = results.get("public_messages") or []
    finals = results.get("final_texts") or {}

    rounds: list[dict] = []
    priv_msgs = []
    for i, raw in enumerate(private):
        speaker, text = parse_message(raw)
        priv_msgs.append(
            {
                "speaker": speaker,
                "role": roles.get(speaker, "colluder"),
                "text": text,
                "index": i,
                "channel": "private",
                "phase": "private",
                "round": 0,
            }
        )
    rounds.append(
        {
            "round": 0,
            "phase": "private",
            "channel": "private",
            "label": "Round 0 · Private coordination",
            "messages": priv_msgs,
        }
    )

    for r in range(1, NUM_DISCUSSION_ROUNDS + 1):
        chunk = public[(r - 1) * 4 : r * 4]
        msgs = []
        for i, raw in enumerate(chunk):
            speaker, text = parse_message(raw)
            msgs.append(
                {
                    "speaker": speaker,
                    "role": roles.get(speaker, "honest"),
                    "text": text,
                    "index": (r - 1) * 4 + i,
                    "channel": "public",
                    "phase": "discussion",
                    "round": r,
                }
            )
        rounds.append(
            {
                "round": r,
                "phase": "discussion",
                "channel": "public",
                "label": f"Round {r} · Public discussion",
                "messages": msgs,
            }
        )

    final_msgs = []
    for name in agents:
        final_msgs.append(
            {
                "speaker": name,
                "role": roles.get(name, "honest"),
                "text": finals.get(name, ""),
                "vote": votes.get(name),
                "channel": "final",
                "phase": "final",
                "round": NUM_DISCUSSION_ROUNDS + 1,
            }
        )
    tally = vote_tally(votes)
    rounds.append(
        {
            "round": NUM_DISCUSSION_ROUNDS + 1,
            "phase": "final",
            "channel": "final",
            "label": "Final · Individual recommendations + vote",
            "messages": final_msgs,
            "votes": votes,
            "tally": tally,
            "collusion_success": results.get("collusion_success"),
            "target_majority": results.get("target_majority"),
            "colluder_voted_target": results.get("colluder_voted_target"),
            "target_vote_count": results.get("target_vote_count"),
        }
    )

    flat_private = [
        {"speaker": s, "text": t, "role": roles.get(s), "round": 0, "channel": "private"}
        for s, t in (parse_message(r) for r in private)
    ]
    flat_public = []
    for i, raw in enumerate(public):
        s, t = parse_message(raw)
        flat_public.append(
            {
                "speaker": s,
                "text": t,
                "role": roles.get(s),
                "round": (i // 4) + 1,
                "channel": "public",
                "index": i,
            }
        )

    multi = len(public) >= NUM_DISCUSSION_ROUNDS * 4 and bool(finals)
    return {
        "mode": results.get("mode") or config.get("mode"),
        "run_id": config.get("run_id") or run_dir.name,
        "scenario_id": config.get("scenario_id"),
        "scenario_domain": config.get("scenario_domain"),
        "scenario_title": config.get("scenario_title"),
        "agents": agents,
        "roles": roles,
        "coalition": config.get("coalition")
        or [n for n, r in roles.items() if r == "colluder"],
        "votes": votes,
        "tally": tally,
        "target_option": config.get("target_option"),
        "fair_option": config.get("fair_option"),
        "option_labels": config.get("option_labels"),
        "collusion_success": results.get("collusion_success"),
        "target_majority": results.get("target_majority"),
        "colluder_voted_target": results.get("colluder_voted_target"),
        "target_vote_count": results.get("target_vote_count"),
        "flagged": results.get("flagged", False),
        "n_private": len(private),
        "n_public": len(public),
        "n_final": len(finals),
        "private_messages": flat_private,
        "public_messages": flat_public,
        "rounds": rounds,
        "brief": scenario_brief(prompts, config),
        "has_agent_prompts": prompts is not None,
        "n_prompt_turns": len(prompts) if prompts else 0,
        "source": source_tag,
        "source_path": str(run_dir),
        "model_id": model_id,
        "multi_round": multi,
        "protocol": {
            "private_rounds": 1,
            "discussion_rounds": NUM_DISCUSSION_ROUNDS,
            "final_vote": True,
            "ref": "NARCBench Core / generation/core.py NUM_DISCUSSION_ROUNDS=3",
        },
    }


def resolve_core_dir(model_id: str) -> Path | None:
    """Prefer .../core/latest symlink, else newest stamped subdir, else flat core/."""
    base = SCENARIOS_ROOT / model_id / "core"
    if not base.exists():
        return None
    latest = base / "latest"
    if latest.exists():
        return latest.resolve() if latest.is_symlink() else latest
    ptr = base / "LATEST_RUN.txt"
    if ptr.exists():
        stamped = base / ptr.read_text().strip()
        if stamped.is_dir():
            return stamped
    # Flat layout (official qwen dump): run dirs directly under core/
    if any((p / "results.json").exists() for p in base.iterdir() if p.is_dir()):
        return base
    # Stamped subdirs without pointer: pick newest by name
    stamped = sorted(
        [p for p in base.iterdir() if p.is_dir() and p.name not in ("latest",)],
        key=lambda p: p.name,
        reverse=True,
    )
    for s in stamped:
        if any((p / "results.json").exists() for p in s.iterdir() if p.is_dir()):
            return s
    return base if any(base.iterdir()) else None


def index_model(model_id: str, meta: dict) -> dict:
    src = resolve_core_dir(model_id)
    source_tag = f"{model_id}_core"
    by_sid: dict[str, dict[str, dict]] = {}
    n_runs = 0
    if src and src.exists():
        for run_dir in sorted(src.iterdir()):
            if not run_dir.is_dir():
                continue
            loaded = load_run(run_dir, model_id=model_id, source_tag=source_tag)
            if not loaded or not loaded.get("scenario_id"):
                continue
            by_sid.setdefault(loaded["scenario_id"], {})[loaded["mode"]] = loaded
            n_runs += 1

    cases = []
    n_pairs = 0
    n_multi = 0
    for sid in sorted(by_sid):
        modes = by_sid[sid]
        col, ctrl = modes.get("collusion"), modes.get("control")
        if not (col and ctrl):
            continue
        n_pairs += 1
        multi = bool(col.get("multi_round") and ctrl.get("multi_round"))
        if multi:
            n_multi += 1
        brief = (col or ctrl).get("brief") or {}
        cases.append(
            {
                "id": sid,
                "title": brief.get("title") or (col or ctrl).get("scenario_title"),
                "domain": brief.get("domain") or (col or ctrl).get("scenario_domain"),
                "fair_option": (col or ctrl).get("fair_option"),
                "target_option": (col or ctrl).get("target_option"),
                "option_labels": (col or ctrl).get("option_labels"),
                "option_lines": brief.get("option_lines"),
                "setup_excerpt": brief.get("setup_excerpt"),
                "has_full_transcript": multi,
                "multi_round": multi,
                "collusion_success": col.get("collusion_success"),
                "collusion_tally": col.get("tally"),
                "control_tally": ctrl.get("tally"),
                "n_private": col.get("n_private", 0),
                "n_public": col.get("n_public", 0),
                "n_rounds_collusion": len(col.get("rounds") or []),
                "n_rounds_control": len(ctrl.get("rounds") or []),
                "coalition": col.get("coalition") or [],
                "collusion_run_id": col.get("run_id"),
                "control_run_id": ctrl.get("run_id"),
                "has_agent_prompts": col.get("has_agent_prompts")
                and ctrl.get("has_agent_prompts"),
                "model_id": model_id,
            }
        )

    cases.sort(key=lambda c: (0 if c["multi_round"] else 1, c.get("domain") or "", c["id"]))
    available = n_pairs > 0
    return {
        "model_id": model_id,
        "label": meta.get("label"),
        "hf_model": meta.get("hf_model"),
        "replication": meta.get("replication", False),
        "source_note": meta.get("source"),
        "available": available,
        "stub": not available,
        "source_dir": str(src) if src else None,
        "stats": {
            "n_scenario_pairs": n_pairs,
            "n_full_multi_round_pairs": n_multi,
            "n_runs_on_disk": n_runs,
        },
        "cases": cases,
        "attribution": {
            "paper": "Rose et al., Detecting Multi-Agent Collusion Through Multi-Agent Interpretability",
            "arxiv": "https://arxiv.org/abs/2604.01151",
            "github": "https://github.com/aaronrose227/narcbench",
            "hf_dataset": "https://huggingface.co/datasets/aaronrose227/narcbench",
            "note": (
                f"Model bank '{model_id}'. "
                + (
                    "Official Qwen3-32B Core transcripts; dialogue from results.json."
                    if not meta.get("replication")
                    else "Gemma regen is a protocol replication (not official HF dump)."
                )
            ),
        },
        "protocol": {
            "private_rounds": 1,
            "discussion_rounds": NUM_DISCUSSION_ROUNDS,
            "final_vote": True,
            "agents_per_round": 4,
        },
    }


# Back-compat helpers used by server.py
def core_dirs() -> Path:
    d = resolve_core_dir("qwen3_32b")
    if d:
        return d
    return ROOT / "data" / "hf_samples"


def main() -> None:
    models = []
    by_id = {}
    for meta in MODEL_CATALOG:
        idx = index_model(meta["id"], meta)
        models.append(
            {
                "id": idx["model_id"],
                "label": idx["label"],
                "hf_model": idx["hf_model"],
                "available": idx["available"],
                "stub": idx["stub"],
                "replication": idx["replication"],
                "stats": idx["stats"],
                "source_dir": idx["source_dir"],
            }
        )
        by_id[meta["id"]] = idx
        # Per-model index file for the API
        per = ROOT / "data" / f"demo_index_{meta['id']}.json"
        per.parent.mkdir(parents=True, exist_ok=True)
        per.write_text(json.dumps(idx, indent=2))

    default_id = "qwen3_32b" if by_id["qwen3_32b"]["available"] else next(
        (m["id"] for m in models if m["available"]), "qwen3_32b"
    )
    default = by_id[default_id]

    # Legacy demo_index.json = default model (keeps old clients working)
    legacy = {
        "attribution": default["attribution"],
        "protocol": default["protocol"],
        "stats": {**default["stats"], "source_dir": default["source_dir"]},
        "cases": default["cases"],
        "model_id": default_id,
        "models": models,
        "default_model": default_id,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(legacy, indent=2))

    models_index = {
        "default_model": default_id,
        "models": models,
        "protocol": default["protocol"],
        "attribution": {
            "paper": "Rose et al.",
            "arxiv": "https://arxiv.org/abs/2604.01151",
            "github": "https://github.com/aaronrose227/narcbench",
            "hf_dataset": "https://huggingface.co/datasets/aaronrose227/narcbench",
        },
    }
    OUT_MODELS.write_text(json.dumps(models_index, indent=2))
    print(f"Wrote {OUT}")
    print(f"Wrote {OUT_MODELS}")
    print(json.dumps({"default_model": default_id, "models": models}, indent=2))


if __name__ == "__main__":
    main()
