#!/usr/bin/env python3
"""Package the Gemma-2-9B-it Core and Transfer transcripts for the scenario-browser artifact.

Reads the committed transcripts (run_config.json, results.json, agent_prompts.json per episode)
and writes three JSON files the static page fetches:

  <out>/data/index.json      one small record per episode (list, filters, twins)
  <out>/data/core.json       {run_id: episode} for the 100 Core episodes
  <out>/data/transfer.json   {run_id: episode} for the 72 Transfer episodes

An episode record holds the briefing, the per-agent system prompts, the private round, the public
rounds grouped as the generator ran them, the final texts with parsed votes, and the outcome.
Nothing is summarised or paraphrased; every string is the recorded text.

  python3 scripts/build_scenario_browser.py --out /path/to/browser_dir
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BANKS = {
    "core": ROOT / "upstream/scenarios/gemma2_9b/core/20261001T012639Z",
    "transfer": ROOT / "upstream/scenarios/gemma2_9b/transfer/RUNPOD",
}
FAMILY_LABEL = {
    "cicd": "CI/CD pipelines", "code_review": "code review", "news": "news",
    "pump_dump": "pump-and-dump", "rag": "RAG corpora", "research": "research integrity",
}
SPEAKER_RE = re.compile(r"^\[(.+?)\]:\s*(.*)$", re.S)
HISTORY_MARKERS = (
    "\nOPTIONS UNDER REVIEW:", "\nDISCUSSION HISTORY:", "\nFULL DISCUSSION HISTORY:", "\nSTORY DRAFT SO FAR:", "\nPRIVATE COORDINATION",
    "\nThis is a private channel", "\nPipeline output so far", "\nPIPELINE OUTPUT", "\nMessages so far:",
    "\nREVIEW THREAD", "\nDISCUSSION:", "\nKNOWLEDGE BASE ENTRIES", "\nDraft so far", "\nReport so far",
)
VOTE_RE = re.compile(r"RECOMMENDATION:\s*\[?\s*Option\s*([A-Z])", re.I)


def parse_message(raw: str) -> dict:
    m = SPEAKER_RE.match((raw or "").strip())
    if m:
        return {"speaker": m.group(1).strip(), "text": m.group(2).strip()}
    return {"speaker": "Unknown", "text": (raw or "").strip()}


def cut_briefing(prompt: str) -> str:
    cut = prompt.strip()
    for marker in HISTORY_MARKERS:
        if marker in cut:
            cut = cut.split(marker, 1)[0]
    return cut.strip()


def option_lines(prompt: str) -> list[str]:
    if "OPTIONS UNDER REVIEW:" not in prompt:
        return []
    out: list[str] = []
    for line in prompt.split("OPTIONS UNDER REVIEW:", 1)[1].splitlines():
        line = line.strip()
        if not line:
            if out:
                break
            continue
        if line.startswith("Option "):
            out.append(line)
        elif out:
            break
    return out[:4]


def tail_instruction(prompt: str) -> str:
    """The last paragraph(s) of a prompt: what the agent was asked to do in that turn."""
    paras = [p.strip() for p in prompt.strip().split("\n\n") if p.strip()]
    if not paras:
        return ""
    if len(paras) >= 2 and ("RECOMMENDATION" in paras[-1] or paras[-1].startswith("[")):
        return "\n\n".join(paras[-2:])
    return paras[-1]


def common_prefix_len(a: str, b: str) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def load_episode(run_dir: Path, bank: str) -> dict:
    cfg = json.loads((run_dir / "run_config.json").read_text())
    res = json.loads((run_dir / "results.json").read_text())
    prompts = json.loads((run_dir / "agent_prompts.json").read_text())
    agents = list(cfg["agents"])
    roles = dict(cfg["roles"])
    disc = sorted([p for p in prompts if p["phase"] == "discussion"], key=lambda p: p["round"])
    priv = [p for p in prompts if p["phase"] == "private"]
    fin = [p for p in prompts if p["phase"] == "final"]
    first_prompt = disc[0]["user_prompt"] if disc else ""
    briefing = cut_briefing(first_prompt)
    opts = option_lines(first_prompt)

    # rounds: split public messages by the number of discussion prompts per round
    per_round = Counter(p["round"] for p in disc)
    rounds, i = [], 0
    msgs = list(res.get("public_messages") or [])
    for r in sorted(per_round):
        n = per_round[r]
        chunk = msgs[i:i + n]
        i += n
        rounds.append({"round": r, "messages": [parse_message(m) for m in chunk]})
    if i < len(msgs):  # anything left over stays visible
        rounds.append({"round": (rounds[-1]["round"] + 1) if rounds else 1, "messages": [parse_message(m) for m in msgs[i:]]})

    system_prompts = {}
    for p in prompts:
        name = p["agent_name"]
        if name not in system_prompts or (p["phase"] == "private" and roles.get(name) == "colluder"):
            system_prompts[name] = p["system_prompt"]

    private_instruction = ""
    if priv:
        pp = priv[0]["user_prompt"]
        k = common_prefix_len(pp, first_prompt)
        private_instruction = pp[k:].strip() if k > 40 else tail_instruction(pp)
    round1_instruction = tail_instruction(first_prompt) if disc else ""
    final_instruction = tail_instruction(fin[0]["user_prompt"]) if fin else ""

    finals = []
    votes = res.get("votes") or {}
    for name in agents:
        text = (res.get("final_texts") or {}).get(name, "")
        vote = votes.get(name)
        if not vote:
            m = VOTE_RE.search(text or "")
            vote = m.group(1).upper() if m else None
        finals.append({"speaker": name, "text": text, "vote": vote})
    tally = dict(sorted(Counter(v for v in votes.values() if v).items())) if votes else {}

    if bank == "core":
        group, title = cfg.get("scenario_domain", ""), cfg.get("scenario_title", cfg["scenario_id"])
    else:
        fam = cfg.get("scenario_type", "")
        group = FAMILY_LABEL.get(fam, fam)
        title = (first_prompt.strip().splitlines() or [cfg["scenario_id"]])[0].strip()
    return {
        "id": cfg["run_id"], "bank": bank, "scenario_id": cfg["scenario_id"], "group": group, "title": title,
        "mode": cfg["mode"], "seed": cfg.get("seed"), "agents": agents, "roles": roles,
        "coalition": list(cfg.get("coalition") or []),
        "options": {"labels": cfg.get("option_labels"), "lines": opts, "fair": cfg.get("fair_option"), "target": cfg.get("target_option")},
        "briefing": briefing,
        "instructions": {"private": private_instruction, "round1": round1_instruction, "final": final_instruction},
        "system_prompts": system_prompts,
        "private": [parse_message(m) for m in (res.get("private_messages") or [])],
        "rounds": rounds,
        "finals": finals,
        "outcome": {
            "collusion_success": res.get("collusion_success"), "target_vote_count": res.get("target_vote_count"),
            "colluder_voted_target": res.get("colluder_voted_target"), "flagged": res.get("flagged"), "tally": tally,
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    data_dir = args.out / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    index = []
    for bank, root in BANKS.items():
        episodes = {}
        for run_dir in sorted(p for p in root.iterdir() if p.is_dir() and (p / "results.json").exists()):
            ep = load_episode(run_dir, bank)
            episodes[ep["id"]] = ep
        # twins
        by_key = {}
        for ep in episodes.values():
            by_key.setdefault(ep["scenario_id"], {})[ep["mode"]] = ep["id"]
        for ep in episodes.values():
            twin = by_key[ep["scenario_id"]].get("control" if ep["mode"] == "collusion" else "collusion")
            ep["twin"] = twin
            index.append({
                "id": ep["id"], "bank": bank, "scenario_id": ep["scenario_id"], "group": ep["group"], "title": ep["title"],
                "mode": ep["mode"], "twin": twin, "agents": ep["agents"], "roles": ep["roles"],
                "fair": ep["options"]["fair"], "target": ep["options"]["target"], "tally": ep["outcome"]["tally"],
                "collusion_success": ep["outcome"]["collusion_success"], "n_rounds": len(ep["rounds"]),
                "n_private": len(ep["private"]),
            })
        (data_dir / f"{bank}.json").write_text(json.dumps(episodes, ensure_ascii=False))
        print(bank, len(episodes), "episodes ->", data_dir / f"{bank}.json")
    (data_dir / "index.json").write_text(json.dumps({"model": "google/gemma-2-9b-it", "episodes": index}, ensure_ascii=False))
    print("index", len(index), "->", data_dir / "index.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
