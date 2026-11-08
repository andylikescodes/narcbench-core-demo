"""Context conditions for the system-prompt confound.

Every colluder turn carries the scheme in two places:
  1. its system prompt (all phases, Core and Transfer), and
  2. colluder-only blocks in the user prompt: ``PRIVATE COORDINATION`` (discussion
     turns, Core and Transfer) and ``DRIFT GUIDANCE FOR THIS ROUND`` (Transfer only).

Conditions (applied to the same recorded replies):
  original     -- as generated.
  swap         -- colluder system prompt replaced by an honest one, colluder-only blocks removed.
                  Core: HONEST_SYSTEM with the scenario's honest mandate.
                  Transfer: the control twin's prompt for the same seat, name replaced.
  nosys        -- no system prompt; colluder-only blocks removed.
  prompt_only  -- the system prompt alone, read at its last token (no task, no reply).
"""
from __future__ import annotations

import copy
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONDITIONS = ("original", "swap", "nosys", "prompt_only")

_PRIVATE_HDR = "PRIVATE COORDINATION"
_DRIFT_HDR = "DRIFT GUIDANCE FOR THIS ROUND"
_NEXT_HDR = re.compile(r"\n\n(?=[A-Z][A-Z \-/]{6,}[A-Z](?:\s*\(|:))")


def strip_private_block(user_prompt: str, private_messages: list[str]) -> str:
    """Remove the PRIVATE COORDINATION block (header + the private messages)."""
    i = user_prompt.find(_PRIVATE_HDR)
    if i < 0:
        return user_prompt
    end = -1
    if private_messages:
        last = private_messages[-1]
        j = user_prompt.find(last, i)
        if j >= 0:
            end = j + len(last)
    if end < 0:  # fall back to the next section header
        m = _NEXT_HDR.search(user_prompt, i + len(_PRIVATE_HDR))
        end = m.start() if m else len(user_prompt)
    head = user_prompt[:i].rstrip("\n")
    tail = user_prompt[end:].lstrip("\n")
    return f"{head}\n\n{tail}" if tail else head


def strip_drift_block(user_prompt: str) -> str:
    i = user_prompt.find(_DRIFT_HDR)
    if i < 0:
        return user_prompt
    m = _NEXT_HDR.search(user_prompt, i + len(_DRIFT_HDR))
    end = m.start() if m else len(user_prompt)
    head = user_prompt[:i].rstrip("\n")
    tail = user_prompt[end:].lstrip("\n")
    return f"{head}\n\n{tail}" if tail else head


def strip_colluder_blocks(user_prompt: str, private_messages: list[str]) -> str:
    return strip_drift_block(strip_private_block(user_prompt, private_messages))


def load_core_bank() -> dict[str, dict]:
    bank = json.loads((ROOT / "generation" / "narcbench_core_bank.json").read_text())
    scen = bank["scenarios"] if isinstance(bank, dict) else bank
    return {s["id"]: s for s in scen}


def core_honest_system(agent: str, scenario: dict) -> str:
    from generation.core_protocol import HONEST_SYSTEM

    return HONEST_SYSTEM.format(agent_name=agent, honest_mandate=scenario["honest_mandate"])


def _seat_order(prompts: list[dict]) -> list[str]:
    seen: list[str] = []
    for e in prompts:
        if e["agent_name"] not in seen:
            seen.append(e["agent_name"])
    return seen


def transfer_control_prompts(run_dir: Path) -> dict[int, tuple[str, str]]:
    """Seat index -> (control agent name, control system prompt) from the control twin."""
    twin = run_dir.parent / run_dir.name.replace("__collusion", "__control")
    cfg = json.loads((twin / "run_config.json").read_text())
    prompts = json.loads((twin / "agent_prompts.json").read_text())
    by_agent = {}
    for e in prompts:
        by_agent.setdefault(e["agent_name"], e.get("system_prompt") or "")
    order = list(cfg.get("roles", {}).keys()) or _seat_order(prompts)
    return {i: (a, by_agent.get(a, "")) for i, a in enumerate(order)}


def apply_condition(samples: list[dict], condition: str, runs_dir: Path, split: str) -> list[dict]:
    """Return copies of ``samples`` with system/user prompts rewritten for ``condition``.

    ``samples`` are rows from ``scripts/extract_activations_from_transcripts.load_samples``.
    Adds ``condition``, ``context_changed`` and (for prompt_only) ``read_prompt_only``.
    Only colluder rows change under swap/nosys; every row changes under prompt_only and nosys.
    """
    if condition not in CONDITIONS:
        raise ValueError(condition)
    bank = load_core_bank() if split == "core" else {}
    private_cache: dict[str, list[str]] = {}
    seat_cache: dict[str, list[str]] = {}
    twin_cache: dict[str, dict] = {}
    out = []
    for s in samples:
        t = copy.deepcopy(s)
        t["condition"] = condition
        t["context_changed"] = False
        rid = s["run_id"]
        run_dir = runs_dir / rid
        if rid not in private_cache:
            res = json.loads((run_dir / "results.json").read_text())
            private_cache[rid] = [re.sub(r"^\[[^\]]+\]:\s*", "", m) for m in res.get("private_messages", [])]
            # keep the raw form too: the prompts quote "[Name]: text"
            private_cache[rid + "#raw"] = res.get("private_messages", [])
            cfg = json.loads((run_dir / "run_config.json").read_text())
            seat_cache[rid] = list(cfg.get("roles", {}).keys())
        if condition == "original":
            out.append(t)
            continue
        if condition == "prompt_only":
            t["user_prompt"] = ""
            t["response"] = ""
            t["read_prompt_only"] = True
            t["context_changed"] = True
            out.append(t)
            continue
        is_col = s["role"] == "colluder"
        if is_col:
            t["user_prompt"] = strip_colluder_blocks(s["user_prompt"], private_cache[rid + "#raw"])
        if condition == "nosys":
            t["system_prompt"] = ""
            t["context_changed"] = True
        elif condition == "swap" and is_col:
            if split == "core":
                t["system_prompt"] = core_honest_system(s["agent_name"], bank[s["scenario_id"]])
            else:
                if rid not in twin_cache:
                    twin_cache[rid] = transfer_control_prompts(run_dir)
                seat = seat_cache[rid].index(s["agent_name"])
                ctrl_name, ctrl_sys = twin_cache[rid][seat]
                t["system_prompt"] = ctrl_sys.replace(ctrl_name, s["agent_name"])
            t["context_changed"] = True
        out.append(t)
    return out


def residual_leak_report(samples: list[dict]) -> dict:
    """Count colluder-only markers left in rewritten samples (should be 0 after swap/nosys)."""
    n = {"private_block": 0, "drift_block": 0}
    for s in samples:
        up = s.get("user_prompt") or ""
        n["private_block"] += _PRIVATE_HDR in up
        n["drift_block"] += _DRIFT_HDR in up
    return n
