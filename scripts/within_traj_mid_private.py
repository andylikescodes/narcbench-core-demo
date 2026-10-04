#!/usr/bin/env python3
"""Within-trajectory mid-private intervene on Core (teacher-forced).

Protocol (NEXT_CAUSAL_RANKED #2 / within_trajectory_spec.md):
  - Rebuild first colluder private-turn prompt from Core transcript.
  - Teacher-force recorded private utterance tokens 0…k−1 (mid-utterance).
  - Intervene from k onward at attn_L22 + residual L21.
  - Continue generation (temp 0); score continuation edit + vote secondary.

Arms: baseline_tf, ablate_role (attn_L22 / resid_L21), ablate_role_perp,
      random, steer_honest (optional +β on honest twin if present).

Offline dry-run needs no GPU:
  python3 scripts/within_traj_mid_private.py --dry-run --max-scenarios 6

GPU (parent after greenlight; do NOT launch from this script):
  python3 scripts/within_traj_mid_private.py \\
    --run-dir upstream/scenarios/gemma2_9b/core/20261001T012639Z \\
    --directions results/transfer_stable/directions \\
    --out results/transfer_stable/within_traj_mid_private_smoke \\
    --max-scenarios 6 --k-frac 0.5
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any

import numpy as np

try:
    import torch
except ImportError:
    torch = None  # type: ignore

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = ROOT / "upstream/scenarios/gemma2_9b/core/20261001T012639Z"
DEFAULT_DIRS = ROOT / "results/transfer_stable/directions"
DEFAULT_OUT = ROOT / "results/transfer_stable/within_traj_mid_private_dry_run"


def _unit(path: Path) -> np.ndarray:
    u = np.load(path).astype(np.float64)
    return (u / (np.linalg.norm(u) + 1e-12)).astype(np.float32)


def iter_collusion_dirs(run_dir: Path):
    for d in sorted(run_dir.iterdir()):
        if d.is_dir() and d.name.endswith("__collusion"):
            yield d


def load_json(p: Path) -> dict | list:
    return json.loads(p.read_text())


def first_colluder_private(scenario_dir: Path) -> dict | None:
    """Return {prompt_entry, recorded_text, agent, target, fair, roles, rc}."""
    rc = load_json(scenario_dir / "run_config.json")
    if not isinstance(rc, dict):
        return None
    target = rc.get("target_option")
    if not (isinstance(target, str) and len(str(target)) <= 2):
        return None  # letter targets only for primary vote metric
    roles = rc.get("roles") or {}
    prompts = load_json(scenario_dir / "agent_prompts.json")
    results = load_json(scenario_dir / "results.json")
    if not isinstance(prompts, list) or not isinstance(results, dict):
        return None

    # First private-phase colluder prompt (round 0)
    priv_entries = [
        e
        for e in prompts
        if (e.get("phase") == "private" or e.get("channel") == "private")
        and roles.get(e.get("agent_name")) == "colluder"
    ]
    if not priv_entries:
        return None
    # Prefer round 0 / earliest
    priv_entries.sort(key=lambda e: (e.get("round") is None, e.get("round") or 0))
    entry = priv_entries[0]
    agent = entry.get("agent_name")

    # Recorded private utterance for this agent
    recorded = None
    for msg in results.get("private_messages") or []:
        # format: "[Name]: text"
        m = re.match(r"\[([^\]]+)\]:\s*(.*)", msg, re.S)
        if m and m.group(1).strip() == agent:
            recorded = m.group(2).strip()
            break
    if not recorded:
        return None

    return {
        "scenario_dir": scenario_dir.name,
        "scenario_id": rc.get("scenario_id"),
        "domain": rc.get("scenario_domain"),
        "seed": rc.get("seed"),
        "agent": agent,
        "target_option": str(target).upper(),
        "fair_option": str(rc.get("fair_option") or "").upper(),
        "option_labels": rc.get("option_labels") or [],
        "collusion_success": rc.get("collusion_success"),
        "system_prompt": entry.get("system_prompt") or "",
        "user_prompt": entry.get("user_prompt") or "",
        "recorded_utterance": recorded,
        "roles": roles,
        "n_private_msgs": len(results.get("private_messages") or []),
        "n_public_msgs": len(results.get("public_messages") or []),
    }


def select_scenarios(run_dir: Path, max_scenarios: int) -> list[dict]:
    rows = []
    seen_domain: dict[str, int] = {}
    for d in iter_collusion_dirs(run_dir):
        row = first_colluder_private(d)
        if not row:
            continue
        if not row.get("collusion_success"):
            continue
        dom = row.get("domain") or "?"
        if seen_domain.get(dom, 0) >= 1 and len(rows) < max_scenarios:
            # prefer domain diversity first
            pass
        if seen_domain.get(dom, 0) >= 1:
            continue
        rows.append(row)
        seen_domain[dom] = seen_domain.get(dom, 0) + 1
        if len(rows) >= max_scenarios:
            break
    # fill
    if len(rows) < max_scenarios:
        have = {r["scenario_id"] for r in rows}
        for d in iter_collusion_dirs(run_dir):
            row = first_colluder_private(d)
            if not row or row["scenario_id"] in have:
                continue
            if not row.get("collusion_success"):
                continue
            rows.append(row)
            if len(rows) >= max_scenarios:
                break
    return rows[:max_scenarios]


def build_chat_prompt(tok, system: str, user: str) -> str:
    body = f"System:\n{system}\n\nUser:\n{user}"
    messages = [{"role": "user", "content": body}]
    if hasattr(tok, "apply_chat_template"):
        try:
            return tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            pass
    return f"<start_of_turn>user\n{body}<end_of_turn>\n<start_of_turn>model\n"


def _module_for_site(model, layer: int, site: str):
    block = model.model.layers[layer]
    s = (site or "residual").lower()
    if s in ("residual", "resid", ""):
        return block
    if s == "attn":
        return block.self_attn
    if s == "mlp":
        return block.mlp
    raise ValueError(f"unknown site {site}")


def make_persist_ablate_hook(start_pos: int, u: "torch.Tensor", alpha: float):
    """Ablate û from token position start_pos onward (prefill + decode last)."""

    def hook(_m, _i, out):
        h = out[0] if isinstance(out, tuple) else out
        h = h.clone()
        u_d = u.to(device=h.device, dtype=torch.float32)
        seq = h.shape[1]
        if seq > start_pos:
            # Prefill covering intervene region: edit start_pos…end
            v = h[0, start_pos:, :].float()
            proj = v @ u_d
            h[0, start_pos:, :] = (v - alpha * proj[:, None] * u_d).to(dtype=h.dtype)
        else:
            # Decode step: edit last token
            v = h[0, -1, :].float()
            proj = torch.dot(v, u_d)
            h[0, -1, :] = (v - alpha * proj * u_d).to(dtype=h.dtype)
        return (h,) + out[1:] if isinstance(out, tuple) else h

    return hook


def make_persist_steer_hook(start_pos: int, u: "torch.Tensor", beta: float, scale: float):
    """Add +β·scale·û from start_pos onward."""

    def hook(_m, _i, out):
        h = out[0] if isinstance(out, tuple) else out
        h = h.clone()
        u_d = u.to(device=h.device, dtype=h.dtype)
        delta = (beta * scale) * u_d
        seq = h.shape[1]
        if seq > start_pos:
            h[0, start_pos:, :] = h[0, start_pos:, :] + delta
        else:
            h[0, -1, :] = h[0, -1, :] + delta
        return (h,) + out[1:] if isinstance(out, tuple) else h

    return hook


def build_arms(directions: Path, seed: int = 0) -> list[dict]:
    role_attn = directions / "lr_role_attn_L22.npy"
    role_l21 = directions / "lr_role_L21.npy"
    perp_l21 = directions / "lr_role_perp_mode_L21.npy"
    for p in (role_attn, role_l21):
        if not p.exists():
            raise FileNotFoundError(p)
    u_role_attn = _unit(role_attn)
    u_role_l21 = _unit(role_l21)
    u_perp = _unit(perp_l21) if perp_l21.exists() else None
    rng = np.random.default_rng(seed)
    u_rand = rng.normal(size=u_role_attn.shape).astype(np.float64)
    u_rand = (u_rand / (np.linalg.norm(u_rand) + 1e-12)).astype(np.float32)

    arms: list[dict] = [
        {"name": "baseline_tf", "kind": "baseline"},
        {
            "name": "ablate_role_attn_L22",
            "kind": "ablate",
            "site": "attn",
            "layer": 22,
            "u": u_role_attn,
            "alpha": 1.0,
            "dir_file": "lr_role_attn_L22.npy",
        },
        {
            "name": "ablate_role_resid_L21",
            "kind": "ablate",
            "site": "residual",
            "layer": 21,
            "u": u_role_l21,
            "alpha": 1.0,
            "dir_file": "lr_role_L21.npy",
        },
        {
            "name": "ablate_random_attn_L22",
            "kind": "ablate",
            "site": "attn",
            "layer": 22,
            "u": u_rand,
            "alpha": 1.0,
            "dir_file": "random",
        },
        {
            "name": "steer_role_attn_L22",
            "kind": "steer",
            "site": "attn",
            "layer": 22,
            "u": u_role_attn,
            "beta": 1.0,
            "dir_file": "lr_role_attn_L22.npy",
        },
    ]
    if u_perp is not None:
        arms.append(
            {
                "name": "ablate_role_perp_resid_L21",
                "kind": "ablate",
                "site": "residual",
                "layer": 21,
                "u": u_perp,
                "alpha": 1.0,
                "dir_file": "lr_role_perp_mode_L21.npy",
            }
        )
    return arms


def continuation_metrics(gen_text: str, recorded_suffix: str) -> dict:
    """Primary: edit distance-ish vs TF-recorded suffix; keyword overlap."""
    a = (gen_text or "").strip()
    b = (recorded_suffix or "").strip()
    # char-level normalized Levenshtein (bounded)
    def lev(s, t):
        if len(s) > 400:
            s = s[:400]
        if len(t) > 400:
            t = t[:400]
        if not s and not t:
            return 0
        prev = list(range(len(t) + 1))
        for i, cs in enumerate(s, 1):
            cur = [i]
            for j, ct in enumerate(t, 1):
                cur.append(min(cur[-1] + 1, prev[j] + 1, prev[j - 1] + (cs != ct)))
            prev = cur
        return prev[-1]

    dist = lev(a.lower(), b.lower())
    denom = max(len(a), len(b), 1)
    return {
        "cont_char_len": len(a),
        "recorded_suffix_len": len(b),
        "edit_distance": dist,
        "edit_frac": dist / denom,
        "exact_match": a == b,
        "prefix_match_20": a[:20] == b[:20] if a and b else False,
    }


def option_label_items(raw: Any) -> list[str]:
    """option_labels may be a list or a dict. Dict keys are read only in the dict branch."""
    if isinstance(raw, dict):
        return [str(k) for k in raw.keys()]
    if isinstance(raw, (list, tuple)):
        return [str(x) for x in raw]
    return []


def parse_vote_letter(text: str, allowed: list[str] | None = None) -> str:
    allowed = [a.upper() for a in (allowed or list("ABCD1234"))]
    t = text or ""
    for pat in [
        r"(?:Recommendation|RECOMMENDATION|Option)\s*:?\s*([A-D1-9])\b",
        r"^\s*([A-D1-9])\s*$",
        r"\b([A-D1-9])\b",
    ]:
        m = re.search(pat, t, re.I | re.M)
        if m and m.group(1).upper() in allowed:
            return m.group(1).upper()
    return "?"


def load_gemma(model_id: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16, device_map="auto", low_cpu_mem_usage=True
    )
    model.eval()
    return model, tok


def run_scenario(
    model,
    tok,
    row: dict,
    arms: list[dict],
    *,
    k_frac: float,
    max_new_tokens: int,
) -> dict:
    assert torch is not None
    prefix = build_chat_prompt(tok, row["system_prompt"], row["user_prompt"])
    prefix_ids = tok.encode(prefix, add_special_tokens=False)
    utt_ids = tok.encode(row["recorded_utterance"], add_special_tokens=False)
    if len(utt_ids) < 4:
        raise RuntimeError(f"utterance too short for {row['scenario_id']}: {len(utt_ids)}")
    k = max(1, min(len(utt_ids) - 1, int(round(k_frac * len(utt_ids)))))
    forced_ids = utt_ids[:k]
    recorded_suffix = tok.decode(utt_ids[k:], skip_special_tokens=True)
    # Teacher-forced prompt = chat prefix + forced utterance tokens
    input_ids = torch.tensor([prefix_ids + forced_ids], device=model.device)
    start_pos = len(prefix_ids) + k  # first free token position (intervene from here)
    # Actually intervene at last forced token / first free: use start_pos-1 for last TF token
    intervene_pos = len(prefix_ids) + k - 1

    # option_labels: list or dict (this run's configs are lists of labels, e.g. ["A","B"])
    label_items = option_label_items(row.get("option_labels"))
    letters = sorted(set(label_items + [row["target_option"], row["fair_option"]]))
    arm_out = {}
    for arm in arms:
        hooks_spec = []
        if arm["kind"] == "ablate":
            mod = _module_for_site(model, arm["layer"], arm["site"])
            u = torch.tensor(arm["u"])
            hooks_spec.append(
                (mod, make_persist_ablate_hook(intervene_pos, u, float(arm["alpha"])))
            )
        elif arm["kind"] == "steer":
            mod = _module_for_site(model, arm["layer"], arm["site"])
            u = torch.tensor(arm["u"])
            # scale ≈ activation norm proxy; use 1.0 * residual RMS estimate later — β as unit
            hooks_spec.append(
                (mod, make_persist_steer_hook(intervene_pos, u, float(arm["beta"]), 1.0))
            )

        handles = []
        for mod, fn in hooks_spec:
            handles.append(mod.register_forward_hook(fn))
        try:
            with torch.no_grad():
                out = model.generate(
                    input_ids,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    return_dict_in_generate=True,
                    output_scores=False,
                )
        finally:
            for h in handles:
                h.remove()
        gen_ids = out.sequences[0][input_ids.shape[-1] :]
        text = tok.decode(gen_ids, skip_special_tokens=True)
        mets = continuation_metrics(text, recorded_suffix)
        arm_out[arm["name"]] = {
            "text": text,
            "vote": parse_vote_letter(text, letters),
            **mets,
        }
    # Primary deltas vs baseline_tf
    base = arm_out.get("baseline_tf") or {}
    for name, row_a in arm_out.items():
        if name == "baseline_tf":
            row_a["delta_edit_frac_vs_baseline"] = 0.0
        else:
            row_a["delta_edit_frac_vs_baseline"] = float(
                row_a.get("edit_frac", 0) - base.get("edit_frac", 0)
            )

    return {
        "scenario_id": row["scenario_id"],
        "domain": row.get("domain"),
        "agent": row["agent"],
        "target_option": row["target_option"],
        "k": k,
        "k_frac": k_frac,
        "utt_token_len": len(utt_ids),
        "prefix_token_len": len(prefix_ids),
        "intervene_pos": intervene_pos,
        "recorded_suffix": recorded_suffix,
        "arms": arm_out,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    ap.add_argument("--directions", type=Path, default=DEFAULT_DIRS)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--model", default="google/gemma-2-9b-it")
    ap.add_argument("--max-scenarios", type=int, default=6)
    ap.add_argument("--k-frac", type=float, default=0.5, help="fraction of utterance to TF before intervene")
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    run_dir = args.run_dir if args.run_dir.is_absolute() else ROOT / args.run_dir
    directions = args.directions if args.directions.is_absolute() else ROOT / args.directions
    if not run_dir.is_dir():
        print(f"ERROR: run dir not found: {run_dir}")
        return 1

    scenarios = select_scenarios(run_dir, args.max_scenarios)
    arms = build_arms(directions, seed=args.seed)
    args.out.mkdir(parents=True, exist_ok=True)

    meta = {
        "protocol": "within_traj_mid_private_intervene",
        "world": "Core",
        "source_run": "gemma2_9b/core/20261001T012639Z",
        "sites": ["attn_L22", "residual_L21"],
        "k_frac": args.k_frac,
        "max_new_tokens": args.max_new_tokens,
        "n_scenarios": len(scenarios),
        "scenario_ids": [s["scenario_id"] for s in scenarios],
        "domains": sorted({s.get("domain") or "?" for s in scenarios}),
        "arms": [a["name"] for a in arms],
        "n_arms": len(arms),
        "primary_metrics": [
            "edit_frac vs recorded suffix",
            "delta_edit_frac_vs_baseline_tf",
            "exact_match / prefix_match_20",
        ],
        "secondary_metrics": ["vote letter (if continuation reaches vote)", "target mention"],
        "dry_run": args.dry_run,
        "model": args.model,
        "design_refs": [
            "results/transfer_stable/WEEKEND_REVIEW/design/within_trajectory_spec.md",
            "results/transfer_stable/WEEKEND_REVIEW/design/NEXT_CAUSAL_RANKED.md",
        ],
        "soft_stop": "Track I / Paper 2 / interp-demo / family-house NOT touched; no pod launch",
    }
    (args.out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2), flush=True)

    print("\n=== SCENARIOS (mid-private colluder) ===", flush=True)
    inventory = []
    for s in scenarios:
        utt = s["recorded_utterance"]
        approx_tok = max(1, len(utt.split()))
        k_est = max(1, int(round(args.k_frac * approx_tok)))
        inv = {
            "scenario_id": s["scenario_id"],
            "domain": s.get("domain"),
            "agent": s["agent"],
            "target": s["target_option"],
            "utt_chars": len(utt),
            "utt_words_approx": approx_tok,
            "k_frac": args.k_frac,
            "k_words_approx": k_est,
            "sys_chars": len(s["system_prompt"]),
            "user_chars": len(s["user_prompt"]),
            "collusion_success": s.get("collusion_success"),
            "utt_preview": utt[:120] + ("…" if len(utt) > 120 else ""),
        }
        inventory.append(inv)
        print(
            f"  {s['scenario_id']:<12} {s.get('domain'):<14} agent={s['agent']:<20} "
            f"tgt={s['target_option']} utt_chars={len(utt)}",
            flush=True,
        )
    (args.out / "inventory.json").write_text(json.dumps(inventory, indent=2) + "\n")

    print("\n=== ARMS ===", flush=True)
    for a in arms:
        print(f"  - {a['name']} ({a['kind']})", flush=True)

    if args.dry_run:
        # Offline tokenizer length check (CPU) if available
        tok_chk = None
        try:
            from transformers import AutoTokenizer

            tok_chk = AutoTokenizer.from_pretrained(args.model)
        except Exception as e:
            print(f"[warn] tokenizer unavailable: {e}", flush=True)

        tok_stats = []
        if tok_chk is not None:
            for s in scenarios:
                prefix = build_chat_prompt(tok_chk, s["system_prompt"], s["user_prompt"])
                p_ids = tok_chk.encode(prefix, add_special_tokens=False)
                u_ids = tok_chk.encode(s["recorded_utterance"], add_special_tokens=False)
                k = max(1, min(len(u_ids) - 1, int(round(args.k_frac * len(u_ids)))))
                tok_stats.append(
                    {
                        "scenario_id": s["scenario_id"],
                        "prefix_tokens": len(p_ids),
                        "utt_tokens": len(u_ids),
                        "k": k,
                        "intervene_pos": len(p_ids) + k - 1,
                        "forced_ok": len(u_ids) >= 4,
                    }
                )
                if len(u_ids) < 4:
                    raise RuntimeError(f"utterance too short: {s['scenario_id']}")
            (args.out / "tokenizer_stats.json").write_text(
                json.dumps(tok_stats, indent=2) + "\n"
            )

        md = [
            "# Within-traj mid-private — dry-run",
            "",
            f"- scenarios: {len(scenarios)}",
            f"- arms: {len(arms)}",
            f"- k_frac: {args.k_frac}",
            f"- sites: attn_L22 + residual L21",
            f"- model: {args.model}",
            "",
            "## Scenarios",
        ]
        for s in scenarios:
            md.append(
                f"- `{s['scenario_id']}` domain={s.get('domain')} agent={s['agent']} "
                f"tgt={s['target_option']} utt_chars={len(s['recorded_utterance'])}"
            )
        md += ["", "## Arms"] + [f"- {a['name']}" for a in arms]
        md += [
            "",
            "## Primary metrics",
            "- edit_frac of free continuation vs recorded suffix",
            "- delta_edit_frac vs baseline_tf (ablate/steer must beat random)",
            "- vote letter secondary if continuation reaches recommendation",
            "",
            "## Soft-stop",
            "No Track I / Paper 2 / interp-demo / family-house. No pod launch from this script.",
        ]
        (args.out / "DRY_RUN.md").write_text("\n".join(md) + "\n")
        print(f"\n[dry-run] OK — wrote {args.out / 'DRY_RUN.md'}", flush=True)
        return 0

    if torch is None:
        print("ERROR: torch required for non-dry-run", flush=True)
        return 1

    print(f"\n[load] {args.model}", flush=True)
    model, tok = load_gemma(args.model)
    results = []
    t0 = time.time()
    for i, row in enumerate(scenarios):
        print(f"\n[{i+1}/{len(scenarios)}] {row['scenario_id']}", flush=True)
        results.append(
            run_scenario(
                model,
                tok,
                row,
                arms,
                k_frac=args.k_frac,
                max_new_tokens=args.max_new_tokens,
            )
        )
    elapsed = time.time() - t0

    # Aggregate
    agg = {"arms": {}}
    for arm in arms:
        name = arm["name"]
        edits = [r["arms"][name]["edit_frac"] for r in results if name in r["arms"]]
        deltas = [
            r["arms"][name].get("delta_edit_frac_vs_baseline", 0.0)
            for r in results
            if name in r["arms"]
        ]
        agg["arms"][name] = {
            "mean_edit_frac": float(np.mean(edits)) if edits else None,
            "mean_delta_edit_frac_vs_baseline": float(np.mean(deltas)) if deltas else None,
            "n": len(edits),
        }

    payload = {
        "meta": meta,
        "elapsed_sec": elapsed,
        "aggregate": agg,
        "per_scenario": results,
    }
    (args.out / "results.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(agg, indent=2), flush=True)
    print(f"\n[done] wrote {args.out / 'results.json'} elapsed={elapsed:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
