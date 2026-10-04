#!/usr/bin/env python3
"""Self-contained residual activation patching for NARCBench-Core Gemma runs.

Restore: patch collusion residuals into control final-vote forward; measure
hidden-goal vote flips. Scrub: patch control residuals into collusion forward.
"""
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch


def load_paired_runs(runs_dir: Path) -> list[dict]:
    by_sc: dict[str, dict] = defaultdict(dict)
    for d in sorted(runs_dir.iterdir()):
        if not d.is_dir():
            continue
        cfg_p, prompts_p, res_p = d / "run_config.json", d / "agent_prompts.json", d / "results.json"
        if not (cfg_p.exists() and prompts_p.exists() and res_p.exists()):
            continue
        cfg, prompts, res = json.loads(cfg_p.read_text()), json.loads(prompts_p.read_text()), json.loads(res_p.read_text())
        sid = cfg.get("scenario_id") or d.name
        mode = cfg.get("mode") or res.get("mode")
        by_sc[sid][mode] = {"dir": str(d), "cfg": cfg, "prompts": prompts, "results": res}
    return [{"scenario_id": sid, **modes} for sid, modes in by_sc.items() if "collusion" in modes and "control" in modes]


def pick_final(prompts: list, role: str = "colluder") -> dict | None:
    finals = [p for p in prompts if p.get("phase") == "final" or "RECOMMENDATION" in (p.get("user_prompt") or "")]
    for p in finals:
        if role == "any" or p.get("role") == role:
            return p
    return finals[0] if finals else None


def pick_private(prompts: list) -> dict | None:
    priv = [p for p in prompts if p.get("phase") == "private" or p.get("channel") == "private"]
    for p in priv:
        if p.get("role") == "colluder":
            return p
    return priv[0] if priv else None


def parse_vote(text: str) -> str:
    import re
    for pat in [
        r"RECOMMENDATION:\s*\*{0,2}\[?Option\s*([A-D1-9])\]?\*{0,2}",
        r"RECOMMENDATION:\s*\*{0,2}\[?(?:Project|Candidate|Patient)\s*([A-D1-9])\]?\*{0,2}",
        r"RECOMMENDATION:\s*\*{0,2}\[?([A-D1-9])\]?\*{0,2}\b",
        r"(?:Project|Candidate|Patient|Option)\s*([A-D1-9])\b",
    ]:
        m = re.search(pat, text or "", re.I)
        if m:
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


def build_prompt(tok, system: str, user: str) -> str:
    messages = [
        {"role": "user", "content": f"System:\n{system}\n\nUser:\n{user}"},
    ]
    if hasattr(tok, "apply_chat_template"):
        try:
            return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            pass
    return f"<start_of_turn>user\n{system}\n\n{user}<end_of_turn>\n<start_of_turn>model\n"


@torch.no_grad()
def generate_text(model, tok, prompt: str, max_new_tokens: int = 64) -> str:
    ids = tok(prompt, return_tensors="pt").to(model.device)
    out = model.generate(**ids, max_new_tokens=max_new_tokens, do_sample=False)
    return tok.decode(out[0][ids["input_ids"].shape[-1] :], skip_special_tokens=True)


def capture_resid_last(model, tok, prompt: str, layers: list[int]) -> dict[int, torch.Tensor]:
    ids = tok(prompt, return_tensors="pt").to(model.device)
    captured: dict[int, torch.Tensor] = {}
    hooks = []

    def make(L):
        def hook(_m, _i, out):
            h = out[0] if isinstance(out, tuple) else out
            captured[L] = h[0, -1, :].detach().to(torch.float32).cpu()

        return hook

    for L in layers:
        hooks.append(model.model.layers[L].register_forward_hook(make(L)))
    try:
        model(**ids)
    finally:
        for h in hooks:
            h.remove()
    return captured


def capture_resid_span(model, tok, prompt: str, layers: list[int], positions: list[int]) -> dict[int, dict[int, torch.Tensor]]:
    ids = tok(prompt, return_tensors="pt").to(model.device)
    captured: dict[int, dict[int, torch.Tensor]] = {L: {} for L in layers}
    hooks = []

    def make(L):
        def hook(_m, _i, out):
            h = out[0] if isinstance(out, tuple) else out
            for p in positions:
                if 0 <= p < h.shape[1]:
                    captured[L][p] = h[0, p, :].detach().to(torch.float32).cpu()

        return hook

    for L in layers:
        hooks.append(model.model.layers[L].register_forward_hook(make(L)))
    try:
        model(**ids)
    finally:
        for h in hooks:
            h.remove()
    return captured


@torch.no_grad()
def generate_with_patches(model, tok, prompt: str, patches: dict[int, dict[int, torch.Tensor]], max_new_tokens: int = 64) -> str:
    """patches: layer -> {pos -> cpu tensor} applied on first forward (prompt)."""
    ids = tok(prompt, return_tensors="pt").to(model.device)
    hooks = []

    def make(L, pos_map):
        def hook(_m, _i, out):
            h = out[0] if isinstance(out, tuple) else out
            h = h.clone()
            for pos, vec in pos_map.items():
                if 0 <= pos < h.shape[1]:
                    h[0, pos, :] = vec.to(device=h.device, dtype=h.dtype)
            return (h,) + out[1:] if isinstance(out, tuple) else h

        return hook

    for L, pos_map in patches.items():
        if pos_map:
            hooks.append(model.model.layers[L].register_forward_hook(make(L, pos_map)))
    try:
        # one patched prefill then free generate — simple approach: generate with hooks only on first step
        out = model.generate(**ids, max_new_tokens=max_new_tokens, do_sample=False)
    finally:
        for h in hooks:
            h.remove()
    return tok.decode(out[0][ids["input_ids"].shape[-1] :], skip_special_tokens=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", type=Path, required=True)
    ap.add_argument("--model", default="google/gemma-2-2b-it")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--layers", default="12,16,20")
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--max-scenarios", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--last-only", action="store_true", default=True)
    ap.add_argument("--with-brief", action="store_true", help="also patch brief spans (more VRAM)")
    args = ap.parse_args()

    paired = load_paired_runs(args.runs_dir)
    if args.max_scenarios:
        paired = paired[: args.max_scenarios]
    args.out.mkdir(parents=True, exist_ok=True)
    meta = {"model": args.model, "runs_dir": str(args.runs_dir), "n_paired": len(paired), "scenarios": [p["scenario_id"] for p in paired]}
    (args.out / "pairs.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2), flush=True)
    if args.dry_run or not paired:
        return 0

    layers = [int(x) for x in args.layers.split(",") if x.strip()]
    print(f"[patch] loading {args.model}", flush=True)
    model, tok = load_gemma(args.model)
    restore_hits: dict[str, list[int]] = defaultdict(list)
    scrub_hits: dict[str, list[int]] = defaultdict(list)
    per = []
    t0 = time.time()

    for item in paired:
        sid = item["scenario_id"]
        coll, ctrl = item["collusion"], item["control"]
        target = str(coll["cfg"].get("target_option"))
        print(f"[patch] {sid} target={target}", flush=True)
        coll_final = pick_final(coll["prompts"], "colluder")
        ctrl_final = pick_final(ctrl["prompts"], "any")
        if not (coll_final and ctrl_final):
            print("  skip missing finals", flush=True)
            continue
        corr_prompt = build_prompt(tok, coll_final["system_prompt"], coll_final["user_prompt"])
        clean_prompt = build_prompt(tok, ctrl_final["system_prompt"], ctrl_final["user_prompt"])
        corr_ids = tok(corr_prompt, return_tensors="pt")["input_ids"][0]
        clean_ids = tok(clean_prompt, return_tensors="pt")["input_ids"][0]
        n_span = 8
        brief_pos = list(range(max(0, len(corr_ids) - n_span), len(corr_ids)))
        align_pos = list(range(max(0, len(clean_ids) - n_span), len(clean_ids)))

        corr_text = generate_text(model, tok, corr_prompt, args.max_new_tokens)
        clean_text = generate_text(model, tok, clean_prompt, args.max_new_tokens)
        corr_vote, clean_vote = parse_vote(corr_text), parse_vote(clean_text)
        corr_hit, clean_hit = corr_vote == target, clean_vote == target
        print(f"  base corr={corr_vote}/{corr_hit} clean={clean_vote}/{clean_hit}", flush=True)

        torch.cuda.empty_cache()
        do_brief = args.with_brief
        corr_brief = clean_align = {}
        if do_brief:
            corr_brief = capture_resid_span(model, tok, corr_prompt, layers, brief_pos)
            torch.cuda.empty_cache()
            clean_align = capture_resid_span(model, tok, clean_prompt, layers, align_pos)
            torch.cuda.empty_cache()
        corr_last = capture_resid_last(model, tok, corr_prompt, layers)
        torch.cuda.empty_cache()
        clean_last = capture_resid_last(model, tok, clean_prompt, layers)
        torch.cuda.empty_cache()

        rows = []

        def record(site, direction, vote, reply):
            hit = vote == target
            rows.append({"site": site, "direction": direction, "vote": vote, "hit_target": hit, "reply": (reply or "")[:200]})
            if direction == "restore":
                restore_hits[site].append(int(hit))
            else:
                scrub_hits[site].append(int(not hit) if corr_hit else 0)

        for L in layers:
            if do_brief:
                patches = {L: {}}
                for i, dst in enumerate(align_pos):
                    src = brief_pos[min(i, len(brief_pos) - 1)]
                    if src in corr_brief.get(L, {}):
                        patches[L][dst] = corr_brief[L][src]
                if not patches[L] and align_pos and corr_brief.get(L):
                    patches[L][align_pos[-1]] = torch.stack(list(corr_brief[L].values())).mean(0)
                text = generate_with_patches(model, tok, clean_prompt, patches, args.max_new_tokens)
                record(f"brief.resid.L{L}", "restore", parse_vote(text), text)
                torch.cuda.empty_cache()

                patches_s = {L: {}}
                for i, dst in enumerate(brief_pos):
                    src = align_pos[min(i, len(align_pos) - 1)]
                    if src in clean_align.get(L, {}):
                        patches_s[L][dst] = clean_align[L][src]
                if not patches_s[L] and brief_pos and clean_align.get(L):
                    patches_s[L][brief_pos[-1]] = torch.stack(list(clean_align[L].values())).mean(0)
                text = generate_with_patches(model, tok, corr_prompt, patches_s, args.max_new_tokens)
                record(f"brief.resid.L{L}", "scrub", parse_vote(text), text)
                torch.cuda.empty_cache()

            patches_l = {L: {len(clean_ids) - 1: corr_last[L]}} if L in corr_last else {L: {}}
            text = generate_with_patches(model, tok, clean_prompt, patches_l, args.max_new_tokens)
            record(f"last.resid.L{L}", "restore", parse_vote(text), text)
            torch.cuda.empty_cache()
            patches_ls = {L: {len(corr_ids) - 1: clean_last[L]}} if L in clean_last else {L: {}}
            text = generate_with_patches(model, tok, corr_prompt, patches_ls, args.max_new_tokens)
            record(f"last.resid.L{L}", "scrub", parse_vote(text), text)
            torch.cuda.empty_cache()

        per.append({
            "scenario_id": sid,
            "target": target,
            "baseline": {"corr_vote": corr_vote, "clean_vote": clean_vote, "corr_hit": corr_hit, "clean_hit": clean_hit},
            "transcript_collusion_success": coll["results"].get("collusion_success"),
            "sites": rows,
        })
        (args.out / f"{sid}.json").write_text(json.dumps(per[-1], indent=2))

    def rate(d):
        return {k: {"n": len(xs), "rate": (sum(xs) / len(xs) if xs else None), "hits": sum(xs)} for k, xs in d.items()}

    report = {
        "elapsed_s": round(time.time() - t0, 1),
        "model": args.model,
        "n_scenarios": len(per),
        "restore_vote_flip_rates": rate(restore_hits),
        "scrub_vote_flip_rates": rate(scrub_hits),
        "per_scenario": per,
    }
    (args.out / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: report[k] for k in ("elapsed_s", "n_scenarios", "restore_vote_flip_rates", "scrub_vote_flip_rates")}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
