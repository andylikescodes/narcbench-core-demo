#!/usr/bin/env python3
"""Extract NARCBench-style last-token activations from Core/Transfer transcripts.

Paper-faithful layout (aaronrose227/narcbench extraction/extract_core.py):
  activations_gen.npz       keys layer_<i>   — residual stream (DecoderLayer out)
  activations_attn_gen.npz  keys layer_<i>   — attention block output (residual write)
  activations_mlp_gen.npz   keys layer_<i>   — MLP block output (residual write)
  metadata_gen.json         rows aligned across all three

Hook documentation (Gemma-2 via HF AutoModelForCausalLM):
  residual  → forward hook on model.model.layers[L]
              = post-(attn+mlp) residual at that layer (same as prior residual caches)
  attn_out  → forward hook on model.model.layers[L].self_attn
              = attention module output BEFORE residual add (the attn residual-write)
  mlp_out   → forward hook on model.model.layers[L].mlp
              = MLP module output BEFORE residual add (the mlp residual-write)
  All taken at the LAST prompt/sequence token (gen-time includes assistant response).

Works from regen dirs that already have run_config.json / agent_prompts.json /
results.json. Does NOT touch running Core pod.

  # CPU metadata dry-run (no model):
  python3 scripts/extract_activations_from_transcripts.py \
    --runs-dir upstream/scenarios/gemma2_9b/core/20261001T012639Z \
    --out data/activations/gemma2_9b/core/20261001T012639Z_mlp_attn --dry-run

  # GPU extract residual+attn+mlp (RunPod):
  python3 scripts/extract_activations_from_transcripts.py \
    --runs-dir ... --out ... --model google/gemma-2-9b-it --layers 19-23 \
    --gen-only --sites residual,attn,mlp
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

SITE_DOC = {
    "residual": "hook model.model.layers[L] output (post-block residual)",
    "attn": "hook model.model.layers[L].self_attn output (attn residual-write, pre-add)",
    "mlp": "hook model.model.layers[L].mlp output (mlp residual-write, pre-add)",
}


def parse_layers(spec: str | None, n_layers: int | None = None) -> list[int]:
    if not spec:
        if n_layers is None:
            return [19, 20, 21, 22, 23]  # Gemma-2-9B mid band default
        mid = n_layers // 2
        return sorted({max(0, mid - 2), mid - 1, mid, mid + 1, min(n_layers - 1, mid + 2)})
    if "-" in spec and "," not in spec:
        a, b = spec.split("-", 1)
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in spec.split(",") if x.strip()]


def parse_sites(spec: str) -> list[str]:
    allowed = {"residual", "attn", "mlp"}
    sites = [s.strip().lower() for s in (spec or "residual").split(",") if s.strip()]
    bad = [s for s in sites if s not in allowed]
    if bad:
        raise ValueError(f"unknown sites {bad}; allowed {sorted(allowed)}")
    # stable order
    order = ["residual", "attn", "mlp"]
    return [s for s in order if s in sites]


def _strip_name_prefix(msg: str) -> str:
    return re.sub(r"^\[[^\]]+\]:\s*", "", msg or "")


def load_samples(runs_dir: Path) -> list[dict[str, Any]]:
    run_dirs = sorted(d for d in runs_dir.iterdir() if d.is_dir())
    samples: list[dict[str, Any]] = []
    missing = []
    for run_dir in run_dirs:
        cfg_p, prompts_p, res_p = (
            run_dir / "run_config.json",
            run_dir / "agent_prompts.json",
            run_dir / "results.json",
        )
        if not all(p.exists() for p in (cfg_p, prompts_p, res_p)):
            missing.append(run_dir.name)
            continue
        cfg = json.loads(cfg_p.read_text())
        prompts = json.loads(prompts_p.read_text())
        results = json.loads(res_p.read_text())
        private_responses = [_strip_name_prefix(m) for m in results.get("private_messages", [])]
        public_responses = [_strip_name_prefix(m) for m in results.get("public_messages", [])]
        final_texts = results.get("final_texts", {})
        private_idx = public_idx = 0
        run_id = cfg.get("run_id") or run_dir.name
        for entry in prompts:
            phase = entry.get("phase") or ""
            channel = entry.get("channel") or ""
            agent = entry["agent_name"]
            # Regen uses phase=private; paper extract uses discussion+private.
            if phase == "private" or (phase == "discussion" and channel == "private"):
                response = private_responses[private_idx] if private_idx < len(private_responses) else ""
                private_idx += 1
                phase_norm, channel_norm = "private", "private"
            elif phase == "discussion" and channel == "public":
                response = public_responses[public_idx] if public_idx < len(public_responses) else ""
                public_idx += 1
                phase_norm, channel_norm = "discussion", "public"
            elif phase == "final":
                response = final_texts.get(agent, "")
                phase_norm, channel_norm = "final", "public"
            else:
                response = ""
                phase_norm, channel_norm = phase, channel
            role = entry.get("role") or (cfg.get("roles") or {}).get(agent, "?")
            scen_dom = cfg.get("scenario_domain") or cfg.get("scenario_type") or run_id.split("__")[0]
            samples.append(
                {
                    "run_id": run_id,
                    "scenario_id": cfg.get("scenario_id", ""),
                    "scenario_domain": scen_dom,
                    "family": scen_dom,
                    "scenario_title": cfg.get("scenario_title", ""),
                    "mode": cfg.get("mode") or results.get("mode"),
                    "agent_name": agent,
                    "role": role,
                    "phase": phase_norm,
                    "channel": channel_norm,
                    "round": entry.get("round", 0),
                    "collusion_success": bool(cfg.get("collusion_success", results.get("collusion_success"))),
                    "target_majority": bool(cfg.get("target_majority", results.get("target_majority"))),
                    "target_option": cfg.get("target_option", ""),
                    "fair_option": cfg.get("fair_option", ""),
                    "label_variant": cfg.get("label_variant", 0),
                    "agent_vote": (cfg.get("votes") or results.get("votes") or {}).get(agent, "?"),
                    "flagged": bool(cfg.get("flagged", False)),
                    "system_prompt": entry.get("system_prompt", ""),
                    "user_prompt": entry.get("user_prompt", ""),
                    "response": response,
                    "model_id": cfg.get("model_id") or results.get("model_id"),
                }
            )
    if missing:
        print(f"[warn] skipped incomplete dirs: {missing[:8]}{'...' if len(missing)>8 else ''}")
    print(f"[load] {len(run_dirs)} dirs → {len(samples)} samples from {runs_dir}")
    return samples


def build_chat_text(tokenizer, sample: dict, mode: str) -> str:
    """Gemma-2-IT: fold system into user; include assistant for gen-time."""
    system = (sample.get("system_prompt") or "").strip()
    user = (sample.get("user_prompt") or "").strip()
    content = f"{system}\n\n{user}" if system else user
    messages = [{"role": "user", "content": content}]
    if mode == "gen":
        messages.append({"role": "assistant", "content": sample.get("response") or ""})
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def metadata_only(samples: list[dict], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = []
    for s in samples:
        meta.append(
            {
                "run_id": s["run_id"],
                "scenario_id": s["scenario_id"],
                "scenario_domain": s["scenario_domain"],
                "mode": s["mode"],
                "agent_name": s["agent_name"],
                "role": s["role"],
                "phase": s["phase"],
                "channel": s["channel"],
                "round": s["round"],
                "collusion_success": s["collusion_success"],
                "target_majority": s["target_majority"],
                "target_option": s["target_option"],
                "fair_option": s["fair_option"],
                "label_variant": s["label_variant"],
                "agent_vote": s["agent_vote"],
                "flagged": s["flagged"],
                "n_tokens": None,
                "has_response": bool(s.get("response")),
                "response_chars": len(s.get("response") or ""),
            }
        )
    path = out_dir / "metadata_gen.preview.json"
    path.write_text(json.dumps(meta, indent=2) + "\n")
    from collections import Counter

    c = Counter(f"{m['mode']}/{m['role']}/{m['channel']}/{m['phase']}" for m in meta)
    summary = {
        "n_samples": len(meta),
        "n_runs": len({m["run_id"] for m in meta}),
        "n_domains": len({m["scenario_domain"] for m in meta if m["scenario_domain"]}),
        "distribution": dict(sorted(c.items())),
        "note": "dry-run preview only; activations_*.npz not written",
    }
    (out_dir / "dry_run_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return path


def _last_token_vec(out) -> "Any":
    """Take last-token hidden from module forward output (tensor or tuple)."""
    h = out[0] if isinstance(out, (tuple, list)) else out
    # attn may return (hidden, attn_weights, ...) — first elem is the write
    if isinstance(h, (tuple, list)):
        h = h[0]
    return h[0, -1, :].detach().float().cpu()


def extract(
    samples: list[dict],
    out_dir: Path,
    *,
    model_id: str,
    layers: list[int],
    modes: tuple[str, ...],
    sites: list[str],
) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[model] loading {model_id}")
    print(f"[sites] {sites}")
    for s in sites:
        print(f"  {s}: {SITE_DOC[s]}")
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16, device_map="auto", low_cpu_mem_usage=True
    )
    model.eval()
    n_layers = int(model.config.num_hidden_layers)
    layers = [L for L in layers if 0 <= L < n_layers]
    print(f"[model] n_layers={n_layers} extracting {layers}")

    layer_mods = model.model.layers
    hook_doc = {
        "sites": {s: SITE_DOC[s] for s in sites},
        "layers": layers,
        "token": "last sequence token (gen includes assistant response)",
        "model_id": model_id,
        "n_hidden_layers": n_layers,
        "hidden_size": int(model.config.hidden_size),
    }
    (out_dir / "extract_hook_doc.json").write_text(json.dumps(hook_doc, indent=2) + "\n")

    for mode in modes:
        # site → layer → list of vectors
        acts: dict[str, dict[int, list]] = {s: {L: [] for L in layers} for s in sites}
        meta: list[dict] = []
        t0 = time.time()
        for i, sample in enumerate(samples):
            boxes: dict[tuple[str, int], Any] = {}
            handles = []

            def make_hook(site: str, L: int):
                def hook(_m, _inp, out):
                    boxes[(site, L)] = _last_token_vec(out)

                return hook

            for L in layers:
                if "residual" in sites:
                    handles.append(layer_mods[L].register_forward_hook(make_hook("residual", L)))
                if "attn" in sites:
                    handles.append(layer_mods[L].self_attn.register_forward_hook(make_hook("attn", L)))
                if "mlp" in sites:
                    handles.append(layer_mods[L].mlp.register_forward_hook(make_hook("mlp", L)))
            try:
                if mode == "gen" and not sample.get("response"):
                    hs = int(model.config.hidden_size)
                    for s in sites:
                        for L in layers:
                            acts[s][L].append(np.zeros(hs, dtype=np.float32))
                    n_tokens = 0
                else:
                    text = build_chat_text(tok, sample, mode)
                    inputs = tok(text, return_tensors="pt")
                    inputs = {k: v.to(model.device) for k, v in inputs.items()}
                    n_tokens = int(inputs["input_ids"].shape[1])
                    with torch.inference_mode():
                        # Forward backbone only — skip lm_head/logits (OOM on long Transfer
                        # prompts on 22GB L4). Hooks on layers/attn/mlp still fire.
                        if hasattr(model, "model"):
                            model.model(**inputs, use_cache=False)
                        else:
                            model(**inputs, use_cache=False)
                    for s in sites:
                        for L in layers:
                            acts[s][L].append(boxes[(s, L)].numpy())
            finally:
                for h in handles:
                    h.remove()
            meta.append(
                {
                    "run_id": sample["run_id"],
                    "scenario_id": sample["scenario_id"],
                    "scenario_domain": sample["scenario_domain"],
                    "mode": sample["mode"],
                    "agent_name": sample["agent_name"],
                    "role": sample["role"],
                    "phase": sample["phase"],
                    "channel": sample["channel"],
                    "round": sample["round"],
                    "collusion_success": sample["collusion_success"],
                    "target_majority": sample["target_majority"],
                    "target_option": sample["target_option"],
                    "fair_option": sample["fair_option"],
                    "label_variant": sample["label_variant"],
                    "agent_vote": sample["agent_vote"],
                    "flagged": sample["flagged"],
                    "n_tokens": n_tokens,
                }
            )
            if (i + 1) % 10 == 0 or i == 0:
                print(f"[{mode}] {i+1}/{len(samples)}", flush=True)

        (out_dir / f"metadata_{mode}.json").write_text(json.dumps(meta, indent=2) + "\n")

        site_to_fname = {
            "residual": f"activations_{mode}.npz",
            "attn": f"activations_attn_{mode}.npz",
            "mlp": f"activations_mlp_{mode}.npz",
        }
        for s in sites:
            arrays = {f"layer_{L}": np.stack(acts[s][L]) for L in layers}
            fname = site_to_fname[s]
            np.savez_compressed(out_dir / fname, **arrays)
            shape = next(iter(arrays.values())).shape
            print(f"[save] {fname} site={s} shape={shape}", flush=True)

        elapsed = time.time() - t0
        print(f"[done] mode={mode} n={len(samples)} sites={sites} elapsed={elapsed:.1f}s", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model", default="google/gemma-2-9b-it")
    ap.add_argument("--layers", default="19-23")
    ap.add_argument("--dry-run", action="store_true", help="metadata only, no GPU")
    ap.add_argument("--gen-only", action="store_true")
    ap.add_argument("--input-only", action="store_true")
    ap.add_argument(
        "--sites",
        default="residual,attn,mlp",
        help="comma sites: residual,attn,mlp (default all three for mlp/attn extract)",
    )
    args = ap.parse_args()
    runs_dir = args.runs_dir if args.runs_dir.is_absolute() else ROOT / args.runs_dir
    out = args.out if args.out.is_absolute() else ROOT / args.out
    samples = load_samples(runs_dir)
    if not samples:
        print("ERROR: no samples", file=sys.stderr)
        return 1
    if args.dry_run:
        metadata_only(samples, out)
        return 0
    layers = parse_layers(args.layers)
    sites = parse_sites(args.sites)
    if args.gen_only:
        modes: tuple[str, ...] = ("gen",)
    elif args.input_only:
        modes = ("input",)
    else:
        modes = ("input", "gen")
    extract(samples, out, model_id=args.model, layers=layers, modes=modes, sites=sites)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
