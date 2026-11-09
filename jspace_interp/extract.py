"""Activation extraction at several read positions and context conditions (GPU for Gemma-2-9B).

For each agent turn and condition, one forward pass reads every position at once (the model is causal,
so positions inside the reply see exactly the prefix they would see at generation time).

Output per (split, condition) folder:
  activations.npz   keys "<position>__layer_<L>"  float16 [n_rows, d]
  metadata.json     one record per row: run/agent/role/mode/phase/round/vote/target, n_tokens,
                    token index per position, fallback flags, condition, context_changed
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from .positions import POSITIONS, gemma_chat_text, token_positions


def _hidden(out):
    h = out[0] if isinstance(out, (tuple, list)) else out
    return h


def extract_rows(samples: list[dict], model, tokenizer, layers: list[int], out_dir: Path, *,
                 positions=POSITIONS, max_tokens: int | None = None, log=print) -> Path:
    import torch

    out_dir.mkdir(parents=True, exist_ok=True)
    device = next(model.parameters()).device
    backbone = model.model if hasattr(model, "model") else model
    layer_mods = backbone.layers
    eot_id = tokenizer.convert_tokens_to_ids("<end_of_turn>")
    bufs: dict[str, list] = {}
    meta = []
    t0 = time.time()
    for i, s in enumerate(samples):
        prompt_only = bool(s.get("read_prompt_only"))
        response = None if prompt_only else (s.get("response") or "")
        text, spans = gemma_chat_text(s.get("system_prompt") or "", s.get("user_prompt") or "", response, tokenizer)
        enc = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
        ids = enc["input_ids"]
        offsets = [tuple(o) for o in enc["offset_mapping"]]
        if max_tokens and len(ids) > max_tokens:  # only for smoke tests; recorded in metadata
            ids, offsets = ids[-max_tokens:], offsets[-max_tokens:]
        eots = [j for j, t in enumerate(ids) if t == eot_id]
        pos, fb = token_positions(offsets, spans, response, eots)
        want = ["prompt_end"] if prompt_only else [p for p in positions if p in pos]
        if prompt_only and "prompt_end" not in pos:
            pos["prompt_end"] = len(ids) - 1
            fb["prompt_end"] = True
        boxes = {}
        hooks = []
        for L in layers:
            def mk(L):
                def hook(_m, _i, out):
                    boxes[L] = _hidden(out)[0].detach()
                return hook
            hooks.append(layer_mods[L].register_forward_hook(mk(L)))
        try:
            with torch.inference_mode():
                backbone(input_ids=torch.tensor([ids], device=device), use_cache=False)
        finally:
            for h in hooks:
                h.remove()
        for L in layers:
            H = boxes[L]
            for p in want:
                if p == "reply_mean":
                    a, b = pos[p]
                    v = H[a:b].float().mean(0)
                else:
                    v = H[pos[p]].float()
                bufs.setdefault(f"{p}__layer_{L}", []).append(v.cpu().numpy().astype(np.float16))
        meta.append({k: s.get(k) for k in ("run_id", "scenario_id", "scenario_domain", "mode", "agent_name", "role",
                                          "phase", "channel", "round", "agent_vote", "target_option", "fair_option",
                                          "condition", "context_changed", "warned", "seat")}
                    | {"n_tokens": len(ids), "positions": {k: v for k, v in pos.items()}, "fallback": fb,
                       "truncated_to": max_tokens if (max_tokens and len(enc["input_ids"]) > max_tokens) else None})
        if (i + 1) % 25 == 0 or i == 0:
            log(f"[extract] {i + 1}/{len(samples)} ({time.time() - t0:.0f}s)")
    np.savez(out_dir / "activations.npz", **{k: np.stack(v) for k, v in bufs.items()})
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=1, default=str) + "\n")
    return out_dir
