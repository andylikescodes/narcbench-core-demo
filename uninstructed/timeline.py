"""Replay of recorded turns and per-token reads in the private room (docs/uninstructed_collusion/DESIGN.md, section 7).

Each room message is replayed in the context in which it was written: one forward pass over system prompt, user
prompt and the recorded reply. The model is causal, so the activation at a reply token is what the model had when it
produced that token. For every message we keep the token just before it starts (the decision point, position -1)
and every token of the reply, at the requested layers.

Per room, extract_room_tokens writes <out>/<run id>__room<k>.npz with
  acts      float16 [T, n_layers, d]   the room's timeline: message 1 (decision point, then its tokens), message 2, ...
  msg       int8    [T]                message index 0..3
  pos       int16   [T]                position in the message: -1 = decision point, 0.. = reply tokens
  token     int32   [T]                token id at that position
  speaker   int8    [T]                seat of the agent who wrote the message
and one record per room in <out>/rooms_meta.json.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from jspace_interp.positions import gemma_chat_text


def turn_samples(run_dir: Path, phases=("room",)) -> list[dict]:
    """The recorded turns of one run in the order they were generated."""
    cfg = json.loads((Path(run_dir) / "run_config.json").read_text())
    out = []
    for e in json.loads((Path(run_dir) / "agent_prompts.json").read_text()):
        if e["phase"] in phases:
            out.append({"run_id": cfg["run_id"], "version": cfg["version"], "scenario_id": cfg["scenario_id"], **e})
    return out


def _reply_span(offsets, spans):
    r0, r1 = spans["response"]
    return [i for i, (a, b) in enumerate(offsets) if b > a and a >= r0 and b <= r1 + 1]


def forward_layers(model, input_ids, layers):
    """One forward pass; the output of each requested block for the single sequence, as [n_tokens, d] tensors."""
    import torch

    backbone = model.model if hasattr(model, "model") else model
    boxes, hooks = {}, []
    for L in layers:
        def mk(L):
            def hook(_m, _i, out):
                boxes[L] = (out[0] if isinstance(out, (tuple, list)) else out)[0].detach()
            return hook
        hooks.append(backbone.layers[L].register_forward_hook(mk(L)))
    try:
        with torch.inference_mode():
            backbone(input_ids=torch.tensor([input_ids], device=next(model.parameters()).device), use_cache=False)
    finally:
        for h in hooks:
            h.remove()
    return boxes


def extract_room_tokens(run_dirs: list[Path], model, tokenizer, layers: list[int], out_dir: Path, *, log=print) -> dict:
    """Replay every room message of the runs and save the per-token timeline of each room. Returns timing figures."""
    out_dir.mkdir(parents=True, exist_ok=True)
    meta, n_ctx, n_tok, n_bytes, t0 = [], 0, 0, 0, time.time()
    for rd in run_dirs:
        cfg = json.loads((Path(rd) / "run_config.json").read_text())
        samples = turn_samples(rd, ("room",))
        for k in range(len(cfg["rooms"])):
            parts = {"acts": [], "msg": [], "pos": [], "token": [], "speaker": []}
            lengths = []
            for s in sorted((x for x in samples if x["room"] == k), key=lambda x: x["turn"]):
                text, spans = gemma_chat_text(s["system_prompt"], s["user_prompt"], s["response"], tokenizer)
                enc = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
                ids, offsets = enc["input_ids"], [tuple(o) for o in enc["offset_mapping"]]
                span = _reply_span(offsets, spans)
                if not span:              # an empty reply: keep the decision point only
                    span = []
                    first = len(ids) - 2
                else:
                    first = span[0]
                idx = [first - 1] + span   # decision point, then the reply tokens
                H = forward_layers(model, ids, layers)
                A = np.stack([H[L][idx].float().cpu().numpy().astype(np.float16) for L in layers], axis=1)   # [n, n_layers, d]
                parts["acts"].append(A)
                parts["msg"] += [s["turn"]] * len(idx)
                parts["pos"] += [-1] + list(range(len(span)))
                parts["token"] += [ids[i] for i in idx]
                parts["speaker"] += [s["seat"]] * len(idx)
                lengths.append(len(span))
                n_ctx += 1
                n_tok += len(ids)
            name = f"{cfg['run_id']}__room{k}.npz"
            acts = np.concatenate(parts["acts"], axis=0)
            np.savez(out_dir / name, acts=acts, msg=np.array(parts["msg"], dtype=np.int8), pos=np.array(parts["pos"], dtype=np.int16),
                     token=np.array(parts["token"], dtype=np.int32), speaker=np.array(parts["speaker"], dtype=np.int8))
            n_bytes += (out_dir / name).stat().st_size
            meta.append({"room_id": f"{cfg['run_id']}#{k}", "run_id": cfg["run_id"], "room": k, "version": cfg["version"],
                         "scenario_id": cfg["scenario_id"], "file": name, "layers": layers, "n_positions": int(acts.shape[0]),
                         "message_tokens": lengths})
        log(f"[replay] {len(meta)} rooms, {n_ctx} contexts ({time.time() - t0:.0f}s)")
    (out_dir / "rooms_meta.json").write_text(json.dumps(meta, indent=1) + "\n")
    el = time.time() - t0
    return {"rooms": len(meta), "contexts": n_ctx, "seconds": round(el, 1), "seconds_per_context": round(el / max(n_ctx, 1), 3),
            "context_tokens_mean": round(n_tok / max(n_ctx, 1)), "bytes": n_bytes, "bytes_per_room": round(n_bytes / max(len(meta), 1)),
            "positions_per_room_mean": round(float(np.mean([m["n_positions"] for m in meta])), 1) if meta else 0}


def replay_last_token(run_dirs: list[Path], model, tokenizer, layers: list[int], phases=("discussion", "final"), *, log=print) -> dict:
    """Replay the public turns and the votes, reading the last token only (the read position of the instructed study).
    Returns timing figures and the activations [n, n_layers, d] (float16) with one record per row."""
    rows, acts, n_tok, t0 = [], [], 0, time.time()
    for rd in run_dirs:
        for s in turn_samples(rd, phases):
            text, _ = gemma_chat_text(s["system_prompt"], s["user_prompt"], s["response"], tokenizer)
            ids = tokenizer(text, add_special_tokens=False)["input_ids"]
            H = forward_layers(model, ids, layers)
            acts.append(np.stack([H[L][-1].float().cpu().numpy().astype(np.float16) for L in layers], axis=0))
            rows.append({k: s[k] for k in ("run_id", "version", "scenario_id", "agent_name", "seat", "phase", "round", "room")})
            n_tok += len(ids)
        log(f"[replay] public turns and votes: {len(rows)} contexts ({time.time() - t0:.0f}s)")
    el = time.time() - t0
    return {"contexts": len(rows), "seconds": round(el, 1), "seconds_per_context": round(el / max(len(rows), 1), 3),
            "context_tokens_mean": round(n_tok / max(len(rows), 1)), "rows": rows, "acts": np.stack(acts) if acts else None}
