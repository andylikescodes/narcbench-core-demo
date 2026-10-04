#!/usr/bin/env python3
"""Matched-prefix activation interchange + direction ablate (Core/Transfer micro-worlds).

Teacher-forced shared-prefix protocol — NOT free multi-round regen.

Default arms (temperature 0, short max_new_tokens):
  baseline_colluder / baseline_honest
  patch colluder↔honest @ attn_L22 / resid_L21
  ablate_role attn_L22 α=1; ablate_role resid L21 α=1
  ablate_mode L21; random ablate @ attn_L22
  ablate_role_perp L21 (piggyback)

Cheap-extras (--extras / --extras-only; Paige queue #4 offline-prep):
  multi-site patch/ablate attn_L22 AND resid_L21 together
  PCA k8 L23 project-out + ambient single-dir ablate @ resid_L23
  role_perp directional component transplant (patch_h2c / patch_c2h donor)

  python3 scripts/matched_prefix_interchange.py \
    --pairs results/transfer_stable/matched_prefix/pairs_transfer.jsonl \
    --directions results/transfer_stable/directions \
    --out results/transfer_stable/MATCHED_PREFIX_cheap_extras_dry_run \
    --extras --dry-run
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


# ---------------------------------------------------------------------------
# Pair I/O
# ---------------------------------------------------------------------------

def load_pairs(path: Path) -> list[dict]:
    text = path.read_text().strip()
    if not text:
        return []
    if path.suffix == ".json":
        obj = json.loads(text)
        if isinstance(obj, list):
            return obj
        return obj.get("pairs") or []
    out = []
    for line in text.splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


def compose_prompt(pair: dict, arm: str) -> str:
    div = pair["divergence"]
    if arm in ("a", "colluder", "arm_a"):
        block = div["arm_a"]["text"]
    else:
        block = div["arm_b"]["text"]
    return pair["shared_stem"] + "\n" + block + "\n" + pair["shared_suffix"]


def shared_prefix_len(ids_a: list[int], ids_b: list[int]) -> int:
    t = 0
    n = min(len(ids_a), len(ids_b))
    while t < n and ids_a[t] == ids_b[t]:
        t += 1
    return t


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
    # Prefer first standalone letter on its own / after Recommendation
    for pat in [
        r"(?:Recommendation|RECOMMENDATION)\s*:?\s*([A-D1-9])\b",
        r"^\s*([A-D1-9])\s*$",
        r"\bOption\s+([A-D1-9])\b",
        r"\b([A-D1-9])\b",
    ]:
        m = re.search(pat, t, re.I | re.M)
        if m:
            v = m.group(1).upper()
            if v in allowed:
                return v
    return "?"


def build_chat_prompt(tok, body: str) -> str:
    messages = [{"role": "user", "content": body}]
    if hasattr(tok, "apply_chat_template"):
        try:
            return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            pass
    return f"<start_of_turn>user\n{body}<end_of_turn>\n<start_of_turn>model\n"


def _unit(path: Path) -> np.ndarray:
    u = np.load(path).astype(np.float64)
    return (u / (np.linalg.norm(u) + 1e-12)).astype(np.float32)


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


# ---------------------------------------------------------------------------
# Capture / patch / ablate hooks
# ---------------------------------------------------------------------------

def capture_activations(
    model,
    tok,
    prompt: str,
    *,
    sites: list[tuple[str, int, str]],
    pos: int,
) -> dict[str, "torch.Tensor"]:
    """Teacher-force one forward; cache activation at `pos` for each (name, layer, site)."""
    assert torch is not None
    ids = tok(prompt, return_tensors="pt").to(model.device)
    seq_len = int(ids["input_ids"].shape[-1])
    if pos < 0:
        pos = seq_len + pos
    if not (0 <= pos < seq_len):
        raise ValueError(f"pos {pos} out of range for seq_len {seq_len}")
    captured: dict[str, torch.Tensor] = {}
    hooks = []

    def make(name: str):
        def hook(_m, _i, out):
            h = out[0] if isinstance(out, tuple) else out
            captured[name] = h[0, pos, :].detach().to(torch.float32).cpu()

        return hook

    for name, layer, site in sites:
        mod = _module_for_site(model, layer, site)
        hooks.append(mod.register_forward_hook(make(name)))
    try:
        with torch.no_grad():
            model(**ids)
    finally:
        for h in hooks:
            h.remove()
    return captured


def make_patch_hook(pos: int, vec: "torch.Tensor"):
    fired = {"done": False}

    def hook(_m, _i, out):
        h = out[0] if isinstance(out, tuple) else out
        # Prefill only: when sequence still includes the shared prefix position
        if h.shape[1] > pos and not fired["done"]:
            h = h.clone()
            h[0, pos, :] = vec.to(device=h.device, dtype=h.dtype)
            fired["done"] = True
            return (h,) + out[1:] if isinstance(out, tuple) else h
        return out

    return hook


def make_ablate_hook(pos: int, u: "torch.Tensor", alpha: float, *, last_only: bool = True):
    """Ablate direction û at position `pos` on prefill (and optionally last token each step)."""

    def hook(_m, _i, out):
        h = out[0] if isinstance(out, tuple) else out
        h = h.clone()
        u_d = u.to(device=h.device, dtype=torch.float32)
        if last_only:
            # Edit last shared pos on prefill; on decode steps edit last token
            if h.shape[1] > pos:
                # first forward covering the intervention site
                v = h[0, pos, :].float()
                proj = torch.dot(v, u_d)
                h[0, pos, :] = (v - alpha * proj * u_d).to(dtype=h.dtype)
            else:
                v = h[0, -1, :].float()
                proj = torch.dot(v, u_d)
                h[0, -1, :] = (v - alpha * proj * u_d).to(dtype=h.dtype)
        else:
            v = h[0].float()
            proj = v @ u_d
            h[0] = (v - alpha * proj[:, None] * u_d).to(dtype=h.dtype)
        return (h,) + out[1:] if isinstance(out, tuple) else h

    return hook



def make_dir_patch_hook(pos: int, u: "torch.Tensor", src_vec: "torch.Tensor"):
    """Directional component transplant: h' = h - (h·û)û + (h_src·û)û at `pos`."""
    fired = {"done": False}

    def hook(_m, _i, out):
        h = out[0] if isinstance(out, tuple) else out
        if h.shape[1] > pos and not fired["done"]:
            h = h.clone()
            u_d = u.to(device=h.device, dtype=torch.float32)
            src = src_vec.to(device=h.device, dtype=torch.float32)
            v = h[0, pos, :].float()
            proj_dest = torch.dot(v, u_d)
            proj_src = torch.dot(src.float(), u_d)
            h[0, pos, :] = (v - proj_dest * u_d + proj_src * u_d).to(dtype=h.dtype)
            fired["done"] = True
            return (h,) + out[1:] if isinstance(out, tuple) else h
        return out

    return hook


def make_project_out_hook(pos: int, basis: "torch.Tensor", alpha: float = 1.0, *, last_only: bool = True):
    """Project-out subspace: h' = h - α * B.T @ (B @ h) with B rows ≈ orthonormal (k, d)."""

    def hook(_m, _i, out):
        h = out[0] if isinstance(out, tuple) else out
        h = h.clone()
        B = basis.to(device=h.device, dtype=torch.float32)  # (k, d)
        if last_only:
            if h.shape[1] > pos:
                v = h[0, pos, :].float()
                coeffs = B @ v  # (k,)
                h[0, pos, :] = (v - alpha * (B.T @ coeffs)).to(dtype=h.dtype)
            else:
                v = h[0, -1, :].float()
                coeffs = B @ v
                h[0, -1, :] = (v - alpha * (B.T @ coeffs)).to(dtype=h.dtype)
        else:
            v = h[0].float()  # (T, d)
            coeffs = v @ B.T  # (T, k)
            h[0] = (v - alpha * (coeffs @ B)).to(dtype=h.dtype)
        return (h,) + out[1:] if isinstance(out, tuple) else h

    return hook


# no_grad applied inside when torch present
def generate_with_hooks(
    model,
    tok,
    prompt: str,
    *,
    hooks_spec: list[tuple[Any, Any]],
    max_new_tokens: int,
    return_first_logits: bool = True,
) -> dict:
    """hooks_spec: list of (module, hook_fn)."""
    assert torch is not None
    ids = tok(prompt, return_tensors="pt").to(model.device)
    handles = []
    for mod, fn in hooks_spec:
        handles.append(mod.register_forward_hook(fn))
    try:
        with torch.no_grad():
            out = model.generate(
                **ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                return_dict_in_generate=True,
                output_scores=return_first_logits,
            )
    finally:
        for h in handles:
            h.remove()
    gen_ids = out.sequences[0][ids["input_ids"].shape[-1] :]
    text = tok.decode(gen_ids, skip_special_tokens=True)
    result: dict[str, Any] = {"text": text, "gen_ids": gen_ids.tolist()}
    if return_first_logits and getattr(out, "scores", None):
        logits0 = out.scores[0][0].float().cpu()
        result["logits0"] = logits0
    return result


def letter_token_ids(tok, letters: list[str]) -> dict[str, list[int]]:
    """Candidate token ids for each vote letter (with/without leading space)."""
    out: dict[str, list[int]] = {}
    for L in letters:
        cands = []
        for s in (L, f" {L}", f"\n{L}"):
            ids = tok.encode(s, add_special_tokens=False)
            if len(ids) == 1:
                cands.append(ids[0])
            elif len(ids) >= 1:
                # take last piece if multi
                cands.append(ids[-1])
        # dedupe
        out[L] = list(dict.fromkeys(cands))
    return out


def p_letter_from_logits(logits: "torch.Tensor", letter_ids: dict[str, list[int]], target: str) -> dict:
    """Softmax over candidate letter tokens; return P(target) and argmax among letters."""
    assert torch is not None
    # Gather unique ids
    all_ids = []
    map_back: list[str] = []
    for L, ids in letter_ids.items():
        for i in ids:
            all_ids.append(i)
            map_back.append(L)
    if not all_ids:
        return {"p_target": None, "argmax_letter": "?", "letter_probs": {}}
    idx = torch.tensor(all_ids, dtype=torch.long)
    vals = logits[idx]
    # Collapse duplicates per letter via logsumexp
    letters = list(letter_ids.keys())
    agg = []
    for L in letters:
        mask = [i for i, m in enumerate(map_back) if m == L]
        if not mask:
            agg.append(torch.tensor(-1e9))
        else:
            agg.append(torch.logsumexp(vals[mask], dim=0))
    agg_t = torch.stack(agg)
    probs = torch.softmax(agg_t, dim=0)
    letter_probs = {L: float(probs[i]) for i, L in enumerate(letters)}
    argmax = letters[int(torch.argmax(probs))]
    p_t = letter_probs.get(target.upper(), letter_probs.get(target, 0.0))
    return {"p_target": p_t, "argmax_letter": argmax, "letter_probs": letter_probs}


# ---------------------------------------------------------------------------
# Arms
# ---------------------------------------------------------------------------

def build_arms(
    directions: Path,
    seed: int = 0,
    *,
    extras: bool = False,
    extras_only: bool = False,
) -> list[dict]:
    """Build arm list. Default = original 11; --extras appends cheap-extras; --extras-only = baselines + extras."""
    role_attn = directions / "lr_role_attn_L22.npy"
    role_l21 = directions / "lr_role_L21.npy"
    mode_l21 = directions / "lr_mode_L21.npy"
    perp_l21 = directions / "lr_role_perp_mode_L21.npy"
    pca_k8 = directions / "pca_contrast_k8_L23.npy"
    pca_amb = directions / "pca_contrast_k8_L23_lr_ambient.npy"
    for p in (role_attn, role_l21, mode_l21):
        if not p.exists():
            raise FileNotFoundError(p)
    u_role_attn = _unit(role_attn)
    u_role_l21 = _unit(role_l21)
    u_mode = _unit(mode_l21)
    u_perp = _unit(perp_l21) if perp_l21.exists() else None
    rng = np.random.default_rng(seed)
    u_rand = rng.normal(size=u_role_attn.shape).astype(np.float64)
    u_rand = (u_rand / (np.linalg.norm(u_rand) + 1e-12)).astype(np.float32)

    baselines: list[dict] = [
        {"name": "baseline_colluder", "kind": "baseline", "prompt_arm": "colluder"},
        {"name": "baseline_honest", "kind": "baseline", "prompt_arm": "honest"},
    ]
    core_arms: list[dict] = [
        {
            "name": "patch_c2h_attn_L22",
            "kind": "patch",
            "prompt_arm": "honest",
            "src_arm": "colluder",
            "site": "attn",
            "layer": 22,
            "cache_key": "attn_L22",
        },
        {
            "name": "patch_h2c_attn_L22",
            "kind": "patch",
            "prompt_arm": "colluder",
            "src_arm": "honest",
            "site": "attn",
            "layer": 22,
            "cache_key": "attn_L22",
        },
        {
            "name": "patch_c2h_resid_L21",
            "kind": "patch",
            "prompt_arm": "honest",
            "src_arm": "colluder",
            "site": "residual",
            "layer": 21,
            "cache_key": "resid_L21",
        },
        {
            "name": "patch_h2c_resid_L21",
            "kind": "patch",
            "prompt_arm": "colluder",
            "src_arm": "honest",
            "site": "residual",
            "layer": 21,
            "cache_key": "resid_L21",
        },
        {
            "name": "ablate_role_attn_L22_colluder",
            "kind": "ablate",
            "prompt_arm": "colluder",
            "site": "attn",
            "layer": 22,
            "u": u_role_attn,
            "alpha": 1.0,
            "dir_file": "lr_role_attn_L22.npy",
        },
        {
            "name": "ablate_role_resid_L21_colluder",
            "kind": "ablate",
            "prompt_arm": "colluder",
            "site": "residual",
            "layer": 21,
            "u": u_role_l21,
            "alpha": 1.0,
            "dir_file": "lr_role_L21.npy",
        },
        {
            "name": "ablate_mode_resid_L21_colluder",
            "kind": "ablate",
            "prompt_arm": "colluder",
            "site": "residual",
            "layer": 21,
            "u": u_mode,
            "alpha": 1.0,
            "dir_file": "lr_mode_L21.npy",
        },
        {
            "name": "ablate_random_attn_L22_colluder",
            "kind": "ablate",
            "prompt_arm": "colluder",
            "site": "attn",
            "layer": 22,
            "u": u_rand,
            "alpha": 1.0,
            "dir_file": "random",
        },
    ]
    if u_perp is not None:
        core_arms.append(
            {
                "name": "ablate_role_perp_resid_L21_colluder",
                "kind": "ablate",
                "prompt_arm": "colluder",
                "site": "residual",
                "layer": 21,
                "u": u_perp,
                "alpha": 1.0,
                "dir_file": "lr_role_perp_mode_L21.npy",
            }
        )

    extra_arms: list[dict] = []
    if extras or extras_only:
        # (1) Multi-site L21+L22 — patch / ablate attn_L22 AND resid_L21 together
        extra_arms.extend(
            [
                {
                    "name": "patch_c2h_attn_L22+resid_L21",
                    "kind": "multi_patch",
                    "prompt_arm": "honest",
                    "src_arm": "colluder",
                    "ops": [
                        {"site": "attn", "layer": 22, "cache_key": "attn_L22"},
                        {"site": "residual", "layer": 21, "cache_key": "resid_L21"},
                    ],
                },
                {
                    "name": "patch_h2c_attn_L22+resid_L21",
                    "kind": "multi_patch",
                    "prompt_arm": "colluder",
                    "src_arm": "honest",
                    "ops": [
                        {"site": "attn", "layer": 22, "cache_key": "attn_L22"},
                        {"site": "residual", "layer": 21, "cache_key": "resid_L21"},
                    ],
                },
                {
                    "name": "ablate_role_attn_L22+resid_L21_colluder",
                    "kind": "multi_ablate",
                    "prompt_arm": "colluder",
                    "ops": [
                        {
                            "site": "attn",
                            "layer": 22,
                            "u": u_role_attn,
                            "alpha": 1.0,
                            "dir_file": "lr_role_attn_L22.npy",
                        },
                        {
                            "site": "residual",
                            "layer": 21,
                            "u": u_role_l21,
                            "alpha": 1.0,
                            "dir_file": "lr_role_L21.npy",
                        },
                    ],
                },
            ]
        )
        # (2) PCA k8 L23 project-out (+ ambient single-dir ablate if present)
        if pca_k8.exists():
            P = np.load(pca_k8).astype(np.float32)
            if P.ndim != 2:
                raise ValueError(f"expected pca basis (k,d), got {P.shape}")
            norms = np.linalg.norm(P, axis=1, keepdims=True) + 1e-12
            P = (P / norms).astype(np.float32)
            extra_arms.append(
                {
                    "name": "project_out_pca_k8_resid_L23_colluder",
                    "kind": "project_out",
                    "prompt_arm": "colluder",
                    "site": "residual",
                    "layer": 23,
                    "basis": P,
                    "alpha": 1.0,
                    "dir_file": "pca_contrast_k8_L23.npy",
                }
            )
        else:
            extra_arms.append(
                {
                    "name": "project_out_pca_k8_resid_L23_colluder__STUB_MISSING",
                    "kind": "baseline",
                    "prompt_arm": "colluder",
                    "missing": "pca_contrast_k8_L23.npy",
                }
            )
        if pca_amb.exists():
            extra_arms.append(
                {
                    "name": "ablate_pca_lr_ambient_resid_L23_colluder",
                    "kind": "ablate",
                    "prompt_arm": "colluder",
                    "site": "residual",
                    "layer": 23,
                    "u": _unit(pca_amb),
                    "alpha": 1.0,
                    "dir_file": "pca_contrast_k8_L23_lr_ambient.npy",
                }
            )
        # (3) role_perp as patch donor — directional component transplant @ resid_L21
        if u_perp is not None:
            extra_arms.extend(
                [
                    {
                        "name": "patch_c2h_role_perp_resid_L21",
                        "kind": "dir_patch",
                        "prompt_arm": "honest",
                        "src_arm": "colluder",
                        "site": "residual",
                        "layer": 21,
                        "cache_key": "resid_L21",
                        "u": u_perp,
                        "dir_file": "lr_role_perp_mode_L21.npy",
                    },
                    {
                        "name": "patch_h2c_role_perp_resid_L21",
                        "kind": "dir_patch",
                        "prompt_arm": "colluder",
                        "src_arm": "honest",
                        "site": "residual",
                        "layer": 21,
                        "cache_key": "resid_L21",
                        "u": u_perp,
                        "dir_file": "lr_role_perp_mode_L21.npy",
                    },
                ]
            )

    if extras_only:
        return baselines + extra_arms
    if extras:
        return baselines + core_arms + extra_arms
    return baselines + core_arms


def cache_sites_for_arms(arms: list[dict]) -> list[tuple[str, int, str]]:
    """Derive capture sites from arms (incl. multi / dir_patch / L23)."""
    needed: dict[str, tuple[str, int, str]] = {
        "attn_L22": ("attn_L22", 22, "attn"),
        "resid_L21": ("resid_L21", 21, "residual"),
    }
    for arm in arms:
        kind = arm.get("kind")
        if kind == "patch":
            key = arm["cache_key"]
            needed[key] = (key, int(arm["layer"]), arm["site"])
        elif kind == "multi_patch":
            for op in arm["ops"]:
                key = op["cache_key"]
                needed[key] = (key, int(op["layer"]), op["site"])
        elif kind == "dir_patch":
            key = arm["cache_key"]
            needed[key] = (key, int(arm["layer"]), arm["site"])
        elif kind in ("ablate", "project_out"):
            if int(arm.get("layer", -1)) == 23:
                needed.setdefault("resid_L23", ("resid_L23", 23, "residual"))
    return list(needed.values())


def load_gemma(model_id: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16, device_map="auto", low_cpu_mem_usage=True
    )
    model.eval()
    return model, tok


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pairs", type=Path, required=True)
    ap.add_argument("--directions", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model", default="google/gemma-2-9b-it")
    ap.add_argument("--max-pairs", type=int, default=12)
    ap.add_argument("--max-new-tokens", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--skip-perp", action="store_true")
    ap.add_argument(
        "--extras",
        action="store_true",
        help="append cheap-extras arms (multi-site L21+L22, PCA k8 L23, role_perp dir_patch)",
    )
    ap.add_argument(
        "--extras-only",
        action="store_true",
        help="baselines + cheap-extras only (omit original single-site interchange arms)",
    )
    args = ap.parse_args()

    pairs = load_pairs(args.pairs)[: args.max_pairs]
    directions = args.directions if args.directions.is_absolute() else (
        Path(__file__).resolve().parents[1] / args.directions
    )
    arms = build_arms(
        directions,
        seed=args.seed,
        extras=bool(args.extras),
        extras_only=bool(args.extras_only),
    )
    if args.skip_perp:
        arms = [a for a in arms if "perp" not in a["name"]]

    args.out.mkdir(parents=True, exist_ok=True)
    arm_names = [a["name"] for a in arms]
    suites = {(p.get("suite") or ("transfer" if "transfer" in str(p.get("source_run_id") or "") else "core")) for p in pairs}
    suite = "transfer" if suites == {"transfer"} else ("mixed" if len(suites) > 1 else "core")
    source_run = (pairs[0].get("source_run_id") if pairs else None) or (
        "gemma2_9b/transfer/RUNPOD" if suite == "transfer" else "gemma2_9b/core/20261001T012639Z"
    )
    meta = {
        "model": args.model,
        "n_pairs": len(pairs),
        "pair_ids": [p["pair_id"] for p in pairs],
        "arms": arm_names,
        "n_arms": len(arms),
        "max_new_tokens": args.max_new_tokens,
        "protocol": "matched_prefix_activation_interchange",
        "suite": suite,
        "world": "Transfer" if suite == "transfer" else ("Mixed" if suite == "mixed" else "Core"),
        "source_run": source_run,
        "sites": ["attn_L22", "residual_L21"]
        + (["attn_L22+resid_L21", "residual_L23_pca_k8"] if (args.extras or args.extras_only) else []),
        "extras": bool(args.extras or args.extras_only),
        "extras_only": bool(args.extras_only),
        "dry_run": args.dry_run,
    }
    (args.out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2), flush=True)

    # Offline tokenizer prefix re-check (CPU)
    try:
        from transformers import AutoTokenizer
        tok_chk = AutoTokenizer.from_pretrained(args.model)
    except Exception as e:
        tok_chk = None
        print(f"[warn] tokenizer unavailable for pre-check: {e}", flush=True)

    print("\n=== PAIRS ===", flush=True)
    for p in pairs:
        pa = compose_prompt(p, "colluder")
        pb = compose_prompt(p, "honest")
        t_star = None
        if tok_chk is not None:
            ids_a = tok_chk.encode(pa, add_special_tokens=False)
            ids_b = tok_chk.encode(pb, add_special_tokens=False)
            t_star = shared_prefix_len(ids_a, ids_b)
            if t_star < 8:
                raise RuntimeError(f"prefix FAIL {p['pair_id']} t*={t_star}")
        print(
            f"  {p['pair_id']}: tgt={p['target_option']} fair={p['fair_option']} "
            f"domain={p.get('domain')} t*={t_star or (p.get('tokenizer_check') or {}).get('t_star')}",
            flush=True,
        )

    print("\n=== ARMS ===", flush=True)
    for a in arms:
        print(f"  - {a['name']} ({a['kind']})", flush=True)

    if args.dry_run:
        md = [
            "# Matched-prefix interchange — dry-run",
            "",
            f"- pairs: {len(pairs)}",
            f"- arms: {len(arms)}",
            f"- model: {args.model}",
            f"- extras: {bool(args.extras or args.extras_only)} (extras_only={bool(args.extras_only)})",
            f"- cache_sites: {[s[0] for s in cache_sites_for_arms(arms)]}",
            "",
            "## Pairs",
        ]
        for p in pairs:
            md.append(
                f"- `{p['pair_id']}` tgt={p['target_option']} domain={p.get('domain')} "
                f"t*={(p.get('tokenizer_check') or {}).get('t_star')}"
            )
        md += ["", "## Arms"] + [f"- {n}" for n in arm_names]
        (args.out / "DRY_RUN.md").write_text("\n".join(md) + "\n")
        print(f"\n[dry-run] OK — wrote {args.out / 'DRY_RUN.md'}", flush=True)
        return 0

    if torch is None:
        print("ERROR: torch required for non-dry-run", flush=True)
        return 1

    print(f"\n[load] {args.model}", flush=True)
    model, tok = load_gemma(args.model)
    cache_sites = cache_sites_for_arms(arms)

    per_pair = []
    t0 = time.time()
    for pi, pair in enumerate(pairs):
        target = str(pair["target_option"]).upper()
        fair = str(pair["fair_option"]).upper()
        letters = sorted(set(option_label_items(pair.get("option_labels")) + [target, fair]))
        body_c = compose_prompt(pair, "colluder")
        body_h = compose_prompt(pair, "honest")
        prompt_c = build_chat_prompt(tok, body_c)
        prompt_h = build_chat_prompt(tok, body_h)
        ids_c = tok.encode(prompt_c, add_special_tokens=False)
        ids_h = tok.encode(prompt_h, add_special_tokens=False)
        t_star = shared_prefix_len(ids_c, ids_h)
        if t_star < 8:
            raise RuntimeError(f"chat-template prefix FAIL {pair['pair_id']} t*={t_star}")
        # Last shared token index
        pos = t_star - 1
        print(
            f"\n[{pi+1}/{len(pairs)}] {pair['pair_id']} t*={t_star} pos={pos} "
            f"tgt={target}",
            flush=True,
        )

        # Capture activations at last shared token on both arms
        cap_c = capture_activations(model, tok, prompt_c, sites=cache_sites, pos=pos)
        cap_h = capture_activations(model, tok, prompt_h, sites=cache_sites, pos=pos)
        caches = {"colluder": cap_c, "honest": cap_h}
        letter_ids = letter_token_ids(tok, letters)

        arm_results = {}
        for arm in arms:
            kind = arm["kind"]
            prompt_arm = arm["prompt_arm"]
            prompt = prompt_c if prompt_arm == "colluder" else prompt_h
            hooks_spec = []
            if kind == "patch":
                src = arm["src_arm"]
                key = arm["cache_key"]
                vec = caches[src][key]
                mod = _module_for_site(model, arm["layer"], arm["site"])
                hooks_spec.append((mod, make_patch_hook(pos, vec)))
            elif kind == "multi_patch":
                src = arm["src_arm"]
                for op in arm["ops"]:
                    vec = caches[src][op["cache_key"]]
                    mod = _module_for_site(model, op["layer"], op["site"])
                    hooks_spec.append((mod, make_patch_hook(pos, vec)))
            elif kind == "dir_patch":
                src = arm["src_arm"]
                key = arm["cache_key"]
                src_vec = caches[src][key]
                u = torch.tensor(arm["u"], dtype=torch.float32)
                mod = _module_for_site(model, arm["layer"], arm["site"])
                hooks_spec.append((mod, make_dir_patch_hook(pos, u, src_vec)))
            elif kind == "ablate":
                u = torch.tensor(arm["u"], dtype=torch.float32)
                mod = _module_for_site(model, arm["layer"], arm["site"])
                hooks_spec.append((mod, make_ablate_hook(pos, u, float(arm["alpha"]))))
            elif kind == "multi_ablate":
                for op in arm["ops"]:
                    u = torch.tensor(op["u"], dtype=torch.float32)
                    mod = _module_for_site(model, op["layer"], op["site"])
                    hooks_spec.append((mod, make_ablate_hook(pos, u, float(op["alpha"]))))
            elif kind == "project_out":
                B = torch.tensor(arm["basis"], dtype=torch.float32)
                mod = _module_for_site(model, arm["layer"], arm["site"])
                hooks_spec.append(
                    (mod, make_project_out_hook(pos, B, float(arm.get("alpha", 1.0))))
                )
            # baseline: no hooks
            gen = generate_with_hooks(
                model, tok, prompt, hooks_spec=hooks_spec, max_new_tokens=args.max_new_tokens
            )
            vote = parse_vote_letter(gen["text"], letters)
            metrics = {"vote": vote, "text": gen["text"][:200]}
            if "logits0" in gen:
                pl = p_letter_from_logits(gen["logits0"], letter_ids, target)
                metrics.update(pl)
            arm_results[arm["name"]] = metrics
            print(
                f"  {arm['name']}: vote={vote} p_tgt={metrics.get('p_target')} "
                f"argmax={metrics.get('argmax_letter')} text={gen['text'][:40]!r}",
                flush=True,
            )

        # Derived flip / delta metrics
        base_c = arm_results.get("baseline_colluder") or {}
        base_h = arm_results.get("baseline_honest") or {}
        derived = {
            "baseline_gap_p": None,
            "flips": {},
        }
        if base_c.get("p_target") is not None and base_h.get("p_target") is not None:
            derived["baseline_gap_p"] = float(base_c["p_target"] - base_h["p_target"])
        for name, m in arm_results.items():
            if name.startswith("baseline"):
                continue
            if "c2h" in name or (name.startswith("patch_") and "honest" in str(arm_results)):
                ref = base_h
                ref_name = "baseline_honest"
            elif "h2c" in name:
                ref = base_c
                ref_name = "baseline_colluder"
            elif "ablate" in name:
                ref = base_c
                ref_name = "baseline_colluder"
            else:
                ref = base_c
                ref_name = "baseline_colluder"
            flip = (m.get("vote") != ref.get("vote")) if m.get("vote") != "?" and ref.get("vote") != "?" else None
            dp = None
            if m.get("p_target") is not None and ref.get("p_target") is not None:
                dp = float(m["p_target"] - ref["p_target"])
            derived["flips"][name] = {
                "vs": ref_name,
                "vote_flip": flip,
                "delta_p_target": dp,
            }

        per_pair.append(
            {
                "pair_id": pair["pair_id"],
                "source_scenario_id": pair.get("source_scenario_id"),
                "domain": pair.get("domain"),
                "target_option": target,
                "fair_option": fair,
                "t_star": t_star,
                "pos": pos,
                "arms": arm_results,
                "derived": derived,
            }
        )
        (args.out / "per_pair.json").write_text(json.dumps(per_pair, indent=2) + "\n")

    # Summary
    summary = summarize(per_pair, arm_names)
    summary["elapsed_sec"] = round(time.time() - t0, 1)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.out / "RESULTS.md").write_text(format_md(summary, per_pair, meta) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    print(f"\n[done] {args.out}", flush=True)
    return 0


def summarize(per_pair: list[dict], arm_names: list[str]) -> dict:
    def mean(xs):
        xs = [x for x in xs if x is not None]
        return float(np.mean(xs)) if xs else None

    out: dict[str, Any] = {"n_pairs": len(per_pair), "by_arm": {}}
    for name in arm_names:
        votes_target = []
        p_tgts = []
        flips = []
        dps = []
        for row in per_pair:
            m = (row.get("arms") or {}).get(name) or {}
            tgt = row["target_option"]
            votes_target.append(1.0 if m.get("vote") == tgt else 0.0 if m.get("vote") not in (None, "?") else None)
            p_tgts.append(m.get("p_target"))
            fl = ((row.get("derived") or {}).get("flips") or {}).get(name) or {}
            flips.append(fl.get("vote_flip"))
            dps.append(fl.get("delta_p_target"))
        out["by_arm"][name] = {
            "mean_p_target": mean(p_tgts),
            "target_vote_rate": mean(votes_target),
            "flip_rate_vs_ref": mean([1.0 if f else 0.0 if f is not None else None for f in flips]),
            "mean_delta_p_vs_ref": mean(dps),
        }
    # Primary comparisons
    by = out["by_arm"]
    out["primary"] = {
        "baseline_gap_p": mean(
            [r.get("derived", {}).get("baseline_gap_p") for r in per_pair]
        ),
        "patch_c2h_attn_vs_honest_dp": (by.get("patch_c2h_attn_L22") or {}).get("mean_delta_p_vs_ref"),
        "patch_h2c_attn_vs_colluder_dp": (by.get("patch_h2c_attn_L22") or {}).get("mean_delta_p_vs_ref"),
        "ablate_role_attn_vs_colluder_dp": (by.get("ablate_role_attn_L22_colluder") or {}).get("mean_delta_p_vs_ref"),
        "ablate_random_attn_vs_colluder_dp": (by.get("ablate_random_attn_L22_colluder") or {}).get("mean_delta_p_vs_ref"),
        "ablate_mode_vs_colluder_dp": (by.get("ablate_mode_resid_L21_colluder") or {}).get("mean_delta_p_vs_ref"),
        "ablate_perp_vs_colluder_dp": (by.get("ablate_role_perp_resid_L21_colluder") or {}).get("mean_delta_p_vs_ref"),
        # cheap-extras
        "patch_h2c_multi_vs_colluder_dp": (by.get("patch_h2c_attn_L22+resid_L21") or {}).get("mean_delta_p_vs_ref"),
        "ablate_multi_vs_colluder_dp": (by.get("ablate_role_attn_L22+resid_L21_colluder") or {}).get("mean_delta_p_vs_ref"),
        "project_out_pca_k8_L23_dp": (by.get("project_out_pca_k8_resid_L23_colluder") or {}).get("mean_delta_p_vs_ref"),
        "patch_h2c_role_perp_dp": (by.get("patch_h2c_role_perp_resid_L21") or {}).get("mean_delta_p_vs_ref"),
        "patch_c2h_role_perp_dp": (by.get("patch_c2h_role_perp_resid_L21") or {}).get("mean_delta_p_vs_ref"),
    }
    return out


def format_md(summary: dict, per_pair: list[dict], meta: dict) -> str:
    lines = [
        "# Matched-prefix activation interchange — results",
        "",
        f"- n_pairs: {summary.get('n_pairs')}",
        f"- model: {meta.get('model')}",
        f"- elapsed_sec: {summary.get('elapsed_sec')}",
        "",
        "## Primary",
        "",
    ]
    for k, v in (summary.get("primary") or {}).items():
        lines.append(f"- **{k}**: {v}")
    lines += ["", "## By arm", ""]
    lines.append("| arm | mean P(target) | target-vote rate | flip rate | ΔP vs ref |")
    lines.append("|---|---:|---:|---:|---:|")
    for name, s in (summary.get("by_arm") or {}).items():
        lines.append(
            f"| {name} | {s.get('mean_p_target')} | {s.get('target_vote_rate')} | "
            f"{s.get('flip_rate_vs_ref')} | {s.get('mean_delta_p_vs_ref')} |"
        )
    lines += ["", "## Per pair (vote / P)", ""]
    for row in per_pair:
        lines.append(f"### {row['pair_id']} (tgt={row['target_option']}, t*={row['t_star']})")
        for an, m in (row.get("arms") or {}).items():
            lines.append(f"- {an}: vote={m.get('vote')} p={m.get('p_target')}")
        lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
