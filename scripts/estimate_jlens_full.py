#!/usr/bin/env python3
"""Estimate the mean layer-to-output Jacobian J_l as Gurnee et al. (arXiv:2607.15495) define it.

    J_l = E_{prompt, t <= t'} [ d h_{N,t'} / d h_{l,t} ]

h_l  = residual stream after decoder block l (the hook site of every recorded activation in this repo).
h_N  = residual stream after the last block, before the final norm (the J-lens applies the norm after J).
The expectation runs over a corpus of prompts and over every ordered pair of positions (source t, target
t' >= t). Position 0 (BOS) is excluded as source and target by default (--skip-first).

Method: two-pass randomized range finder for the MEAN Jacobian (Halko, Martinsson and Tropp 2011), rank r.
  pass 1   Y = mean_p M_p Omega              forward-mode: r JVPs per prompt and layer, batched
  pass 2   B = mean_p Q^T M_p,  Q = orth(Y)  reverse-mode: r VJPs per prompt and layer, batched
  J_l ~= Q B
M_p is the pair-mean of prompt p's position-to-position Jacobians. One JVP with the same tangent v at every
source position returns sum_{t <= t'} J_{t'<-t} v at each target t', so the pair-mean is the sum over
targets divided by the number of pairs; one VJP with the same cotangent u at every target returns the
matching row sums at each source. The sketch is exact when r = d.

Outputs (in --out): J_layer_{L}_fp16.npy (d x d), J_layer_{L}_factors.npz (Q, B, singular values of Y),
prompts.json (the exact prompt texts), jlens_meta.json (settings, timings, diagnostics, self-check).

    python3 scripts/estimate_jlens_full.py --model google/gemma-2-9b-it --layers 19-23 \
        --n-prompts 1000 --rank 256 --max-len 128 --out data/j_basis/gemma2_9b/jlens_full
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch


# ----------------------------------------------------------------------------- helpers
def parse_layers(spec: str) -> list[int]:
    spec = spec.strip()
    if "-" in spec and "," not in spec:
        a, b = spec.split("-", 1)
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in spec.split(",") if x.strip()]


def get_base(model):
    return model.model if hasattr(model, "model") else model


def load_bundle(model_id: str, dtype: str, attn: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[dtype]
    tok = AutoTokenizer.from_pretrained(model_id)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch_dtype, attn_implementation=attn, low_cpu_mem_usage=True
    ).to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, tok, device


def build_prompts(tok, mix: str, seed: int) -> list[dict]:
    """mix = 'wikitext:500,alpaca:500'. Deterministic for a given seed. Needs the `datasets` package."""
    from datasets import load_dataset

    rng = random.Random(seed)
    out: list[dict] = []
    for part in mix.split(","):
        name, cnt = part.split(":")
        cnt = int(cnt)
        if name == "wikitext":
            # The canonical dataset moved under a namespace; huggingface_hub >= 1.0 rejects the bare "wikitext" id.
            ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="test+validation")
            paras = [t.strip() for t in ds["text"] if len(t.split()) >= 40 and not t.strip().startswith("=")]
            for t in rng.sample(paras, cnt):
                out.append({"source": "wikitext-103-raw-v1 test+validation", "text": t})
        elif name == "alpaca":
            ds = load_dataset("tatsu-lab/alpaca", split="train")
            for i in rng.sample(range(len(ds)), cnt):
                r = ds[i]
                user = r["instruction"] + ("\n\n" + r["input"] if r["input"].strip() else "")
                text = tok.apply_chat_template(
                    [{"role": "user", "content": user}, {"role": "assistant", "content": r["output"]}],
                    tokenize=False,
                )
                out.append({"source": "tatsu-lab/alpaca, Gemma chat template, reference answer", "text": text})
        else:
            raise SystemExit(f"unknown prompt source {name!r}")
    rng.shuffle(out)
    return out


def tokenize(tok, text: str, max_len: int, device):
    add_special = not (tok.bos_token and text.startswith(tok.bos_token))
    enc = tok(text, return_tensors="pt", truncation=True, max_length=max_len, add_special_tokens=add_special)
    return {k: v.to(device) for k, v in enc.items()}


def capture_layer_outputs(base, inputs, layers: list[int]) -> dict[int, torch.Tensor]:
    """One forward pass; returns {L: residual after block L, shape (1, T, d)}."""
    store: dict[int, torch.Tensor] = {}

    def mk(L):
        def hook(_m, _i, out):
            h = out[0] if isinstance(out, (tuple, list)) else out
            store[L] = h.detach()
        return hook

    handles = [base.layers[L].register_forward_hook(mk(L)) for L in layers]
    try:
        with torch.no_grad():
            base(**inputs, use_cache=False)
    finally:
        for h in handles:
            h.remove()
    return store


def make_f(base, inputs, L: int):
    """f(H) = residual after the LAST block (pre final norm, shape (T, d)) when block L's output is replaced by H."""

    def f(H: torch.Tensor) -> torch.Tensor:
        Hb = H if H.dim() == 3 else H.unsqueeze(0)
        box: dict = {}

        def swap(_m, _i, out):
            if isinstance(out, (tuple, list)):
                return (Hb.to(out[0].dtype),) + tuple(out[1:])
            return Hb.to(out.dtype)

        def grab(_m, _i, out):
            box["y"] = out[0] if isinstance(out, (tuple, list)) else out

        h1 = base.layers[L].register_forward_hook(swap)
        h2 = base.layers[-1].register_forward_hook(grab)
        try:
            base(**inputs, use_cache=False)
        finally:
            h1.remove()
            h2.remove()
        return box["y"][0]

    return f


def n_pairs(T: int, skip: int) -> float:
    n = T - skip
    return n * (n + 1) / 2.0


def pair_mean_jvp(f, H0: torch.Tensor, V: torch.Tensor, skip: int, mode: str) -> torch.Tensor:
    """Rows of V (c, d) are tangents v; returns (c, d) with row i = M v_i, M the pair-mean Jacobian."""
    T, d = H0.shape[-2], H0.shape[-1]
    P = n_pairs(T, skip)
    mask = torch.zeros(T, 1, device=H0.device, dtype=H0.dtype)
    mask[skip:] = 1

    def one(v):
        dH = (v.to(H0.dtype)[None, :] * mask).unsqueeze(0)
        _, jv = torch.func.jvp(f, (H0,), (dH,))
        return jv[skip:].sum(0).float() / P

    if mode == "vmap":
        return torch.func.vmap(one)(V)
    if mode == "loop":
        return torch.stack([one(v) for v in V])
    outs = []
    for v in V:  # double-backward fallback
        dH = (v.to(H0.dtype)[None, :] * mask).unsqueeze(0)
        _, jv = torch.autograd.functional.jvp(f, H0, dH)
        outs.append(jv[skip:].sum(0).float() / P)
    return torch.stack(outs)


def pair_mean_vjp_rows(f, H0: torch.Tensor, Us: list[torch.Tensor], skip: int, batched: bool) -> list[torch.Tensor]:
    """For each chunk U (c, d) of cotangents u, returns (c, d) with row i = u_i^T M. One forward, several grads."""
    H = H0.detach().clone().requires_grad_(True)
    y = f(H)  # (T, d)
    T = y.shape[0]
    P = n_pairs(T, skip)
    mask = torch.zeros(T, 1, device=y.device, dtype=y.dtype)
    mask[skip:] = 1
    results = []
    for ci, U in enumerate(Us):
        keep = ci < len(Us) - 1
        G = U.to(y.dtype)[:, None, :] * mask[None]  # (c, T, d)
        if batched:
            g = torch.autograd.grad(y, H, grad_outputs=G, is_grads_batched=True, retain_graph=True)[0]  # (c, 1, T, d)
            rows = g[:, 0, skip:, :].sum(1).float() / P
        else:
            rows = torch.stack(
                [torch.autograd.grad(y, H, grad_outputs=G[i], retain_graph=True)[0][0, skip:, :].sum(0).float() / P
                 for i in range(G.shape[0])]
            )
        results.append(rows)
        if not keep:
            del y
    return results


# ----------------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="google/gemma-2-9b-it")
    ap.add_argument("--layers", default="19-23")
    ap.add_argument("--n-prompts", type=int, default=1000)
    ap.add_argument("--rank", type=int, default=256)
    ap.add_argument("--chunk", type=int, default=16, help="tangents or cotangents per batched pass")
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--skip-first", type=int, default=1, help="positions excluded as source and target (BOS)")
    ap.add_argument("--holdout", type=int, default=20, help="prompts kept out of the estimate for the self-check")
    ap.add_argument("--prompt-mix", default="wikitext:510,alpaca:510")
    ap.add_argument("--prompts-json", type=Path, default=None, help="use these prompts instead of building them")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--attn", default="eager", help="attention implementation; eager is forward-AD safe and exact for Gemma 2")
    ap.add_argument("--mode", default="auto", choices=["auto", "vmap", "loop", "double"])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--status", type=Path, default=None, help="progress file for a watcher")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    layers = parse_layers(args.layers)
    args.out.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    def status(**kw):
        if args.status:
            kw.setdefault("elapsed_s", round(time.time() - t_start, 1))
            args.status.write_text(json.dumps(kw) + "\n")
        print("[status]", json.dumps(kw), flush=True)

    status(phase="load", model=args.model)
    model, tok, device = load_bundle(args.model, args.dtype, args.attn)
    base = get_base(model)
    d = int(model.config.hidden_size)
    r = min(args.rank, d)

    # prompts
    pj = args.out / "prompts.json"
    if args.prompts_json and args.prompts_json.exists():
        prompts = json.loads(args.prompts_json.read_text())
    elif pj.exists() and args.resume:
        prompts = json.loads(pj.read_text())
    else:
        status(phase="prompts", mix=args.prompt_mix)
        prompts = build_prompts(tok, args.prompt_mix, args.seed)
    if isinstance(prompts[0], str):
        prompts = [{"source": "given", "text": t} for t in prompts]
    need = args.n_prompts + args.holdout
    if len(prompts) < need:
        print(f"[warn] only {len(prompts)} prompts for n={args.n_prompts} + holdout={args.holdout}", flush=True)
    train = prompts[: args.n_prompts]
    hold = prompts[args.n_prompts : args.n_prompts + args.holdout]
    pj.write_text(json.dumps({"train": train, "holdout": hold}, ensure_ascii=False, indent=1) + "\n")
    N = len(train)

    # sketch matrices
    g = torch.Generator(device="cpu").manual_seed(args.seed)
    Omega = torch.randn(d, r, generator=g).to(device)
    Omega, _ = torch.linalg.qr(Omega, mode="reduced")
    Y = {L: torch.zeros(d, r, device=device, dtype=torch.float32) for L in layers}
    ck1 = args.out / "ckpt_pass1.pt"
    done1 = 0
    if args.resume and ck1.exists():
        ck = torch.load(ck1, map_location=device)
        Omega = ck["Omega"].to(device)
        Y = {int(k): v.to(device) for k, v in ck["Y"].items()}
        done1 = int(ck["done"])
        print(f"[resume] pass 1 from prompt {done1}", flush=True)

    mode = args.mode
    batched_vjp = True
    seq_lens: list[int] = []
    chunk = args.chunk

    def run_pass1_prompt(text: str):
        nonlocal mode, chunk
        inputs = tokenize(tok, text, args.max_len, device)
        H0s = capture_layer_outputs(base, inputs, layers)
        T = int(inputs["input_ids"].shape[-1])
        for L in layers:
            f = make_f(base, inputs, L)
            H0 = H0s[L]
            for start in range(0, r, chunk):
                V = Omega[:, start : start + chunk].T.contiguous()
                while True:
                    try:
                        if mode == "auto":
                            try:
                                rows = pair_mean_jvp(f, H0, V, args.skip_first, "vmap")
                                mode = "vmap"
                            except Exception as e:  # noqa: BLE001
                                print(f"[mode] vmap jvp unavailable ({type(e).__name__}: {str(e)[:120]}); trying loop", flush=True)
                                try:
                                    rows = pair_mean_jvp(f, H0, V, args.skip_first, "loop")
                                    mode = "loop"
                                except Exception as e2:  # noqa: BLE001
                                    print(f"[mode] forward-mode jvp unavailable ({type(e2).__name__}); using double-backward", flush=True)
                                    rows = pair_mean_jvp(f, H0, V, args.skip_first, "double")
                                    mode = "double"
                        else:
                            rows = pair_mean_jvp(f, H0, V, args.skip_first, mode)
                        break
                    except torch.cuda.OutOfMemoryError:
                        torch.cuda.empty_cache()
                        if chunk <= 1:
                            raise
                        chunk = max(1, chunk // 2)
                        print(f"[oom] pass 1: chunk -> {chunk}", flush=True)
                        V = V[:chunk]
                        continue
                Y[L][:, start : start + V.shape[0]] += rows.T / N
        return T

    status(phase="pass1", n=N, done=done1, rank=r, layers=layers)
    t1 = time.time()
    for pi in range(done1, N):
        T = run_pass1_prompt(train[pi]["text"])
        seq_lens.append(T)
        if (pi + 1) % 5 == 0 or pi + 1 == N:
            per = (time.time() - t1) / max(pi + 1 - done1, 1)
            status(phase="pass1", n=N, done=pi + 1, mode=mode, chunk=chunk, seq=T,
                   s_per_prompt=round(per, 2), eta_s=round(per * (N - pi - 1)))
        if (pi + 1) % 25 == 0 or pi + 1 == N:
            torch.save({"Omega": Omega.cpu(), "Y": {str(L): Y[L].cpu() for L in layers}, "done": pi + 1}, ck1)
    pass1_s = time.time() - t1

    Q = {}
    svals = {}
    for L in layers:
        Q[L], _ = torch.linalg.qr(Y[L], mode="reduced")
        svals[L] = torch.linalg.svdvals(Y[L]).cpu()

    B = {L: torch.zeros(r, d, device=device, dtype=torch.float32) for L in layers}
    ck2 = args.out / "ckpt_pass2.pt"
    done2 = 0
    if args.resume and ck2.exists():
        ck = torch.load(ck2, map_location=device)
        B = {int(k): v.to(device) for k, v in ck["B"].items()}
        done2 = int(ck["done"])
        print(f"[resume] pass 2 from prompt {done2}", flush=True)

    status(phase="pass2", n=N, done=done2)
    t2 = time.time()
    for pi in range(done2, N):
        inputs = tokenize(tok, train[pi]["text"], args.max_len, device)
        H0s = capture_layer_outputs(base, inputs, layers)
        for L in layers:
            f = make_f(base, inputs, L)
            Us = [Q[L][:, s : s + chunk].T.contiguous() for s in range(0, r, chunk)]
            while True:
                try:
                    rows_list = pair_mean_vjp_rows(f, H0s[L], Us, args.skip_first, batched_vjp)
                    break
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    if chunk <= 1:
                        raise
                    chunk = max(1, chunk // 2)
                    Us = [Q[L][:, s : s + chunk].T.contiguous() for s in range(0, r, chunk)]
                    print(f"[oom] pass 2: chunk -> {chunk}", flush=True)
                except Exception as e:  # noqa: BLE001
                    if batched_vjp:
                        print(f"[mode] batched vjp unavailable ({type(e).__name__}: {str(e)[:120]}); looping", flush=True)
                        batched_vjp = False
                    else:
                        raise
            s = 0
            for rows in rows_list:
                B[L][s : s + rows.shape[0]] += rows / N
                s += rows.shape[0]
        if (pi + 1) % 5 == 0 or pi + 1 == N:
            per = (time.time() - t2) / max(pi + 1 - done2, 1)
            status(phase="pass2", n=N, done=pi + 1, batched_vjp=batched_vjp, chunk=chunk,
                   s_per_prompt=round(per, 2), eta_s=round(per * (N - pi - 1)))
        if (pi + 1) % 25 == 0 or pi + 1 == N:
            torch.save({"B": {str(L): B[L].cpu() for L in layers}, "done": pi + 1}, ck2)
    pass2_s = time.time() - t2

    # assemble, diagnose, self-check
    status(phase="finish")
    meta = {
        "model": args.model, "layers": layers, "n_prompts": N, "holdout": len(hold), "rank": r,
        "chunk_final": chunk, "max_len": args.max_len, "skip_first": args.skip_first, "dtype": args.dtype,
        "attn": args.attn, "jvp_mode": mode, "batched_vjp": batched_vjp, "seed": args.seed,
        "prompt_mix": args.prompt_mix, "hidden_size": d,
        "seq_len_mean": float(np.mean(seq_lens)) if seq_lens else None,
        "pairs_total_est": float(sum(n_pairs(T, args.skip_first) for T in seq_lens)) if seq_lens else None,
        "pass1_s": round(pass1_s, 1), "pass2_s": round(pass2_s, 1),
        "definition": ("J_l = E_{prompt, t<=t'} d h_{N,t'} / d h_{l,t}; h_l = residual after block l, "
                       "h_N = residual after the last block before the final norm; prompts weighted equally, "
                       "pairs weighted equally within a prompt; BOS excluded; two-pass randomized range finder of rank r."),
        "layer_meta": {},
    }
    eye = torch.eye(d, device=device)
    for L in layers:
        J = (Q[L] @ B[L]).float()
        np.save(args.out / f"J_layer_{L}_fp16.npy", J.half().cpu().numpy())
        np.savez(args.out / f"J_layer_{L}_factors.npz", Q=Q[L].cpu().numpy(), B=B[L].cpu().numpy(),
                 svals_Y=svals[L].numpy())
        s = svals[L]
        energy = (s ** 2).cumsum(0) / (s ** 2).sum()
        meta["layer_meta"][str(L)] = {
            "fro": float(J.norm()), "diff_from_I_rel": float((J - eye).norm() / eye.norm()),
            "sval_top": float(s[0]), "sval_last": float(s[-1]), "sval_tail_ratio": float(s[-1] / s[0]),
            "energy_at": {str(k): float(energy[min(k, len(s)) - 1]) for k in (16, 32, 64, 128, 256, 512) if k <= len(s)},
        }
    # self-check on held-out prompts: how well Q B predicts a new prompt's pair-mean Jacobian along random directions
    if hold:
        k = 8
        Vt = torch.randn(k, d, generator=g).to(device)
        Vt = Vt / Vt.norm(dim=1, keepdim=True)
        for L in layers:
            approx = (Q[L] @ (B[L] @ Vt.T)).T  # (k, d)
            per_prompt, mean_m = [], torch.zeros_like(approx)
            for hp in hold:
                inputs = tokenize(tok, hp["text"], args.max_len, device)
                H0 = capture_layer_outputs(base, inputs, [L])[L]
                m = pair_mean_jvp(make_f(base, inputs, L), H0, Vt, args.skip_first, mode if mode != "auto" else "loop")
                per_prompt.append(float((m - approx).norm() / m.norm()))
                mean_m += m / len(hold)
            meta["layer_meta"][str(L)]["selfcheck_rel_err_per_prompt_mean"] = float(np.mean(per_prompt))
            meta["layer_meta"][str(L)]["selfcheck_rel_err_of_holdout_mean"] = float((mean_m - approx).norm() / mean_m.norm())
    (args.out / "jlens_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    status(phase="done", seconds=round(time.time() - t_start, 1))
    print(json.dumps(meta["layer_meta"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
