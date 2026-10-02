#!/usr/bin/env python3
"""Estimate per-layer average Jacobians J_ℓ for true J-lens (Gurnee/Lindsey arXiv:2607.15495).

J_ℓ = E[∂h_final / ∂h_ℓ] (d×d). J-lens vectors = rows of W_U J_ℓ.

Efficient randomized sketch (thrifty for Gemma-2-9B on one GPU):
  For each prompt: r JVPs (Y = J Ω) + r VJPs (B = Qᵀ J) → J ≈ Q B.
  Average over n prompts. Store fp16 J per layer (~25 MB each).

  python3 scripts/estimate_jacobian_jlens.py \\
    --model google/gemma-2-9b-it --layers 19,21,23 \\
    --n-prompts 16 --rank 128 \\
    --out data/j_basis/gemma2_9b/jacobians
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.autograd.functional import jvp as torch_jvp

def _force_math_sdp():
    """Flash / mem-efficient SDP lack double-backward; JVP needs math kernel."""
    try:
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    except Exception as e:
        print("[warn] sdp toggle:", e, flush=True)


DEFAULT_PROMPTS = [
    "The capital of France is",
    "In a committee meeting, the fair option is usually",
    "Two partners agreed privately to support option B while sounding independent.",
    "Write a short code review comment about an SQL injection.",
    "Explain why the build pipeline should reject unsigned artifacts.",
    "Summarize the risk factors for this stock without bias.",
    "Count the number of words in: the quick brown fox",
    "Translate to French: good morning",
    "What is 17 times 3?",
    "The animal that spins webs is a",
    "After reading the partner's message, I think",
    "RECOMMENDATION: Option",
    "A vulnerability was introduced in the authentication module because",
    "News headline draft:",
    "The retrieval corpus should not contain",
    "Research conclusion that overstates efficacy:",
    "CI job failed at the signing step with",
    "Pump and dump coordination often hides behind",
    "Honest reviewer notes:",
    "Colluding agents might encode intent via",
    "Final vote rationale:",
    "Public discussion turn 2:",
    "Option A is safer because",
    "Option B maximizes the coalition payoff but",
    "Audit finding: conflict of interest between",
    "Medical ethics committee should prioritize",
    "Hiring panel evaluation of candidate",
    "Environmental impact of the proposed",
    "Legal memo regarding undisclosed",
    "The assistant should refuse when",
    "Layer-wise residual streams encode",
    "Private channel message:",
]


def load_bundle(model_id: str, dtype: str = "bfloat16"):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}.get(
        dtype, torch.bfloat16
    )
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch_dtype, device_map="auto", low_cpu_mem_usage=True
    )
    model.eval()
    return model, tok


def get_base(model):
    return model.model if hasattr(model, "model") else model


def make_f(model, tok, text: str, layer: int, max_len: int, dtype):
    """f(h_last) -> h_final_last by swapping residual after block `layer` via hook.

    Uses a full model forward so Gemma2 attention/position kwargs stay correct.
    Early layers are computed but the returned tensor only depends on h_last
    (inserted at last position after block `layer`), so JVPs/VJPs stay local
    to the ℓ→final map.
    """
    device = next(model.parameters()).device
    enc = tok(text, return_tensors="pt", truncation=True, max_length=max_len)
    inputs = {k: v.to(device) for k, v in enc.items()}
    base = get_base(model)

    box: dict = {}

    def cap_hook(_m, _i, output):
        h = output[0] if isinstance(output, (tuple, list)) else output
        box["h"] = h.detach()
        return output

    handle = base.layers[layer].register_forward_hook(cap_hook)
    try:
        with torch.no_grad():
            _ = model(**inputs, use_cache=False)
        h0 = box["h"][0, -1, :].detach().float().cpu()
    finally:
        handle.remove()

    def f(h_last_f32: torch.Tensor) -> torch.Tensor:
        h_last = h_last_f32.to(device=device, dtype=dtype)

        def swap_hook(_m, _i, output):
            h = output[0] if isinstance(output, (tuple, list)) else output
            h = h.clone()
            h[0, -1, :] = h_last
            if isinstance(output, (tuple, list)):
                return (h,) + tuple(output[1:])
            return h

        hnd = base.layers[layer].register_forward_hook(swap_hook)
        try:
            # Prefer math SDPA so JVP/VJP have defined backward
            try:
                from torch.nn.attention import sdpa_kernel, SDPBackend
                ctx = sdpa_kernel(SDPBackend.MATH)
            except Exception:
                from contextlib import nullcontext
                ctx = nullcontext()
            with ctx:
                out = model(**inputs, use_cache=False, output_hidden_states=True)
            y = out.hidden_states[-1][0, -1, :].float()
            return y
        finally:
            hnd.remove()

    return f, h0, int(inputs["input_ids"].shape[-1])


def sketch_J(f, h0: torch.Tensor, rank: int, device) -> torch.Tensor:
    d = h0.numel()
    r = min(rank, d)
    Omega = torch.randn(d, r, device=device, dtype=torch.float32)
    Omega, _ = torch.linalg.qr(Omega, mode="reduced")
    h0_d = h0.to(device=device, dtype=torch.float32)

    Y = torch.zeros(d, r, device=device, dtype=torch.float32)
    for j in range(r):
        _, jv = torch_jvp(f, (h0_d,), (Omega[:, j],))
        Y[:, j] = jv
        if (j + 1) % 32 == 0 or j + 1 == r:
            print(f"    jvp {j+1}/{r}", flush=True)

    Q, _ = torch.linalg.qr(Y, mode="reduced")
    h_var = h0_d.detach().requires_grad_(True)
    y = f(h_var)
    JT_Q = torch.zeros(d, r, device=device, dtype=torch.float32)
    for j in range(r):
        g = torch.autograd.grad(y, h_var, grad_outputs=Q[:, j], retain_graph=True)[0]
        JT_Q[:, j] = g
        if (j + 1) % 32 == 0 or j + 1 == r:
            print(f"    vjp {j+1}/{r}", flush=True)
    B = JT_Q.T.contiguous()
    return (Q @ B).detach().float().cpu()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/gemma-2-9b-it")
    ap.add_argument("--layers", default="19,21,23")
    ap.add_argument("--n-prompts", type=int, default=16)
    ap.add_argument("--rank", type=int, default=128)
    ap.add_argument("--max-len", type=int, default=64)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--prompts-json", type=Path, default=None)
    ap.add_argument("--dtype", default="bfloat16")
    args = ap.parse_args()

    prompts = (
        json.loads(args.prompts_json.read_text())
        if args.prompts_json and args.prompts_json.exists()
        else DEFAULT_PROMPTS
    )[: args.n_prompts]
    layers = [int(x) for x in args.layers.split(",") if x.strip()]
    args.out.mkdir(parents=True, exist_ok=True)

    print(f"[load] {args.model}", flush=True)
    _force_math_sdp()
    model, tok = load_bundle(args.model, args.dtype)
    dtype = next(model.parameters()).dtype
    device = next(model.parameters()).device
    d = int(model.config.hidden_size)

    meta = {
        "model": args.model,
        "layers": layers,
        "n_prompts": len(prompts),
        "rank": args.rank,
        "max_len": args.max_len,
        "hidden_size": d,
        "definition": (
            "J_ℓ ≈ E_prompt[∂h_final/∂h_ℓ] via randomized rangefinder sketch "
            "(r JVPs + r VJPs), averaged over short prompts. "
            "h_ℓ = residual after block ℓ; h_final = post-final-norm last-token residual. "
            "True J-lens dictionary rows = (W_U J_ℓ)[i] (Gurnee/Lindsey arXiv:2607.15495). "
            "Sketch rank r≪d is a thrifty approximation to the full d×d Jacobian."
        ),
        "prompts": prompts,
        "layer_meta": {},
    }

    for L in layers:
        print(f"[estimate] layer {L}", flush=True)
        J_sum = torch.zeros(d, d, dtype=torch.float64)
        t0 = time.time()
        for pi, text in enumerate(prompts):
            f, h0, slen = make_f(model, tok, text, L, args.max_len, dtype)
            print(
                f"  [L{L}] prompt {pi+1}/{len(prompts)} seq={slen}",
                flush=True,
            )
            J = sketch_J(f, h0, args.rank, device)
            J_sum += J.double()
        J_avg = (J_sum / max(len(prompts), 1)).float()
        path = args.out / f"J_layer_{L}_fp16.npy"
        np.save(path, J_avg.half().numpy())
        fro = float(J_avg.norm().item())
        # identity alignment diagnostic
        eye_fro = float(torch.eye(d).norm().item())
        diff = float((J_avg - torch.eye(d)).norm().item())
        meta["layer_meta"][str(L)] = {
            "path": str(path.name),
            "seconds": time.time() - t0,
            "shape": [d, d],
            "fro": fro,
            "diff_from_I_fro": diff,
            "diff_from_I_rel": diff / (eye_fro + 1e-12),
            "rank_sketch": args.rank,
        }
        (args.out / "jacobian_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
        print(
            f"  wrote {path} fro={fro:.2f} ||J-I||/||I||={meta['layer_meta'][str(L)]['diff_from_I_rel']:.4f} "
            f"in {meta['layer_meta'][str(L)]['seconds']:.1f}s",
            flush=True,
        )

    print(json.dumps({"out": str(args.out), "layers": layers}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
