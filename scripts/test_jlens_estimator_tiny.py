#!/usr/bin/env python3
"""Check scripts/estimate_jlens_full.py against an exact Jacobian on a tiny Gemma-2 model (CPU, seconds).

Verifies, for one short prompt and then for the mean over two prompts:
  1. causality: the position-to-position Jacobian is zero for target < source;
  2. the all-positions JVP trick returns the exact pair-mean Jacobian M (columns);
  3. the all-positions VJP trick returns the exact M (rows);
  4. the two-pass sketch with rank r = d reconstructs M, and the pass-1/pass-2 accumulation
     over prompts reconstructs the mean of the per-prompt M's.
Run:  python3 scripts/test_jlens_estimator_tiny.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import estimate_jlens_full as E  # noqa: E402

CANDIDATES = [
    "hf-internal-testing/tiny-random-Gemma2ForCausalLM",
    "hf-internal-testing/tiny-random-GemmaForCausalLM",
    "hf-internal-testing/tiny-random-LlamaForCausalLM",
]
TEXTS = ["The quick brown fox jumps over the lazy dog near the river bank.",
         "Two partners agreed privately to support option B while sounding independent in the meeting."]


def exact_pair_mean(f, H0, skip):
    """Brute force: full Jacobian of f over all positions, then the mean over pairs t <= t' (both >= skip)."""
    Jfull = torch.autograd.functional.jacobian(lambda H: f(H), H0)  # (T, d, 1, T, d)
    T, d = H0.shape[-2], H0.shape[-1]
    Jfull = Jfull[:, :, 0]  # (T', d_out, T, d_in)
    acausal = max(float(Jfull[tp, :, t, :].abs().max()) for t in range(T) for tp in range(T) if t > tp) if T > 1 else 0.0
    M = torch.zeros(d, d)
    n = 0
    for tp in range(skip, T):
        for t in range(skip, tp + 1):
            M += Jfull[tp, :, t, :]
            n += 1
    return M / n, acausal


def main() -> int:
    torch.manual_seed(0)
    model = tok = None
    device = "cpu"
    try:  # a random three-block Gemma 2 built from a config: no download, and the tested map crosses two real blocks
        from transformers import Gemma2Config, Gemma2ForCausalLM
        cfg = Gemma2Config(vocab_size=256, hidden_size=32, intermediate_size=64, num_hidden_layers=3,
                           num_attention_heads=4, num_key_value_heads=2, head_dim=8, max_position_embeddings=64,
                           sliding_window=6, attn_implementation="eager")
        model = Gemma2ForCausalLM(cfg).eval()
        for prm in model.parameters():
            prm.requires_grad_(False)
        print("model: random Gemma2 from config (3 blocks, d=32, sliding window 6 on the local layers)")
    except Exception as e:  # noqa: BLE001
        print(f"[info] config build failed ({type(e).__name__}: {str(e)[:100]}); trying Hub tiny models")
        for mid in CANDIDATES:
            try:
                model, tok, device = E.load_bundle(mid, "float32", "eager")
                print("model:", mid)
                break
            except Exception as e2:  # noqa: BLE001
                print(f"[skip] {mid}: {type(e2).__name__}: {str(e2)[:100]}")
    if model is None:
        print("no tiny model available"); return 2
    base = E.get_base(model)
    d = int(model.config.hidden_size)
    nl = len(base.layers)
    L = 0 if nl < 3 else 1
    skip = 1
    print(f"hidden {d}, layers {nl}, testing layer {L} (map crosses {nl - 1 - L} later block(s)), skip {skip}")
    gen = torch.Generator().manual_seed(1)

    def make_inputs(T):
        ids = torch.randint(5, model.config.vocab_size, (1, T), generator=gen)
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}
    ok = True

    def check(name, got, want, tol):
        nonlocal ok
        err = float((got - want).norm() / (want.norm() + 1e-12))
        flag = "PASS" if err < tol else "FAIL"
        ok &= err < tol
        print(f"  [{flag}] {name}: rel err {err:.2e} (tol {tol})")

    Ms = []
    for T in (15, 16):
        inputs = make_inputs(T) if tok is None else E.tokenize(tok, TEXTS[T - 15], 32, device)
        H0 = E.capture_layer_outputs(base, inputs, [L])[L]
        T = H0.shape[1]
        f = E.make_f(base, inputs, L)
        M, acausal = exact_pair_mean(f, H0, skip)
        Ms.append(M)
        print(f"prompt T={T}: exact pair-mean |M|={float(M.norm()):.3f}, acausal max {acausal:.1e}")
        ok &= acausal < 1e-6
        I = torch.eye(d)
        for mode in ("loop", "vmap"):
            try:
                rows = E.pair_mean_jvp(f, H0, I, skip, mode)  # row i = M e_i
                check(f"jvp trick ({mode})", rows.T, M, 1e-4)
            except Exception as e:  # noqa: BLE001
                print(f"  [info] jvp mode {mode} unavailable here: {type(e).__name__}: {str(e)[:120]}")
        rows = E.pair_mean_jvp(f, H0, I, skip, "double")
        check("jvp trick (double-backward)", rows.T, M, 1e-4)
        for batched in (True, False):
            try:
                out = E.pair_mean_vjp_rows(f, H0, [I[: d // 2], I[d // 2 :]], skip, batched)
                check(f"vjp trick (batched={batched})", torch.cat(out), M, 1e-4)
            except Exception as e:  # noqa: BLE001
                print(f"  [info] vjp batched={batched} unavailable here: {type(e).__name__}: {str(e)[:120]}")

    # two-pass sketch over both prompts at full rank, as the estimator accumulates it
    Omega, _ = torch.linalg.qr(torch.randn(d, d))
    Y = torch.zeros(d, d)
    pre = []
    gen.manual_seed(1)
    for T in (15, 16):
        inputs = make_inputs(T) if tok is None else E.tokenize(tok, TEXTS[T - 15], 32, device)
        H0 = E.capture_layer_outputs(base, inputs, [L])[L]
        f = E.make_f(base, inputs, L)
        pre.append((inputs, H0, f))
        Y += E.pair_mean_jvp(f, H0, Omega.T.contiguous(), skip, "loop").T / len(TEXTS)
    Q, _ = torch.linalg.qr(Y)
    B = torch.zeros(d, d)
    for inputs, H0, f in pre:
        rows = E.pair_mean_vjp_rows(f, H0, [Q.T.contiguous()], skip, False)[0]
        B += rows / len(TEXTS)
    J = Q @ B
    check("two-pass sketch (r = d) vs mean of exact M", J, (Ms[0] + Ms[1]) / 2, 1e-3)
    print("ALL PASS" if ok else "SOME CHECKS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
