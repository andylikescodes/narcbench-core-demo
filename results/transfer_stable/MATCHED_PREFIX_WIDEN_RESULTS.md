> **Withdrawn.** This is an Oct 1 matched-prefix "first positive" writeup. It used the pre-fix harness: the patch sat on the last shared token, and the scorer compared patches to the honest baseline, so the reported honest→colluder ΔP is the untouched gap rather than a causal effect. Withdrawn pending a re-run on the fixed harness. Correlational claims in `papers/transfer-stable-collusion-2026-10/main.pdf` are a separate, earlier result. See `MIGRATION_HANDOFF.md`.

# Matched-prefix Core widen — results

- Pod `fczidrsh55t44p` (L4 SECURE, $0.49/hr): **done / ok**, exit 0; **terminated** (pods empty; spend/hr ≈ $0.019 storage only).
- Finished ~**8:46 PM PT** Oct 1, 2026; compute elapsed **174s**; approx spend **~$0.09** (balance launch $25.52 → now **~$25.43**).
- Local: `results/transfer_stable/matched_prefix_20261002T033634/`
- Model: gemma-2-9b-it · n_pairs=**33** · Core role-flip widen

## Primary ΔP / vote / flips

| arm | mean P(target) | target-vote rate | flip rate | ΔP vs ref |
|---|---:|---:|---:|---:|
| baseline_colluder | 0.651 | 0.606 | — | — |
| baseline_honest | 0.032 | 0.030 | — | — |
| patch_c2h_attn_L22 | 0.032 | 0.030 | **0.0** | **~0** |
| patch_h2c_attn_L22 | 0.651 | 0.606 | **0.576** | **+0.619** |
| patch_c2h_resid_L21 | 0.032 | 0.030 | **0.0** | **~0** |
| patch_h2c_resid_L21 | 0.651 | 0.606 | **0.576** | **+0.619** |
| ablate_role_attn_L22 | 0.651 | 0.606 | 0.0 | ~0 |
| ablate_role_resid_L21 | 0.650 | 0.606 | 0.0 | ~0 |
| ablate_mode_resid_L21 | 0.651 | 0.606 | 0.0 | ~0 |
| ablate_random_attn_L22 | 0.650 | 0.606 | 0.0 | ~0 |
| ablate_role_perp_resid_L21 | 0.650 | 0.606 | 0.0 | ~0 |

**vs prior smoke (n=10):** h2c ΔP ≈ +0.48 / flip 0.4 → widen h2c ΔP ≈ **+0.62** / flip **0.58** (same qualitative pattern, stronger on fuller set). Ablations still flat.

**Takeaway:** Honest→colluder patch restores colluder-level P/votes on both attn L22 and resid L21; colluder→honest matches honest; role/mode/random/perp ablations on colluder do not move votes.
