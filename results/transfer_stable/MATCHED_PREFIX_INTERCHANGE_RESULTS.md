> **Withdrawn.** This is an Oct 1 matched-prefix "first positive" writeup. It used the pre-fix harness: the patch sat on the last shared token, and the scorer compared patches to the honest baseline, so the reported honest→colluder ΔP is the untouched gap rather than a causal effect. Withdrawn pending a re-run on the fixed harness. Correlational claims in `papers/transfer-stable-collusion-2026-10/main.pdf` are a separate, earlier result. See `MIGRATION_HANDOFF.md`.

# Matched-prefix interchange Core smoke — results

- Pod `klmkfu4dubgc39` (L4): **done / ok**, exit 0; **terminated** (no billable pods).
- Finished ~**7:45 PM PT** Oct 1, 2026; compute elapsed **54s**; approx spend **~$0.09**; balance **~$25.54**.
- Local: `results/transfer_stable/matched_prefix_20261001T194538/`
- Model: gemma-2-9b-it · n_pairs=10 · Core role-flip pairs

## Primary ΔP / vote / flips

| arm | mean P(target) | target-vote rate | flip rate | ΔP vs ref |
|---|---:|---:|---:|---:|
| baseline_colluder | 0.582 | 0.50 | — | — |
| baseline_honest | 0.101 | 0.10 | — | — |
| patch_c2h_attn_L22 | 0.101 | 0.10 | **0.0** | **0.0** |
| patch_h2c_attn_L22 | 0.582 | 0.50 | **0.4** | **+0.481** |
| patch_c2h_resid_L21 | 0.101 | 0.10 | **0.0** | **0.0** |
| patch_h2c_resid_L21 | 0.582 | 0.50 | **0.4** | **+0.481** |
| ablate_role_attn_L22 | 0.582 | 0.50 | 0.0 | ~0 |
| ablate_role_resid_L21 | 0.582 | 0.50 | 0.0 | ~0 |
| ablate_mode_resid_L21 | 0.582 | 0.50 | 0.0 | ~0 |
| ablate_random_attn_L22 | 0.582 | 0.50 | 0.0 | ~0 |
| ablate_role_perp_resid_L21 | 0.581 | 0.50 | 0.0 | ~0 |

**Takeaway:** Honest→colluder patch restores colluder-level ΔP (~+0.48) with 40% vote flips; colluder→honest matches honest (no flip). Role / mode / random / perp ablations on the colluder are flat.
