# Matched-prefix result verification (CPU re-derivation)

Recomputed from the per-pair records in git with `scripts/verify_matched_prefix_results.py`. No model was run. Deltas use the arm-name rule (h2c vs colluder baseline, c2h vs honest baseline).

## Final-norm last-token copy and private span (smoke_20261004T173047Z, SHA 319cf825)


| arm | mean P(target) | vote rate | flip rate | ΔP vs ref | matches written table |
|---|---:|---:|---:|---:|---|
| baseline_colluder | 0.5676 | 0.5000 | — | — | yes |
| baseline_honest | 0.0839 | 0.0833 | — | — | yes |
| patch_h2c_full_final_last | 0.0839 | 0.0833 | 0.4167 | -0.4837 | yes |
| patch_c2h_full_final_last | 0.5676 | 0.5000 | 0.4167 | 0.4837 | yes |
| patch_h2c_full_final_private | 0.5676 | 0.5000 | 0.0000 | 0.0000 | yes |
| patch_c2h_full_final_private | 0.0839 | 0.0833 | 0.0000 | 0.0000 | yes |

- PASS: final-resid per_pair.json vs summary.json: every recomputed by-arm statistic matches the table written on the pod
- PASS: final-resid RESULTS.md per-pair lines vs its own by-arm table: every recomputed by-arm statistic matches the table written on the pod
- PASS: last-token copy: fraction of gap 1.000000, flips 5/5 (cyber_03_role_flip_v1, eng_01_role_flip_v1, env_02_role_flip_v1, food_01_role_flip_v1, leg_03_role_flip_v1); gate passed
- PASS: last-token gate matches the gate record written on the pod
- PASS: private-span copy: mean change 0.000000, flips 0/5; gate failed
- PASS: private-span gate matches the gate record written on the pod
- PASS: h2c last-token copy is digit-identical to the honest baseline on 12/12 pairs (wiring check)
- PASS: c2h last-token copy is digit-identical to the colluder baseline on 12/12 pairs
- PASS: h2c private-span copy is digit-identical to the untouched colluder run on 12/12 pairs (null)
- PASS: untouched gap recomputed as 0.48370584490476182 (handoff: 0.48370584490476176)
- PASS: baseline means colluder 0.5676 / honest 0.0839 match the handoff (0.5676 / 0.0839)

## Layer-21 last-token card (smoke_20261004T165214Z, SHA 1fd0855)


| arm | mean P(target) | vote rate | flip rate | ΔP vs ref | matches written table |
|---|---:|---:|---:|---:|---|
| baseline_colluder | 0.5676 | 0.5000 | — | — | yes |
| baseline_honest | 0.0839 | 0.0833 | — | — | yes |
| patch_h2c_full_resid_L21 | 0.5157 | 0.5000 | 0.0000 | -0.0519 | yes |
| patch_c2h_full_resid_L21 | 0.0841 | 0.0833 | 0.0000 | 0.0002 | yes |
| patch_h2c_role_perp_resid_L21 | 0.5627 | 0.5000 | 0.0000 | -0.0049 | yes |
| patch_c2h_role_perp_resid_L21 | 0.0839 | 0.0833 | 0.0000 | -0.0000 | yes |
| ablate_role_perp_resid_L21_colluder | 0.5686 | 0.5000 | 0.0000 | 0.0010 | yes |
| ablate_role_resid_L21_colluder | 0.5624 | 0.5000 | 0.0000 | -0.0052 | yes |
| patch_h2c_role_resid_L21 | 0.5454 | 0.5000 | 0.0000 | -0.0222 | yes |
| patch_c2h_role_resid_L21 | 0.0839 | 0.0833 | 0.0000 | 0.0000 | yes |
| patch_h2c_random_resid_L21 | 0.5674 | 0.5000 | 0.0000 | -0.0002 | yes |
| patch_c2h_random_resid_L21 | 0.0839 | 0.0833 | 0.0000 | -0.0000 | yes |
| project_out_pca_k8_resid_L23_colluder | 0.5838 | 0.5833 | 0.0833 | 0.0162 | yes |
| ablate_pca_lr_ambient_resid_L23_colluder | 0.5805 | 0.5000 | 0.0000 | 0.0129 | yes |

- PASS: l21-last RESULTS.md per-pair lines vs summary.json: every recomputed by-arm statistic matches the table written on the pod
- PASS: l21-last RESULTS.md per-pair lines vs its own by-arm table: every recomputed by-arm statistic matches the table written on the pod
- PASS: full residual h2c at L21 last token: mean change -0.0519 vs colluder, flip rate 0 (handoff: about -0.052, no flips)
- PASS: L21 full copy fails the final-residual bar: fraction of gap 0.1074 < 0.5, flips 0/5
- PASS: hook was live: patched p differs from the colluder baseline on 12/12 pairs
- direction arm patch_h2c_role_perp_resid_L21: mean change -0.00493 (not interpreted; mid-layer site)
- direction arm patch_h2c_role_resid_L21: mean change -0.02216 (not interpreted; mid-layer site)
- direction arm patch_h2c_random_resid_L21: mean change -0.00018 (not interpreted; mid-layer site)

## Withdrawn role-perp confirm (smoke_20261004T150244Z, SHA a0c34db): the bug, re-derived


| arm | mean P(target) | vote rate | flip rate | ΔP vs ref | matches written table |
|---|---:|---:|---:|---:|---|
| baseline_colluder | 0.5676 | 0.5000 | — | — | n/a |
| baseline_honest | 0.0839 | 0.0833 | — | — | n/a |
| patch_h2c_role_perp_resid_L21 | 0.5676 | 0.5000 | 0.0000 | 0.0000 | n/a |
| patch_c2h_role_perp_resid_L21 | 0.0839 | 0.0833 | 0.0000 | 0.0000 | n/a |
| ablate_role_perp_resid_L21_colluder | 0.5676 | 0.5000 | 0.0000 | -0.0000 | n/a |
| ablate_role_resid_L21_colluder | 0.5675 | 0.5000 | 0.0000 | -0.0001 | n/a |
| patch_h2c_role_resid_L21 | 0.5676 | 0.5000 | 0.0000 | 0.0000 | n/a |
| patch_c2h_role_resid_L21 | 0.0839 | 0.0833 | 0.0000 | 0.0000 | n/a |
| patch_h2c_random_resid_L21 | 0.5676 | 0.5000 | 0.0000 | 0.0000 | n/a |
| patch_c2h_random_resid_L21 | 0.0839 | 0.0833 | 0.0000 | 0.0000 | n/a |
| project_out_pca_k8_resid_L23_colluder | 0.5676 | 0.5000 | 0.0000 | -0.0000 | n/a |
| ablate_pca_lr_ambient_resid_L23_colluder | 0.5672 | 0.5000 | 0.0000 | -0.0004 | n/a |

- PASS: patch_h2c_role_perp_resid_L21 is digit-identical to the colluder baseline on 12/12 pairs: the patch did nothing (last shared token)
- PASS: patch_h2c_role_perp_resid_L21: with the fixed scorer the change vs colluder is 0.000000
- PASS: patch_h2c_role_perp_resid_L21: the table written by the buggy scorer reports 0.48370584490476176 = the untouched gap (compared to the honest baseline)
- PASS: patch_h2c_role_resid_L21 is digit-identical to the colluder baseline on 12/12 pairs: the patch did nothing (last shared token)
- PASS: patch_h2c_role_resid_L21: with the fixed scorer the change vs colluder is 0.000000
- PASS: patch_h2c_role_resid_L21: the table written by the buggy scorer reports 0.48370584490476176 = the untouched gap (compared to the honest baseline)
- PASS: patch_h2c_random_resid_L21 is digit-identical to the colluder baseline on 12/12 pairs: the patch did nothing (last shared token)
- PASS: patch_h2c_random_resid_L21: with the fixed scorer the change vs colluder is 0.000000
- PASS: patch_h2c_random_resid_L21: the table written by the buggy scorer reports 0.48370584490476176 = the untouched gap (compared to the honest baseline)
- PASS: the buggy 'flip rate' 0.4167 = 5/12 pairs whose untouched votes already disagree

## Oct 1 'first positive' writeups (pre-fix harness): reported ΔP equals the untouched gap

- PASS: MATCHED_PREFIX_INTERCHANGE_RESULTS.md patch_h2c_attn_L22: mean P 0.582 equals the colluder baseline 0.582 (patch did nothing); reported ΔP 0.481 equals the untouched gap 0.481
- PASS: MATCHED_PREFIX_INTERCHANGE_RESULTS.md patch_h2c_resid_L21: mean P 0.582 equals the colluder baseline 0.582 (patch did nothing); reported ΔP 0.481 equals the untouched gap 0.481
- PASS: MATCHED_PREFIX_INTERCHANGE_RESULTS.md patch_c2h_attn_L22: mean P 0.101 equals the honest baseline 0.101 (patch did nothing)
- PASS: MATCHED_PREFIX_INTERCHANGE_RESULTS.md patch_c2h_resid_L21: mean P 0.101 equals the honest baseline 0.101 (patch did nothing)
- PASS: MATCHED_PREFIX_WIDEN_RESULTS.md patch_h2c_attn_L22: mean P 0.651 equals the colluder baseline 0.651 (patch did nothing); reported ΔP 0.619 equals the untouched gap 0.619
- PASS: MATCHED_PREFIX_WIDEN_RESULTS.md patch_h2c_resid_L21: mean P 0.651 equals the colluder baseline 0.651 (patch did nothing); reported ΔP 0.619 equals the untouched gap 0.619
- PASS: MATCHED_PREFIX_WIDEN_RESULTS.md patch_c2h_attn_L22: mean P 0.032 equals the honest baseline 0.032 (patch did nothing)
- PASS: MATCHED_PREFIX_WIDEN_RESULTS.md patch_c2h_resid_L21: mean P 0.032 equals the honest baseline 0.032 (patch did nothing)
- PASS: MATCHED_PREFIX_TRANSFER_RESULTS.md patch_h2c_attn_L22: mean P 0.799 equals the colluder baseline 0.799 (patch did nothing); reported ΔP 0.727 equals the untouched gap 0.727
- PASS: MATCHED_PREFIX_TRANSFER_RESULTS.md patch_h2c_resid_L21: mean P 0.799 equals the colluder baseline 0.799 (patch did nothing); reported ΔP 0.727 equals the untouched gap 0.727
- PASS: MATCHED_PREFIX_TRANSFER_RESULTS.md patch_c2h_attn_L22: mean P 0.072 equals the honest baseline 0.072 (patch did nothing)
- PASS: MATCHED_PREFIX_TRANSFER_RESULTS.md patch_c2h_resid_L21: mean P 0.072 equals the honest baseline 0.072 (patch did nothing)

**0 failing check(s).**
