# Claims checklist — Paige LaTeX / PDF (author: Andy Liang)

**Written:** 2026-10-01 ~5:20 PM PT  
**Number sheet:** `CLAIMS_NUMBER_SHEET.md` + `.json`  
**Soft-stop / no-GPU:** honored. Do not invent numbers; cite sheet + source paths.  
**Arc:** Andy’s 5-beat study spine.

**Ready legend:** `PASS` = numbers + caveats locked for typeset · `PARTIAL` = main claim OK, optional fig/table pending polish · `FAIL` = do not typeset as result

---

## Locked safety claim (abstract + conclusion)

> **Monitorable, Transfer-stable coordination signature (attn_L22 / non-J residual) — editability unproven under tested interventions; matched-prefix + checkpoint assays are the next causal designs.**

Must appear (or paraphrase without strengthening “edit”) in abstract and conclusion.  
**Ready:** PASS

---

## Beat 1 — Probeable (prior) → where + transfer (ours)

**Claim (one sentence):** Core-trained collusion-role probes transfer to Transfer finals as a shared moderately low-rank residual direction (full band primary 0.902; LR ≡ full; contrast PCA k≈4–8 reaches 0.964), locating transferable geometry beyond a diffuse full-space probe.

| Item | Detail |
|---|---|
| Supporting tables/figs | HARD full row; subspace/direction table; optional low-rank bar |
| Source paths | `TRANSFER_FULL_HARD_METRICS.md`; `SUBSPACE_AND_DIRECTION.md`; `tables/hard_metrics.csv`; `layer_sweep_residual.csv` |
| Must-include caveats | Primary protocol is Core **discussion** → Transfer **final**; n=72 HARD |
| Ready | **PASS** |

---

## Beat 2 — J fails Transfer; non-J / attn_L22 is Transfer-stable

**Claim (one sentence):** Under Core→Transfer, J-only role readout is weak (~0.55–0.60) while complement ≈ full (~0.89–0.90), and among residual/MLP/attn sites the Transfer-stable locus peaks at **attn_L22 primary ≈ 0.999** (attn ≫ mlp; attn_L22 > residual L21/L22).

| Item | Detail |
|---|---|
| Supporting tables/figs | J vs full vs complement bar; residual layer sweep; mlp/attn site×layer table; `fig_j_vs_full_vs_complement.png`; `fig_residual_vs_attn_vs_mlp.png` |
| Source paths | `TRANSFER_FULL_HARD_METRICS.md`; `LAYER_SWEEP.md`; `MLP_ATTN_PROBES.md`; `tables/hard_metrics.*`; `tables/mlp_attn_site_aurocs.*`; `tables/layer_sweep_residual.*` |
| Must-include caveats | “attn_L22” = attention **block residual-write**, not a named head; residual mid-band also Transfer-stable (L21 0.941) |
| Ready | **PASS** |

---

## Beat 3 — Role↔mode entanglement → coordination/setup vs pure lying atom

**Claim (one sentence):** Role and mode cross-predict on Transfer (mode→role ~0.90; cos(role, mode_runavg) ~0.70), and after residualizing mode out of role, role_perp still transfers (L21 0.796; L22 0.837) with mode leak reduced — supporting a **coordination/setup** signature rather than a pure lying atom, without reducing role to task-ID only.

| Item | Detail |
|---|---|
| Supporting tables/figs | Controls cross-test table; residualize before/after table; `tables/controls_residualize.csv` |
| Source paths | `CONTROLS_OFFLINE.md`; `RESIDUALIZE_MODE.md`; `05_CONTROLS_AND_RESIDUALIZE.md` |
| Must-include caveats | Labels are mode (collusion/control) and role (colluder/honest), never collusion_success for this control pass; entanglement ≠ explanation of causal nulls |
| Ready | **PASS** |

---

## Beat 4 — Causal null under divergent histories → readability ≠ control

**Claim (one sentence):** Five free-regen last-token residual-write ablation smokes (Transfer residual α=1 and α-sweep; Core residual; attn_L22 final; attn_L22 discussion±final) all returned null on n=6 (votes/collusion_success tied to baseline; flips=0), so near-ceiling linear readability does **not** imply vote control under these protocols.

| Item | Detail |
|---|---|
| Supporting tables/figs | Master null table (5 rows); `fig_causal_null_flat.png`; `tables/causal_nulls.csv` |
| Source paths | `NEGATIVE_CAUSAL_APPENDIX.md`; `CAUSAL_SMOKE_RESULTS.md`; `CAUSAL_ALPHA_CORE_RESULTS.md`; `MLP_ATTN_CAUSAL_RESULTS.md`; `DISCUSSION_PHASE_CAUSAL_RESULTS.md`; `04_NEGATIVE_CAUSAL_APPENDIX.md` |
| Must-include caveats | Weak causal test under **divergent histories** / free regen — does **not** prove epiphenomenality; does block “we can steer by scrubbing this direction at last token under free regen”; causal GPU paused |
| Ready | **PASS** |

---

## Beat 5 — Safety claim + next methods

**Claim (one sentence):** We endorse a **monitorable Transfer-stable coordination signature** (attn_L22 / non-J residual; role survives mode peel) and explicitly **do not** claim editability yet; next causal designs are matched-prefix interchange and checkpoint / within-trajectory assays, not residual-α chasing.

| Item | Detail |
|---|---|
| Supporting tables/figs | Claim box in abstract/conclusion; pointer to `design/matched_prefix_spec.md`, `within_trajectory_spec.md`, `checkpoint_assays_spec.md` |
| Source paths | `01_EXECUTIVE_SUMMARY.md`; `02_RESEARCH_RETHINK.md`; `06_METHODS_LADDER.md`; `CLAIM_CORRELATIONAL.md` |
| Must-include caveats | Soft-stop: no Track I / Paper 2 / interp-demo; no further free-regen α chase without redesign approval |
| Ready | **PASS** |

---

## Supplement / robustness (optional in main PDF; OK in appendix)

| Item | One-liner | Numbers | Ready |
|---|---|---|---|
| Cross-phase discussion eval | Core discussion → Transfer **discussion** still transfers | residual band **0.959**; attn_L22 disc. **0.997** | **PASS** (`tables/cross_phase.*`, `fig_cross_phase_residual.png`) |
| Core domain-holdout → Transfer | LDO Core fit does not collapse Transfer L21 | mean **0.939** (min 0.927) vs baseline 0.941 | **PASS** (`tables/domain_holdout_L21.*`) — note Core domains ≠ Transfer families |
| Family AUROC @ L21 | All six Transfer families show strong Peak/Split | cicd/code_review/rag ceiling; news Peak 0.861 | **PASS** (`tables/family_auroc_L21.*`) |

---

## Typeset blockers (none for correlational + negative causal arc)

| Potential blocker | Status |
|---|---|
| Missing HARD / attn_L22 numbers | Cleared — sheet frozen |
| Causal positive result required for paper | **Not required** — paper arc is monitorable + null under tested edits |
| Head-level attn claim | **Do not claim** |
| Editability claim | **Do not claim** |

---

## Paige handoff checklist

- [ ] Pull numbers only from `CLAIMS_NUMBER_SHEET.md` / `.json`
- [ ] Embed safety claim wording above in abstract + conclusion
- [ ] Include caveats rows for beats 2–4
- [ ] Use `tables/` CSVs and `figures/*.png` as figure sources (or re-plot from CSV identically)
- [ ] Author line: **Andy Liang**
- [ ] Soft-stop lanes absent from related-work overclaim
- [ ] No GPU / no new causal results invented

**Overall ready to typeset (5-beat arc):** **PASS**
