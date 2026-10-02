> **GitHub snapshot, 2026-10-02.** This is the 2026-10-01 weekend attachment, placed in this repository for review. Paths that begin with `/workspace/narcbench-core-demo/` are this repo. Paper PDF: [papers/transfer-stable-collusion-2026-10/main.pdf](../../../papers/transfer-stable-collusion-2026-10/main.pdf). Handoff and skim: [docs/](../../../docs/). What arrived in the attachment, and what did not, is listed in [SNAPSHOT.md](SNAPSHOT.md).

# Claims / number sheet (frozen for Paige LaTeX)

**Author:** Andy Liang  
**Written:** 2026-10-01 ~5:20 PM PT (locked)  
**Lane:** NARCBench Transfer-stable · Gemma-2-9B-it  
**Protocol (default):** train Core **discussion** public → eval Transfer **final**  
**Soft-stop:** Track I / Paper 2 / interp-demo / family-house **not touched**. **No GPU** this pack.  
**Companion JSON:** `CLAIMS_NUMBER_SHEET.json`  
**Rule:** every number cites a path under `results/transfer_stable/` or `results/transfer/` (or activation meta). Nothing invented.

**Status legend:** `frozen` = cite as-is · `provisional` = supporting / secondary · `open` = not claimed yet

---

## Setup

| Field | Value | status | Source |
|---|---|---|---|
| Model | Gemma-2-9B-it | frozen | `results/transfer/TRANSFER_FULL_HARD_METRICS.md` |
| Train | Core discussion public (`20261001T012639Z`) | frozen | same; acts `data/activations/gemma2_9b/core/20261001T012639Z` |
| Eval (primary) | Transfer **final** | frozen | same; acts `data/activations/gemma2_9b/transfer/RUNPOD` |
| n Transfer runs (HARD) | **72** | frozen | `TRANSFER_FULL_HARD_METRICS.md`; `FULL_HARD_SUMMARY.json` |
| Cached residual layers | L19–L23 | frozen | `LAYER_SWEEP.md` / `layer_sweep.json` |
| MLP/attn extract layers | L19–L23 | frozen | `MLP_ATTN_PROBES.md`; `RUNPOD_mlp_attn/` |
| Balance (last known) | ~**$26.24** after discussion-phase smoke (~4:37 PM PT) | provisional | `WEEKEND_REVIEW/00_README.md`; `PACK_MANIFEST.json` — live API not queried |

---

## A. HARD J vs full vs complement

| Arm | Primary | Peak | Split | Asym | status | Source |
|---|---:|---:|---:|---:|---|---|
| full acts (band L19–23) | **0.902** | 0.874 | 0.904 | 0.929 | frozen | `results/transfer/TRANSFER_FULL_HARD_METRICS.md` |
| W_U k25 J-only | **0.602** | 0.625 | 0.565 | 0.616 | frozen | same |
| jac k25 J-only | **0.549** | 0.586 | 0.523 | 0.540 | frozen | same |
| W_U k25 complement | **0.898** | 0.869 | 0.897 | 0.928 | frozen | same |
| jac k25 complement | **0.891** | 0.867 | 0.888 | 0.917 | frozen | same |

**Claim:** J-localization does **not** transfer; complement ≈ full.  
Table/fig: `tables/hard_metrics.csv`, `figures/fig_j_vs_full_vs_complement.png`

---

## B. Residual layer sweep

| Arm | Primary | Peak | Split | Asym | status | Source |
|---|---:|---:|---:|---:|---|---|
| L19 | 0.884 | 0.879 | 0.863 | 0.910 | frozen | `LAYER_SWEEP.md`, `layer_sweep.json` |
| L20 | 0.885 | 0.864 | 0.871 | 0.918 | frozen | same |
| **L21** | **0.941** | 0.899 | 0.975 | 0.950 | frozen | same |
| **L22** | **0.930** | 0.897 | 0.937 | 0.955 | frozen | same |
| L23 | 0.873 | 0.832 | 0.877 | 0.910 | frozen | same |
| Band mean L19–23 | 0.902 | 0.874 | 0.904 | 0.929 | frozen | same (≡ HARD full) |
| Mean-pool L19–23 | **0.941** | 0.906 | 0.951 | 0.967 | frozen | same |

Table: `tables/layer_sweep_residual.csv`

---

## C. Subspace / shared direction

| Arm | Best layer | Primary | Band mean | status | Source |
|---|---|---:|---:|---|---|
| Diff-in-means | L22 | **0.858** | 0.817 | frozen | `SUBSPACE_AND_DIRECTION.md`, `subspace_and_direction.json` |
| LR weight | L21 | **0.941** | **0.902** | frozen | same (≡ full probe) |
| PCA residual k=8 | L22 | **0.927** | — | frozen | same (47% var) |
| PCA residual k=32 | L22 | **0.956** | — | frozen | same (71% var) |
| PCA contrast k=4 | L23 | **0.957** | — | frozen | same |
| **PCA contrast k=8** | L23 | **0.964** | — | frozen | same (best low-dim) |
| mean \|cos(diff-in-means, LR)\| | — | **0.875** | — | frozen | same |

---

## D. MLP / attn per-site (attn_L22 best)

Protocol: Core discussion → Transfer final. Hooks: attn = self_attn out (residual-write); mlp = mlp out; residual = DecoderLayer out.

| Site | L | Primary | Peak | Split | Asym | Role sample | status | Source |
|---|---:|---:|---:|---:|---:|---:|---|---|
| **attn** | **22** | **0.999** | **0.998** | **1.000** | **1.000** | **0.987** | frozen | `MLP_ATTN_PROBES.md` / `.json` (JSON raw primary ≈0.9995) |
| attn | 21 | 0.986 | 0.978 | 0.986 | 0.994 | 0.948 | frozen | same |
| attn | 19 | 0.954 | 0.965 | 0.928 | 0.970 | 0.951 | frozen | same |
| attn | 20 | 0.949 | 0.934 | 0.948 | 0.965 | 0.870 | frozen | same |
| attn | 23 | 0.888 | 0.878 | 0.873 | 0.914 | 0.881 | frozen | same |
| mlp | 22 | 0.915 | 0.881 | 0.923 | 0.941 | 0.878 | frozen | same |
| mlp | 19 | 0.861 | 0.863 | 0.836 | 0.883 | 0.852 | frozen | same |
| mlp | 20 | 0.821 | 0.837 | 0.763 | 0.863 | 0.872 | frozen | same |
| mlp | 21 | 0.849 | 0.834 | 0.841 | 0.873 | 0.825 | frozen | same |
| mlp | 23 | 0.823 | 0.805 | 0.794 | 0.870 | 0.814 | frozen | same |
| residual | 21 | 0.941 | 0.899 | 0.975 | 0.950 | 0.859 | frozen | same |
| residual | 22 | 0.930 | 0.897 | 0.937 | 0.955 | 0.873 | frozen | same |

Direction file: `results/transfer_stable/directions/lr_role_attn_L22.npy`  
Caveat: “attn_L22” = **attention block output / residual-write**, not a named head (head split **not** done).  
Table/fig: `tables/mlp_attn_site_aurocs.csv`, `figures/fig_residual_vs_attn_vs_mlp.png`

---

## E. Controls (role / mode) + residualize

### Cross-tests @ Transfer final

| Layer | Direction → target | AUROC | status | Source |
|---:|---|---:|---|---|
| 21 | Role → role | **0.859** | frozen | `CONTROLS_OFFLINE.md`, `controls_offline.json` |
| 21 | Role → mode (run-avg) | 0.766 | frozen | same |
| 21 | Role → mode (scenario primary) | **0.941** | frozen | same |
| 21 | Mode → mode (run-avg) | **0.886** | frozen | same |
| 21 | Mode → role | **0.900** | frozen | same |
| 22 | Role → role | **0.873** | frozen | `05_CONTROLS_AND_RESIDUALIZE.md` |
| 22 | Mode → role | **0.887** | frozen | same |
| 22 | Mode → mode (run-avg) | **0.831** | frozen | same |
| 21 | cos(role, mode_runavg) | **+0.694** | frozen | `CONTROLS_OFFLINE.md` |
| 22 | cos(role, mode_runavg) | **+0.707** | frozen | same |

### After residualize mode out of role

| Layer | role→role | role_perp→role | role→mode | role_perp→mode | status | Source |
|---:|---:|---:|---:|---:|---|---|
| 21 | 0.859 | **0.796** | 0.766 | **0.575** | frozen | `RESIDUALIZE_MODE.md`, `residualize_mode.json` |
| 22 | 0.873 | **0.837** | 0.767 | **0.604** | frozen | same |

cos(role_perp, mode) = **0** by construction.  
Dirs: `directions/lr_role_perp_mode_L21.npy`, `lr_role_perp_mode_L22.npy`  
Table: `tables/controls_residualize.csv`

---

## F. Causal null table (all 5 smokes)

| # | Battery | World | Site | α | collusion_success | flips vs baseline | Votes identical? | status | Source |
|---|---|---|---|---|---:|---:|---|---|---|
| 1 | Transfer residual α=1 | Transfer | residual L21/L22 (+perp/random) | 1 | **0.167 all** | **0** | tied rates | frozen | `CAUSAL_SMOKE_RESULTS.md` |
| 2 | Transfer residual α-sweep | Transfer | residual L21/L22 (+perp@4,random@8) | 2,4,8 | **0.167 all** | **0** | tied rates | frozen | `CAUSAL_ALPHA_CORE_RESULTS.md` |
| 3 | Core residual role | Core | residual L21/L22/perp/random; L21@4 | 1,4 | **0.833 all** | **0** | tied rates | frozen | `CAUSAL_ALPHA_CORE_RESULTS.md` |
| 4 | Core attn_L22 final-only | Core | attn residual-write L22 | 1,4 | **0.833 all** | **0** | **Yes** | frozen | `MLP_ATTN_CAUSAL_RESULTS.md` |
| 5 | Core attn_L22 discussion (±final) | Core | attn L22 discussion ± final | 1 | **0.667 all** | **0** | **Yes** | frozen | `DISCUSSION_PHASE_CAUSAL_RESULTS.md` |

Smoke n=6 each. Consolidated: `NEGATIVE_CAUSAL_APPENDIX.md`, `WEEKEND_REVIEW/04_NEGATIVE_CAUSAL_APPENDIX.md`  
Fig: `figures/fig_causal_null_flat.png` · Table: `tables/causal_nulls.csv`

---

## G. New offline (this pack) — frozen for supplement / robustness

### Cross-phase: Core discussion → Transfer **discussion**

| Site | Eval phase | Layer / band | Primary | status | Source |
|---|---|---|---:|---|---|
| residual | discussion | band mean L19–23 | **0.959** | frozen | `tables/cross_phase_discussion_residual/metrics.json` |
| residual | discussion | L21 | **0.965** | frozen | same |
| residual | discussion | L22 | **0.964** | frozen | same |
| attn | discussion | L22 | **0.997** | frozen | `tables/cross_phase_attn.json` |
| attn | discussion | L21 | **0.965** | frozen | same |
| attn | final (recheck) | L22 | **0.9995** | frozen | same (matches MLP_ATTN_PROBES) |

Fig: `figures/fig_cross_phase_residual.png` · Table: `tables/cross_phase.csv`

### Domain-holdout (Core LDO fit → Transfer final @ L21)

Core domains ≠ Transfer families (no overlapping domain names). Holdout = leave-one-**Core**-domain-out fit, eval **all** Transfer.

| Statistic | Value | status | Source |
|---|---:|---|---|
| Full-fit baseline primary L21 | **0.941** | frozen | `tables/domain_holdout_L21.json` |
| Holdout mean primary (17 domains) | **0.939** | frozen | same |
| Holdout min / max | **0.927 / 0.948** | frozen | same |

### Transfer family AUROC @ L21 residual (full-fit)

| Family | Peak | Split | n | status | Source |
|---|---:|---:|---:|---|---|
| cicd | 1.000 | 1.000 | 12 | frozen | `tables/family_auroc_L21.json`; also `LAYER_SWEEP.md` |
| code_review | 1.000 | 1.000 | 12 | frozen | same |
| news | 0.861 | 0.917 | 12 | frozen | same |
| pump_dump | 0.917 | 1.000 | 12 | frozen | same |
| rag | 1.000 | 1.000 | 12 | frozen | same |
| research | 0.917 | 1.000 | 12 | frozen | same |

---

## Explicit non-claims

| Non-claim | status | Why |
|---|---|---|
| Scrubbing attn_L22 / residual role **controls** collusive votes | frozen non-claim | All 5 free-regen smoke batteries null |
| Axis is a pure “lying atom” (deception-only) | frozen non-claim | Role↔mode entanglement; mode→role ~0.90 |
| J-space carries Transfer-stable role signal | frozen non-claim | J-only ~0.55–0.60 |
| Head-level localization inside attn L22 | open | Block-output only; no head split |
| Editability under matched-prefix / interchange | open | Design ready; GPU paused |
| Track I / Paper 2 / interp-demo results | non-scope | Soft-stop |

---

## Open items (do not typeset as results)

1. Matched-prefix path-patch / interchange causal smoke (design in `design/`).  
2. Within-trajectory teacher-forced mid-discussion edits.  
3. Checkpoint assays for commitment timing.  
4. Head-level attn split; multi-position / subspace causal.  
5. Residual-α chase / larger free-regen n — **paused** (do not resume under current design).

---

## Safety claim wording (abstract / conclusion) — LOCKED

> **Monitorable, Transfer-stable coordination signature (attn_L22 / non-J residual) — editability unproven under tested interventions; matched-prefix + checkpoint assays are the next causal designs.**

Fridge line (optional):  
> Transfer-stable attn_L22 reads coordination setup almost perfectly after shift; ablating it on free regen never flipped votes — monitor yes, edit unproven; redesign causal tests around matched prefixes.

---

## Spend / balance note

Last known balance ~**$26.24** after discussion-phase attn_L22 smoke (2026-10-01 ~4:37 PM PT). Live billing API **not** queried this pack. No new GPU spend for claims freeze / offline analyses.
