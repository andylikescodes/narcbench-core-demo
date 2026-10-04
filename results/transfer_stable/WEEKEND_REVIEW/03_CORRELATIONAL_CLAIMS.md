# Correlational evidence — where the signal lives, what transfers

**Written:** 2026-10-01 ~5:15 PM PT  
**Role in the study:** evidence backbone for rethink beats 1–3 (probeable→localized; J-fail; role↔mode).  
**Rule:** numbers from existing artifacts only. Causal nulls live in `04_NEGATIVE_CAUSAL_APPENDIX.md`.

**Protocol (default):** train Core **discussion public** → eval Transfer **final**; Gemma-2-9B-it.

---

## Narrative map (not a flat table dump)

| Study question | Answer from this file | Headline number |
|---|---|---|
| Does a Core probe transfer at all? | Yes — full residual band | primary **0.902** |
| Does **J** carry that Transfer signal? | **No** — J-only weak; complement ≈ full | J **~0.55–0.60** vs complement **~0.89–0.90** |
| Where in depth? | Mid-band residual | L21 **0.941**, L22 **0.930** |
| Which block write-in? | **Attention ≫ MLP**; attn beats residual at best layer | **attn_L22 ≈ 0.999** |
| Shared low-rank direction? | Yes — LR = full; contrast PCA k≈4–8 strong | PCA contrast k8 **0.964** |
| Deception-pure vs setup? | **Mixed** coordination geometry; role survives mode peel | role_perp→role **0.796–0.837** |

Safety bridge: these support a **monitorable Transfer-stable coordination signature**; they do **not** by themselves support editability (`02_RESEARCH_RETHINK.md`).

---

## A. HARD Transfer J-fail (beat 2 background)

Source: `results/transfer/TRANSFER_FULL_HARD_METRICS.md`

| Arm | Primary (FULL n=72) | Peak | Split | Asym |
|---|---:|---:|---:|---:|
| full acts (band L19–23) | **0.902** | 0.874 | 0.904 | 0.929 |
| W_U k25 J-only | **0.602** | 0.625 | 0.565 | 0.616 |
| jac k25 J-only | **0.549** | 0.586 | 0.523 | 0.540 |
| W_U k25 complement | **0.898** | 0.869 | 0.897 | 0.928 |
| jac k25 complement | **0.891** | 0.867 | 0.888 | 0.917 |

**Claim:** J-localization does **not** transfer; transferable collusion-role signal is **non-J**.

Also: `FULL_HARD_SUMMARY.json`, `INTERIM_VS_BASELINES.md`.

---

## B. Residual depth — L21–L22 (beat 2)

Source: `LAYER_SWEEP.md`, `layer_sweep.json`

| Arm | Primary | Peak | Split | Asym |
|---|---:|---:|---:|---:|
| L19 | 0.884 | 0.879 | 0.863 | 0.910 |
| L20 | 0.885 | 0.864 | 0.871 | 0.918 |
| **L21** | **0.941** | 0.899 | 0.975 | 0.950 |
| **L22** | **0.930** | 0.897 | 0.937 | 0.955 |
| L23 | 0.873 | 0.832 | 0.877 | 0.910 |
| Band mean L19–23 | 0.902 | 0.874 | 0.904 | 0.929 |
| Mean-pooled L19–23 | **0.941** | 0.906 | 0.951 | 0.967 |

**Claim:** Transfer-stable residual signal at every cached layer; peaks **L21–L22**.

---

## C. Shared direction + low-rank subspace (beat 1–2)

Source: `SUBSPACE_AND_DIRECTION.md`, `subspace_and_direction.json`

| Arm | Best layer | Primary | Notes |
|---|---|---:|---|
| Diff-in-means | L22 | **0.858** | band mean primary **0.817** |
| LR weight | L21 | **0.941** | band mean **0.902** ≡ HARD full |
| PCA residual k=8 | L22 | **0.927** | 47% var |
| PCA residual k=32 | L22 | **0.956** | 71% var |
| PCA contrast k=4 | L23 | **0.957** | |
| **PCA contrast k=8** | L23 | **0.964** | best low-dim |

Mean |cos(diff-in-means, LR)| across layers ≈ **0.875**.

**Claim:** transferable geometry is a **moderately low-rank shared direction** — monitorable with a single Core-fit vector, not only a high-D probe.

---

## D. Attention vs MLP vs residual — attn_L22 primary (beat 2 climax)

Source: `MLP_ATTN_PROBES.md`, `MLP_ATTN_PROBES.json`, `directions/meta_mlp_attn.json`  
Hooks: attn = self_attn out (residual-write); mlp = mlp out; residual = DecoderLayer out.

| Site | L | Primary | Peak | Split | Asym | Role sample |
|---|---:|---:|---:|---:|---:|---:|
| **attn** | **22** | **0.999** | **0.998** | **1.000** | **1.000** | **0.987** |
| attn | 21 | 0.986 | 0.978 | 0.986 | 0.994 | 0.948 |
| attn | 19 | 0.954 | 0.965 | 0.928 | 0.970 | 0.951 |
| attn | 20 | 0.949 | 0.934 | 0.948 | 0.965 | 0.870 |
| attn | 23 | 0.888 | 0.878 | 0.873 | 0.914 | 0.881 |
| mlp | 22 | 0.915 | 0.881 | 0.923 | 0.941 | 0.878 |
| mlp | 19–21,23 | 0.82–0.86 | — | — | — | — |
| residual | 21 | 0.941 | 0.899 | 0.975 | 0.950 | 0.859 |
| residual | 22 | 0.930 | 0.897 | 0.937 | 0.955 | 0.873 |

Direction: `directions/lr_role_attn_L22.npy`

**Where claim (correlational):** among tested sites, **attention L22 residual-write** is the primary Transfer-stable role locus (attn ≫ mlp; attn_L22 > residual L21/L22). Head-level split inside the attn module was **not** done this lane — “attn_L22” means the **attention block output / residual-write**, not a named head.

---

## E. Role↔mode controls + residualize (beat 3)

Sources: `CONTROLS_OFFLINE.md`, `RESIDUALIZE_MODE.md` (+ JSON). Detail in `05_CONTROLS_AND_RESIDUALIZE.md`.

### Cross-tests @ L21 Transfer final

| Direction → target | AUROC |
|---|---:|
| Role → role | **0.859** |
| Role → mode (run-avg) | 0.766 |
| Role → mode (scenario primary) | **0.941** |
| Mode → mode (run-avg) | **0.886** |
| Mode → role | **0.900** |

Cos(role, mode_runavg): L21 **+0.694**, L22 **+0.707**.

### After residualize mode out of role

| Layer | role→role | role_perp→role | role→mode | role_perp→mode |
|---:|---:|---:|---:|---:|
| 21 | 0.859 | **0.796** | 0.766 | **0.575** |
| 22 | 0.873 | **0.837** | 0.767 | **0.604** |

cos(role_perp, mode) = **0** by construction.

**Mechanism claim (correlational):** **coordination/setup entanglement** with a residual **role** axis that still transfers — monitoring target is a **coalition-setup signature**, not a guaranteed pure lying atom.

---

## F. Synthesis paragraph

Starting from “collusion is probeable,” this lane shows **what transfers** and **where it lives**: not J under Core→Transfer, but non-J residual mid-band and especially **attention L22**, as a **low-rank shared direction** entangled with collusion-run mode yet not reducible to it. That is the correlational half of the safety claim in `02_RESEARCH_RETHINK.md`. The causal half is negative under current protocols — see `04_NEGATIVE_CAUSAL_APPENDIX.md`.

---

## Pointer index

| Topic | MD | JSON / dirs |
|---|---|---|
| HARD J-fail | `../../transfer/TRANSFER_FULL_HARD_METRICS.md` | `FULL_HARD_SUMMARY.json` |
| Interim | `../INTERIM_VS_BASELINES.md` | `offline_summary.json` |
| Layers | `../LAYER_SWEEP.md` | `layer_sweep.json` |
| Subspace | `../SUBSPACE_AND_DIRECTION.md` | `subspace_and_direction.json` |
| Controls | `../CONTROLS_OFFLINE.md` | `controls_offline.json` |
| Residualize | `../RESIDUALIZE_MODE.md` | `residualize_mode.json` |
| MLP/Attn | `../MLP_ATTN_PROBES.md` | `MLP_ATTN_PROBES.json` |
| Claim card | `../CLAIM_CORRELATIONAL.md` | — |
