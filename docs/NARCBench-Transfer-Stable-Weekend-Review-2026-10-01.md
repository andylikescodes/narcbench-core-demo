> **GitHub snapshot, 2026-10-02.** Paths that begin with `/workspace/narcbench-core-demo/` are this repository. Paper PDF: [papers/transfer-stable-collusion-2026-10/main.pdf](../papers/transfer-stable-collusion-2026-10/main.pdf). Study pack: [results/transfer_stable/WEEKEND_REVIEW/](../results/transfer_stable/WEEKEND_REVIEW/). Attachment inventory: [SNAPSHOT.md](../results/transfer_stable/WEEKEND_REVIEW/SNAPSHOT.md).

# NARCBench Transfer-stable — Mechanism Study (Weekend Review)
**Date:** 2026-10-01 ~5:15 PM PT (rewritten as mechanism study; numbers unchanged)  
**For:** Andy  
**Lane:** Transfer-stable / Core→Transfer collusion · Gemma-2-9B-it  
**Soft-stop:** Track I / Paper 2 / interp-demo / family-house **not touched**. Further causal GPU **paused**.

**Entry:** `/workspace/andy-biz/NARCBench-Transfer-Stable-Weekend-Review-2026-10-01.md`  
**Companion inventory:** `/workspace/andy-biz/NARCBench-Transfer-Stable-Cache-Inventory-2026-10-01.md`

---

## 1. Motivation vs the original probe paper

Rose et al. (*Detecting Multi-Agent Collusion Through Multi-Agent Interpretability*, arXiv:2604.01151) show that **full residual / activation probes** can separate colluding vs non-colluding agents on NARCBench-style committee games — often near-ceiling in-distribution. That establishes **monitorability of collusion from hidden state**, but leaves open:

- **Where** in representation space the signal lives (e.g. verbalizable / J-space vs complement).
- Whether a **shared direction** transfers **across scenario families** (Core → Transfer), not only within Core leave-domain-out.
- Whether the readable geometry is **role/deception-specific**, **task/mode-generic**, or entangled.
- Whether the direction is **editable** — ablation/steer changes votes — or only correlational.

This lane is a **mechanism follow-on**, not a re-run of the paper’s full-activation baseline. We keep the NARCBench protocol family (train Core discussion public → scenario Peak / Split / Asym; labels `mode` / `role`; never Transfer `collusion_success`, which is all-false) and ask *what kind of object* transfers.

**One-line motivation:** *Can we locate a Transfer-stable coordination signature beyond “full acts AUROC is high,” and is it the sort of thing a safety monitor could claim — or only a probe?*

---

## 2. Hypotheses (pre-registered flavor; tested with existing numbers)

| # | Hypothesis | Prediction if true | Status from this lane |
| ---: | --- | --- | --- |
| **H1** | Collusion signal is **J-localized** (verbalizable / logit-lens subspace) | J-only Transfer AUROC ≈ full; complement weak | **Rejected on Transfer** — J-only weak; complement ≈ full |
| **H2** | A **shared residual direction** transfers Core→Transfer | Core-fit LR / dim / low-rank subspace beats J-only and approaches full | **Supported** — LR band = full; contrast-PCA k≈4–8 best |
| **H3** | Signal peaks mid-band residual; **attn write** may dominate MLP | L21–L22 residual strong; attn > mlp at matched layers | **Supported correlationally** — L21/L22 best residual; attn_L22 ≈0.999 |
| **H4** | Direction is **role/deception-specific**, orthogonal to collusion-vs-control mode | Role→role high; mode→role ≈ chance; cos(role,mode)≈0 | **Rejected / mixed** — both directions hit both targets; cos≈0.70; residualize partially helps |
| **H5** | Ablating the role direction at last-token residual/attn **steers votes** | Ablate drops collusion_success / flips vs baseline & random | **Not supported under tested free-regen smokes** (n=6 nulls); assay design weakens the negative (see §5) |

Core in-distribution was already easy (Stage 1 HARD_PASS primary 1.0; J and complement both near-full on Core). The scientifically interesting cut is **Transfer OOD** + **specificity** + **causal editability**.

---

## 3. Where + transfer results (facts only)

**Protocol:** train Core **discussion public** → eval Transfer **final**, HARD n=72 (unless noted). Primary = mean(Peak, Split, Asym).

### 3a. J vs complement vs full (HARD Transfer)

| Arm | Primary AUROC | Peak | Split | Asym |
| --- | ---: | ---: | ---: | ---: |
| Full acts (band L19–23) | **0.902** | 0.874 | 0.904 | 0.929 |
| W_U k25 J-only | 0.602 | 0.625 | 0.565 | 0.616 |
| jac k25 J-only | 0.549 | 0.586 | 0.523 | 0.540 |
| W_U k25 complement | 0.898 | 0.869 | 0.897 | 0.928 |
| jac k25 complement | 0.891 | 0.867 | 0.888 | 0.917 |

**Mechanism read:** Transferable signal is **not** confined to J-space. Complement carries essentially the full Transfer probe. H1 fails on the OOD cut even though Core J-only can look strong at larger k.

### 3b. Residual geography and low-rank shared direction

| Arm | Primary | Peak | Split | Asym | Notes |
| --- | ---: | ---: | ---: | ---: | --- |
| L21 full residual | **0.941** | 0.899 | 0.975 | 0.950 | best single layer |
| L22 full residual | **0.930** | 0.897 | 0.937 | 0.955 | |
| L19 / L20 / L23 residual | 0.884 / 0.885 / 0.873 | — | — | — | all ≫ J-only |
| Mean-pool L19–23 | **0.941** | 0.906 | 0.951 | 0.967 | ties best layer |
| Diff-in-means best (L22) | **0.858** | 0.824 | 0.877 | 0.871 | one vector |
| LR weight best (L21) | **0.941** | 0.899 | 0.975 | 0.950 | = full probe @ layer |
| LR weight band mean | **0.902** | — | — | — | ≡ HARD full |
| PCA residual k=8 @ L22 | **0.927** | 0.893 | 0.954 | 0.934 | ~47% var |
| PCA residual k=32 @ L22 | **0.956** | 0.951 | 0.948 | 0.970 | ~71% var |
| PCA contrast k=4 @ L23 | **0.957** | 0.958 | 0.951 | 0.962 | |
| PCA contrast k=8 @ L23 | **0.964** | 0.946 | 0.965 | 0.981 | **best low-dim** |

Mean |cos(diff-in-means, LR)| across layers ≈ **0.875**. H2/H3 (residual geography + shared direction) hold correlationally.

### 3c. Block-internal site: attn vs MLP vs residual

Hooks: attn = self_attn residual-write; mlp = mlp residual-write; residual = DecoderLayer out. Same Core→Transfer role protocol.

| Site | L | Primary | Peak | Split | Asym | Role-sample |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **attn** | **22** | **0.999** | 0.998 | 1.000 | 1.000 | 0.987 |
| attn | 21 | 0.986 | 0.978 | 0.986 | 0.994 | 0.948 |
| attn | 19 | 0.954 | 0.965 | 0.928 | 0.970 | 0.951 |
| mlp | 22 | 0.915 | 0.881 | 0.923 | 0.941 | 0.878 |
| mlp | 21 | 0.849 | 0.834 | 0.841 | 0.873 | 0.825 |
| residual | 21 | 0.941 | 0.899 | 0.975 | 0.950 | 0.859 |
| residual | 22 | 0.930 | 0.897 | 0.937 | 0.955 | 0.873 |

**Mechanism read:** Among tested writes, **attention mid-late layers** carry the strongest linear Transfer role signal; MLP is weaker; residual sits between / near attn depending on layer. Correlational localization ≠ causal sufficiency (§5).

---

## 4. Entanglement / coordination-state interpretation

Controls ask: is the Transfer-stable geometry a clean **colluder-vs-honest deception axis**, a **collusion-run vs control-run mode axis**, or a blended **coordination state**?

### 4a. Cross-tests @ L21 (Transfer final)

| Direction → target | AUROC | Reading |
| --- | ---: | --- |
| Role → role | **0.859** | within-collusion deception transfers |
| Role → mode (run-avg) | 0.766 | moderate mode leak |
| Role → mode (scenario Peak/Split) | **0.941** | asymmetry scores pull mode |
| Mode → mode (run-avg) | **0.886** | condition transfers |
| Mode → role | **0.900** | mode geometry recovers role almost as well |

Cosine (ambient): role LR vs mode run-avg ≈ **+0.70** (L21/L22). Sample-level mode LR is nearly orthogonal to role (cos≈0) but is a weaker / different aggregation — run-avg is the fairer one-vector-per-run arm.

PCA-contrast k=4/8 @ L23 shows the same pattern (e.g. k=8: role→role 0.882, mode→role 0.912, cos(role,mode) 0.762).

### 4b. Residualize mode out of role

| Layer | role→role | role_perp→role | role→mode | role_perp→mode | cos→0 by construction |
| ---: | ---: | ---: | ---: | ---: | --- |
| 21 | 0.859 | **0.796** | 0.766 | **0.575** | yes |
| 22 | 0.873 | **0.837** | 0.767 | **0.604** | yes |

Mode_perp still separates mode (L21 0.819) while mode→role collapses toward chance (0.553).

### 4c. Interpretation (careful)

The geometry looks like an overlapping **coordination / collusion-condition state**, not a surgically pure deception fingerprint:

- Role still transfers after removing the mode projection → **not reducible to “just task.”**
- Mode training alone recovers role at ~0.90 → **not a clean deception-only axis.**
- Practical implication: a monitor that thresholds this direction will likely fire on **both** “who is the colluder” and “is this a collusion-conditioned run,” unless further disentangling succeeds causally (not shown).

This is consistent with multi-agent collusion being a **distributed committee state** (who coordinates + that coordination is happening) rather than a single lie-token feature in J-space.

---

## 5. Causal nulls + why divergent histories weaken classical tracing

### 5a. What was tried (n=6; last-token residual-write ablate; free regen)

| # | Intervention | World | collusion_success (all arms) | Flips vs baseline |
| ---: | --- | --- | ---: | ---: |
| 1 | Residual role L21/L22 α=1 (+ perp / random) | Transfer | **0.167** | 0 |
| 2 | Residual α ∈ {2,4,8} L21/L22 | Transfer | **0.167** | 0 |
| 3 | Residual role L21/L22 α=1; L21 α=4 | Core | **0.833** | 0 |
| 4 | attn_L22 final-only α=1, α=4 (+ random) | Core | **0.833** | 0 (votes identical) |
| 5 | attn_L22 discussion-only; discussion+final α=1 | Core | **0.667** | 0 (votes identical `{B:6,2:3,1:1,A:2}`) |

Discussion-phase null (#5) argues against “only locked at final” as the sole excuse for #4. Under these hooks/α/n, free-regen ablate did not move committee votes. Causal GPU on this harness is **paused** (`CLAIM_CORRELATIONAL.md`, `NEGATIVE_CAUSAL_APPENDIX.md`).

### 5b. Why this is a weak classical causal-tracing negative

Probes were fit on **fixed historical last-token activations**. Causal arms **regenerated** discussion and/or finals with hooks:

1. **Divergent histories.** Changing early tokens changes later context. The post-edit trajectory is not the trajectory on which the direction was measured — classical activation patching assumes matched prefixes.
2. **Edit class.** Last-token residual-write at one site ≠ patch/scrub of cached activations at aligned positions under teacher forcing.
3. **n=6 / coarse outcome.** Binary collusion_success (Transfer often lacks letter `target_option`) has low power; no Δlogit / Peak–Split collapse / local probe-score assays.
4. **Readable ≠ causally sufficient under this edit.** attn_L22 AUROC≈0.999 coexists with null vote movement — expected if the probe tracks a coordination state distributed across tokens/history, not a single final write that gates the vote.

**Claim level:** *Under free-regen last-token ablations, we did not find vote control.* **Not claimed:** *no causal substrate exists.* Residual-α chasing on the current harness is the wrong next experiment.

---

## 6. Candidate AI-safety claim (careful — do not overclaim)

**Supported (correlational, Gemma-2-9B-it, this protocol):**

1. **Transferable coordination signature.** A Core-fit residual / attn direction monitors collusion-related structure on held-out Transfer families at high AUROC (full ~0.90; attn_L22 near ceiling; low-rank contrast subspace k≈4–8 ≥ full).
2. **J-local fingerprint is an overclaim risk.** Treating “collusion lives in verbalizable / J-space” as the Transfer story fails here: J-only ~0.55–0.60 while complement ≈ full. Safety narratives that equate collusion with sayable features would **miss** the transferable mass.
3. **Readability ≠ control.** Near-ceiling linear monitors did not, under tested regen ablations, yield vote steering. Deployable “erase the collusion direction → safe committee” is **not** evidenced.

**Not supported / not claimed:**

- Causal necessity or sufficiency of attn_L22 / L21 residual for collusion success.
- Disentangled “deception-only” monitor (entanglement with mode remains).
- Cross-model / closed-model generality; only Gemma-2-9B-it Core→Transfer here.
- That free-regen nulls refute all mechanistic interventions.

**Candidate one-liner for weekend framing (max strength allowed by data):**  
*We can monitor a Transfer-stable, mostly non-J coordination signature in open-weight committee agents; we should not advertise it as a J-local fingerprint or as an off-switch until matched-prefix causal assays say otherwise.*

---

## 7. Systematic next — matched-prefix, within-trajectory, behavioral assays

Stop optimizing free-regen `collusion_success` on n=6. Redesign so the causal question shares **prefix and trajectory** with the correlational measurement.

### 7a. Matched-prefix / teacher-forced

- Freeze recorded turns (`agent_prompts.json`); teacher-force the historical transcript.
- Activation **patch / scrub** at chosen `(layer, token_span)`: copy collusion site into matched control, or scrub collusion with control residuals.
- Temperature 0; measure ΔP / Δlogit on target option letters where available (Core letter `target_option`).

### 7b. Within-trajectory

- Same multi-round path; sparse sites: last token before vote; partner-message read; colluder public-turn onset.
- Report **local** role-probe score before/after edit on the same tokens, separately from end vote.
- Prefer full prefix lock over regenerating discussion (discussion smoke reused honest msgs but still rewrote colluder discussion).

### 7c. Behavioral assays (co-primary with success)

| Assay | Role |
| --- | --- |
| Δlogit / ΔP(target) | Continuous, temperature-0 |
| Peak / Split / Asym collapse under scrub | Ties edit to the monitor metric |
| Mid-discussion coordination / partner-alignment shift | Coordination state, not only final vote |
| Site probe score Δ | Did we move the claimed representation? |

### 7d. Explicit non-goals until redesign is approved

- No residual-α sweeps or n-expansion on free-regen.
- No Track I / Paper 2 / interp-demo from this lane.
- No GPU until a named matched-prefix card + budget is approved.
- Offline still free: Transfer discussion-phase correlational eval; Core domain-holdout → Transfer family AUROC.

### 7e. Weekend decision prompts

1. Approve matched-prefix Core smoke (attn_L22 or residual L21) before any regen scale-up?
2. Lock ΔP(target) + Peak/Split collapse as co-primaries?
3. Compare `role_perp` vs ambient role under matched-prefix for disentangle?
4. Ship correlational Transfer-stable + negative regen appendix, or wait for one positive patch?

---

## 8. Artifact index

### Entry / inventory (andy-biz)

| Doc | Path |
| --- | --- |
| **This mechanism study / weekend pack** | `/workspace/andy-biz/NARCBench-Transfer-Stable-Weekend-Review-2026-10-01.md` |
| Cache inventory | `/workspace/andy-biz/NARCBench-Transfer-Stable-Cache-Inventory-2026-10-01.md` |

### Transfer-stable results root

`/workspace/narcbench-core-demo/results/transfer_stable/`

| Artifact | Rel path |
| --- | --- |
| Claim (corr vs causal) | `CLAIM_CORRELATIONAL.md` |
| Negative causal appendix | `NEGATIVE_CAUSAL_APPENDIX.md` |
| Interim vs HARD baselines | `INTERIM_VS_BASELINES.md` |
| Controls offline | `CONTROLS_OFFLINE.md`, `controls_offline.json` |
| Residualize mode | `RESIDUALIZE_MODE.md`, `residualize_mode.json` |
| Layer sweep | `LAYER_SWEEP.md`, `layer_sweep.json` |
| Subspace + direction | `SUBSPACE_AND_DIRECTION.md`, `subspace_and_direction.json` |
| Gate cards (do not launch) | `GATE_CARDS.md` |
| Plan + inventory | `PLAN_AND_INVENTORY.md` |
| MLP/attn probes | `MLP_ATTN_PROBES.md`, `MLP_ATTN_PROBES.json` |
| MLP/attn causal (final) | `MLP_ATTN_CAUSAL_RESULTS.md` |
| Discussion-phase causal | `DISCUSSION_PHASE_CAUSAL_RESULTS.md` |
| Residual α=1 Transfer smoke | `CAUSAL_SMOKE_RESULTS.md` |
| Residual α-sweep + Core | `CAUSAL_ALPHA_CORE_RESULTS.md` |
| Causal plan | `CAUSAL_PLAN.md` |
| Pipeline summary | `MLP_ATTN_PIPELINE_SUMMARY.json` |
| Directions | `directions/` (`lr_role_attn_L22.npy`, `lr_role_L21.npy`, `lr_role_perp_mode_L21.npy`, …, `meta.json`, `meta_mlp_attn.json`) |
| Smoke run dirs | `causal_steer_smoke_*`, `causal_steer_alpha_sweep_*`, `causal_steer_core_smoke_*`, `mlp_attn_smoke_*`, `discussion_attn_smoke_*` |

### HARD Transfer baselines (prior)

| Artifact | Path |
| --- | --- |
| FULL HARD metrics | `/workspace/narcbench-core-demo/results/transfer/TRANSFER_FULL_HARD_METRICS.md` |
| Per-arm metrics | `results/transfer/FULL_core_to_transfer_*/metrics.json` |

### Canonical activation caches

| Cache | Path |
| --- | --- |
| Core residual L19–23 | `data/activations/gemma2_9b/core/20261001T012639Z/` |
| Transfer residual L19–23 | `data/activations/gemma2_9b/transfer/RUNPOD/` |
| Core MLP/attn | `data/activations/gemma2_9b/core/20261001T012639Z_mlp_attn/` |
| Transfer MLP/attn | `data/activations/gemma2_9b/transfer/RUNPOD_mlp_attn/` |
| W_U / Jacobians | `data/j_basis/gemma2_9b/` |

### Scripts (reference)

`scripts/transfer_stable_offline.py`, `transfer_stable_controls.py`, `transfer_stable_residualize.py`, `causal_steer_transfer.py`, `runpod_launch_causal_steer.py`, `eval_transfer_from_core.py`, `train_collusion_probe.py`

---

*Rewritten 2026-10-01 ~5:15 PM PT as a mechanism study from existing `results/transfer_stable/*` + HARD Transfer baselines + cache inventory. No new numbers. No GPU. Soft-stop honored.*
