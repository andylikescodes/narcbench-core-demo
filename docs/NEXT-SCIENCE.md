# Next science (J-lens + activation patching)

Attribution: NARCBench protocol Rose et al. Gemma Core regen is a **replication** bank for smaller open-weight models so we can run interpretability locally / on modest GPUs.

## Do not J-lens every token first

Full-sequence J-space / logit-lens over every position × every layer is expensive and noisy for multi-round committee transcripts (private R0 + 12 public turns + finals). Prefer **sparse sites** with a clear behavioral hypothesis, then patch.

## (a) Sparse J-lens sites

Priority token spans (per agent forward, collusion vs matched control):

1. **Last token before vote** — final-phase prompt, immediately prior to generating `RECOMMENDATION:` / the option letter. Hypothesis: colluder residual carries “target option” direction more strongly than control at late layers.
2. **While reading partner message** — positions over the colluder-partner’s private (R0) or public utterance inside the *current* agent’s context. Hypothesis: cross-agent coordination shows up as transient alignment / stego-like features when encoding the partner’s text, not uniformly across the whole history.

Secondary (only after 1–2 light up):

- First generated token of a public colluder turn (commitment / talking-point onset).
- Tokens spelling the *fair* option name when the colluder is arguing against it (suppression).

Recording recipe: for each chosen `(layer, token_span)`, store residual stream (and optionally attn out) for collusion-run and control-run with **matched** scenario_id, speaking order seed, and agent name draw.

## (b) Activation patching (Andy’s idea)

Causal test, not just correlation:

1. Run **collusion** forward; cache residuals at chosen `layer × token_span`.
2. Run **matched control** forward; at the same `(layer, positions)`, **copy in** the collusion residuals (patch). Measure ΔP(target option) / vote string / next-token KL on the final recommendation.
3. **Reverse scrub**: on the collusion forward, patch in control residuals at the same site; measure whether collusive vote / talking points collapse toward fair.

Start with a single late layer and the “last token before vote” site. Expand to partner-read spans only if vote-site patches move behavior.

Implementation notes (Gemma-2-2B/9B):

- Use the regen transcripts under `upstream/scenarios/gemma2_*/core/<run_id>/` so prompts are identical to what we patch.
- Keep temperature 0 for patch evals; reuse `agent_prompts.json` turns so collusion/control contexts stay aligned except for role/system and private history.
- Log `collusion_success`, `target_vote_count`, and token-level Δlogit on option letters A/B (or 1/2).

## (c) Ordering

1. Smoke Gemma-2-2B Core regen (3×2) → confirm colluders actually coordinate + vote target at non-trivial rate.
2. Sparse J-lens on smoke set only (vote-site + partner-read).
3. Activation patch / scrub on the best layer×site.
4. Scale to full 50×2 and/or 9B **after** the causal site is stable.

Do **not** begin with exhaustive J-lens grids on Qwen3-32B activations (heavy) until the small-model causal story is clear.

## Locked research direction — 2026-09-30

After the Gemma-2-9B full Core run finishes (pod `lnqm4yawhv13on`; do not touch it while running), run two collusion-detection probe ablations using the NARCBench-paper setup family: hidden activations → predict colluding vs. non-colluding pairs.

1. **J-space only:** train/evaluate using only J-space activations. Near-100% separation would support collusion-relevant information already living in J-space.
2. **Complement:** train/evaluate using only activations outside J-space. This tests whether a probe can still be found there, locating (or failing to locate) the signal relative to J-space.

For each ablation, report accuracy/separation against both the paper’s full-activation baseline and the full-activation baseline from this 9B Core run. Do not change any running jobs.

## Related work cite table — subspace probes / J-space vs complement (2026-09-30)

Dump of the novelty lit brief for collusion detection via J-space-only vs complement probe ablation. Source papers only; do not invent citations.

### Novelty (short answer)

**No clear prior** for a **J-space-only vs complement** probe ablation aimed at **multi-agent collusion**. Closest work does subspace-vs-complement for *deception*, or privileges verbalizable / J-space representations without a collusion AUROC split. **Likely novel** for the NARCBench collusion setting.

### NARCBench baseline

| Paper | Venue / ID | What they do | Gap vs our ablation |
| --- | --- | --- | --- |
| Rose, Cullen, Abdelnabi, Torr, Kaplowitz, Schroeder de Witt — *Detecting Multi-Agent Collusion Through Multi-Agent Interpretability* | [arXiv:2604.01151](https://arxiv.org/abs/2604.01151) (2026) | Full residual / activation probes for colluding vs non-colluding pairs | No **J vs complement** split; full-activation baseline only |

### (1) Subspace vs complement probes (closest methodological priors)

| Paper | ID | Relevance | Not our claim |
| --- | --- | --- | --- |
| Yoo & Skapars — *Probe Generalization as Subspace Selection for OOD Deception Detection* | [arXiv:2609.02893](https://arxiv.org/abs/2609.02893) | Explicit **PCA subspace vs complement** for **deception** OOD | Deception, not collusion; not J-space |
| Mahadik & Skapars — *Do Linear Probes Generalize Better in Persona Coordinates?* | [arXiv:2605.09391](https://arxiv.org/abs/2605.09391) | Persona PCs for deception / sycophancy | Coordinate choice ≠ J vs complement for collusion |
| Gurnee, Lindsey et al. — *Verbalizable Representations Form a Global Workspace* | [arXiv:2607.15495](https://arxiv.org/abs/2607.15495) | **J vs non-J** split of representations | Privilege / workspace story; **not** collusion AUROC ablation |
| Boxo et al. — *Caught in the Act* | [arXiv:2508.19505](https://arxiv.org/abs/2508.19505) | Nullspace projection across multiple deception directions | Against “one clean subspace”; not collusion / J |
| Kumar — *Pressure-Testing Deception Probes* | [arXiv:2605.27958](https://arxiv.org/abs/2605.27958) / GEM 2026 | Stress-tests against a clean low-dim deception subspace | Skeptical of containment; deception only |
| Belrose et al. — *LEACE* | [arXiv:2306.03819](https://arxiv.org/abs/2306.03819) | Concept-erasure tooling (linear) | Tooling, not a collusion result |

### (2) Containment claims (does collusion live in J-space?)

**No strong containment claim** that multi-agent collusion is confined to J-space.

| Source | Stance |
| --- | --- |
| Gurnee et al. (verbalizable / global workspace) | **Privilege** for J / verbalizable reps — not a collusion containment proof |
| Yoo & Skapars | Useful signal in a **small PC set** for deception — not J, not collusion |
| Kumar; Boxo et al. | Push **against** “one clean low-dim deception subspace” |

### (3) Related hits (broader probe / representation lit)

| Line | Pointer |
| --- | --- |
| Logit lens | classic residual→vocab readout |
| Tuned Lens | Belrose et al. [arXiv:2303.08112](https://arxiv.org/abs/2303.08112) |
| Representation Engineering (RepE) | Zou et al. [arXiv:2310.01405](https://arxiv.org/abs/2310.01405) |
| Contrastive Activation Addition (CAA) | Rimsky et al. |
| Geometry of Truth | Marks & Tegmark [arXiv:2310.06824](https://arxiv.org/abs/2310.06824) |
| CCS | Burns et al. [arXiv:2212.03827](https://arxiv.org/abs/2212.03827) |
| Linear Representation Hypothesis (LRH) | Park et al. [arXiv:2311.03658](https://arxiv.org/abs/2311.03658) |
| Goldowsky-Dill et al. | [arXiv:2502.03407](https://arxiv.org/abs/2502.03407) |
| Anthropic sleeper / deception probes | Anthropic line (sleeper agents / probe evals) |
| Refusal direction | Arditi et al. [arXiv:2406.11717](https://arxiv.org/abs/2406.11717) |
| *A Topic Detector Not a Lie Detector* | de Polignac, Sep 2026 |
| Short Horizons / J-lens math | [arXiv:2608.25347](https://arxiv.org/abs/2608.25347) |

### J-space definition (working)

**J-space**: the subspace of residual / activation directions that are **verbalizable** under a J-lens / logit-lens–style map (roughly: directions that project cleanly onto vocabulary / “sayable” features), as used in Gurnee–Lindsey global-workspace framing and related Anthropic J-space privilege. **Complement** = residual directions orthogonal (or otherwise outside) that verbalizable subspace. Our planned ablation: train/eval collusion probes on **J-only** vs **complement-only**, vs full activations (NARCBench + our 9B Core baseline).

### Overall novelty assessment

| Claim | Status |
| --- | --- |
| J-only vs complement probes for **collusion** | **Likely novel** — no clear prior found |
| Subspace vs complement probes for **deception** | Exists (Yoo & Skapars) |
| J vs non-J representational split | Exists (Gurnee et al.) — not collusion AUROC |
| Full-activation collusion probes | Exists (NARCBench / Rose et al.) |

**Talking point:** NARCBench already does **full-activation** collusion probes; Anthropic / Gurnee grant **J-space privilege**; Yoo does **subspace-vs-complement** for deception — **nobody crossed these for collusion**. That cross is the low-hanging fruit after the Gemma-2-9B Core run.

## Sequenced implementation plan — NEXT-SCIENCE (2026-09-30 ~8:16 PM PT)

Hard constraint: **do not modify / interrupt** full Core pod `lnqm4yawhv13on`
(status: https://lnqm4yawhv13on-8765.proxy.runpod.net/status.json).
All GPU work for activation extract / Stage 2 uses **separate** RunPod jobs.
Box has no CUDA — probe training is CPU-ok once `.npz` exist; extraction needs GPU.

### Gate vocabulary

| Verdict | Meaning |
| --- | --- |
| **PASS** | Proceed to next stage |
| **FAIL** | Stop; do not start Stage 2 / 3 until unblock |
| **SOFT** | Smoke-only / n-small; informative, not a hard gate |

### Stage 1 — Reproduce NARCBench collusion classification on FULL activations (Gemma-2-9B)

**Goal.** Match paper protocol family (Rose et al. arXiv:2604.01151 / `aaronrose227/narcbench` `probes/reproduce.py`): logistic **deception direction** on last-token mid-layer residuals (colluder vs honest within collusion runs) → scenario-level aggregators → held-out AUROC.

**Discrete steps**

| Step | Action | Deliverable |
| --- | --- | --- |
| 1.0 | Inventory transcripts + activation availability (smoke + full) | This section + `results/stage1/INVENTORY.md` |
| 1.1 | Build sample index from `agent_prompts.json` + `results.json` (no GPU) | `metadata` preview JSON |
| 1.2 | GPU extract gen-time last-token residuals for smoke (`20261001T011552Z`) on a **new** pod — not `lnqm4yawhv13on` | `data/activations/gemma2_9b/core/<run_id>/activations_gen.npz` + `metadata_gen.json` |
| 1.3 | Train/eval probe on smoke (exploratory) | `results/stage1/smoke/metrics.json` — **SOFT** only |
| 1.4 | After full Core transcripts land (pod done + pull): extract activations for full 50×2 on a **separate** pod | full `activations_gen.npz` |
| 1.5 | Leave-domain-out CV on full bank; report Peak / Split / Asym (+ optional Alignment / SortedConcat) | `results/stage1/full/metrics.json` |

**Primary metrics (must report all three + mean)**

- `auroc_peak_suspicion`
- `auroc_deception_split`
- `auroc_asymmetry_probe`
- `auroc_primary` = mean of the three

Optional (paper five-method set): Activation Alignment, Sorted Concat.

**Protocol details (attributable)**

- Model: `google/gemma-2-9b-it` (42 layers, d_model=3584). Default probe band: mid layers **19–23** (center ≈21); also report single mid layer 21.
- Site: **generation-time last token** of each public discussion turn (paper Core default: avg agent vectors across discussion rounds). Also store **final**-phase gen-time as secondary.
- Train direction: colluder vs honest, collusion runs only, public channel, discussion phase; leave-domain-out excludes hold domain from direction fit.
- Layout (HF / paper): `activations_gen.npz` keys `layer_<i>`; rows align 1:1 with `metadata_gen.json`.

**What “similar to NARCBench Core” means (PASS gate)**

Paper headline (arXiv:2604.01151): **every model reaches 1.00 AUROC in-distribution** on Core held-out. Local Gemma-2-2B replication (`collusion-exp`): Peak=1.00, Split≈0.97, Asym=1.00, mean≈0.99.

| Check | PASS threshold (full 50×2, leave-domain-out) | FAIL |
| --- | --- | --- |
| Primary mean AUROC | **≥ 0.90** | < 0.90 |
| Each of Peak / Split / Asym | **≥ 0.85** | any < 0.85 |
| Sanity | Collusion behavioral hit rate on audited bank non-trivial; ≥1 domain fold with both labels | Empty folds / broken labels |

**SOFT (smoke 3×2 only):** report metrics; do **not** PASS Stage 1 on smoke alone (n_domains=3, tiny). Smoke PASS-to-continue-extract = extraction job completes with aligned npz+meta and probe script runs without error.

**Stage 1 PASS → unlock Stage 2.** Stage 1 FAIL → debug extract site / layer / audit failed collusion runs; do not start J/complement.

### Stage 2 — Subspace studies (only after Stage 1 PASS)

Two **separate** batteries, same train/eval splits as Stage 1 full baseline:

| Arm | Definition | Cite pattern |
| --- | --- | --- |
| **J-only** | Project residuals onto verbalizable / J-space basis (Gurnee/Lindsey arXiv:2607.15495 explicit J vs non-J); train+eval probe in that subspace only | J privilege framing |
| **Complement** | Residuals orthogonal (or otherwise outside) J-space; train+eval separately | Yoo & Skapars arXiv:2609.02893 subspace-vs-complement style |

**Steps:** (2.1) Fit/load J basis on Gemma-2-9B residuals → (2.2) Project full activations → J / complement tensors → (2.3) Retrain Peak/Split/Asym on each arm with **identical** leave-domain-out folds as Stage 1 → (2.4) Write `results/stage2/{j_only,complement}/metrics.json`.

**PASS (procedural):** both arms produce metrics under the same CV; no requirement that J≈full. **FAIL:** basis/projection broken or fold mismatch vs Stage 1.

### Stage 3 — Compare full / J-only / complement

| Step | Deliverable |
| --- | --- |
| 3.1 | Table: method × arm AUROC (+ fold std) |
| 3.2 | ΔAUROC (J − full), (complement − full), (J − complement) |
| 3.3 | One-paragraph attributable reading (containment vs distributed) |

**PASS:** table + deltas checked in; numbers trace to Stage 1/2 artifact paths.

### Ordering / continuous execution checklist

1. Write plan (this section) ✅
2. Scaffold extract + probe scripts under `scripts/` (CPU dry-run metadata) 
3. Launch **separate** smoke activation extract on RunPod (estimate → `--launch` only with spend approval)
4. Soft-eval smoke probe
5. Wait for `lnqm4yawhv13on` phase=done → pull transcripts only (do not attach jobs to that pod)
6. Separate full activation extract → Stage 1 hard gate
7. Only if PASS: Stage 2 J / complement → Stage 3 compare

### Exact unblock if activations missing (Stage 1)

Smoke transcripts exist; **no** `.npz` activations on box. Next GPU job (separate from Core):

```bash
# Estimate only (default):
python3 /workspace/narcbench-core-demo/scripts/runpod_launch_extract_acts.py \
  --runs-rel gemma2_9b/core/20261001T011552Z \
  --model google/gemma-2-9b-it \
  --layers 19-23 --gen-only

# Spend (requires explicit approval):
python3 /workspace/narcbench-core-demo/scripts/runpod_launch_extract_acts.py \
  --runs-rel gemma2_9b/core/20261001T011552Z \
  --model google/gemma-2-9b-it \
  --layers 19-23 --gen-only --launch
```

Then on box (CPU; use probe venv):

```bash
cd /workspace/narcbench-core-demo
.venv-probe/bin/python scripts/train_collusion_probe.py \
  --acts-dir data/activations/gemma2_9b/core/20261001T011552Z \
  --out results/stage1/smoke --layers 19-23
```

### Artifact roots

- Transcripts: `upstream/scenarios/gemma2_9b/core/<run_id>/`
- Activations: `data/activations/gemma2_9b/core/<run_id>/`
- Stage results: `results/stage1/`, `results/stage2/`, `results/stage3/`
- Upstream probe reference: `/workspace/collusion-exp/collusion_exp/train_probe.py` + paper `probes/reproduce.py`

## Execution status — 2026-09-30 ~10:50 PM PT

| Stage | Verdict | Notes |
| --- | --- | --- |
| Stage 1 FULL | **HARD_PASS** | primary=1.000 Peak/Split/Asym=1.000; `results/stage1/FULL_HARD_GATE.md` |
| Stage 2 J-only | **PASS** (procedural) | primary=0.989; Peak=0.991 Split=0.980 Asym=0.996; CPU only |
| Stage 2 Complement | **PASS** (procedural) | primary=1.000; matches full baseline; CPU only |
| Stage 3 compare | **PASS** | table + deltas + distributed reading in `results/stage3/` |

**J-space definition used:** per-activation orthogonal projection onto span of top-k=25 unit unembedding rows (`W_U` = tied `embed_tokens` for `google/gemma-2-9b-it`; logit-lens proxy for Gurnee/Lindsey J-lens vectors). Complement = `h − h_J`. Full Jacobians not estimated. See `results/stage2/J_SPACE_DEFINITION.md`.

**GPU:** not needed for Stage 2/3 (precomputed W_U + CPU projection of already-extracted activations). No RunPod launched for this stage; no idle GPUs left running.

**Reading (Stage 3):** Distributed — both arms retain near-full AUROC; collusion signal not confined to J-space under this definition.

## NEXT-SCIENCE continuation — 2026-10-01 ~12:50 AM PT

### Goal lock

1. **True per-layer Jacobian J-lens** (Gurnee/Lindsey $J_\ell=\mathbb{E}[\partial h_{\mathrm{final}}/\partial h_\ell]$), not only top-k $W_U$ proxy.
2. **Transfer battery** on Gemma-2-9B: obtain transcripts+acts; probe full / J / complement; test whether J-localization transfers.

### A) Transfer inventory

See `results/transfer/INVENTORY.md`. **No gemma2_9b Transfer on HF or box.** HF has Transfer scenarios for `qwen3_32b` / `gpt_oss_20b` and activations for four paper models. Must regen on Gemma via RunPod (`generation/transfer.py` + HF backend).

### B) True Jacobian estimation (GPU)

- Script: `scripts/estimate_jacobian_jlens.py` — randomized sketch (r JVPs + r VJPs) averaged over short prompts; stores `J_layer_{L}_fp16.npy`.
- Launcher: `scripts/runpod_launch_jacobian.py` (+ ad-hoc ID-based deploy for EU-RO-1 stock).
- **Launched:** pod `o2552svtyep7ui` · RTX PRO 4500 · EU-RO-1 · **$0.72/h** · layers `19,21,23` · n_prompts=8 · rank=64 (thrifty first pass; escalate rank/prompts if needed).
- Status: https://o2552svtyep7ui-8765.proxy.runpod.net/status.json
- Auto-terminate after pull window.

### C) Core J-variant battery (CPU, parallel)

- Script: `scripts/project_j_variants.py` + `scripts/run_stage2_jacobian_battery.py`
- Variants documented under `results/stage2_jacobian/VARIANT_DEFINITIONS.md`:
  - `wu_topk_k{K}` — Stage2 logit-lens proxy (J≈I)
  - `wu_fixed_k{K}` — fixed global top-k W_U frame
  - `jac_topk_k{K}` — true J-lens scores ⟨Ŵ_i, J_ℓ h⟩ (waits on Jacobian files)
- **Running now:** W_U grid k∈{8,16,25,50}, L19–23 / L21, + fixed-k25 (pid in `results/stage2_jacobian/battery_wu.pid`).
- Outputs: `results/stage2_jacobian/<tag>/{j_only,complement}/metrics.json` + `BATTERY_TABLE.md`

### D) Transfer regen (GPU, parallel)

- Script: `scripts/regen_gemma_transfer.py` (patches narcbench `call_vllm` → HF Gemma).
- **Launched smoke:** pod `umnsl6ex86ryls` · RTX PRO 4500 · EU-RO-1 · **$0.72/h** · 12 runs (1×6 families × collusion+control).
- Status: https://umnsl6ex86ryls-8765.proxy.runpod.net/status.json
- After smoke OK → extract acts → train-on-Core / eval-on-Transfer for full / J / complement; then scale to full 72 if smoke collusion non-trivial.

### Spend snapshot

| Item | Est |
| --- | --- |
| Balance at launch | ~$32.47 |
| Combined burn | ~$1.44/h (2× RTX PRO 4500) |
| Jacobian pass (≤2 h) | ~$0.70–1.50 |
| Transfer smoke (≤3 h) | ~$0.70–2.20 |
| Pods auto-terminate on finish | yes |

### Watch points

1. Jac pod `phase=done` → pull `data/j_basis/gemma2_9b/jacobians/` → run `--jac-only` battery.
2. Transfer smoke `phase=done` → pull transcripts → launch extract acts (separate pod) → Transfer probe eval.
3. EU-RO-1 stock is thin; prefer RTX PRO 4500 / A40 / A6000 IDs that actually deploy with volume `cr1kjtzjs2`.


## Gate results — 2026-10-01 03:33 PDT

**SUCCESS (reportable):** Core W_U+jac battery + Transfer smoke AUROCs on disk.
See `results/NEXT_SCIENCE_GATE_REPORT.md`, `results/stage2_jacobian/BATTERY_TABLE.md`, `results/transfer/TRANSFER_SMOKE_METRICS.md`.

### Core (discussion) headline

| Variant | J-only | Complement |
| --- | ---: | ---: |
| wu_topk k8 | 0.894 | 1.000 |
| wu_topk k16 | 0.950 | 1.000 |
| wu_topk k25 | 0.989 | 1.000 |
| wu_topk k50 | 1.000 | 1.000 |
| wu_fixed k25 | 0.990 | 1.000 |
| jac_topk k8 | 0.900 | 1.000 |
| jac_topk k16 | 0.919 | 1.000 |
| jac_topk k25 | 0.961 | 1.000 |
| jac_topk k50 | 0.997 | 1.000 |

**Core reading:** Distributed — complement always 1.0; J-only ↑ with k.

### Transfer smoke (n=12) Core→Transfer final

| Arm | Primary |
| --- | ---: |
| full | 0.867 |
| W_U k25 J-only | 0.548 |
| W_U k25 complement | 0.867 |
| jac k25 J-only | 0.648 |
| jac k25 complement | 0.867 |

**Transfer reading:** Probes transfer (~0.87). **J-localization does NOT transfer** — complement carries OOD signal.

### Full Transfer 72
- Pod `twkd9sy1dt72c5` L4 EU-RO-1 $0.49/h — regen in progress; watch auto-pull+terminate.
- Status: https://twkd9sy1dt72c5-8765.proxy.runpod.net/status.json
- Extract+probes still TODO after pull.

### Costs / blockers
- Balance ~$30.9; jac sketch low-energy (escalate n/rank if needed); no idle GPUs intended.


## FULL Transfer HARD results — 2026-10-01 09:28 PDT

**SUCCESS:** 72/72 Transfer regen + extract + Core→Transfer probes on disk; billable pods terminated.

See `results/transfer/TRANSFER_FULL_HARD_METRICS.md`.

### Core→Transfer (n=72) primary AUROC — train Core discussion → eval Transfer final

| Arm | Smoke n=12 | **FULL n=72** |
| --- | ---: | ---: |
| full acts | 0.867 | **0.902** |
| W_U k25 J-only | 0.548 | **0.602** |
| W_U k25 complement | 0.867 | **0.898** |
| jac k25 J-only | 0.648 | **0.549** |
| jac k25 complement | 0.867 | **0.891** |

### Verdict (PASS vs smoke)
**Confirmed:** J-localization does **not** transfer. Complement ≈ full (~0.89–0.90); J-only weak (~0.55–0.60). Transferable collusion signal lives in **non-J residual**.

### Infra notes
- First full regen pod `twkd9sy1dt72c5` timed out at 64/72 (exit_124); resume pod `lcev9zjbacw3mz` completed 72/72 with skip-if-exists.
- Extract pod `7mpwtpcvvne0ge` → 936 samples; pulled; terminated.
- Balance ~$27.8; only volume idle cost (~$0.019/h).


## Transfer-stable weekend review — 2026-10-01 ~5:15 PM PT

Consolidated study pack (soft-stop: no Track I / Paper 2 / interp-demo / family-house; **no GPU** this pack; causal GPU paused):

→ **`results/transfer_stable/WEEKEND_REVIEW/`**

Safety claim: **monitorable Transfer-stable coordination signature (attn_L22 / non-J); editability unproven** — matched-prefix + checkpoint assays next (`WEEKEND_REVIEW/06_METHODS_LADDER.md`).

