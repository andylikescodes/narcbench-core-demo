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

## 2026-11-04 — Full J-lens estimation prepared; launch pending credentials

**Why.** No Gemma run estimated J_ℓ as Gurnee et al. (arXiv:2607.15495) define it, an expectation over about a thousand prompts and every pair of positions. The two arms recorded on 1 October used stand-ins: the unembedding rows with J = I (the logit lens), and an eight-prompt, rank-64, last-token sketch at layers 19, 21 and 23 (the frozen Transfer metrics list exactly those three layers; the nearest-layer fallback in `project_j_variants.py` was never exercised). Until the estimate exists no J-space result is claimed: the "Collusion Signatures Beyond J-Space" deck withholds the J-space results as of today (v19, 20 slides: the paper's replication on Gemma-2-9B, Core 1.000 and Transfer 0.902, stays; the J-space, site, role-versus-mode, robustness, findings and safety slides are out until the estimate exists), and the Gemma-2-9B deck names both arms as proxies.

**What exists now.**
- `scripts/estimate_jlens_full.py` — J_ℓ = E_{prompt, t ≤ t'} ∂h_{N,t'}/∂h_{ℓ,t}; h_ℓ = residual after block ℓ (the hook site of every recorded activation), h_N = residual after the last block before the final norm, BOS excluded as source and target, prompts weighted equally and pairs weighted equally within a prompt. Two-pass randomized range finder (Halko et al. 2011) for the mean Jacobian: one JVP with the same tangent at every source position returns the sum over sources at each target, so the pair-mean is one batched forward-mode pass per tangent; the rows come from batched VJPs with the same cotangent at every target. Defaults: 1,000 prompts (510 wikitext-103 paragraphs, 510 alpaca turns in the Gemma chat template, 20 held out), rank 256, chunk 32, max 128 tokens, layers 19–23, bf16 weights with fp32 accumulators, eager attention (forward-AD safe, exact Gemma-2 soft-capping). Checkpoints every 25 prompts; `--resume`. Outputs per layer: `J_layer_{L}_fp16.npy`, `J_layer_{L}_factors.npz` (Q, B, singular values of Y), plus `prompts.json` and `jlens_meta.json` (settings, timings, spectrum energy at 16…256, ‖J − I‖/‖I‖, and a held-out self-check: relative error of Q B against the pair-mean Jacobian of unseen prompts along random directions).
- `scripts/test_jlens_estimator_tiny.py` — exact-Jacobian check on a random three-block Gemma-2 (d = 32, sliding window on the local layers): causality holds, the JVP trick (vmap, loop, double-backward) and the VJP trick (batched, looped) match the exact pair-mean to ≤ 1.5e-7 relative, and the two-pass sketch at full rank reconstructs the mean over two prompts to 2.8e-7. ALL PASS, 4 Nov 2026, CPU.
- `scripts/runpod_launch_jlens_full.py` — plan at the defaults: 80,000 batched passes (1,000 prompts × 2 passes × 8 chunks × 5 layers). At an assumed 0.6 s per batched pass that is 15.8 h, about $14 on an RTX 6000 Ada at $0.89/h; the assumption is unmeasured, so the pod runs a smoke first (8 prompts, rank 32, one layer, ≤ 60 min) and sizes the job to 90% of the 12-hour cap: rank 256 → 128, then prompts 1,000 → 600 → 400. The chosen configuration and the projection are written to `status.json`. Results go to the network volume (`/workspace/jobs/narcbench-results/j_basis/gemma2_9b/jlens_full/`) and are served as `results.tgz` for 45 minutes before the pod terminates itself.

**Not launched from this machine.** `RUNPOD_API_KEY` and `HF_TOKEN` are not in this environment (the 1 October launches ran from the box, reading `/home/box/.secrets`). To launch, in a shell with the key exported:

```bash
python3 scripts/runpod_launch_jlens_full.py --launch
```

then `--status <pod id>` to watch and `--pull <pod id>` to fetch into `data/j_basis/gemma2_9b/jlens_full/` (ignored by git). One billable GPU at a time, as before.

**After the estimate (CPU, to write).** `scripts/jspace_decompose.py`: J-lens vectors = rows of W_U J_ℓ (with the final norm applied to J_ℓ h before scoring); per-activation sparse nonnegative pursuit with about 25 atoms → J component and complement for the Core and Transfer caches; then the three arms through `train_collusion_probe.py` / `eval_transfer_from_core.py` with the pre-registered controls of `docs/jspace/JSPACE_SEPARABILITY_DESIGN.md`. Needs the activation caches (RunPod volume) or a re-extraction.

## 2026-11-04 — Full J-lens estimation launched (pod rt0eeu5t3kff55)

- Launched 2026-11-05 03:44 UTC (19:44 PST) with `scripts/runpod_launch_jlens_full.py --gpu <type> --launch`, one GPU type per call so the A100/H100 entries of the preference list could be skipped (a 12-hour run on them does not fit under the balance). The four 48 GB types (RTX 6000 Ada, L40S, A40, RTX A6000) had no supply in the volume's data centre on either cloud, and `NVIDIA RTX PRO 4500` is no longer a valid gpuTypeId. Result: pod `rt0eeu5t3kff55`, RTX 4090 (24 GB), SECURE, EU-RO-1, $0.89/h (the launcher's table assumed $0.74).
- Gates at launch: balance $24.60, no other pods on the account (volume idle $0.019/h), clean tree at c2a7c67. Worst case: 12 h cap + up to 1 h smoke + 45 min grace ≈ $12.2.
- Keys came from the repo's git-ignored `.env` (`.gitignore:10`) through a scratch wrapper that loads only RUNPOD_API_KEY and HF_TOKEN into the launcher's environment; no value was typed into chat or printed.
- Watch: `python3 scripts/runpod_launch_jlens_full.py --status rt0eeu5t3kff55` (status.json, projection.json, status_full.json). Pull: `--pull rt0eeu5t3kff55` → `data/j_basis/gemma2_9b/jlens_full/` (git-ignored). The pod self-terminates after the grace window; checkpoints and outputs live on the network volume under `/workspace/jobs/narcbench-results/j_basis/gemma2_9b/jlens_full/` and are copied to `/workspace/jobs/narcbench-core-demo/data/j_basis/gemma2_9b/jlens_full/` at the end.
- On a 24 GB card the estimator may halve the tangent chunk on OOM; the smoke's projection then shrinks (n, rank) to fit 90% of the cap. The smoke outcome and the chosen (n, rank) are appended below once status.json reports them.
- **First pod failed in the smoke and was terminated by hand at 03:51 UTC (about 3 minutes billed, ≈ $0.05).** `job.log`: `huggingface_hub.errors.HfUriError: Invalid HF URI 'hf://datasets/wikitext@…'. Repository id must be 'namespace/name', got 'wikitext'` — the pod's fresh `datasets` install pulls huggingface_hub ≥ 1.0, which no longer accepts the bare canonical dataset id. Fix in `scripts/estimate_jlens_full.py`: `load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="test+validation")` (the alpaca source already used `tatsu-lab/alpaca`). Checked locally in the scratch venv (datasets 5.1.0, huggingface_hub 2.4.0) with the Gemma tokenizer: the smoke mix gives 8 + 8 prompts, the full mix 1,020 distinct prompts, BOS added, 64 tokens at max-len 64.
- Launcher corrected against the live `gpuTypes` catalogue: `NVIDIA RTX PRO 4500` → `NVIDIA RTX PRO 4500 Blackwell`; `NVIDIA L40` and `NVIDIA RTX PRO 5000 Blackwell` added to the 48 GB group; the rate table now holds the secure-cloud prices (RTX 4090 $0.89, RTX 6000 Ada $0.99, L40S $1.09, A40/A6000/L4 $0.59, A100 $1.79, H100 PCIe $2.89). New gates, run on `--launch` and on the new `--preflight`: balance ≥ `--min-balance` (20), no billable pod already running unless `--allow-existing`, and each GPU type's worst case (rate × (cap + 1 h smoke + grace) = 13.75 h) must leave `--reserve` ($5) of the balance, which drops the A100/H100 entries at today's balance. Preflight at relaunch: balance $24.51, no pods, nine eligible types.
- **Relaunched 2026-11-05 03:52 UTC (19:52 PST) with the fixed estimator: pod `i84t3tev3ighsr`, RTX PRO 4500 Blackwell (32 GB, the October sketch's GPU class), SECURE, EU-RO-1, $0.72/h; worst case 13.75 h ≈ $9.90.** The six 48 GB types had no supply in EU-RO-1 again. Record: `results/jspace/jlens_full_pod.json`. Watch: `python3 scripts/runpod_launch_jlens_full.py --status i84t3tev3ighsr`; pull: `--pull i84t3tev3ighsr`. The smoke outcome and the chosen (n, rank) follow below.
- **Pod `i84t3tev3ighsr` never started its container: 19 minutes "Rented by User" with `runtime: null` and the status port down, so it was terminated at 04:11 UTC.** The balance did not move during the wait ($24.49 before and after), so the pull was not billed. Likely a slow or stuck host: the first pod (a different host) had its container up within a minute. Launcher: new `--min-download` (Mbps, default 800; 0 disables) passed as `minDownload` to `podFindAndDeployOnDemand`, so placement skips slow hosts.
- **Third pod, launched 2026-11-05 04:11 UTC (20:11 PST) with `--min-download 800`: `6arnxdpuarte35`, RTX PRO 4500 Blackwell, SECURE, EU-RO-1, $0.72/h.** The 48 GB types had no supply for the third time. Record: `results/jspace/jlens_full_pod.json`. Watch: `python3 scripts/runpod_launch_jlens_full.py --status 6arnxdpuarte35`; pull: `--pull 6arnxdpuarte35`. Smoke outcome and chosen (n, rank) follow.
- **Third pod `6arnxdpuarte35` landed on the same host (podHostId suffix 64412270) and again had no container after 9 minutes; terminated at 04:21 UTC, unbilled. Fourth pod, launched 04:22 UTC (20:22 PST) with `--gpu "NVIDIA GeForce RTX 4090"` to leave that host pool: `tt4hvhr7xgvsnu`, RTX 4090 (24 GB), SECURE, EU-RO-1, $0.89/h; worst case 13.75 h ≈ $12.2 against a $24.49 balance.** Record: `results/jspace/jlens_full_pod.json`. Watch: `python3 scripts/runpod_launch_jlens_full.py --status tt4hvhr7xgvsnu`; pull: `--pull tt4hvhr7xgvsnu`. On 24 GB the weights take ~18.5 GB; the estimator halves the tangent chunk on OOM and the smoke's projection sizes (n, rank) accordingly.
- **Pod `tt4hvhr7xgvsnu` booted in 2 min and the smoke passed at 04:26 UTC:** 8 prompts, rank 32, layer 21, max-len 64 → 5.2 s of GPU passes (jvp_mode vmap, batched VJP), 112 s wall including the model load; holdout self-check relative error 0.95–0.97, as expected for 8 prompts at rank 32. The boot script's projection chose (1,000 prompts, rank 256) with "3.61 h", but that formula divided by the 128/64 length ratio instead of multiplying (fixed in the launcher: `per_unit = s/(8·32)·(max_len/64)`), so the honest smoke-based projection is ≈ 14 h with warm-up included, against the 12 h cap. The full run (pass 1 from 04:27 UTC, 40 batched passes per prompt per pass) reports seconds per prompt in `status_full.json`; the decision whether to let it run or restart with a larger cap follows from that measured rate, recorded below.
- **Measured rate, 04:29 UTC, after 5 prompts of pass 1: 13.8 s per prompt at chunk 32 (no OOM), vmap JVPs, ETA for pass 1 ≈ 3.8 h; pass 2 (one graph forward plus batched VJPs per chunk) expected at 1–1.5× that, so the whole job should finish in ≈ 8.5–10.5 h (≈ 13:00–15:00 UTC, 05:00–07:00 PST on 5 Nov), inside the 12 h cap. Decision: let it run with (1,000, 256).** Expected cost ≈ 9.5 h × $0.89 + 45 min grace ≈ $9, leaving ≈ $15 of the $24.49 balance. A session cron checks `status.json` every 20 minutes to pull `results.tgz` inside the grace window; the outputs also persist on the network volume.

## 2026-11-04 (evening) — Anthropic's released Jacobian-lens code and a published lens for gemma-2-9b-it

Found while the pod ran, at the user's prompting. **Code:** `github.com/anthropics/jacobian-lens` (package `jlens`, Apache-2.0, "Reference implementation. Not maintained"). `jlens.fit` estimates the full d×d mean Jacobian per source layer by reverse mode: one forward with the graph retained, then one-hot cotangents on `dim_batch` output dimensions at every valid target position per backward pass (d/dim_batch backward passes per prompt); gradient at source t = Σ_{t′≥t} ∂h_final[t′]/∂h_ℓ[t], averaged over source positions (not over pairs), then an unweighted mean over prompts; valid positions skip the first 16 and the last; target = output of the last block (before the final norm); fp32 accumulation, fp16 files; checkpoint/resume; `JacobianLens.merge` for sharded fits; `apply` = `model.unembed(J_ℓ h)`. No J-space decomposition code is released (jlens/ has fitting, lens, hf, hooks, protocol, vis, examples). **Published lens for our model:** `huggingface.co/neuronpedia/jacobian-lens`, file `gemma-2-9b-it/jlens/Salesforce-wikitext/gemma-2-9b-it_jacobian_lens.pt` (1,053 MB fp16 = 41 source layers × 3584²; keys J / n_prompts / source_layers / d_model), fit by Neuronpedia with Anthropic's code on 2026-06-12 on a B200: wikitext-103 train, max_chars 2000, max_seq_len 128, dim_batch 128, bf16 model, requested 1,000 prompts, **early-stopped at 453** (stop_at_delta 0.002; final mean relative change 0.00154; final identity distance 0.288). Apache-2.0 per the repo metadata (the lens derives from Gemma weights, so Gemma's terms also apply). **Versus our running estimate (pod tt4hvhr7xgvsnu):** same estimand and target point; ours is a rank-256 sketch (theirs full rank), pair-mean (theirs source-mean; a per-prompt scalar that the norm in the readout removes, so only prompt weighting differs), BOS-only skip (theirs 16), 1,000 wikitext+alpaca chat-templated prompts (theirs 453 wikitext). **Consequence:** the published lens is the J_ℓ to use for the Gemma J-space analysis (the paper's own code, full rank, every layer, converged); our run is at most an independent cross-check of corpus sensitivity. What neither provides is the J-space decomposition itself (gradient pursuit over the rows of W_U J_ℓ, k ≈ 25, orthogonal projection onto the chosen span; complement = remainder), which still has to be written here.
- **Decision (user, 2026-11-05 04:44 UTC = 20:44 PST 4 Nov): stop our estimate and use the published lens.** Pod `tt4hvhr7xgvsnu` terminated at 04:44 UTC after 75 of 1,000 prompts of pass 1 (13.0 s/prompt, chunk 32, no OOM); its partial checkpoints remain on the network volume under `/workspace/jobs/narcbench-results/j_basis/gemma2_9b/jlens_full/`. The 20-minute session check was cancelled. Tonight's four pods cost $0.39 in total (balance $24.60 → $24.21). The published lens is being downloaded into `data/j_basis/neuronpedia/gemma-2-9b-it/jlens/Salesforce-wikitext/` (git-ignored) with its `config.yaml` and convergence CSV; verification follows below.
- **Published lens downloaded and verified (04:46 UTC).** `gemma-2-9b-it_jacobian_lens.pt`, 1,053,307,540 bytes, sha256 dcc03a24e762…; loads with `torch.load(weights_only=True)` in 0.4 s; keys `J`, `d_model`, `n_prompts`, `source_layers`; n_prompts = 453, d_model = 3,584, source layers 0–40 (41 fp16 matrices). The convergence CSV shows 128-token prompts with 111 valid positions each (skip 16 + last), ≈ 13 s per prompt on the B200, identity distance falling from 0.308 to 0.288 and the mean relative change reaching 0.0015 at prompt 453. Band layers 19–23: ‖J‖_F = 24.9, 25.5, 27.1, 29.1, 30.4 (‖I‖_F = 59.9); ‖J − I‖_F/‖I‖_F = 0.997, 0.991, 0.986, 0.974, 0.953; mean diagonal 0.087, 0.098, 0.115, 0.143, 0.174; max |entry| ≈ 1.09–1.36. Singular-value spectra for 19/21/23 follow.
- **Spectra of the published lens (fp32 svdvals on CPU, 5 s per layer).** Layer 19: top singular values 8.65, 4.88, 4.04, 3.61, 3.27 …; Frobenius energy in the top 64 / 128 / 256 / 512 = 0.606 / 0.758 / 0.883 / 0.956. Layer 21: 9.62, 5.10, 4.14, 3.76, 3.32 …; 0.563 / 0.704 / 0.834 / 0.926. Layer 23: 9.60, 4.95, 4.20, 3.84, 3.46 …; 0.465 / 0.595 / 0.734 / 0.861. The mean Jacobian is moderately low-rank (one dominant direction, then a slow tail): a rank-256 sketch like the one we stopped would have carried 73–88% of ‖J‖_F² in this band, so the full published matrix is the right J_ℓ to decompose against. **Next (CPU, to write): `scripts/jspace_decompose.py`** on this lens — J-lens readout softmax(W_U · norm(J_ℓ h)) with the Gemma final norm and softcap as in `jlens.hf.unembed`; dictionary = rows of W_U J_ℓ (final-norm weight folded in) restricted per activation to a candidate set from the readout; nonnegative gradient pursuit to k = 25 atoms; h_J = orthogonal projection onto the chosen span, h_C = h − h_J (paper's appendix formalisation); then the pre-registered arms and controls of `docs/jspace/JSPACE_SEPARABILITY_DESIGN.md`. The Core/Transfer activation caches (`activations_gen.npz`, layers 19–23) are on the RunPod volume, not on this Mac, so the reruns need them copied down first.

## 2026-11-04 (late evening) — `scripts/jspace_decompose.py`: the paper's J-space split, built against the published lens

**What exists now.** `scripts/jspace_decompose.py` (CPU; numpy + scipy, torch only to read the .pt lens) takes a paper-layout activation cache, a lens in the `jlens` save format and the model's unembedding and final-norm weight, and writes `j_only/` and `complement/` caches in the same layout plus per-activation tokens, coefficients, readout top-25 and FVE. Definitions, as registered in the addendum of `docs/jspace/JSPACE_SEPARABILITY_DESIGN.md`: readout softcap(W_U · norm(J_ℓ h)) with Gemma's (1 + γ) RMSNorm scale and the softcap 30, as `jlens.hf.unembed`; dictionary v_i = J_ℓᵀ((1 + γ) ⊙ W_U[i]); candidates = top 1,024 tokens by ⟨v_i, h⟩/‖v_i‖; nonnegative OMP to k = 25 (NNLS refit each step), filled to exactly k with the next-best candidates when the pursuit stops early (recorded); h_J = orthogonal projection onto the chosen span, h_C = h − h_J. Options: `--method gp|topk`, `--select raw`, `--no-fill`, k, candidates. `scripts/test_jspace_decompose_synthetic.py`: ALL PASS (planted-atom recovery with coefficients to 3.5e-4, additivity to 1e-5, orthogonality to 1e-8 of ‖h‖², readout order = raw-score order, `rms_norm` = transformers' `Gemma2RMSNorm` to 2.4e-7, chunked atom norms, end-to-end batch).

**Inputs fetched (git-ignored).** `data/j_basis/gemma2_9b/`: `unembed_fp16.npy` [256000, 3584] (the tied `model.embed_tokens.weight`, 1.835 GB, row norms mean 1.72), `unembed_row_norms_fp32.npy`, `final_norm_weight_fp32.npy` (`model.norm.weight`, mean 2.56, min −0.98, max 7.88, so the scale 1 + γ runs from 0.016 to 8.9). `scripts/fetch_unembed_rows.py` gained a bearer header from `HF_TOKEN` for gated repos and 1-D tensor support (`--tensor model.norm.weight`). Atom norms per layer are cached under the lens's `derived/` (11 s per layer).

**Checks on the real lens and weights (layer 21).** For h = the J-lens vector of ' Paris', ' seven', ' dog', ' yes', ' because' (scaled to norm 200), the pursuit picks that token first with FVE_J = 1.000 in all five cases (unit selection); raw-score selection picks ' dogs' before ' dog' once, a first sign that the normalisation choice matters. Timings on the M1: candidates + readout ≈ 50 ms per activation, pursuit ≈ 70–90 ms, so a 10,000-activation cache takes ≈ 20–25 min per layer, ≈ 2 h for the band. End-to-end CLI run on a synthetic 40-row cache at layers 21 and 23: planted rows FVE_J 1.000 with the right first token, 3-atom mixtures 0.93–0.97, additivity error 1e-6, outputs in the paper layout. **Dictionary geometry worth knowing:** J-lens vector norms run from 0.02 to 11 (median 3.5) and the largest belong to `<end_of_turn>` and `<unused…>` tokens, which therefore dominate the raw readout of any non-activation direction; the atoms share a strong common component, so on isotropic Gaussian inputs the nonnegative pursuit finds few positive correlations (mean 16–18 of 25 steps before filling), the most extreme cosines are negative, and the 25-atom span explains 0.2–0.5% of variance, below the 0.7% of a random 25-dimensional subspace (verified: the first chosen atom is exactly the vocabulary's best positive cosine, 0.0012 vs a negative extreme of 0.0047). Real activations are not isotropic noise; the pre-registered randsub/shuffled/PCA controls remain the right comparison.

**Not done.** Nothing has been run on real activations: the Gemma Core and Transfer caches (`activations_gen.npz`, layers 19–23) are on the RunPod network volume, not on this Mac. Next: copy them down (a short pod in EU-RO-1 with the volume mounted and an HTTP server, a few cents), run `jspace_decompose.py` on Core and Transfer at k = 25 (unit and raw selection), then the arms Jpaper_25 / Cpaper_25 through the probe pipeline with the controls. Uncommitted: this script and its test, the fetch-helper additions, the launcher/estimator fixes, the README rows, the design addendum, this log.

## 2026-11-05 (early, 07:00–09:00 UTC) — The paper's J-space run on the Gemma caches: pulled, decomposed, probed

**Commit.** Tonight's earlier work (decomposition script and test, fetch-helper auth, launcher gates, design addendum, run log) went in as `2046442` on `jspace-separability` at the user's "Commit and continue the work".

**Caches pulled from the RunPod volume** with the new `scripts/runpod_pull_volume_files.py` (cheapest secure-cloud pod in EU-RO-1 with the volume mounted, `python:3.12-slim` image, no GPU work; the pod inventories every `activations_gen.npz`/`metadata_gen.json` on the volume, serves `/workspace` read-only with `/sha256?path=` and `/ls?path=` endpoints and Range support, and is terminated once every checksum verifies; it also terminates itself after `--max-minutes`). First pod `9ph32fuklll0wc` (RTX PRO 4500, ≈$0.10) found nothing because the old scripts' path `/workspace/narcbench-core-demo/data/activations/…` holds no data (that clone is 3.9 MB); the data lives under `/workspace/jobs/narcbench-results/activations/gemma2_9b/…` (and a second copy in `/workspace/jobs/narcbench-core-demo/`, 493 MB). Second pod `fu2kvy935mvcxs` (RTX PRO 4000, $0.57/h) copied Core `20261001T012639Z` (1,700 rows, 70.9 MB) and Transfer `RUNPOD` (936 rows, 39.1 MB) with metadata in 343 s, ≈$0.05, sha256-verified; record `results/jspace/volume_pull_gemma_acts.json`, inventory under `data/volume_pull/_inventory/20261105T073331Z/` (ignored). Only RTX PRO 4500/4000 had supply in EU-RO-1 at 07:00 UTC; balance after: $23.96.

**Decomposition** (`scripts/jspace_decompose.py`, published Neuronpedia lens, k = 25, unit-normalised selection, nonnegative OMP, fill to 25, `--shuffled-seed 0`; outputs under `<cache>/variants/jpaper_nnomp_k25/{j_only,complement,shuffled}`, ignored). ≈0.1 s per activation per layer on the M1; Core 22 min, Transfer 13 min. Summary: `results/jspace/gemma2_9b_jpaper_nnomp_k25/decomp_summary.{md,json}` (`scripts/jspace_decompose_summary.py`, new).

| split | FVE_J L19 | L20 | L21 | L22 | L23 | shuffled-atoms FVE (L19–23) | early stop | steps / filled (mean) |
|---|---|---|---|---|---|---|---|---|
| Core (1,700) | 0.0098 | 0.0097 | 0.0104 | 0.0108 | 0.0128 | 0.0103–0.0132 | 97–100% | 16.4–17.7 / 7.3–8.6 |
| Transfer (936) | 0.0124 | 0.0125 | 0.0132 | 0.0134 | 0.0148 | 0.0126–0.0138 | 77–95% | 18.5–21.0 / 4.0–6.5 |

The paper's J component is 1.0–1.5% of the variance at every layer (random 25-dim subspace: 0.70%; the W_U top-25 frame: 5.7% Core / 5.3% Transfer); every single activation is under the paper's 10% ceiling, and the interquartile range is ±0.001. The nonnegative pursuit stops early on almost every activation (16–21 positive atoms of 25). The chosen atoms are nearly a fixed set: at Core L21 `' decision'` is in 90% of the 1,700 turns' sets, `'<eos>'` 86%, `' prioritizing'` 82%, `' recommendation'`/`' recommendations'` 76%/73%, `' eloquently'` 71%, with multilingual `' تاکید'`, `' diría'`, `' pungkasnya'` at 41–57%; on Transfer L21 `'?'` 72%, `'<eos>'` 64%, `' pungkasnya'` 59%, `':'` 47%, `' discussions'` 45%. The J-lens readout's top-1 token is `'<eos>'` for 98% of Core turns and 70% of Transfer turns (`'<end_of_turn>'` 12%): these are end-of-turn positions. Another turn's atoms explain as much variance as the turn's own (shuffled FVE ≥ own FVE on Core).

**Probe pipeline** (`scripts/jspace_separability.py`, now with `--external-arm NAME=k:CORE:XFER[+C]`; arms `Jpaper` = h_J, `C_Jpaper` = h − h_J, `shuffled_paper` = projection onto a deranged other turn's atoms, all at k = 25 beside the unchanged W_U arms and controls; 2,000 bootstraps, 2,000 permutations, seed 42; 1,052 s). Plumbing check first: a half-scaled copy of the activations entered as an external arm reproduced the full arm to every decimal. Output `results/jspace/gemma2_9b_jpaper_nnomp_k25/{summary.json,RESULTS.md,episode_scores.json}`; per-layer intervals in `arm_layer_cis.{md,json}` (`scripts/jspace_arm_layer_cis.py`, new; same stratified episode bootstrap, post hoc per layer).

*Reproduction.* `full` 0.902 [0.863, 0.935] and the W_U `J` 0.602 [0.550, 0.658] on Core → Transfer equal the frozen 1 October numbers (FROZEN_STATS: 0.902 and 0.602 ± 0.067), so this Mac's pipeline reproduces the pod's.

*A2, Core → Transfer finals, band mean L19–23, k = 25 (72 episodes; 95% bootstrap intervals; permutation p):*

| arm | primary | interval | p | FVE core / xfer |
|---|---|---|---|---|
| full | 0.902 | [0.863, 0.935] | <0.001 | 1 / 1 |
| **Jpaper** | **0.755** | [0.709, 0.796] | <0.001 | 0.011 / 0.013 |
| **C_Jpaper** | **0.891** | [0.850, 0.927] | <0.001 | 0.989 / 0.987 |
| shuffled_paper (another turn's atoms) | 0.559 | [0.511, 0.611] | 0.026 | 0.011 / 0.013 |
| J (W_U top-25) | 0.602 | [0.550, 0.658] | 0.002 | 0.057 / 0.053 |
| C_J | 0.898 | [0.859, 0.931] | <0.001 | |
| randsub | 0.647 | [0.602, 0.693] | <0.001 | 0.007 / 0.007 |
| shuffled (W_U) | 0.549 | [0.503, 0.597] | 0.039 | |
| randrows | 0.466 | [0.413, 0.521] | 0.88 | |
| pca_25 | 0.856 | [0.807, 0.898] | <0.001 | 0.64 / 0.38 |
| contrast | 0.690 | [0.634, 0.742] | <0.001 | |

Paired differences: Jpaper − randsub **+0.108 [+0.052, +0.160]**; Jpaper − shuffled_paper **+0.196 [+0.136, +0.252]**; Jpaper − J(W_U) +0.153 [+0.094, +0.208]; C_Jpaper − Jpaper +0.136 [+0.083, +0.190]; C_Jpaper − full −0.011 [−0.019, −0.006]; for comparison J(W_U) − randsub −0.045 [−0.091, +0.003].

*Per layer (A2 primary, Jpaper vs the two dimension-matched controls):* L19 0.564 | randsub 0.538 | shuffled_paper 0.548; L20 0.709 | 0.696 | 0.589; L21 0.752 | 0.750 | 0.472; L22 0.852 | 0.753 | 0.612; L23 **0.897** | 0.497 | 0.575 (full at L23: 0.873). Jpaper − randsub per layer: +0.03, +0.01, +0.00, +0.10 [+0.01, +0.19], +0.40 [+0.28, +0.51]; Jpaper − shuffled_paper: +0.02, +0.12, +0.28 [+0.15, +0.41], +0.24 [+0.14, +0.34], +0.32 [+0.20, +0.44]. The paper's J component carries nothing beyond a random subspace at L19–21 and a clear transferable signal at L22–23.

*A1 Core leave-one-domain-out:* full 1.000, Jpaper 0.995 (per-fold full − Jpaper 0.001, 1 of 17 folds), C_Jpaper 1.000, J(W_U) 0.989, randsub 0.981, shuffled_paper 0.965. *A3 within-Transfer leave-one-family-out:* full 0.977 [0.963, 0.988], Jpaper 0.854 [0.820, 0.887], C_Jpaper 0.978, shuffled_paper 0.608 [0.562, 0.653], J(W_U) 0.744, randsub 0.742 [0.694, 0.786], pca 0.902, contrast 0.915; Jpaper − randsub +0.112 [+0.060, +0.165], Jpaper − shuffled_paper +0.246 [+0.192, +0.298]. *A4a Core discussion → Core final:* full 0.998, Jpaper 0.930, J(W_U) 0.872, randsub 0.912. *A4b Core → Transfer discussion turns:* full 0.959, Jpaper 0.613 [0.567, 0.657], C_Jpaper 0.956, randsub 0.602, shuffled_paper 0.589, J(W_U) 0.504; Jpaper − randsub +0.010 [−0.043, +0.063].

**Readings, by the rules fixed in the design (section 6), applied to the paper arm.** (1) In distribution, J-space separates colluders (0.995 ≥ 0.90; mean per-fold gap 0.001). (2) The "transferable signal is not in J-space" reading does **not** hold for the paper's J-space: condition (ii) fails outright (Jpaper is above randsub by +0.108 with an interval excluding zero, and above the stronger shuffled-atoms control by +0.196), condition (i) is just missed (C − J = +0.136, interval excludes zero, estimate under 0.15), and (iii) misses zero by a hair but is within the 0.02 tolerance recorded on 31 October (C − full = −0.011). The design's alternative clause applies: **the paper's J-space carries a part of the transferable colluder signal, and the complement carries most of it** (0.891 vs 0.902). This differs from every W_U-frame result in this lane, where J-only sat with the random controls; the W_U frame was a poor stand-in for the Jacobian-lens dictionary on this model. (4) Neither exclusive reading of rule 4: role information is present in the paper's J-space on Transfer finals (0.854 ≥ 0.80) and the Core direction transfers in part (0.755 > 0.65). (5) A4a holds up (0.930) while A4b falls to the random level (0.613 ≈ 0.602), so the J component's transferable signal is specific to final turns: on discussion turns across the family change it carries nothing beyond a random subspace. (6) pca_25 (0.856) and contrast (0.690) do not reach full; dimension still matters.

**Caveats, stated now.** The random fixed subspace is one Haar draw per layer and swings between 0.50 and 0.75 across layers; the shuffled-atoms control (same dictionary, another turn's set, drawn per activation) is the tighter comparison and gives the same verdict. The lens was fitted on WikiText at 128 tokens; it is applied here at the last token of long chat turns, outside its fitting range. Unit-normalised selection and filling to exactly k are design choices (the 5 Nov addendum registers raw selection, plain top-k, gradient pursuit, k ∈ {8, 16, 50} and no-fill as variants, reported, not selected on). The J component is ≈1% of the variance; a probe on it reaching 0.85–0.90 at L22–23 says the signal there sits in a low-variance direction inside the J-lens cone. A mechanism worth testing, not claimed: colluders' and honest agents' final statements differ in which option they recommend, and the J-lens atoms `' decision'`, `' recommendation'`, `' prioritizing'` read exactly that. One pre-registered run, one model.

**Running now (started 08:40 UTC, ≈3.5 h):** scratchpad `jpaper_run/run_jpaper_variants.sh` decomposes Core and Transfer under `--select raw`, `--method topk`, k = 50 and k = 16, then one probe run enters all of them as external arms at their k beside the k = 25 primary → `results/jspace/gemma2_9b_jpaper_variants/` (+ `decomp_summary`). Gradient pursuit (`gp`), k = 8 and `--no-fill` are not queued.

**Not done.** Figures, the Gemma report, the decks and STATUS section 8a's one-line summary still describe the W_U-frame result only; the variant results are pending; nothing is pushed.

## 2026-11-05 (late morning, 09:50–11:45 UTC) — Robustness variants of the paper's J-space on the Gemma caches

**What ran.** The variants registered in the design addendum of 5 November (reported, not selected on), each a new decomposition of the same Core (1,700 rows) and Transfer (936 rows) caches under the published Neuronpedia lens, entered as external arms at their own k into the unchanged probe pipeline (seed 42, 2,000 bootstraps, 2,000 permutations). Batch 1 (`results/jspace/gemma2_9b_jpaper_variants/`, commit 4c575f7, probe run 09:53–10:11 UTC): atom selection by raw score (`Jpaper_raw`), the k largest correlations without pursuit (`Jpaper_topk`), nonnegative OMP at k = 50 (`Jpaper_k50`) and k = 16 (`Jpaper_k16`), beside the k = 25 primary (`Jpaper`, which reproduces the committed run to the digit: 0.755 [0.709, 0.796]). Batch 2 (`results/jspace/gemma2_9b_jpaper_variants_all/`; decompositions 10:12–11:04 UTC, 4–14 min per split on the M1; probe run 11:04–11:24 UTC): gradient pursuit (`Jpaper_gp`), k = 8 (`Jpaper_k8`), the early-stopped pursuit without filling (`Jpaper_nofill`), the nonnegative cone fit Σ cᵢ vᵢ of the primary's own atoms and coefficients in place of the span projection (`Jpaper_cone`, rebuilt from the saved arrays by `scripts/jspace_cone_cache.py`), plus `shuffled_paper`: all ten paper arms with complements in one run, which reproduces every batch 1 number. Post-hoc companions: `scripts/jspace_arm_layer_cis.py --out-stem` (per-layer bootstrap views per k: `arm_layer_cis.md` at k = 25 and `arm_layer_cis_k8.md` in the combined directory, `_k16`, `_k50` and `_k25_variants` in the batch 1 directory) and the new `scripts/jspace_arm_pairs.py` (paired bootstrap differences between arms that sit in different k blocks, same stratified episode bootstrap, seed 43: `arm_pairs.md`, `arm_pairs_vs_full.md`). No GPU, no pod.

**Decomposition statistics** (`decomp_summary.md` in each results directory; FVE_J = ‖h_J‖²/‖h‖², mean over activations; steps = atoms the pursuit chose before stopping; filled = atoms added afterwards to reach k):

| variant | k | FVE_J Core L19→L23 | FVE_J Transfer L19→L23 | steps / filled (Core) | early stop (Core / Transfer) |
|---|---|---|---|---|---|
| primary (unit selection, nnomp, fill) | 25 | 0.0098 → 0.0128 | 0.0124 → 0.0148 | 16.4–17.7 / 7.3–8.6 | 97–100% / 77–95% |
| raw selection | 25 | 0.0120 → 0.0179 | 0.0149 → 0.0209 | 18.3–20.1 / 4.9–6.7 | 91–96% / 71–86% |
| top-k (no pursuit) | 25 | 0.0097 → 0.0127 | 0.0122 → 0.0142 | 25 / 0 | — |
| gradient pursuit | 25 | 0.0098 → 0.0129 | 0.0124 → 0.0148 | 15.6–17.3 / 7.7–9.4 | 97–100% / 80–97% |
| nnomp k = 50 | 50 | 0.0120 → 0.0155 | 0.0149 → 0.0176 | 16.4–17.8 / 32.2–33.6 | 100% / 100% |
| nnomp k = 16 | 16 | 0.0093 → 0.0122 | 0.0119 → 0.0144 | 15.1–15.4 / 0.6–0.9 | 24–38% / 8–18% |
| nnomp k = 8 | 8 | 0.0092 → 0.0120 | 0.0117 → 0.0141 | 8.0 / 0 | 0% / 0% |
| nnomp, no fill | 25 | 0.0093 → 0.0122 | 0.0119 → 0.0144 | 16.4–17.7 / 0 | 97–100% / 77–95% |
| cone fit of the primary | 25 | 0.010 (probe's mean FVE) | 0.013 | the primary's atoms | |

Every variant keeps the J component at 0.9–2.1% of the variance, under the paper's 10% ceiling for every activation. The nonnegative pursuit stops after 16–21 positive atoms whatever k is offered, so k = 50 is the same ≈17 pursuit atoms plus ≈33 filled atoms, k = 16 is ≈15 pursuit atoms with ≈1 filled, k = 8 is the pursuit cut at its first 8 atoms, and the no-fill arm is the primary's pursuit atoms alone (its pursuit statistics equal the primary's). Raw-score selection picks larger-norm atoms (FVE up by a third) and the same frequent tokens. In the cone fit (nonnegative least squares over all 25 atoms) the fill atoms receive exactly zero weight on every activation of both splits (checked at L19, L21 and L23 in `decomp_tokens.npz`: the pursuit stopped because no remaining atom correlated positively with the residual, so the fit cannot use them), and 0.4–0.9% of the pursuit atoms are zeroed as well; the cone arm is therefore the pursuit atoms' nonnegative fit, which is why it tracks the no-fill arm everywhere below.

**A2, Core discussion → Transfer finals, band mean L19–23, 72 episodes** (primary AUROC; 95% bootstrap intervals; `randsub`/`shuffled` are the k-matched W_U-pipeline controls of the same run; the last column pairs each variant with the k = 25 primary, across k blocks where needed, `arm_pairs.md`):

| arm | k | primary | interval | − randsub | − shuffled (W_U) | − Jpaper (k = 25) |
|---|---|---|---|---|---|---|
| full | — | 0.902 | [0.863, 0.935] | | | |
| Jpaper (primary) | 25 | 0.755 | [0.709, 0.796] | +0.108 [+0.052, +0.160] | +0.206 [+0.148, +0.258] | — |
| Jpaper_raw | 25 | 0.740 | [0.695, 0.782] | +0.094 [+0.036, +0.148] | +0.191 [+0.134, +0.248] | −0.014 [−0.052, +0.025] |
| Jpaper_topk | 25 | 0.751 | [0.714, 0.786] | +0.104 [+0.054, +0.153] | +0.202 [+0.151, +0.249] | −0.004 [−0.046, +0.038] |
| Jpaper_gp | 25 | 0.756 | [0.708, 0.801] | +0.109 [+0.054, +0.161] | +0.206 [+0.147, +0.262] | +0.001 [−0.025, +0.026] |
| Jpaper_k50 | 50 | 0.721 | [0.684, 0.757] | +0.082 [+0.030, +0.133] | +0.045 [−0.008, +0.097] | −0.034 [−0.073, +0.005] |
| Jpaper_k16 | 16 | 0.803 | [0.756, 0.844] | +0.202 [+0.141, +0.260] | +0.250 [+0.194, +0.303] | +0.048 [+0.013, +0.085] |
| Jpaper_k8 | 8 | 0.768 | [0.720, 0.809] | +0.196 [+0.143, +0.245] | +0.172 [+0.112, +0.228] | +0.013 [−0.027, +0.055] |
| **Jpaper_nofill** | 25 | **0.811** | [0.761, 0.856] | +0.164 [+0.106, +0.215] | +0.262 [+0.201, +0.315] | **+0.056 [+0.019, +0.093]** |
| **Jpaper_cone** | 25 | **0.809** | [0.761, 0.854] | +0.162 [+0.105, +0.214] | +0.260 [+0.199, +0.315] | **+0.055 [+0.019, +0.091]** |
| shuffled_paper (another turn's atoms) | 25 | 0.559 | [0.511, 0.611] | −0.088 [−0.155, −0.018] | +0.010 [−0.061, +0.084] | −0.196 [−0.252, −0.136] |
| randsub | 8 / 16 / 25 / 50 | 0.572 / 0.600 / 0.647 / 0.638 | | | | |
| J (W_U top-k) | 8 / 16 / 25 / 50 | 0.664 / 0.583 / 0.602 / 0.604 | | | | |

Complements: C_Jpaper 0.891 (− full −0.011 [−0.019, −0.006]), C_Jpaper_raw 0.885 (−0.017 [−0.025, −0.010]), C_Jpaper_topk 0.891 (−0.011 [−0.019, −0.004]), C_Jpaper_gp 0.893 (−0.009 [−0.016, −0.004]), C_Jpaper_k50 0.892 (−0.010 [−0.018, −0.004]), C_Jpaper_k16 0.899 (−0.003 [−0.009, +0.001]), C_Jpaper_k8 0.899 (−0.003 [−0.008, +0.001]), C_Jpaper_nofill 0.899 (−0.003 [−0.008, +0.001]), C_Jpaper_cone 0.899 (−0.003 [−0.008, +0.001]); C − J for the no-fill and cone arms +0.089 [+0.031, +0.148] and +0.090 [+0.033, +0.150] (primary +0.136 [+0.083, +0.190]). Against the same-dictionary control `shuffled_paper` (k = 25): primary +0.196 [+0.136, +0.252], gp +0.196 [+0.133, +0.256], k = 16 +0.243 [+0.178, +0.307], k = 8 +0.208 [+0.140, +0.273], no fill +0.251 [+0.178, +0.320], cone +0.250 [+0.180, +0.319]. Other cross-k pairs: k16 − k50 +0.082 [+0.037, +0.123]; k8 − k16 −0.035 [−0.067, −0.005]; no fill − k16 +0.008 [−0.011, +0.028]; no fill − cone +0.001 [−0.004, +0.006]. Against the larger arms (`arm_pairs_vs_full.md`, post hoc): no fill − full −0.092 [−0.150, −0.036] (primary − full −0.147 [−0.199, −0.095]); no fill − pca_25 −0.045 [−0.107, +0.016]; k16 − pca_16 −0.053 [−0.110, +0.001].

*Per layer (A2 primary, L19 → L23):* Jpaper_nofill 0.710 / 0.799 / 0.723 / 0.882 / **0.940**; Jpaper_cone 0.708 / 0.805 / 0.706 / 0.888 / 0.940; Jpaper_k16 0.689 / 0.804 / 0.680 / 0.892 / 0.947; Jpaper_k8 0.657 / 0.722 / 0.705 / 0.857 / 0.898; Jpaper_gp 0.613 / 0.623 / 0.751 / 0.889 / 0.900; Jpaper_k50 0.614 / 0.564 / 0.728 / 0.828 / 0.870; Jpaper_raw 0.551 / 0.565 / 0.841 / 0.898 / 0.847; Jpaper_topk 0.539 / 0.662 / 0.862 / 0.866 / 0.825; primary 0.564 / 0.709 / 0.752 / 0.852 / 0.897; full 0.884 / 0.885 / 0.941 / 0.930 / 0.873; randsub (k = 25) 0.538 / 0.696 / 0.750 / 0.753 / 0.497. The margin over the random subspace is largest at L23 under every variant: +0.30 to +0.52 (no fill and cone +0.442 [+0.313, +0.558], k = 16 +0.522 [+0.395, +0.632], gp and k = 8 +0.40, k = 50 +0.300 [+0.208, +0.397]). At L21 no variant is above the random level. At L19–20 the arms that carry fill atoms sit at the random level, as the primary did, while the arms without them are above it: at L19 no fill +0.171 [+0.053, +0.280], cone +0.170 [+0.051, +0.276], k = 16 +0.154 [+0.046, +0.262], k = 8 +0.179 [+0.067, +0.287], and over shuffled_paper +0.162 [+0.013, +0.306] (no fill); at L20 k = 8 +0.216 [+0.084, +0.346], no fill +0.103 [−0.010, +0.217] over randsub and +0.210 [+0.061, +0.354] over shuffled_paper. At L23 the no-fill and k = 16 arms, ≈1% of the variance, are level with the full activation (no fill − full +0.067 [−0.005, +0.131]; k16 − full +0.074 [+0.007, +0.137]; primary − full +0.024 [−0.040, +0.088]) and above pca_25 there (+0.132 [+0.047, +0.209]); over the band, full stays 0.09–0.10 above them.

**A3, within-Transfer leave-one-family-out, band mean, 72 episodes:**

| arm | k | primary | interval | − randsub | − shuffled_paper | − Jpaper (k = 25) |
|---|---|---|---|---|---|---|
| full | — | 0.977 | [0.963, 0.988] | | | |
| Jpaper (primary) | 25 | 0.854 | [0.820, 0.887] | +0.112 [+0.060, +0.165] | +0.246 [+0.192, +0.298] | — |
| Jpaper_raw | 25 | 0.796 | [0.753, 0.835] | +0.054 [−0.004, +0.110] | | −0.057 [−0.089, −0.026] |
| Jpaper_topk | 25 | 0.785 | [0.749, 0.821] | +0.044 [−0.011, +0.099] | | −0.068 [−0.108, −0.028] |
| Jpaper_gp | 25 | 0.843 | [0.807, 0.876] | +0.101 [+0.049, +0.156] | +0.235 [+0.186, +0.284] | −0.011 [−0.035, +0.013] |
| Jpaper_k50 | 50 | 0.871 | [0.839, 0.898] | +0.049 [+0.004, +0.099] | | +0.017 [−0.013, +0.049] |
| Jpaper_k16 | 16 | 0.844 | [0.809, 0.881] | +0.107 [+0.059, +0.160] | +0.237 [+0.188, +0.286] | −0.009 [−0.038, +0.019] |
| Jpaper_k8 | 8 | 0.781 | [0.740, 0.818] | +0.128 [+0.073, +0.182] | +0.173 [+0.120, +0.223] | −0.073 [−0.107, −0.040] |
| Jpaper_nofill | 25 | 0.841 | [0.807, 0.876] | +0.099 [+0.051, +0.148] | +0.233 [+0.185, +0.281] | −0.013 [−0.036, +0.009] |
| Jpaper_cone | 25 | 0.840 | [0.805, 0.876] | +0.098 [+0.051, +0.147] | +0.233 [+0.184, +0.281] | −0.013 [−0.038, +0.010] |
| shuffled_paper | 25 | 0.608 | [0.562, 0.653] | −0.134 [−0.190, −0.075] | — | |
| randsub | 8 / 16 / 25 / 50 | 0.653 / 0.737 / 0.742 / 0.822 | | | | |
| C_Jpaper_* | | 0.976–0.978 | − full within ±0.002 | | | |

Over the W_U shuffled control every variant is positive here too: raw +0.136 [+0.087, +0.184], top-k +0.125 [+0.064, +0.182], gp +0.182 [+0.130, +0.236], k = 50 +0.112 [+0.060, +0.164], k = 16 +0.207 [+0.157, +0.259], k = 8 +0.214 [+0.153, +0.271], no fill +0.181 [+0.133, +0.230], cone +0.180 [+0.133, +0.229]. Within Transfer the fill atoms cost nothing (no fill − primary −0.013 [−0.036, +0.009]) and the smaller or cruder frames lose a little (k = 8 −0.073 [−0.107, −0.040], raw −0.057, top-k −0.068 against the primary).

**A4b, Core discussion → Transfer discussion** (band means; the phase-matched, family-changed control): Jpaper 0.613, raw 0.636, top-k 0.673, gp 0.621, k = 50 0.650, k = 16 0.637, k = 8 0.619, no fill 0.625, cone 0.621, shuffled_paper 0.589, against randsub 0.602 (k = 25) / 0.552 (k = 50) / 0.613 (k = 16) / 0.550 (k = 8) and full 0.959; complements 0.954–0.960. Margins over randsub: primary +0.010 [−0.043, +0.063], raw +0.033 [−0.020, +0.086], top-k +0.071 [+0.013, +0.128], gp +0.018 [−0.031, +0.068], k = 50 +0.098 [+0.031, +0.167], k = 16 +0.023 [−0.029, +0.069], k = 8 +0.069 [+0.014, +0.122], no fill +0.023 [−0.025, +0.069], cone +0.018 [−0.028, +0.062]. At the band level the J-space arms' discussion-turn transfer stays at or near the random level under every variant. Every positive band margin and every arm's per-layer excess comes from L23 alone, where each variant is well above the random subspace (no fill 0.832 against randsub 0.461, +0.371 [+0.286, +0.456]; cone +0.363; primary +0.310 [+0.219, +0.402]; k = 8 +0.289; gp +0.270) while at L19–22 the arms are at or below it (several margins negative, e.g. no fill −0.124 [−0.216, −0.024] at L22).

**A4a, Core discussion → Core final:** full 0.998; Jpaper 0.930, raw 0.907, top-k 0.926, gp 0.937, k = 50 0.942, k = 16 0.957, k = 8 0.921, no fill 0.944, cone 0.943; shuffled_paper 0.761; randsub 0.912 (k = 25). **A1, Core leave-one-domain-out:** full 1.000; Jpaper 0.995, raw 0.989, top-k 0.989, gp 0.997, k = 50 1.000, k = 16 0.997, k = 8 1.000, no fill 0.993, cone 0.994 (mean per-fold full − arm 0.000–0.009); shuffled_paper 0.965; complements 1.000. **A5, geometry (centre layer, k = 25):** |cos| between the full probe direction and the direction each arm learns: Jpaper 0.105, cone 0.105, no fill 0.109, gp 0.107, raw 0.107, top-k 0.082, shuffled_paper 0.081, W_U J 0.042; complements 0.845–0.892.

**Readings, by the rules fixed in the design (section 6), applied to each variant.** (1) In distribution every variant separates colluders (A1 0.989–1.000, per-fold gaps under 0.01). (2) The "transferable signal is not in J-space" reading fails under all nine decompositions. Condition (ii) fails everywhere: every paper arm is above its k-matched random subspace with an interval excluding zero (smallest margin k = 50, +0.082 [+0.030, +0.133]) and above the same-dictionary shuffled-atoms control wherever that pair was computed (+0.196 to +0.251). Condition (i), C − J at least 0.15 with an interval excluding zero, is missed narrowly by the primary (+0.136) and clearly by the no-fill and cone arms (+0.089, +0.090). Condition (iii), C − full containing zero, holds outright for the no-fill, cone, k = 16 and k = 8 arms (−0.003, intervals containing zero) and within the 0.02 tolerance of the 31 October amendment for the rest (largest gap raw, −0.017 [−0.025, −0.010]). The alternative clause therefore applies under every variant: **the paper's J-space carries a part of the transferable colluder signal and the complement carries most of it.** How large the part is depends on one choice only. The selection normalisation and the pursuit algorithm do not move it (against the primary: raw −0.014, top-k −0.004, gp +0.001, intervals straddling zero). Filling the early-stopped pursuit to exactly k atoms lowers it: the arms that carry no fill atoms (no fill, cone, k = 16) score 0.80–0.81 against the primary's 0.755 (+0.056 [+0.019, +0.093]), the arm with ≈33 fill atoms (k = 50) scores 0.721 (−0.034 [−0.073, +0.005]), and the pursuit cut at 8 atoms sits between (0.768, +0.013 [−0.027, +0.055]). The transferable part sits in the atoms the nonnegative pursuit chooses; the atoms appended afterwards to reach k add directions that the Core-fit probe uses and that do not transfer. (4) Neither exclusive reading of rule 4 applies under any variant: A3 is far above its permutation null everywhere (0.78–0.87, p < 0.001) and A2 is above 0.65 everywhere (0.72–0.81), so role information is present in the paper's J-space on Transfer finals and the Core direction transfers in part, under every decomposition (three variants, raw 0.796, top-k 0.785 and k = 8 0.781, sit just under the 0.80 figure the first reading uses). (5) A4a 0.91–0.96 against A4b 0.61–0.67 (random 0.55–0.61): the drop is the family change under every variant, and at the band level the J component's cross-family transfer is a final-turn effect, with the L23 exception noted above. (6) pca_25 (0.856) stays above the primary (−0.101 [−0.157, −0.043]) but not above the no-fill arm (−0.045 [−0.107, +0.016]) at the band level, and at L23 the no-fill and k = 16 arms are above it and level with full; dimension still matters at L19–21, where every J-space arm sits well below full.

**Caveats.** The pre-registered arm remains k = 25 with fill, 0.755; the variants are reported beside it, not in its place. The no-fill, cone and k = 16 results are three views of one fact (no fill − k16 +0.008 [−0.011, +0.028]; no fill − cone +0.001 [−0.004, +0.006]), not three confirmations. The paper's appendix minimises over |S| = k exactly, which the fill implements, while its prose describes a nonnegative cone, which the cone arm implements; the two readings of the paper differ by 0.055 on this measure and the cone reading is the stronger one. The cross-k comparisons (k = 8, 16, 50 against 25) and all per-layer statements are post hoc and share the same 72 episodes; the random fixed subspace is one Haar draw per layer and per k and swings between 0.46 and 0.78 across layers, so margins over randsub at different k are not comparable with each other, only the paired differences between paper arms are. "Shuffled" in the A2 table is the W_U-pipeline shuffled control; the same-dictionary control is shuffled_paper, paired only where stated. The L19–20 margins of the no-fill arms rest on intervals that exclude zero narrowly (lower bounds +0.01 to +0.07). The lens was fitted on WikiText at 128 tokens and is applied at the last token of long chat turns. One model, one run per arm, every arm on the same episodes.

**Not done.** Figures, the Gemma report (`papers/jspace-gemma2-9b-2026-11/`), the decks and STATUS section 8a's one-line summary still describe the W_U-frame result only; the "Collusion Signatures Beyond J-Space" deck still says J_ℓ is being estimated the paper's way. Nothing is pushed.
