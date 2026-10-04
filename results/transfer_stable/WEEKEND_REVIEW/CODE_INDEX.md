> **GitHub path map.** In this repository, `/workspace/narcbench-core-demo/` is the repo root. `/workspace/andy-biz/papers/transfer-stable-collusion-2026-10/` is `papers/transfer-stable-collusion-2026-10/`. `/workspace/andy-biz/NARCBench-Weekend-FINAL-HANDOFF-2026-10-01.md` and `NARCBench-Transfer-Stable-Weekend-Review-2026-10-01.md` are under `docs/`. `data/activations/` and other `.npz` caches are gitignored.

# CODE_INDEX — Transfer-stable / NARCBench pipeline (weekend review)

**Written:** 2026-10-01 ~5:41 PM PT  
**Scope:** Code available to review under `/workspace/narcbench-core-demo/` plus the local upstream clone and related runners that this lane actually uses.  
**Rules for this index:** No invented runners. If a GPU/causal redesign is design-only, it is listed under **Missing / not implemented**. Soft-stop: Track I / Paper 2 / interp-demo / family-house untouched. **No GPU** for this documentation pass.

**Canonical copy:** this file.  
**Paper tree pointer:** `/workspace/andy-biz/papers/transfer-stable-collusion-2026-10/CODE_INDEX.md` (symlink).

---

## Pipeline map (how pieces connect)

```
scenarios (Core / Transfer regen)
    → activations extract (residual ± attn/mlp)
        → probes / Core→Transfer eval
            → J-space / complement projection
                → transfer_stable offline (layer / subspace / controls / residualize)
                    → causal steer/ablate (GPU; currently paused)
                        → WEEKEND_REVIEW + paper
```

**Default CPU env for offline probes:** `/workspace/narcbench-core-demo/.venv-probe/bin/python`  
**Repo root for relative commands below:** `cd /workspace/narcbench-core-demo`

**Canonical caches used by Transfer-stable analyses**

| Asset | Path |
|---|---|
| Core residual acts | `data/activations/gemma2_9b/core/20261001T012639Z/` |
| Transfer residual acts | `data/activations/gemma2_9b/transfer/RUNPOD/` |
| Core mlp/attn acts | `data/activations/gemma2_9b/core/20261001T012639Z_mlp_attn/` |
| Transfer mlp/attn acts | `data/activations/gemma2_9b/transfer/RUNPOD_mlp_attn/` |
| Jacobians (L19,21,23) | `data/j_basis/gemma2_9b/jacobians/` |
| Role/mode directions | `results/transfer_stable/directions/` |
| Core transcripts (full) | `upstream/scenarios/gemma2_9b/core/20261001T012639Z/` (100 runs) |
| Transfer transcripts | `upstream/scenarios/gemma2_9b/transfer/RUNPOD/` (72 runs) |

---

## Packages / libraries (not pip-installed as one package)

There is **no** `pyproject.toml` / `setup.py` for `narcbench-core-demo`. Code is scripts + local modules + a cloned upstream repo.

| Path | What it is |
|---|---|
| `generation/` | Local Core regen package: `core_protocol.py` (R0→R1–3→final), `hf_backend.py` (HF Gemma chat), `narcbench_core_bank.json` (50 Core scenarios). |
| `app/` | Presenter: `server.py` + `viewer.html` — replay indexed transcripts (CPU). |
| `.venv-probe/` | CPU probe venv (sklearn/numpy) used by offline probe scripts. |
| `/workspace/narcbench-upstream/repo/` | Clone of `aaronrose227/narcbench`: official `generation/{core,transfer,stego}.py`, `extraction/`, `probes/`, `analysis/`, `config.py`, `scripts/run_model.sh`. |
| `zeabur-deploy/` | Deploy snapshot of presenter + data (not the science pipeline). |
| **Related, not this lane’s runners** | `/workspace/collusion-exp/` (earlier collusion circuit/steer prototypes); `/workspace/jspace-stego-demo/` (stego J-lens demo). Do not treat as Transfer-stable harnesses. |

---

## 1. Scenarios (generation / download / index / presenter)

### Local Core bank + protocol

| Path | What it does | How to run | Inputs → outputs |
|---|---|---|---|
| `generation/core_protocol.py` | Rose-style Core multi-round protocol (private R0 → 3 public rounds → final vote); backend-agnostic `generate(system,user)`. | Imported by regen scripts (not a CLI). | Bank JSON + generate callable → per-run `run_config.json` / `agent_prompts.json` / `results.json` via callers. |
| `generation/hf_backend.py` | HF transformers Gemma-2-IT load + chat generate (system folded into first user turn). | Imported by `regen_gemma_core.py` / RunPod wrappers. | Model id → `HFBundle` + generate fn. **Needs GPU** for real regen. |
| `generation/narcbench_core_bank.json` | 50 Core scenario definitions + name pool. | Loaded by `load_bank()`. | — |
| `scripts/regen_gemma_core.py` | Regenerate Core transcripts on Gemma-2-\*-IT using local `generation/`. | Smoke shape test (no GPU): `python3 scripts/regen_gemma_core.py --smoke --dry-run` · Real smoke (GPU): `python3 scripts/regen_gemma_core.py --smoke --model google/gemma-2-9b-it` · Full: `--all` | Bank + model → `upstream/scenarios/<slug>/core/<run_id>/` |
| `scripts/runpod_launch_gemma_core.py` | Estimate / optionally launch disposable RunPod Core regen. | Estimate (default): `python3 scripts/runpod_launch_gemma_core.py --smoke --model google/gemma-2-9b-it` · Spend: add `--launch` (needs Andy approval). | Launch card / pod status → transcripts on volume then pull. |
| `scripts/download_core_scenarios.py` | Download **official Qwen3-32B Core JSON only** from HF (no activations). | `python3 scripts/download_core_scenarios.py` | HF `aaronrose227/narcbench` → `upstream/scenarios/qwen3_32b/core/` |
| `scripts/build_index.py` | Index Core transcripts into demo JSON indexes. | `python3 scripts/build_index.py` | `upstream/scenarios/*/core/` → `data/demo_index*.json` |
| `app/server.py` | HTTP presenter for indexed cases. | `python3 app/server.py` → `http://127.0.0.1:8765/` | Index JSON → browser replay. |
| `scripts/runpod_launch_presenter.py` | Optional RunPod host for presenter (:8765, safety-capped hours). | `python3 scripts/runpod_launch_presenter.py` (estimate); `--launch` to spend. | Tar of presenter → remote :8765. |

### Transfer regen (wraps upstream)

| Path | What it does | How to run | Inputs → outputs |
|---|---|---|---|
| `/workspace/narcbench-upstream/repo/generation/transfer.py` | Official Transfer scenario generators (6 families; relational collusion). Expects `config` + vLLM-style client in upstream. | Normally via wrapper below, not bare. | → Transfer run dirs. |
| `scripts/regen_gemma_transfer.py` | HF Gemma wrapper around upstream `transfer.py` (installs stub `config`). | Smoke: `python3 scripts/regen_gemma_transfer.py --smoke --model google/gemma-2-9b-it` · Full 72: `--full` | Upstream transfer defs + HF → `upstream/scenarios/gemma2_9b/transfer/<tag>/` (72 runs for full). **Needs GPU.** |
| `scripts/runpod_launch_gemma_transfer.py` | Estimate / launch Transfer regen on RunPod. | `python3 scripts/runpod_launch_gemma_transfer.py` · `--full --launch` when approved. | Pod → Transfer transcripts on volume. |
| `scripts/finish_transfer_full.sh` | Post-Transfer helper: locate RUNPOD transcripts; optionally `--launch-extract`. | `bash scripts/finish_transfer_full.sh` · or `... --launch-extract` | Transcripts count; optional extract launch. |

### Upstream official (reference; used selectively)

| Path | What it does | How to run | Notes |
|---|---|---|---|
| `narcbench-upstream/repo/generation/core.py` | Official Core generator (Qwen/vLLM-oriented, large). | Upstream README / `scripts/run_model.sh` | Local Gemma Core uses `generation/core_protocol.py` instead. |
| `narcbench-upstream/repo/generation/stego.py` | Official stego scenarios. | Upstream | **Not used** by Transfer-stable lane. |
| `narcbench-upstream/repo/scripts/run_model.sh` | Upstream model runner shell. | See upstream README | Reference only. |
| `narcbench-upstream/repo/config.py` | Upstream paths / model defaults. | Imported by upstream scripts | `regen_gemma_transfer.py` stubs a minimal `config` when wrapping. |

---

## 2. Activations (extract)

| Path | What it does | How to run | Inputs → outputs |
|---|---|---|---|
| `scripts/extract_activations_from_transcripts.py` | Last-token extract from finished Core/Transfer runs. Sites: `residual`, `attn`, `mlp` (paper-faithful npz layout). | Dry metadata: `python3 scripts/extract_activations_from_transcripts.py --runs-dir upstream/scenarios/gemma2_9b/core/20261001T012639Z --out /tmp/acts_dry --dry-run` · Real extract **needs GPU** (usually via RunPod launcher). | Runs dir (`run_config`/`agent_prompts`/`results.json`) → `activations_gen.npz` (+ attn/mlp npz) + `metadata_gen.json` (+ `extract_hook_doc.json`). |
| `scripts/runpod_launch_extract_acts.py` | Separate RunPod job for residual (gen) extract. | Estimate: `python3 scripts/runpod_launch_extract_acts.py --runs-rel gemma2_9b/transfer/RUNPOD --layers 19-23 --gen-only` · `--launch` to spend. | Volume acts under jobs/narcbench-results (then pull/sync to `data/activations/...`). |
| `scripts/runpod_launch_extract_mlp_attn.py` | **One** pod: Core then Transfer residual+attn+mlp extract (model loaded once). | `python3 scripts/runpod_launch_extract_mlp_attn.py` · `--launch` when approved. | → `.../core/<id>_mlp_attn/` and `.../transfer/RUNPOD_mlp_attn/`. |
| `scripts/watch_core_then_full_extract.py` | Poll Core regen pod; on done, pull + launch **new** full extract pod (does not attach to Core pod). | `python3 scripts/watch_core_then_full_extract.py` | Pod status → extract launch. |
| `scripts/pull_pod_results.py` | Download `results.tgz` from finished job proxy; optional terminate. | `python3 scripts/pull_pod_results.py --pod <id> --extract-to results/... [--terminate] [--require-done]` | Proxy tarball → local extract tree. |
| `narcbench-upstream/repo/extraction/extract_core.py` | Official Core activation extract. | Upstream CLI | **Not the runner used here**; local extract script mirrors its layout. |
| `narcbench-upstream/repo/extraction/extract_transfer.py` | Official Transfer extract. | Upstream | Same — reference; local script used on Gemma transcripts. |
| `narcbench-upstream/repo/extraction/extract_stego.py` / `extract_token_traces.py` | Official stego / token-trace extract. | Upstream | **Not used** in Transfer-stable lane. |

**Also present (legacy / patching, not the Transfer-stable extract):** `scripts/activation_patch_core.py` — residual activation **patching** restore/scrub on Core runs (GPU). Launch: `scripts/runpod_launch_activation_patch.py`. Separate from role-direction causal steer.

---

## 3. Probes (train / Core→Transfer eval)

| Path | What it does | How to run | Inputs → outputs |
|---|---|---|---|
| `scripts/train_collusion_probe.py` | CPU LR deception direction + Peak / Split / Asymmetry (leave-domain-out when ≥3 domains). | `.venv-probe/bin/python scripts/train_collusion_probe.py --acts-dir data/activations/gemma2_9b/core/20261001T012639Z --out results/stage1/full --layers 19-23 --phase discussion` | Acts npz + metadata → metrics JSON/MD under `--out`. |
| `scripts/eval_transfer_from_core.py` | **Paper protocol:** train direction on Core; zero-shot Peak/Split/Asym on Transfer. | `.venv-probe/bin/python scripts/eval_transfer_from_core.py --core-acts data/activations/gemma2_9b/core/20261001T012639Z --transfer-acts data/activations/gemma2_9b/transfer/RUNPOD --out results/transfer/FULL_core_to_transfer_full --layers 19-23 --core-phase discussion --transfer-phase final` | Core+Transfer acts → HARD metrics under `--out`. |
| `scripts/mlp_attn_probes_offline.py` | Core→Transfer probes comparing residual vs attn vs mlp sites (CPU). | `.venv-probe/bin/python scripts/mlp_attn_probes_offline.py --core-acts data/activations/gemma2_9b/core/20261001T012639Z_mlp_attn --transfer-acts data/activations/gemma2_9b/transfer/RUNPOD_mlp_attn --residual-core data/activations/gemma2_9b/core/20261001T012639Z --residual-transfer data/activations/gemma2_9b/transfer/RUNPOD --out results/transfer_stable/MLP_ATTN_PROBES.json` | mlp/attn (+ optional residual) dirs → `MLP_ATTN_PROBES.json` / `.md`. |
| `scripts/run_transfer_full_probes.sh` | Batch: full acts eval + W_U/jac k25 project + J-only/complement Transfer evals. | `bash scripts/run_transfer_full_probes.sh` | Uses default Core/Transfer act paths → `results/transfer/FULL_*`. |
| `scripts/compare_stage3.py` | Compare full vs J-only vs complement probe AUROCs. | `.venv-probe/bin/python scripts/compare_stage3.py --full <path> --j <path> --c <path> --out <path>` | Three metric dirs → comparison JSON/MD. |
| `narcbench-upstream/repo/probes/reproduce*.py` | Official probe variants (LR, LAT, diffMeans, mlp, multiHead, multiLayer) + `roc_analysis.py`, `comparison_plot.py`. | See upstream README | Reference implementations; local `train_collusion_probe.py` / `eval_transfer_from_core.py` are what this lane ran. |

---

## 4. J-space / projection / Jacobians

| Path | What it does | How to run | Inputs → outputs |
|---|---|---|---|
| `scripts/project_j_complement.py` | Project residuals into J-only vs complement using **W_U top-k** logit-lens proxy (CPU). | `.venv-probe/bin/python scripts/project_j_complement.py --acts-dir data/activations/gemma2_9b/core/20261001T012639Z --k 25 --layers 19-23` | Acts → sibling `j_only/` + `complement/` (or `--out-root`). |
| `scripts/project_j_variants.py` | Multiple defs: `wu_topk`, `jac_topk` (needs J_ℓ), `wu_fixed`. | `.venv-probe/bin/python scripts/project_j_variants.py --acts-dir data/activations/gemma2_9b/transfer/RUNPOD --variant wu_topk --k 25 --layers 19-23 --out-root data/activations/gemma2_9b/transfer/RUNPOD/variants/wu_topk_k25` · jac: add `--variant jac_topk --jac-dir data/j_basis/gemma2_9b/jacobians --layers 19,21,23` | Acts (+ optional J npy) → `variants/<tag>/{j_only,complement}/`. |
| `scripts/estimate_jacobian_jlens.py` | Estimate per-layer average Jacobians J_ℓ (randomized JVP/VJP sketch). | GPU: `python3 scripts/estimate_jacobian_jlens.py --model google/gemma-2-9b-it --layers 19,21,23 --n-prompts 16 --rank 128 --out data/j_basis/gemma2_9b/jacobians` | Prompts + model → `J_layer_*_fp16.npy` + meta. **Needs GPU.** |
| `scripts/runpod_launch_jacobian.py` | Estimate / launch Jacobian job on RunPod. | `python3 scripts/runpod_launch_jacobian.py` · `--launch` when approved. | → jacobians on volume / `data/j_basis/...`. |
| `scripts/run_stage2_jacobian_battery.py` | CPU battery: project + probe grid over W_U / jac cells. | `.venv-probe/bin/python scripts/run_stage2_jacobian_battery.py --wu-only` · or `--jac-only --jac-dir data/j_basis/gemma2_9b/jacobians` | Core acts + variants → `results/stage2_jacobian/<tag>/`. |
| `scripts/watch_jac_and_xfer.py` | Poll Jacobian + Transfer smoke pods; remind pull paths. | `python3 scripts/watch_jac_and_xfer.py` | Status prints only. |
| `data/j_basis/gemma2_9b/` | Cached unembed + norms + jacobians used by projection. | — | Inputs to project scripts. |

---

## 5. Causal (steer / ablate / patch)

| Path | What it does | How to run | Inputs → outputs |
|---|---|---|---|
| `scripts/causal_steer_transfer.py` | Causal steer/ablate of Transfer-stable role directions on Gemma-2-9B-it (residual / attn / mlp writes; final and/or discussion phases). Suites: `default`, `alpha_sweep`, `core_smoke`, `mlp_attn_smoke`, `discussion_attn_smoke`. | Intended on GPU via launcher (below). Direct: `python3 scripts/causal_steer_transfer.py --runs-dir <transfer_runs> --directions results/transfer_stable/directions --out results/transfer_stable/<run> --suite discussion_attn_smoke --layers 21,22` | Runs + direction `.npy` → per-suite results JSON/MD + transcripts of regen. **Needs GPU. Currently paused** pending matched-prefix redesign. |
| `scripts/runpod_launch_causal_steer.py` | Estimate / launch causal suites on RunPod (incl. `alpha_core`, mlp/attn, discussion). | `python3 scripts/runpod_launch_causal_steer.py --suite discussion_attn_smoke` · `--launch` when approved. | Estimate/launch cards under `results/transfer_stable/` + volume pulls. |
| `scripts/activation_patch_core.py` | Earlier Core residual **activation patch** restore/scrub (vote flips). | GPU via `runpod_launch_activation_patch.py` or direct with `--runs-dir` / `--out`. | Core runs → patch metrics under `--out`. **Not** the Transfer-stable role-direction suite. |
| `scripts/within_traj_mid_private.py` | Teacher-forced within-trajectory mid-private intervene on Core (attn L22 + residual L21). `option_labels` may be a list or a dict. | `python3 scripts/within_traj_mid_private.py --dry-run --max-scenarios 6` (needs direction `.npy`, not in git). GPU via the launcher below. | Core run + directions → continuation edit + vote metrics under `--out`. |
| `scripts/runpod_launch_within_traj.py` | Estimate / launch that smoke. Create body is a tiny boot that checks out a git SHA. No job tarball in GraphQL. | `python3 scripts/runpod_launch_within_traj.py` · `--launch` only after the cost gate. | Results on the network volume under `jobs/narcbench-results/within_traj_mid_private/`. |
| `scripts/runpod_launch_matched_prefix.py` | Estimate / launch matched-prefix interchange. Same tiny git-SHA create. Refuses `--launch` while `scripts/matched_prefix_interchange.py` is absent from the SHA. | `python3 scripts/runpod_launch_matched_prefix.py` | Results on the volume under `jobs/narcbench-results/matched_prefix/`. |

**Results already on disk (for review, not re-run):**  
`results/transfer_stable/CAUSAL_*`, `MLP_ATTN_CAUSAL_*`, `DISCUSSION_PHASE_CAUSAL_*`, `NEGATIVE_CAUSAL_APPENDIX.md`, and pulled smoke dirs.

---

## 6. Analysis / Transfer-stable / paper

### Offline Transfer-stable analysis (CPU)

| Path | What it does | How to run | Inputs → outputs |
|---|---|---|---|
| `scripts/transfer_stable_offline.py` | Layer sweep L19–23, subspaces (diff-means / LR / PCA), shared direction vs full/J baselines. | `.venv-probe/bin/python scripts/transfer_stable_offline.py --core-acts data/activations/gemma2_9b/core/20261001T012639Z --transfer-acts data/activations/gemma2_9b/transfer/RUNPOD --out results/transfer_stable --layers 19-23` | Acts → `layer_sweep.json`, `subspace_and_direction.json`, MD summaries. |
| `scripts/transfer_stable_controls.py` | Role vs mode directions, cross-test, PCA-contrast. | `.venv-probe/bin/python scripts/transfer_stable_controls.py --core-acts .../core/20261001T012639Z --transfer-acts .../transfer/RUNPOD --out results/transfer_stable/controls_offline.json --layers 21,22` | Acts → `controls_offline.json` + `CONTROLS_OFFLINE.md`. |
| `scripts/transfer_stable_residualize.py` | Residualize mode⊥role / role⊥mode; export intervention directions. | `.venv-probe/bin/python scripts/transfer_stable_residualize.py --core-acts .../core/20261001T012639Z --transfer-acts .../transfer/RUNPOD --out results/transfer_stable/residualize_mode.json --layers 21,22` | Acts → `residualize_mode.json` + `.npy` under `directions/`. |
| `results/transfer_stable/WEEKEND_REVIEW/scripts/matched_prefix_scaffold.py` | **Dry-run only:** list Core stems for matched-prefix micro-worlds (no GPU). | `python3 results/transfer_stable/WEEKEND_REVIEW/scripts/matched_prefix_scaffold.py` | Core run dir → `design/matched_prefix_candidates.json`. |

### Docs / metrics / packs (read, not run)

| Path | Role |
|---|---|
| `results/transfer/TRANSFER_FULL_HARD_METRICS.md` | HARD n=72 full / J / complement AUROCs. |
| `results/transfer_stable/*.md` / `*.json` | Layer sweep, controls, residualize, mlp/attn probes, causal results, claim notes. |
| `results/transfer_stable/WEEKEND_REVIEW/` | Consolidated study pack (`00`–`07`, CLAIMS_*, tables/, figures/, design/). |
| `docs/NEXT-SCIENCE.md` | Science follow-ups / methods ladder notes. |
| `docs/RUNPOD.md` | RunPod estimate/launch conventions. |
| `/workspace/andy-biz/papers/transfer-stable-collusion-2026-10/` | LaTeX + `main.pdf` (weekend review PDF). Build: `pdflatex main && bibtex main && pdflatex main && pdflatex main`. |
| `/workspace/andy-biz/NARCBench-Weekend-FINAL-HANDOFF-2026-10-01.md` | FINAL handoff paths + soft-stops. |

### Upstream analysis (reference)

| Path | Role |
|---|---|
| `narcbench-upstream/repo/analysis/stego_correlations.py` | Stego correlations — **not** Transfer-stable. |

---

## 7. Missing / not implemented (do not invent)

These are called out so weekend review does not assume a runner exists:

| Desired capability | Status |
|---|---|
| **Matched-prefix causal interchange harness** | **GPU runner still absent** (`scripts/matched_prefix_interchange.py`). Slim launcher `scripts/runpod_launch_matched_prefix.py` checks out a git SHA and refuses `--launch` until that file is in the SHA. Specs + CPU scaffold remain under `WEEKEND_REVIEW/`. |
| **Within-trajectory causal runner** | **Present:** `scripts/within_traj_mid_private.py`. Launcher `scripts/runpod_launch_within_traj.py` checks out a git SHA (estimate unless `--launch`). Direction `.npy` files stay on the network volume, not in git. Spec: `WEEKEND_REVIEW/design/within_trajectory_spec.md`. |
| **Checkpoint-assay runner** | **Missing.** Spec only: `WEEKEND_REVIEW/design/checkpoint_assays_spec.md`. |
| Official upstream `extract_core.py` / `extract_transfer.py` on Gemma | **Not wired** as the lane’s launcher; local `extract_activations_from_transcripts.py` is what ran. |
| Official upstream vLLM Core regen for Gemma | **Not used**; local `generation/core_protocol.py` + HF. |
| Installable `narcbench-core-demo` Python package | **None** (scripts + `generation/` on `PYTHONPATH` / `sys.path` inserts). |
| Further free-regen residual-α chasing | **Paused by policy** (causal GPU paused); harness exists but should not be re-launched under soft-stop. |

---

## 8. Soft-stops / ops notes

- **No GPU** for this CODE_INDEX write; do not `--launch` RunPod from review alone.
- **Track I / Paper 2 / interp-demo / family-house:** soft-stopped — not part of this tree’s edits.
- RunPod secrets (if ever launching): `/home/box/.secrets/interp-explorer.env` (see `docs/RUNPOD.md`). Shared GraphQL helper historically: `/workspace/collusion-exp/scripts/runpod_graphql.py`.
- Volume paths seen in launchers: `/workspace/jobs/narcbench-results/` (acts, causal, upstream sync).

---

## Quick “start reviewing code” order

1. `generation/core_protocol.py` + `scripts/regen_gemma_core.py` / `regen_gemma_transfer.py`  
2. `scripts/extract_activations_from_transcripts.py`  
3. `scripts/eval_transfer_from_core.py` + `train_collusion_probe.py`  
4. `scripts/project_j_variants.py` + `estimate_jacobian_jlens.py`  
5. `scripts/transfer_stable_offline.py` → `transfer_stable_controls.py` → `transfer_stable_residualize.py` → `mlp_attn_probes_offline.py`  
6. `scripts/causal_steer_transfer.py` (+ launcher) — read for nulls; do not launch  
7. `WEEKEND_REVIEW/` narrative + `CLAIMS_NUMBER_SHEET.md` + paper `main.tex`
