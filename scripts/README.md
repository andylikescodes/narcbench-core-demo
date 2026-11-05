# Scripts in this repository

Commands below are the ones in [`CODE_INDEX.md`](../CODE_INDEX.md). Run them from the repo root. Probe scripts expect activation caches under `data/activations/`, which are gitignored and are not in this checkout. Do not pass `--launch` on the RunPod scripts from a review clone.

## Presenter (CPU, no activations)

```bash
python3 scripts/build_index.py
python3 app/server.py
```

`build_index.py` reads `upstream/scenarios/<model>/core/` and writes `data/demo_index.json`, `data/models_index.json`, and the per-model indexes. `app/server.py` serves the walkthrough on port 8765 unless `PORT` is set. Transfer FULL hard metrics already on disk are shown at `/?view=transfer`.

## Scenario regen

| Script | How CODE_INDEX says to run it |
|---|---|
| `scripts/regen_gemma_core.py` | `python3 scripts/regen_gemma_core.py --smoke --dry-run` (no GPU). Real regen needs a GPU. |
| `scripts/regen_gemma_transfer.py` | `python3 scripts/regen_gemma_transfer.py --smoke --model google/gemma-2-9b-it` (GPU) |
| `scripts/download_core_scenarios.py` | `python3 scripts/download_core_scenarios.py` |
| `generation/core_protocol.py`, `generation/hf_backend.py`, `generation/narcbench_core_bank.json` | Imported by the regen scripts |

## Activations, probes, J-space, Transfer-stable analysis

| Script | How CODE_INDEX says to run it |
|---|---|
| `scripts/extract_activations_from_transcripts.py` | `python3 scripts/extract_activations_from_transcripts.py --runs-dir upstream/scenarios/gemma2_9b/core/20261001T012639Z --out /tmp/acts_dry --dry-run` |
| `scripts/train_collusion_probe.py` | `.venv-probe/bin/python scripts/train_collusion_probe.py --acts-dir data/activations/gemma2_9b/core/20261001T012639Z --out results/stage1/full --layers 19-23 --phase discussion` |
| `scripts/eval_transfer_from_core.py` | `.venv-probe/bin/python scripts/eval_transfer_from_core.py --core-acts data/activations/gemma2_9b/core/20261001T012639Z --transfer-acts data/activations/gemma2_9b/transfer/RUNPOD --out results/transfer/FULL_core_to_transfer_full --layers 19-23 --core-phase discussion --transfer-phase final` |
| `scripts/mlp_attn_probes_offline.py` | `.venv-probe/bin/python scripts/mlp_attn_probes_offline.py --core-acts data/activations/gemma2_9b/core/20261001T012639Z_mlp_attn --transfer-acts data/activations/gemma2_9b/transfer/RUNPOD_mlp_attn --residual-core data/activations/gemma2_9b/core/20261001T012639Z --residual-transfer data/activations/gemma2_9b/transfer/RUNPOD --out results/transfer_stable/MLP_ATTN_PROBES.json` |
| `scripts/run_transfer_full_probes.sh` | `bash scripts/run_transfer_full_probes.sh` |
| `scripts/compare_stage3.py` | `.venv-probe/bin/python scripts/compare_stage3.py --full <path> --j <path> --c <path> --out <path>` |
| `scripts/project_j_complement.py` | `.venv-probe/bin/python scripts/project_j_complement.py --acts-dir data/activations/gemma2_9b/core/20261001T012639Z --k 25 --layers 19-23` |
| `scripts/project_j_variants.py` | `.venv-probe/bin/python scripts/project_j_variants.py --acts-dir data/activations/gemma2_9b/transfer/RUNPOD --variant wu_topk --k 25 --layers 19-23 --out-root data/activations/gemma2_9b/transfer/RUNPOD/variants/wu_topk_k25` |
| `scripts/transfer_stable_offline.py` | `.venv-probe/bin/python scripts/transfer_stable_offline.py --core-acts data/activations/gemma2_9b/core/20261001T012639Z --transfer-acts data/activations/gemma2_9b/transfer/RUNPOD --out results/transfer_stable --layers 19-23` |
| `scripts/transfer_stable_controls.py` | `.venv-probe/bin/python scripts/transfer_stable_controls.py --core-acts data/activations/gemma2_9b/core/20261001T012639Z --transfer-acts data/activations/gemma2_9b/transfer/RUNPOD --out results/transfer_stable/controls_offline.json --layers 21,22` |
| `scripts/transfer_stable_residualize.py` | `.venv-probe/bin/python scripts/transfer_stable_residualize.py --core-acts data/activations/gemma2_9b/core/20261001T012639Z --transfer-acts data/activations/gemma2_9b/transfer/RUNPOD --out results/transfer_stable/residualize_mode.json --layers 21,22` |
| `results/transfer_stable/WEEKEND_REVIEW/scripts/matched_prefix_scaffold.py` | `python3 results/transfer_stable/WEEKEND_REVIEW/scripts/matched_prefix_scaffold.py` |

`estimate_jacobian_jlens.py` and the `runpod_launch_*.py` / `pull_pod_results.py` / `watch_*.py` / `finish_transfer_full.sh` helpers are in `scripts/` as well. Estimate mode is the default; `--launch` spends money and needs `RUNPOD_API_KEY` outside the repo.

## Causal

`scripts/causal_steer_transfer.py` and `scripts/activation_patch_core.py` are present. `CODE_INDEX.md` marks the causal GPU battery as paused.

Within-trajectory mid-private (teacher-forced) is `scripts/within_traj_mid_private.py`. Offline shape check:

```bash
python3 scripts/within_traj_mid_private.py --dry-run --max-scenarios 6
```

That dry-run still needs direction `.npy` files under `results/transfer_stable/directions/` (those files are not in git). The slim launchers are estimate-only unless `--launch` is passed. A create checks out a public git SHA and runs the smoke script that lives in this repo (`scripts/runpod_within_traj_boot.sh` or `scripts/runpod_matched_prefix_boot.sh`). The GraphQL env is the SHA plus volume paths. It does not contain `JOB_BOOT_B64`, `JOB_PAYLOAD_B64`, or a job tarball. Direction vectors stay on the volume. See [`MIGRATION_HANDOFF.md`](../MIGRATION_HANDOFF.md).

```bash
python3 scripts/runpod_launch_within_traj.py
python3 scripts/runpod_launch_matched_prefix.py
```

Matched-prefix interchange is `scripts/matched_prefix_interchange.py`. Cheap extras (multi-site, PCA, role-perpendicular patch):

```bash
python3 scripts/runpod_launch_matched_prefix.py --extras-only
```

Role-perp confirm writes at each prompt's last token: a full residual copy, the perp dir_patch both ways, perp vs role ablate, role and same-norm random controls, and PCA. h2c is scored against the colluder baseline and c2h against the honest baseline. It scores every row of the volume pairs file (`--max-pairs 0`):

```bash
python3 scripts/runpod_launch_matched_prefix.py --role-perp-confirm
```

Final-residual controls are a separate suite. They copy the pre-logit residual (the hidden state the lm_head reads) on the first 12 core pairs: last token first, then the private-instruction span only if that copy passes. No direction files:

```bash
python3 scripts/runpod_launch_matched_prefix.py --final-resid-controls
```

Final-site direction card (added 2026-10-04, open question 1 of the migration handoff): at that same final-norm last-token site, transplant one direction's component both ways next to the full-residual ceiling. Directions: role L21 and role-perp L21 from the volume (required), mode L21, attn L22 and contrast-PCA k8 (used if present), a same-norm random vector, and a leave-one-pair-out difference of means measured at the site. Each direction also gets a project-out ablation on the colluder prompt:

```bash
python3 scripts/runpod_launch_matched_prefix.py --final-resid-directions
```

Last-token layer sweep (open question 2): copy the whole last-token residual both ways at several depths to find the first depth whose copy carries the vote. No direction files:

```bash
python3 scripts/runpod_launch_matched_prefix.py --last-token-layer-sweep --sweep-layers 24,27,30,33,36,39,41,final
```

Private-span patch (open question 3): copy the private-note span residuals, the last-token residual, and both together, both ways, at residual, attention or MLP sites or the final norm. This is the valid replacement for the withdrawn Oct 1 design, whose last-shared-token patch was zero by construction. No direction files:

```bash
python3 scripts/runpod_launch_matched_prefix.py --private-span-patch --span-sites 21,attn22,final
```

`--pairs-on-volume /workspace/...` points any card at a different pairs file on the volume. `--pairs-from-repo data/matched_prefix/pairs_core_v2.jsonl` has the pod copy that file from the pinned checkout onto the volume first (the destination defaults to `/workspace/jobs/narcbench-data/matched_prefix/<name>`), so the 50-pair widen set never has to be uploaded by hand:

```bash
python3 scripts/build_matched_prefix_pairs.py --all-letter-targets --out data/matched_prefix/pairs_core_v2_all50.jsonl
python3 scripts/runpod_launch_matched_prefix.py --final-resid-directions --pairs-from-repo data/matched_prefix/pairs_core_v2_all50.jsonl --max-pairs 0
```

While a pod runs, the boot serves `/status.json`, `/job.log`, `/cmd.log` and `/out/{meta.json,summary.json,per_pair.json,RESULTS.md}` on port 8765 (RunPod proxy `https://<pod>-8765.proxy.runpod.net`), and appends the result files to `job.log` at the end. `scripts/watch_matched_prefix_pod.py` polls a pod, saves everything into a handoff directory, and can terminate the pod once the files are safe:

```bash
python3 scripts/watch_matched_prefix_pod.py --card results/transfer_stable/MATCHED_PREFIX_LAYER_SWEEP_LAUNCH.json --dest docs/handoff/<date>-layer-sweep --follow --poll 60 --terminate-after-fetch
python3 scripts/verify_matched_prefix_results.py --run-dir docs/handoff/<date>-layer-sweep
```

Those estimates check out a git SHA and read direction `.npy` files from the network volume. Checkpoint assay remains a spec under `results/transfer_stable/WEEKEND_REVIEW/design/`.

## Pairs, verification (CPU, nothing outside git)

| Script | What it does |
|---|---|
| `scripts/build_matched_prefix_pairs.py` | Rebuilds matched-prefix role-flip pairs from the Core transcripts: the 12 recorded scenarios by default (`data/matched_prefix/pairs_core_v2.jsonl`, committed), or `--all-letter-targets` for all 50. Character-level checks always; `--tokenizer google/gemma-2-9b-it` adds the token-level prefix check where the tokenizer is available. |
| `scripts/verify_matched_prefix_results.py` | Re-derives every by-arm statistic, delta and gate of the matched-prefix runs in git from their per-pair records, and shows that the withdrawn writeups' "effects" equal their own untouched baselines. Writes `results/transfer_stable/VERIFICATION_2026-10-04.md` with `--out`. |
| `scripts/verify_claims_sheet.py` | Cross-checks `CLAIMS_NUMBER_SHEET.json` against `WEEKEND_REVIEW/tables/`, the raw probe metrics under `results/transfer/`, the HARD markdown table, and the paper source. |

Both verifiers run inside the test suite (`python3 -m unittest discover -s tests`).

Earlier `*.bak*` copies of the extract and causal harnesses were removed on 2026-10-04; they remain in git history at commit `8eaca99`.

## J-space separability (CPU; official NARCBench activations or any cache in the paper layout)

Pre-registered design: [`docs/jspace/JSPACE_SEPARABILITY_DESIGN.md`](../docs/jspace/JSPACE_SEPARABILITY_DESIGN.md). Results: `results/jspace/`.

| Script | What it does |
|---|---|
| `scripts/download_narcbench_activations.py` | Downloads the paper's own activations (`aaronrose227/narcbench` on Hugging Face; Core and Transfer for `qwen3_32b`, `deepseek_r1_32b`, `gpt_oss_20b`; 14 to 210 MB per split) into `data/activations/official/` (gitignored). |
| `scripts/fetch_unembed_rows.py` | Fetches only the `lm_head.weight` tensor of a model by HTTP range requests on its safetensors shard and writes `data/j_basis/<tag>/unembed_fp16.npy` + row norms (gitignored). `python3 scripts/fetch_unembed_rows.py Qwen/Qwen3-32B-AWQ data/j_basis/qwen3_32b_awq` |
| `scripts/jspace_separability.py` | The analysis: J-only and complement at k = 8, 16, 25, 50, 100 next to dimension-matched controls (random fixed subspace, another activation's frame, random unembedding rows, PCA, supervised contrast), Core leave-one-domain-out (A1), Core → Transfer (A2), within-Transfer leave-one-family-out (A3), phase-versus-family split (A4), probe geometry (A5); bootstrap intervals and permutation tests over episodes; the readings fixed in the design. Writes `summary.json`, `RESULTS.md`, `episode_scores.json`. `--external-arm NAME=k:CORE_DIR:XFER_DIR[+C]` enters a precomputed, row-aligned arm (e.g. `jspace_decompose.py`'s `j_only/` or `shuffled/`) at frame size k beside the built-in arms, with the same probes, bootstrap and permutation inference; `+C` adds its complement. |
| `scripts/jspace_offline_stats.py` | Hanley–McNeil intervals and contrasts for the frozen Gemma-2-9B numbers, from `results/transfer/FULL_core_to_transfer_*/metrics.json` alone → `results/jspace/gemma2_9b_frozen/`. |
| `scripts/jspace_figures.py` | Figures from the summaries → `results/jspace/figures/`. |
| `scripts/jspace_gemma_figures.py` | Gemma-only figures from `results/jspace/gemma2_9b_frozen/frozen_stats.json` → `results/jspace/gemma2_9b_frozen/figures/` and the Gemma paper's `figures/`; since 5 Nov also the paper's J-space figures (arms with intervals, per layer, the nine variants) from the pre-registered run summaries → `results/jspace/gemma2_9b_jpaper_nnomp_k25/figures/` and the paper's `figures/`. |
| `scripts/build_scenario_browser.py` | Packages the Gemma-2-9B Core and Transfer transcripts (briefing, system prompts, private round, public rounds, finals, outcome) into `data/{index,core,transfer}.json` for the scenario-browser artifact: `python3 scripts/build_scenario_browser.py --out <dir>`. |
| `scripts/estimate_jlens_full.py` | The Jacobian lens as Gurnee et al. define it: J_ℓ = E over prompts and over every pair of positions of ∂h_N/∂h_ℓ, with h_N the residual after the last block before the final norm and BOS excluded. Two-pass randomized sketch of the mean Jacobian (batched forward-mode JVPs for the range, batched VJPs for the rows), 1,000 prompts (wikitext-103 paragraphs and alpaca turns in the Gemma chat template), rank 256, layers 19–23; checkpoints and `--resume`; writes `J_layer_{L}_fp16.npy`, the factors, `prompts.json` and `jlens_meta.json` with a held-out self-check. GPU job. |
| `scripts/test_jlens_estimator_tiny.py` | CPU check of the estimator against an exact Jacobian on a random three-block Gemma-2 (d = 32): causality, the all-positions JVP and VJP tricks in every mode, and the full-rank two-pass sketch. All pass on 4 Nov 2026 (relative error below 1e-6). |
| `scripts/runpod_launch_jlens_full.py` | Self-contained RunPod launcher for the estimator. Estimate-only by default; `--preflight` runs the launch gates read-only (balance ≥ $20, no billable pod already running, each GPU type’s 13.75 h worst case must leave $5 of the balance); `--launch` spends (`RUNPOD_API_KEY` from the environment; `HF_TOKEN` only if the weights are not cached on the network volume). On the pod: a timing smoke, then auto-sizing (rank 256 → 128, prompts 1000 → 600 → 400) to fit 90% of `--max-minutes`, checkpoints to the volume, `results.tgz` on port 8765, self-termination after the pull window. `--status`, `--pull` and `--terminate` take a pod id. |
| `scripts/jspace_decompose.py` | The paper's J-space split of recorded activations against a Jacobian lens in Anthropic's `jlens` save format (the published Neuronpedia lens for `gemma-2-9b-it` by default). Readout softcap(W_U · norm(J_ℓ h)) with Gemma's (1 + γ) RMSNorm scale, as `jlens.hf.unembed`; dictionary = J-lens vectors v_i = J_ℓᵀ(s ⊙ W_U[i]) (rows of W_U diag(s) J_ℓ); per activation a greedy nonnegative pursuit over the top-1024 candidates picks k = 25 atoms (`--method nnomp`, or `gp` gradient pursuit, or plain `topk`; atoms compared unit-normalised by default, `--select raw` for raw scores); h_J = orthogonal projection onto their span, h_C = h − h_J. Writes `j_only/`, `complement/` and `shuffled/` (each activation projected onto another activation's chosen atoms, the design's shuffled-frame control; `--shuffled-seed -1` disables) in the paper layout plus the chosen tokens, coefficients, readout top-25 and FVE per activation. CPU; needs `data/j_basis/gemma2_9b/` (unembedding, row norms, `final_norm_weight_fp32.npy`) and the lens under `data/j_basis/neuronpedia/`. |
| `scripts/test_jspace_decompose_synthetic.py` | Synthetic checks for the decomposition (seconds, CPU): planted-atom recovery with exact coefficients, orthogonality and additivity of the split, readout order = raw-score order, `rms_norm` = transformers' `Gemma2RMSNorm`, chunked atom norms, end-to-end batch. |
| `scripts/runpod_pull_volume_files.py` | Copies files off the RunPod network volume with a short, cheap pod (no GPU work): the cheapest secure-cloud type on offer in EU-RO-1 mounts the volume, serves the requested paths with a sha256 manifest over its HTTP proxy (Range-capable, so downloads resume), and is terminated after every checksum verifies; it also terminates itself after `--max-minutes`. Plan-only by default; `--launch` spends (`RUNPOD_API_KEY` from the environment; gates: balance ≥ $20, no billable pod running). `--spec gemma_acts` = the Gemma-2-9B Core and Transfer caches into `data/activations/gemma2_9b/`; `--path REMOTE[=LOCAL]` adds files. Record in `results/jspace/volume_pull_<spec>.json`. |
| `scripts/jspace_decompose_summary.py` | Summarises `jspace_decompose.py` runs for the run log and the paper: per split and layer the FVE_J distribution, the shuffled-control FVE, pursuit steps and filled atoms, FVE_J by mode / phase / role, and the most chosen atoms decoded with a `tokenizer.json`. `--variant LABEL=DIR` (repeatable) → `<out>/decomp_summary.{json,md}`. |
| `scripts/jspace_arm_layer_cis.py` | Post-hoc per-layer view of a `jspace_separability.py` run: the same stratified episode bootstrap (B = the run's, seed = run seed + 1) applied per layer and to extra paired differences (e.g. `Jpaper-randsub`, `Jpaper-shuffled_paper`) for A2, A3 and A4b, from `episode_scores.json` → `<run>/arm_layer_cis.{json,md}` (`--out-stem` for a second view of the same run, e.g. another k). |
| `scripts/jspace_arm_pairs.py` | Post-hoc paired differences between arms of a `jspace_separability.py` run that sit in different k blocks (`--pairs Jpaper_k16@16-Jpaper@25,...`), with the run's own stratified episode bootstrap (seed = run seed + 1), band mean and per layer, for A2, A3 and A4b → `<run>/arm_pairs.{json,md}`. The run's own inference pairs arms within one k only. |
| `scripts/jspace_cone_cache.py` | Rebuilds the registered cone-fit variant of a `jspace_decompose.py` run from its saved atoms and coefficients (cone_i = Σ_j c_ij v_{t_ij}) and writes `<variant>/cone/activations_gen.npz` in the paper layout for `--external-arm`; no pursuit is redone. |

```bash
python3 scripts/download_narcbench_activations.py
python3 scripts/fetch_unembed_rows.py Qwen/Qwen3-32B-AWQ data/j_basis/qwen3_32b_awq
python3 scripts/jspace_separability.py --core-acts data/activations/official/qwen3_32b/core \
    --transfer-acts data/activations/official/qwen3_32b/transfer --unembed data/j_basis/qwen3_32b_awq \
    --layers 26-30 --ks 8,16,25,50,100 --out results/jspace/qwen3_32b --tag qwen3_32b
python3 scripts/jspace_offline_stats.py
python3 scripts/jspace_figures.py
python3 scripts/jspace_gemma_figures.py
```

Layers: Qwen3-32B and DeepSeek-R1-32B 26–30, GPT-OSS-20B 10–14, Gemma-2-9B 19–23. The Gemma caches are not in git; the Colab notebook (section 7b) writes its extraction in this layout and runs the same script. `tests/test_jspace_separability.py` runs the whole pipeline on synthetic data in about 20 s. Needs `numpy`, `scipy`, `scikit-learn` (`requests`, `huggingface_hub` for the downloads; `tokenizers` to decode frame tokens; `matplotlib` for figures).
