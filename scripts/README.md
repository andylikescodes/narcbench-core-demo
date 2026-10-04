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
