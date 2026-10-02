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

`scripts/causal_steer_transfer.py` and `scripts/activation_patch_core.py` are present. `CODE_INDEX.md` marks the causal GPU battery as paused. Matched-prefix, within-trajectory, and checkpoint assay runners are specs under `results/transfer_stable/WEEKEND_REVIEW/design/`, not GPU entry points.

Files named `*.bak` and `*.bak_*` are earlier copies of the extract and causal harnesses from the science tree.
