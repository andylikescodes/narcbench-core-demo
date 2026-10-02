# Scripts in this repository

## Runnable here (CPU presenter)

From the repo root:

```bash
python3 scripts/build_index.py
python3 app/server.py
```

`build_index.py` reads `upstream/scenarios/<model>/core/` and writes `data/demo_index.json`, `data/models_index.json`, and the per-model indexes. `app/server.py` serves the walkthrough on port 8765 unless `PORT` is set (Zeabur uses 8080). Transfer FULL hard metrics already on disk are shown at `/?view=transfer`.

Deploy steps stay in [`README_DEPLOY.md`](../README_DEPLOY.md).

## Probe, causal, and scenario runners

[`CODE_INDEX.md`](../CODE_INDEX.md) is the 2026-10-01 map of the local science workspace. It names runners such as:

| Role | Script named in CODE_INDEX |
|---|---|
| Core regen | `scripts/regen_gemma_core.py` |
| Transfer regen | `scripts/regen_gemma_transfer.py` |
| Activation extract | `scripts/extract_activations_from_transcripts.py` |
| Core probe | `scripts/train_collusion_probe.py` |
| Core → Transfer eval | `scripts/eval_transfer_from_core.py` |
| MLP / attn probes | `scripts/mlp_attn_probes_offline.py` |
| J projection | `scripts/project_j_variants.py` |
| Offline Transfer-stable | `scripts/transfer_stable_offline.py`, `scripts/transfer_stable_controls.py`, `scripts/transfer_stable_residualize.py` |
| Causal steer / ablate | `scripts/causal_steer_transfer.py` |

Those sources are not in this GitHub tree and were not in the weekend attachment. They are not recreated here. `CODE_INDEX.md` also marks matched-prefix, within-trajectory, and checkpoint assay runners as missing on the science machine (design only).

Do not launch RunPod or other GPU jobs from this review. Causal free-regen work is paused pending a matched-prefix redesign. Activation caches under `data/activations/` are not in the commit.
