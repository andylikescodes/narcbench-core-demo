# Transfer FULL (n=72) — HARD Core→Transfer metrics
Updated: 2026-10-01 09:28 PDT

Protocol: train on Core discussion (`20261001T012639Z`), eval Transfer **final** phase.
Acts: `data/activations/gemma2_9b/transfer/RUNPOD` (936 samples, layers 19–23).

| Arm | Smoke (n=12) | **FULL (n=72)** | Peak | Split | Asym |
| --- | ---: | ---: | ---: | ---: | ---: |
| full acts | 0.867 | **0.902** | 0.874 | 0.904 | 0.929 |
| W_U k25 J-only | 0.548 | **0.602** | 0.625 | 0.565 | 0.616 |
| W_U k25 complement | 0.867 | **0.898** | 0.869 | 0.897 | 0.928 |
| jac k25 J-only | 0.648 | **0.549** | 0.586 | 0.523 | 0.540 |
| jac k25 complement | 0.867 | **0.891** | 0.867 | 0.888 | 0.917 |

## Verdict
- Full acts primary: **0.902** (transfer holds on n=72).
- W_U J-only: **0.602** vs complement **0.898**
- jac J-only: **0.549** vs complement **0.891**
- **PASS confirmation of smoke finding:** J-localization does **NOT** transfer; complement ≈ full; J-only near chance / weak.

## Artifacts
- Metrics dirs: `results/transfer/FULL_core_to_transfer_*`
- Transcripts: `upstream/scenarios/gemma2_9b/transfer/RUNPOD` (72/72)
- Activations: `data/activations/gemma2_9b/transfer/RUNPOD`
