# NARCBench Core demo

Public walkthrough of NARCBench Core transcripts, plus the 1 October 2026 Transfer-stable weekend review (Andy Liang).

Live presenter: <https://narcbench-demo.zeabur.app>  
Deploy notes: [`README_DEPLOY.md`](README_DEPLOY.md)

## Weekend review

Transfer-stable collusion on Gemma-2-9B-it, Core discussion → Transfer. The review reading, in the author’s locked wording, is a monitorable Transfer-stable coordination signature (`attn_L22` / non-J residual), with editability unproven under the tested free-regen interventions. Cite figures from the claims sheet and the HARD metrics file.

| Start here | Path |
|---|---|
| Code map | [`CODE_INDEX.md`](CODE_INDEX.md) (same file: [`results/transfer_stable/WEEKEND_REVIEW/CODE_INDEX.md`](results/transfer_stable/WEEKEND_REVIEW/CODE_INDEX.md), [`papers/transfer-stable-collusion-2026-10/CODE_INDEX.md`](papers/transfer-stable-collusion-2026-10/CODE_INDEX.md)) |
| Paper PDF (14 pp) | [`papers/transfer-stable-collusion-2026-10/main.pdf`](papers/transfer-stable-collusion-2026-10/main.pdf) |
| Study pack | [`results/transfer_stable/WEEKEND_REVIEW/`](results/transfer_stable/WEEKEND_REVIEW/) |
| Claims sheet | [`results/transfer_stable/WEEKEND_REVIEW/CLAIMS_NUMBER_SHEET.md`](results/transfer_stable/WEEKEND_REVIEW/CLAIMS_NUMBER_SHEET.md) |
| FINAL handoff | [`docs/NARCBench-Weekend-FINAL-HANDOFF-2026-10-01.md`](docs/NARCBench-Weekend-FINAL-HANDOFF-2026-10-01.md) |
| Mechanism skim | [`docs/NARCBench-Transfer-Stable-Weekend-Review-2026-10-01.md`](docs/NARCBench-Transfer-Stable-Weekend-Review-2026-10-01.md) |
| HARD n=72 metrics already in the repo | [`results/transfer/TRANSFER_FULL_HARD_METRICS.md`](results/transfer/TRANSFER_FULL_HARD_METRICS.md) |
| Attachment inventory | [`results/transfer_stable/WEEKEND_REVIEW/SNAPSHOT.md`](results/transfer_stable/WEEKEND_REVIEW/SNAPSHOT.md) |

### What you can run

From the repo root, CPU only:

```bash
python3 scripts/build_index.py
python3 app/server.py
```

`build_index.py` indexes `upstream/scenarios/*/core/` into `data/demo_index*.json`. The presenter listens on `:8765` (or `PORT`) and includes the Transfer FULL hard view at `/?view=transfer`. Details: [`scripts/README.md`](scripts/README.md).

`CODE_INDEX.md` also names the probe, causal, and scenario runners used on the local science workspace (`eval_transfer_from_core.py`, `train_collusion_probe.py`, `transfer_stable_offline.py`, `causal_steer_transfer.py`, `regen_gemma_core.py`, `extract_activations_from_transcripts.py`, and the RunPod launchers). Those files are not in this repository. See [`scripts/README.md`](scripts/README.md).

### Left out of this commit

- `data/activations/**` and other multi-GB `.npz` caches
- Secrets, `.env`, and RunPod keys
- Track I, Paper 2, and interp-demo material

Spine notes `01`–`07`, `tables/`, `figures/`, `design/`, and `main.tex` were not in the weekend attachment, so they were not filled in. The inventory is [`SNAPSHOT.md`](results/transfer_stable/WEEKEND_REVIEW/SNAPSHOT.md).

Transcripts already in the tree stay: Qwen3-32B Core, Gemma-2-2B Core, Gemma-2-9B Core (`20261001T012639Z`), and Gemma-2-9B Transfer `RUNPOD` (72 runs).
