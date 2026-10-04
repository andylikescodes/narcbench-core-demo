# NARCBench Core demo

Public walkthrough of NARCBench Core transcripts, plus the 1 October 2026 Transfer-stable weekend review (Andy Liang).

Live presenter: <https://narcbench-demo.zeabur.app>  
Deploy notes: [`README_DEPLOY.md`](README_DEPLOY.md)

## Weekend review

Transfer-stable collusion on Gemma-2-9B-it, Core discussion → Transfer. The review reading, in the author’s locked wording, is a monitorable Transfer-stable coordination signature (`attn_L22` / non-J residual), with editability unproven under the tested free-regen interventions. Cite figures from the claims sheet and the HARD metrics file.

| Start here | Path |
|---|---|
| **Current status, review, and roadmap (living entry point)** | [`STATUS.md`](STATUS.md) |
| Migration handoff (4 Oct 2026 record) | [`MIGRATION_HANDOFF.md`](MIGRATION_HANDOFF.md) |
| CPU verification of every matched-prefix number in git | [`results/transfer_stable/VERIFICATION_2026-10-04.md`](results/transfer_stable/VERIFICATION_2026-10-04.md) |
| Code map | [`CODE_INDEX.md`](CODE_INDEX.md) (same file: [`results/transfer_stable/WEEKEND_REVIEW/CODE_INDEX.md`](results/transfer_stable/WEEKEND_REVIEW/CODE_INDEX.md), [`papers/transfer-stable-collusion-2026-10/CODE_INDEX.md`](papers/transfer-stable-collusion-2026-10/CODE_INDEX.md)) |
| Paper PDF (14 pp) | [`papers/transfer-stable-collusion-2026-10/main.pdf`](papers/transfer-stable-collusion-2026-10/main.pdf) |
| Study pack | [`results/transfer_stable/WEEKEND_REVIEW/`](results/transfer_stable/WEEKEND_REVIEW/) |
| Claims sheet | [`results/transfer_stable/WEEKEND_REVIEW/CLAIMS_NUMBER_SHEET.md`](results/transfer_stable/WEEKEND_REVIEW/CLAIMS_NUMBER_SHEET.md) |
| FINAL handoff | [`docs/NARCBench-Weekend-FINAL-HANDOFF-2026-10-01.md`](docs/NARCBench-Weekend-FINAL-HANDOFF-2026-10-01.md) |
| Mechanism skim | [`docs/NARCBench-Transfer-Stable-Weekend-Review-2026-10-01.md`](docs/NARCBench-Transfer-Stable-Weekend-Review-2026-10-01.md) |
| HARD n=72 metrics already in the repo | [`results/transfer/TRANSFER_FULL_HARD_METRICS.md`](results/transfer/TRANSFER_FULL_HARD_METRICS.md) |
| Paper sources | [`papers/transfer-stable-collusion-2026-10/main.tex`](papers/transfer-stable-collusion-2026-10/main.tex), [`refs.bib`](papers/transfer-stable-collusion-2026-10/refs.bib) |
| Runners | [`scripts/README.md`](scripts/README.md) |
| Attachment inventory | [`results/transfer_stable/WEEKEND_REVIEW/SNAPSHOT.md`](results/transfer_stable/WEEKEND_REVIEW/SNAPSHOT.md) |

### Path map

`CODE_INDEX.md` was written against the science machine. In this checkout:

| CODE_INDEX path | This repo |
|---|---|
| `/workspace/narcbench-core-demo/` | repository root |
| `/workspace/andy-biz/papers/transfer-stable-collusion-2026-10/` | `papers/transfer-stable-collusion-2026-10/` |
| `/workspace/andy-biz/NARCBench-*.md` handoff and skim | `docs/` |
| `scripts/`, `generation/`, `app/`, `docs/RUNPOD.md`, `docs/NEXT-SCIENCE.md` | same relative paths |
| `data/activations/` | not committed |

### What you can run

CPU checks that need nothing outside git (28 tests, two verifiers, the pair builder):

```bash
python3 -m unittest discover -s tests
python3 scripts/verify_claims_sheet.py
python3 scripts/verify_matched_prefix_results.py
python3 scripts/build_matched_prefix_pairs.py --out /tmp/pairs_core_v2.jsonl
```

From the repo root, CPU presenter:

```bash
python3 scripts/build_index.py
python3 app/server.py
```

`build_index.py` indexes `upstream/scenarios/*/core/` into `data/demo_index*.json`. The presenter listens on `:8765` (or `PORT`) and includes the Transfer FULL hard view at `/?view=transfer`.

Probe, causal, and scenario runners are in `scripts/` (`eval_transfer_from_core.py`, `train_collusion_probe.py`, `transfer_stable_offline.py`, `causal_steer_transfer.py`, `regen_gemma_core.py`, `extract_activations_from_transcripts.py`, and the RunPod launchers). Core regen imports `generation/`. Exact commands are in [`scripts/README.md`](scripts/README.md) and [`CODE_INDEX.md`](CODE_INDEX.md). Those probe commands read `data/activations/`, which is not in git. Do not pass `--launch`.

Paper rebuild, from `papers/transfer-stable-collusion-2026-10/`:

```bash
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

### Left out of this commit

- `data/activations/**` and other multi-GB `.npz` caches
- Secrets, `.env`, and RunPod keys
- Track I, Paper 2, and interp-demo material
- Sibling `results/transfer_stable/` writeups outside `WEEKEND_REVIEW/` (`LAYER_SWEEP.md`, causal smoke dirs, `directions/`). Numbers for review are in the claims sheet, `WEEKEND_REVIEW/tables/`, and `results/transfer/TRANSFER_FULL_HARD_METRICS.md`.

Transcripts already in the tree stay: Qwen3-32B Core, Gemma-2-2B Core, Gemma-2-9B Core (`20261001T012639Z`), and Gemma-2-9B Transfer `RUNPOD` (72 runs).
