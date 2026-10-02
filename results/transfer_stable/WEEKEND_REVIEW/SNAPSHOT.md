# Weekend attachment inventory

**Placed:** 2026-10-02, for GitHub review of Andy Liang’s 2026-10-01 Transfer-stable collusion package.  
**Rule:** files below are copies of the weekend attachment or paths already in this repo. Missing pack files were not reconstructed, and no metrics were added.

## In this snapshot

| Item | Repo path | Source |
|---|---|---|
| Pack entry | `results/transfer_stable/WEEKEND_REVIEW/00_README.md` | Weekend attachment |
| Frozen claims sheet | `results/transfer_stable/WEEKEND_REVIEW/CLAIMS_NUMBER_SHEET.md` | Weekend attachment |
| Code map | `results/transfer_stable/WEEKEND_REVIEW/CODE_INDEX.md` | Weekend attachment |
| Same code map | `CODE_INDEX.md` and `papers/transfer-stable-collusion-2026-10/CODE_INDEX.md` | Symlinks to the pack copy |
| Review PDF (14 pp) | `papers/transfer-stable-collusion-2026-10/main.pdf` | Weekend attachment (440368 bytes; pdfTeX 2026-10-01 17:21 PT; author Andy Liang; title *Transfer-Stable Collusion Representations*) |
| FINAL handoff | `docs/NARCBench-Weekend-FINAL-HANDOFF-2026-10-01.md` | Weekend attachment |
| Mechanism skim | `docs/NARCBench-Transfer-Stable-Weekend-Review-2026-10-01.md` | Weekend attachment |
| HARD n=72 metrics | `results/transfer/TRANSFER_FULL_HARD_METRICS.md` and `results/transfer/FULL_*/metrics.json` | Already in the repo |
| Transcripts | `upstream/scenarios/` (Qwen Core 100, Gemma-2-2B Core 100, Gemma-2-9B Core 100, Gemma-2-9B Transfer `RUNPOD` 72) | Already in the repo |
| Presenter | `app/server.py`, `app/viewer.html`, `scripts/build_index.py` | Already in the repo |

## Named by the pack, not in the attachment

These paths are cited by `00_README.md`, `CODE_INDEX.md`, or the handoff. They were not uploaded with this package, so they are not in the commit:

- Spine notes `01_EXECUTIVE_SUMMARY.md` through `07_OPEN_QUESTIONS.md`
- `CLAIMS_CHECKLIST.md`, `CLAIMS_NUMBER_SHEET.json`, `PACK_MANIFEST.json`
- `tables/`, `figures/`, `design/`, `WEEKEND_REVIEW/scripts/matched_prefix_scaffold.py`
- `papers/transfer-stable-collusion-2026-10/main.tex` and figure sources
- Sibling writeups under `results/transfer_stable/` (`LAYER_SWEEP.md`, `MLP_ATTN_PROBES.md`, `NEGATIVE_CAUSAL_APPENDIX.md`, `directions/`, causal smoke dirs, and the rest listed in the skim’s artifact index)
- Probe, causal, scenario, and RunPod runner sources named in `CODE_INDEX.md` (`eval_transfer_from_core.py`, `train_collusion_probe.py`, `transfer_stable_offline.py`, `causal_steer_transfer.py`, `regen_gemma_*.py`, `extract_activations_from_transcripts.py`, and the other launchers)

The PDF and `CLAIMS_NUMBER_SHEET.md` are the number sources that did arrive. `results/transfer/TRANSFER_FULL_HARD_METRICS.md` is the HARD table already stored here.

## Left out on purpose

- `data/activations/**` and other multi-GB `.npz` activation caches
- Secrets, `.env`, and RunPod keys
- Track I, Paper 2, and interp-demo material
