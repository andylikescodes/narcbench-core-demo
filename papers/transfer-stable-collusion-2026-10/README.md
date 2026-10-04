# Transfer-stable collusion representations — weekend review PDF

**Author:** Andy Liang  
**Date:** 2026-10-01 (PT)  
**PDF:** `main.pdf` (14 pages)  
**Science freeze:** folded in from `WEEKEND_REVIEW/CLAIMS_NUMBER_SHEET` (+ checklist / tables / figures)

## Build
```bash
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

## Sources
Numbers only from:
- `/workspace/narcbench-core-demo/results/transfer_stable/WEEKEND_REVIEW/CLAIMS_NUMBER_SHEET.md` (+ `.json`)
- `/workspace/narcbench-core-demo/results/transfer_stable/WEEKEND_REVIEW/CLAIMS_CHECKLIST.md`
- `/workspace/narcbench-core-demo/results/transfer_stable/WEEKEND_REVIEW/tables/` + `figures/`
- `/workspace/andy-biz/NARCBench-Transfer-Stable-Weekend-Review-2026-10-01.md`
- `/workspace/narcbench-core-demo/results/transfer/` HARD metrics

No invented metrics. Soft-stop: Track I / Paper 2 / interp-demo untouched; no GPU this write-up.

## Science-freeze numbers added in this rebuild
- Cross-phase Core→Transfer **discussion** residual band primary **0.959**; attn_L22 discussion **0.997**
- Core domain-holdout @ L21 mean primary **0.939** (vs full-fit 0.941)
- Transfer family AUROC @ L21 (publication CSV)
- WEEKEND_REVIEW figures: `fig_cross_phase_residual`, `fig_causal_null_flat`, `fig_j_vs_full_vs_complement`, `fig_residual_vs_attn_vs_mlp`

## Code inventory

Symlink to weekend code map: [`CODE_INDEX.md`](./CODE_INDEX.md) → `WEEKEND_REVIEW/CODE_INDEX.md`.

## GitHub checkout

On this repository, `/workspace/narcbench-core-demo/` is the repo root and this directory is `papers/transfer-stable-collusion-2026-10/`. `CODE_INDEX.md` is a relative symlink to `results/transfer_stable/WEEKEND_REVIEW/CODE_INDEX.md`. The handoff and skim live under `docs/`.
