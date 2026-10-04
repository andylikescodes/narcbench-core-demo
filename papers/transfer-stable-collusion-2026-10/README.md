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

On this repository, `/workspace/narcbench-core-demo/` is the repo root and this directory is `papers/transfer-stable-collusion-2026-10/`. `CODE_INDEX.md` is a relative symlink to `results/transfer_stable/WEEKEND_REVIEW/CODE_INDEX.md`. The handoff and skim live under `docs/`. `main.pdf` and `main.tex` here are the 1 Oct 2026 box copies.

## Dating

Correlational claims in `main.pdf` (J versus complement, residual L21/L22, attention L22, free-regen causal nulls) predate the matched-prefix scorer and site bug found on 4 Oct 2026, and they predate the final-residual wiring suite. The PDF treats matched-prefix interchange as next work. It does not contain the later "first positive" interchange tables. Those tables are `results/transfer_stable/MATCHED_PREFIX_INTERCHANGE_RESULTS.md`, `MATCHED_PREFIX_WIDEN_RESULTS.md`, and `MATCHED_PREFIX_TRANSFER_RESULTS.md`. They are withdrawn pending a re-run on the fixed harness. See `MIGRATION_HANDOFF.md`.
