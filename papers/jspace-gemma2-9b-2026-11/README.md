# J-space probes on Gemma-2-9B-it: method and results — paper

**Author:** Andy Liang  
**Date:** 4 November 2026  
**PDF:** `main.pdf` (A4)  
**Source:** `main.tex`, `refs.bib`, `figures/`

A methods-first report on one model, `google/gemma-2-9b-it`, written to be shared. It states two results (a probe restricted to J-space does not transfer to new scenario families; a probe restricted to the complement transfers as well as the unrestricted probe) and then gives the method in six steps with the exact settings behind every number: how the hidden states were read, how each of the two J-space estimates was computed (the top-25 output rows, and the rank-64 eight-prompt Jacobian sketch), how each hidden state was split, how the probe was trained and scored, the two test protocols, and the intervals. A short closing section says what the Gemma numbers do not show.

## Build

```bash
tectonic main.tex
```

(`brew install tectonic` on macOS.) With a full TeX Live: `pdflatex main && bibtex main && pdflatex main && pdflatex main`.

## Sources

Every number comes from the repository (branch `jspace-separability`):

- `results/transfer/TRANSFER_FULL_HARD_METRICS.md`, `results/transfer/FULL_core_to_transfer_*/metrics.json` — the frozen 1 October Transfer metrics
- `results/jspace/gemma2_9b_frozen/FROZEN_STATS.md`, `frozen_stats.json` — Hanley–McNeil intervals, contrasts, per layer, per family, the Core battery (`scripts/jspace_offline_stats.py`)
- `docs/NEXT-SCIENCE.md` — the run log for the Jacobian pod and the Core battery
- `results/transfer_stable/WEEKEND_REVIEW/CLAIMS_NUMBER_SHEET.md` — the locked claims sheet, verified 4 October
- `scripts/extract_activations_from_transcripts.py`, `project_j_complement.py`, `project_j_variants.py`, `estimate_jacobian_jlens.py`, `train_collusion_probe.py`, `eval_transfer_from_core.py` — the six steps of Section 4
- `figures/` — produced by `scripts/jspace_gemma_figures.py` from `frozen_stats.json`; copies in `results/jspace/gemma2_9b_frozen/figures/`

Nothing in this paper was launched on a GPU; it documents the 1 October runs and adds no new experiment.
