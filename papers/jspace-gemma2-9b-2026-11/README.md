# J-space probes on Gemma-2-9B-it: method and results — paper

**Author:** Andy Liang  
**Date:** 4 November 2026, revised 5 November 2026  
**PDF:** `main.pdf` (A4)  
**Source:** `main.tex`, `refs.bib`, `figures/`

A methods-first report on one model, `google/gemma-2-9b-it`, written to be shared. It states three results and then gives the method step by step with the exact settings behind every number. (1) A probe restricted to the two output-row stand-ins for J-space of 1 October (the top-25 unembedding rows; a rank-64 eight-prompt Jacobian sketch) does not transfer to new scenario families, and with the controls run on 5 November it sits at the level of a random 25-dimensional subspace. (2) A probe restricted to the complement transfers as well as the unrestricted probe under every estimate. (3) A probe restricted to the paper's own J-space, built on 5 November with the published Jacobian lens and the paper's nonnegative pursuit, transfers in part: 0.755 [0.709, 0.796] against 0.647 for a random subspace and 0.559 for another turn's atoms, with the complement at 0.891 against 0.902; the result holds under eight registered variants of the decomposition, and without the paper's fill-to-k step the J-space probe scores 0.811. The closing section says what the Gemma numbers do not show.

## Build

```bash
tectonic main.tex
```

(`brew install tectonic` on macOS.) With a full TeX Live: `pdflatex main && bibtex main && pdflatex main && pdflatex main`.

## Sources

Every number comes from the repository (branch `jspace-separability`):

- `results/transfer/TRANSFER_FULL_HARD_METRICS.md`, `results/transfer/FULL_core_to_transfer_*/metrics.json` — the frozen 1 October Transfer metrics
- `results/jspace/gemma2_9b_frozen/FROZEN_STATS.md`, `frozen_stats.json` — Hanley–McNeil intervals, contrasts, per layer, per family, the Core battery (`scripts/jspace_offline_stats.py`)
- `results/jspace/gemma2_9b_jpaper_nnomp_k25/` — the 5 November run of the paper's J-space (estimate C) with the controls, within-Transfer and discussion-turn probes: `RESULTS.md`, `summary.json`, `episode_scores.json`, `decomp_summary.md`, `arm_layer_cis.md`
- `results/jspace/gemma2_9b_jpaper_variants/`, `results/jspace/gemma2_9b_jpaper_variants_all/` — the eight registered variants of estimate C, with cross-k paired differences (`arm_pairs.md`) and per-layer intervals
- `docs/jspace/JSPACE_SEPARABILITY_DESIGN.md` — the pre-registered design, its reading rules, and the 5 November addendum that registered estimate C and its variants before the run
- `docs/NEXT-SCIENCE.md` — the run log (1 October pod and Core battery; 4–5 November lens, pull, decomposition, probe runs, variants)
- `results/transfer_stable/WEEKEND_REVIEW/CLAIMS_NUMBER_SHEET.md` — the locked claims sheet, verified 4 October
- `scripts/extract_activations_from_transcripts.py`, `project_j_complement.py`, `project_j_variants.py`, `estimate_jacobian_jlens.py`, `train_collusion_probe.py`, `eval_transfer_from_core.py` — the six steps of Section 3 for estimates A and B; `scripts/runpod_pull_volume_files.py`, `jspace_decompose.py`, `jspace_cone_cache.py`, `jspace_separability.py`, `jspace_arm_layer_cis.py`, `jspace_arm_pairs.py` — estimate C and its inference
- `figures/` — produced by `scripts/jspace_gemma_figures.py` from `frozen_stats.json` and from the 5 November summaries; copies in `results/jspace/gemma2_9b_frozen/figures/` and `results/jspace/gemma2_9b_jpaper_nnomp_k25/figures/`

The 1 October runs are documented, not rerun. The 5 November runs are CPU runs on the same activation caches (copied off the RunPod volume for about $0.15 in pods); the Jacobian lens is the published one, not a new estimate.
