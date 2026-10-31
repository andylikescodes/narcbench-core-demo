# Can J-space tell the colluders apart? — paper

**Author:** Andy Liang  
**Date:** 31 October 2026  
**PDF:** `main.pdf` (17 pages, A4)  
**Source:** `main.tex`, `refs.bib`, `figures/`

A pre-registered separability analysis of the verbalizable (J-space) subspace in NARCBench collusion probes, run on the paper's own published activations for Qwen3-32B, DeepSeek-R1-Distill-Qwen-32B and GPT-OSS-20B, with the frozen Gemma-2-9B numbers as background. Sections: introduction; the problem and the research question; related work; methodology (data, arms, probe pipeline, analyses A1–A5, inference, pre-registration and its one amendment, exploratory E1); hypotheses and the readings fixed in advance; results; what the results say about the question; conclusion; reproduction; per-layer appendix.

## Build

No TeX distribution is needed; [tectonic](https://tectonic-typesetting.github.io) fetches the packages on first run:

```bash
tectonic main.tex
```

(`brew install tectonic` on macOS.) With a full TeX Live: `pdflatex main && bibtex main && pdflatex main && pdflatex main`.

## Sources

Every number comes from the repository (branch `jspace-separability`):

- `docs/jspace/JSPACE_SEPARABILITY_DESIGN.md` — the design, written before the runs
- `results/jspace/<model>/summary.json` and `RESULTS.md` — per-model outputs of `scripts/jspace_separability.py`
- `results/jspace/CROSS_MODEL_TABLES.md`, `CROSS_MODEL_TABLES_E1_gamma.md` — generated tables
- `results/jspace/gemma2_9b_frozen/FROZEN_STATS.md` — Hanley–McNeil intervals for the 1 October Gemma numbers
- `results/jspace/RESULTS.md`, `E1_GAMMA.md` — the narrative write-ups the paper condenses
- `figures/` — copies of `results/jspace/figures/*.pdf` and `results/jspace/figures_e1/fig_k25_cross_model.pdf` (as `fig_k25_cross_model_e1.pdf`), produced by `scripts/jspace_figures.py`

Nothing in the paper was launched on a GPU; all runs were CPU-only on the published activations.
