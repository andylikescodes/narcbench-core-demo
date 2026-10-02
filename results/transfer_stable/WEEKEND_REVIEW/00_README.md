# Weekend review pack — Transfer-stable as a coherent study

**Written:** 2026-10-01 ~5:15 PM PT (upgraded narrative pass)  
**Claims freeze:** 2026-10-01 ~5:20 PM PT — `CLAIMS_NUMBER_SHEET.*` + `CLAIMS_CHECKLIST.md` locked for Paige LaTeX (author: Andy Liang)  
**Lane:** NARCBench Transfer-stable · Gemma-2-9B-it · Core→Transfer  
**Soft-stop:** Track I / Paper 2 / interp-demo / family-house **not touched**.  
**GPU:** none this pack. Causal GPU **paused** pending matched-prefix redesign.  
**Balance (last known):** ~$26.24 after discussion-phase smoke (~4:37 PM PT).

This pack is the **consolidated study entry point** — deeper than a metric dump or a one-site probe paper. It answers, in order: **where** collusion-role information lives (J vs residual vs attention), **what transfers** under distribution shift, how that geometry relates to **coordination vs a pure lying atom**, why free-regen ablate nulls mean **readability ≠ control**, and what **AI-safety claim** we can (and cannot) make.

Numbers are copied from sibling artifacts under `results/transfer_stable/` and `results/transfer/`; nothing invented.

---

## For Paige (LaTeX handoff) — START HERE

| File | Role |
|---|---|
| **`CLAIMS_NUMBER_SHEET.md`** + **`.json`** | Frozen citable numbers (every row sourced + status) |
| **`CLAIMS_CHECKLIST.md`** | 5-beat claim checklist + safety wording + ready PASS/FAIL |
| **`tables/`** | Publication CSV/JSON (HARD, layer sweep, mlp_attn, controls, causal nulls, cross-phase, family, holdout) |
| **`CODE_INDEX.md`** | Runnable code map (tools/packages/pipelines); paper symlink |
| **`figures/`** | PNG bars: J vs full vs complement; residual/attn/mlp; causal nulls; cross-phase |

**Safety claim (locked):** Monitorable, Transfer-stable coordination signature (attn_L22 / non-J residual) — editability unproven under tested interventions; matched-prefix + checkpoint assays are the next causal designs.

---

## Study spine (read in this order)

| # | File | Arc beat |
|---|---|---|
| 1 | **01_EXECUTIVE_SUMMARY.md** | Full arc in plain English + safety claim |
| 2 | **02_RESEARCH_RETHINK.md** | Paper-shaped rethink: where / what transfers / mechanism claim |
| 3 | **03_CORRELATIONAL_CLAIMS.md** | Evidence for beats 1–3 (J-fail → attn_L22 → role↔mode) with numbers |
| 4 | **05_CONTROLS_AND_RESIDUALIZE.md** | Coordination/setup entanglement vs deception-pure axis |
| 5 | **04_NEGATIVE_CAUSAL_APPENDIX.md** | Beat 4: null table → readability ≠ editability |
| 6 | **06_METHODS_LADDER.md** | Beat 5 next: matched-prefix → within-trajectory → checkpoints |
| 7 | **07_OPEN_QUESTIONS.md** | Ranked Qs; GPU pause |
| — | `CLAIMS_NUMBER_SHEET.md` / `CLAIMS_CHECKLIST.md` | **Paige typeset lock** |
| — | `tables/` · `figures/` | Publication exports |
| — | `design/*` | Specs + pair templates + dry-run candidates |
| — | **`CODE_INDEX.md`** | Code inventory for weekend review (scenarios → acts → probes → causal → analysis); symlink also at `/workspace/andy-biz/papers/transfer-stable-collusion-2026-10/CODE_INDEX.md` |
| — | `PACK_MANIFEST.json` | Paths / soft-stop / balance note |

---

## Pack layout

```
WEEKEND_REVIEW/
  00_README.md … 07_OPEN_QUESTIONS.md
  CLAIMS_NUMBER_SHEET.md + .json
  CLAIMS_CHECKLIST.md
  tables/   hard_metrics, layer_sweep, mlp_attn, controls, causal_nulls, cross_phase, family, holdout
  figures/  fig_j_vs_full_vs_complement, fig_residual_vs_attn_vs_mlp, fig_causal_null_flat, fig_cross_phase_residual
  PACK_MANIFEST.json
  design/   matched_prefix_*, within_trajectory_spec, checkpoint_assays_spec
  scripts/  matched_prefix_scaffold.py
  CODE_INDEX.md   ← code paths / how to run / gaps
```

Sibling sources unchanged (do not delete): `../CLAIM_CORRELATIONAL.md`, `../INTERIM_VS_BASELINES.md`, `../MLP_ATTN_PROBES.md`, `../NEGATIVE_CAUSAL_APPENDIX.md`, `../../transfer/TRANSFER_FULL_HARD_METRICS.md`, etc.

## Science freeze → PDF

Locked numbers from `CLAIMS_NUMBER_SHEET.md` / checklist folded into `/workspace/andy-biz/papers/transfer-stable-collusion-2026-10/main.pdf` (14 pages, rebuilt 2026-10-01 ~5:21 PM PT). Track I soft-stopped.
