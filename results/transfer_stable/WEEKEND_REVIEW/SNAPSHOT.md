# Weekend attachment inventory

**Placed:** 2026-10-02, for GitHub review of Andy Liang’s 2026-10-01 Transfer-stable collusion package.  
**Rule:** files below are copies from the weekend archive or paths already in this repo. No metrics were added. Activation caches and secrets were not committed.

## Path map (`CODE_INDEX.md`)

| Path in CODE_INDEX | Path in this repo |
|---|---|
| `/workspace/narcbench-core-demo/` | repository root |
| `/workspace/andy-biz/papers/transfer-stable-collusion-2026-10/` | `papers/transfer-stable-collusion-2026-10/` |
| `/workspace/andy-biz/NARCBench-Weekend-FINAL-HANDOFF-2026-10-01.md` | `docs/NARCBench-Weekend-FINAL-HANDOFF-2026-10-01.md` |
| `/workspace/andy-biz/NARCBench-Transfer-Stable-Weekend-Review-2026-10-01.md` | `docs/NARCBench-Transfer-Stable-Weekend-Review-2026-10-01.md` |
| `scripts/…`, `generation/…`, `app/…`, `docs/RUNPOD.md`, `docs/NEXT-SCIENCE.md` | same relative paths |
| `results/transfer_stable/WEEKEND_REVIEW/…` | same relative path |
| `data/activations/…`, `data/j_basis/…` | not in git (`.npz` and `data/activations/` are gitignored) |

`papers/transfer-stable-collusion-2026-10/CODE_INDEX.md` and the repo-root `CODE_INDEX.md` are relative symlinks to `results/transfer_stable/WEEKEND_REVIEW/CODE_INDEX.md`.

## In this snapshot

| Item | Repo path |
|---|---|
| Spine `00`–`07`, claims sheet + JSON, checklist, pack manifest | `results/transfer_stable/WEEKEND_REVIEW/` |
| Design specs and matched-prefix scaffold | `WEEKEND_REVIEW/design/`, `WEEKEND_REVIEW/scripts/matched_prefix_scaffold.py` |
| Publication tables and figures | `WEEKEND_REVIEW/tables/`, `WEEKEND_REVIEW/figures/` |
| Code map | `WEEKEND_REVIEW/CODE_INDEX.md` |
| Paper sources, bibliography, figures, PDF | `papers/transfer-stable-collusion-2026-10/` (`main.tex`, `refs.bib`, `main.bbl`, `fig_*`, `main.pdf`) |
| Probe / causal / scenario / RunPod runners | `scripts/` |
| Core regen package the runners import | `generation/` |
| RunPod and next-science notes named by the code index | `docs/RUNPOD.md`, `docs/NEXT-SCIENCE.md` |
| FINAL handoff and mechanism skim | `docs/NARCBench-Weekend-FINAL-HANDOFF-2026-10-01.md`, `docs/NARCBench-Transfer-Stable-Weekend-Review-2026-10-01.md` |
| HARD n=72 metrics and transcripts | `results/transfer/`, `upstream/scenarios/` |
| Presenter | `app/server.py`, `app/viewer.html` |

## Still not in the repository

`CODE_INDEX.md` also names sibling writeups and caches that were not in the archive:

- `results/transfer_stable/` analysis files outside `WEEKEND_REVIEW/` (`LAYER_SWEEP.md`, `MLP_ATTN_PROBES.md`, `NEGATIVE_CAUSAL_APPENDIX.md`, `directions/`, causal smoke directories)
- `data/activations/**` and other `.npz` caches
- `data/j_basis/`
- `.venv-probe/`
- Upstream clone at `/workspace/narcbench-upstream/repo/`
- Track I, Paper 2, and interp-demo trees

Matched-prefix, within-trajectory, and checkpoint **GPU runners** are still design-only, as `CODE_INDEX.md` section 7 says. The CPU scaffold is `WEEKEND_REVIEW/scripts/matched_prefix_scaffold.py`.

## Left out on purpose

- `data/activations/**` and other multi-GB `.npz` activation caches
- Secrets, `.env`, and RunPod keys (launchers read `RUNPOD_API_KEY` from the environment)
- Track I, Paper 2, and interp-demo material
