> **Status update, 4 October 2026 (review pass).** Pull request 1 was merged into `main` and pull request 2 was closed, so the branch instructions below are complete. The living entry point is now [`STATUS.md`](STATUS.md). Every matched-prefix number cited below was re-derived from the per-pair records in git (`results/transfer_stable/VERIFICATION_2026-10-04.md`). Open question 1 (directions at the final site) and open question 2 (where the vote enters the last-token residual) now have cards in the harness and launcher; neither has been launched. The rest of this file is the 4 Oct record and is unchanged.

# NARCBench / Transfer-stable migration handoff

Written 4 October 2026. This file is the trunk note for leaving the Cursor / RunPod setup. Pull request #1 on `andylikescodes/narcbench-core-demo` is the single migration branch. Do not open a second PR for this lane, and do not merge pull request #2 on top of this branch.

GPU stays off. Nothing in this note was launched.

## Where the code is

- Repo: https://github.com/andylikescodes/narcbench-core-demo
- This PR (the trunk): https://github.com/andylikescodes/narcbench-core-demo/pull/1
- Branch: `cursor/transfer-stable-weekend-review-8f77`
- Code map: `CODE_INDEX.md` (the file itself is `results/transfer_stable/WEEKEND_REVIEW/CODE_INDEX.md`)
- Runners: `scripts/README.md`
- Paper: `papers/transfer-stable-collusion-2026-10/` (`main.tex`, `refs.bib`, `main.pdf`). `main.pdf` and `main.tex` match the 1 Oct 2026 box copies. Correlational claims in that PDF predate the matched-prefix scorer/site bug and the wiring suite. The PDF does not contain the later "first positive" interchange tables.
- Weekend pack: `results/transfer_stable/WEEKEND_REVIEW/` (spine `00`–`07`, claims sheet, tables, figures)
- HARD n=72 table already on `main` and in this branch: `results/transfer/TRANSFER_FULL_HARD_METRICS.md`
- 4 Oct matched-prefix narrative and the raw files it cites: `docs/handoff/2026-10-04-matched-prefix/HANDOFF.md`

Older handoffs from 1 Oct are `docs/NARCBench-Weekend-FINAL-HANDOFF-2026-10-01.md` and `docs/NARCBench-Transfer-Stable-Weekend-Review-2026-10-01.md`.

## Pipeline (standing)

Develop on Cursor cloud and GitHub. A RunPod pod, if anyone boots one later, clones or checks out a public SHA of this repo (or starts from a Docker image that already contains that checkout). The GraphQL create body stays a few kilobytes. Cloudflare has rejected create bodies around 140KB. A body of 100,606 bytes still succeeded on 2026-10-03. Stay far under that.

The within-trajectory and matched-prefix launchers follow the stricter shape that was on pull request #2:

- The smoke script lives in the repo: `scripts/runpod_within_traj_boot.sh` and `scripts/runpod_matched_prefix_boot.sh`.
- The create env carries the commit SHA, the repo URL, and volume paths (plus short scalars such as model name, seed, and suite name).
- The create env does not carry `JOB_BOOT_B64`, `JOB_PAYLOAD_B64`, a boot script, or a job tarball.
- Estimate mode is the default. `--launch` is what spends money. Do not pass it from this handoff.

`RUNPOD_API_KEY` and `HF_TOKEN` come from the environment. They are not in git.

Soft-stop: do not touch Track I, Paper 2, or interp-demo. Do not edit those repos from this note.

## Pull request #2

https://github.com/andylikescodes/narcbench-core-demo/pull/2 (head `2b42a183`, branch `cursor/within-traj-git-boot-90e4`) is an earlier within-trajectory git boot. Its useful pattern is the in-repo smoke script and the small create env. That pattern is now on this branch for both slim launchers.

Do not merge #2 as it stands. It does not contain the weekend pack, the matched-prefix harness, or the later scorer fix, and merging it would drop those. Close #2 once this branch is the one people use. Its tests were brought over as `tests/test_within_traj_launch.py`.

## What is not in git

Do not commit these. They live on RunPod network volume `cr1kjtzjs2`, mounted at `/workspace` (datacenter EU-RO-1):

| What | Volume path |
|---|---|
| Matched-prefix pairs (12 Core rows used by the wiring suite) | `/workspace/jobs/narcbench-data/matched_prefix/pairs.jsonl` |
| Direction `.npy` files (role, role-perp, mode, PCA, and the other vectors `transfer_stable_residualize.py` writes) | `/workspace/jobs/narcbench-data/transfer_stable/directions/` |
| Core transcripts, if the volume copy is the one a pod should read | `/workspace/jobs/narcbench-data/gemma2_9b/core/20261001T012639Z` |
| Model cache used by matched-prefix boots | `/workspace/jobs/hf-cache` |
| Model cache used by the within-trajectory boot | `/workspace/jobs/narcbench-hf` |

The same Core transcripts are also in this repo at `upstream/scenarios/gemma2_9b/core/20261001T012639Z`. Activation caches (`data/activations/`, `*.npz`) and `data/j_basis/` are gitignored. They are multi-GB and stay off GitHub. Older box copies of sibling `results/transfer_stable/` writeups (layer sweep, causal smoke dirs, `directions/`) were not in the archive that built this PR. The review numbers that were copied are in the claims sheet, `WEEKEND_REVIEW/tables/`, and `results/transfer/TRANSFER_FULL_HARD_METRICS.md`.

## Science arc (done)

Numbers below are the ones already written in this repo or in the 4 Oct handoff. This note does not add a new measurement.

**HARD Transfer.** J-space does not carry the Transfer signal. Full activations and the complement do. From `results/transfer/TRANSFER_FULL_HARD_METRICS.md` (Core discussion → Transfer final, n=72): full primary AUROC 0.902; W_U k25 J-only 0.602 versus complement 0.898; Jacobian k25 J-only 0.549 versus complement 0.891.

**Transfer-stable hunt (correlational).** Residual L21/L22, contrast PCA around k=8 at L23, and attention L22 read the role shift. The weekend executive summary records residual L21 near 0.941 and attention L22 near 0.999 on the primary transfer metric. Free-regen causal smokes (residual, attention, discussion) were null: votes stayed with the baseline and with a random direction. Readability is not the same as editability. See `results/transfer_stable/WEEKEND_REVIEW/01_EXECUTIVE_SUMMARY.md` and `04_NEGATIVE_CAUSAL_APPENDIX.md`.

**Within-trajectory mid-private smoke.** The recorded result directory is `/workspace/jobs/narcbench-results/within_traj_mid_private/smoke_20261004T031403Z`. The migration notes describe it as weak / near-null: edit-fraction baselines about 0.52, with small deltas. Those raw files are not in git, and this note does not recompute them.

**Matched-prefix claims that are withdrawn.** Two bugs in `scripts/matched_prefix_interchange.py`, present at `ab6934f` (cheap-extras harness) and `a0c34db` (role-perp confirm):

1. The patch was applied at the last shared token (`t* - 1`). The honest and colluder prompts are the same there, so a 1D transplant is exactly zero.
2. The scorer compared every patch to the honest baseline whenever `baseline_honest` was already in the results dict. Honest→colluder was never scored against the colluder baseline. The reported change, about +0.48, was the untouched gap. The "flips" were baseline disagreements.

Withdraw the cheap-extras "perp copies the gap" claim and the role-perp confirm "random matches" claim. They are not effects. The withdrawn confirm run is `smoke_20261004T150244Z` (SHA `a0c34db645ba2904545967839f759febee167ece`). Its RESULTS file is `docs/handoff/2026-10-04-matched-prefix/withdrawn-role-perp-RESULTS.md`.

The Oct 1 "first positive" matched-prefix writeups used that same pre-fix harness. They are in the repo only as a record, and they are withdrawn pending a re-run on the fixed harness:

- `results/transfer_stable/MATCHED_PREFIX_INTERCHANGE_RESULTS.md` — Core smoke, n=10, pod `klmkfu4dubgc39`. Reported honest→colluder ΔP `+0.481` on attention L22 and residual L21.
- `results/transfer_stable/MATCHED_PREFIX_WIDEN_RESULTS.md` — Core widen, n=33, pod `fczidrsh55t44p`. Reported honest→colluder ΔP `+0.619`.
- `results/transfer_stable/MATCHED_PREFIX_TRANSFER_RESULTS.md` — Transfer smoke, n=12, pod `gy5yigd2nrwsua`. Reported honest→colluder ΔP `0.727`.

Those ΔP figures match the untouched baseline gap in each file. Do not cite the takeaways as causal effects.

**After the fix.** SHA `1fd085579aab7a868280a43caf09df1d232bffa1` (`1fd0855`) scores honest→colluder against the colluder baseline and colluder→honest against the honest baseline, from the arm name only, and writes at each prompt's last token. The layer-21 full residual at that last token did not carry the vote (smoke `smoke_20261004T165214Z`, pod `06rbunmpwe9sqs`). Mean change was about −0.05 versus the colluder baseline, with no vote flips. Files: `docs/handoff/2026-10-04-matched-prefix/l21-last-RESULTS.md` and the summary and meta next to it. Volume copy: `/workspace/jobs/narcbench-results/matched_prefix_role_perp_confirm/`.

**Wiring suite.** SHA `319cf82580df09ac5e79c7017d8a0d7ed1116550` (`319cf825`). Smoke `smoke_20261004T173047Z`. The copy is the final-norm output (`model.model.norm` in the run meta: `final_norm_pre_lm_head`), the hidden state the output head turns into the next token, at the last prompt token. Full residual honest↔colluder copy moved the entire untouched gap (about 0.48; the logged honest→colluder change is `−0.48370584490476176`, fraction of the gap 1.0) and flipped all 5/5 disagreeing pairs (cyber_03, eng_01, env_02, food_01, leg_03). Patched probabilities matched the source baseline digit for digit. That is a wiring check. The private-instruction span at that same final depth failed: mean change 0, flips 0/5. The harness is validated. Direction arms were not run at this site, and they were not interpreted. Files: `docs/handoff/2026-10-04-matched-prefix/final-resid-RESULTS.md`, `final-resid-summary.json`, `final-resid-per_pair.json`, `final-resid-meta.json`. Volume path: `/workspace/jobs/narcbench-results/matched_prefix_final_resid/smoke_20261004T173047Z`.

Cheap-extras output, if a pod wrote it, is under `/workspace/jobs/narcbench-results/matched_prefix_cheap_extras/`. Treat it as withdrawn if it used the pre-fix harness.

## Open questions (next platform)

1. At the working site (final-layer residual, last prompt token), do the Transfer-stable directions — role, role-perpendicular, a random direction of the same norm, and the other 1D vectors already on the volume — move P relative to the full-residual ceiling? This suite was not run. The code on this branch does not add that card.
2. Where does the private instruction enter that final last-token residual? The private-instruction span at the final depth was null, so the answer is earlier (layers or attention paths), not a copy of that span at the final norm.
3. Re-run the Oct 1 "first positive" matched-prefix numbers on the fixed harness before treating them as science. The copies under `results/transfer_stable/MATCHED_PREFIX_*_RESULTS.md` are withdrawn. They used the buggy site and scorer.
4. Within-trajectory versus matched-prefix: keep exploring after a clean harness, or refocus on the site that already moves the vote?
5. Before any future pod: confirm `pairs.jsonl` and the direction `.npy` files are on the volume at the paths above. Box copies lived under `results/transfer_stable/` and are not a substitute for the volume.

## Key SHAs

| SHA | What it is |
|---|---|
| `ab6934f` | Matched-prefix cheap-extras harness, before the site and scorer fix |
| `a0c34db` | Role-perp confirm card, still the buggy site and scorer |
| `1fd0855` | Scorer uses the arm name only; writes move to each prompt's last token |
| `319cf825` | Final-residual wiring suite (last-token copy passed; private span failed) |

## Launchers on this branch

Estimate only:

```bash
python3 scripts/runpod_launch_within_traj.py
python3 scripts/runpod_launch_matched_prefix.py --final-resid-controls
```

Guards that do not create a pod:

```bash
python3 -m unittest tests.test_within_traj_launch tests.test_matched_prefix_launch
```
