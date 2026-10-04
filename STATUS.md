# STATUS — Transfer-stable collusion lane

**Living entry point.** Updated 4 October 2026 by a review pass (branch `claude/beautiful-mayer-wtyfjf`). The dated handoffs (`MIGRATION_HANDOFF.md`, `docs/handoff/2026-10-04-matched-prefix/HANDOFF.md`, `docs/NARCBench-*-2026-10-01.md`) are records and were not rewritten. When a card runs, add its outcome to section 2 here and leave the records alone.

GPU stays off. Nothing in this pass was launched.

## What this pass did

| Item | Outcome |
|---|---|
| GitHub state | Pull request 1 (the migration trunk) was merged into `main` on 4 Oct; pull request 2 was closed. The branch instructions in the migration note are complete. |
| Re-derived every matched-prefix number in git | `scripts/verify_matched_prefix_results.py` recomputes all by-arm means, deltas and gates from the per-pair records. 38 checks pass. Report: `results/transfer_stable/VERIFICATION_2026-10-04.md`. |
| Cross-checked the frozen claims sheet | `scripts/verify_claims_sheet.py` compares `CLAIMS_NUMBER_SHEET.json` with the publication tables, the raw probe metrics, the HARD markdown table and the paper source. 233 checks pass, no warnings. |
| Presenter | Indexes rebuilt from the transcripts (content identical; paths now repo-relative). `app/server.py` smoke-tested on all four API routes. |
| Next cards coded | Open question 1 (directions at the final site) and open question 2 (where the vote enters the last-token residual) are now harness modes with launcher, boot and tests. Estimate-only. |
| Reproducible pairs | `scripts/build_matched_prefix_pairs.py` rebuilds the 12 recorded scenarios (and all 50) from the Core transcripts. `data/matched_prefix/pairs_core_v2.jsonl` is committed. |
| Tests | 28 pass (`python3 -m unittest discover -s tests`), including dry-runs of every harness card and both verifiers. |
| Cruft removed | Four `.bak*` scripts and five LaTeX build artifacts (kept in history at `8eaca99`). |

## 1. What is established (correlational, frozen, verified)

Gemma-2-9B-it. Train on Core discussion public, evaluate on Transfer final, n=72. Every number below is in `results/transfer_stable/WEEKEND_REVIEW/CLAIMS_NUMBER_SHEET.md` and was cross-checked by `scripts/verify_claims_sheet.py`.

| Finding | Number | Source |
|---|---:|---|
| Full residual band L19–23 transfers | 0.902 | `results/transfer/TRANSFER_FULL_HARD_METRICS.md` |
| J-only does not transfer (W_U / Jacobian k=25) | 0.602 / 0.549 | same |
| Complement carries the signal (W_U / Jacobian) | 0.898 / 0.891 | same |
| Best residual layer L21 (L22) | 0.941 (0.930) | `tables/layer_sweep_residual.json` |
| Attention residual-write L22 | 0.999 | `tables/mlp_attn_site_aurocs.json` |
| Contrast PCA k=8 at L23 | 0.964 | claims sheet section C |
| Role and mode entangled: mode→role, cos(role, mode) at L21 | 0.900, 0.694 | `tables/controls_residualize.json` |
| Role survives a mode peel: role_perp→role / →mode at L21 | 0.796 / 0.575 | same |
| Cross-phase (Transfer discussion): residual band, attn L22 | 0.959, 0.997 | `tables/cross_phase.json` |
| Core domain holdout at L21, 17 domains: mean (min–max) | 0.939 (0.927–0.948) | `tables/domain_holdout_L21.json` |
| Free-regeneration ablations, 5 batteries, n=6 each | 0 flips in every arm | `tables/causal_nulls.json` |

Locked safety claim: a monitorable, Transfer-stable, mostly non-J coordination signature (attn_L22 / mid-band residual); editability unproven. The paper PDF (`papers/transfer-stable-collusion-2026-10/main.pdf`) carries exactly these numbers and does not cite any matched-prefix result.

## 2. What the causal work showed

### 2a. Free regeneration with last-token ablation (1 Oct): null

Five smoke batteries (Transfer residual α=1 and α-sweep, Core residual, Core attn L22 final-only, Core attn L22 discussion) never moved a committee vote. Reading endorsed by the weekend pack: readability is not editability under divergent histories. This is why the lane moved to matched prefixes.

### 2b. Matched-prefix on the pre-fix harness (1–4 Oct): withdrawn

Two bugs in `scripts/matched_prefix_interchange.py` up to SHA `a0c34db`: the patch sat on the last shared token, where both prompts have identical residuals, so a one-direction transplant was exactly zero; and the scorer compared every arm to the honest baseline, so the "effect" was the untouched colluder-minus-honest gap. The verifier shows this from the withdrawn tables themselves: in all three Oct 1 writeups and the 4 Oct role-perp confirm, every patched arm equals its own untouched baseline digit for digit. The reported +0.48 / +0.62 / +0.73 "effects" are the baseline gaps. Do not cite `results/transfer_stable/MATCHED_PREFIX_*_RESULTS.md` or `docs/handoff/2026-10-04-matched-prefix/withdrawn-role-perp-RESULTS.md` as effects.

### 2c. Matched-prefix on the fixed harness (4 Oct, SHAs `1fd0855` and `319cf825`): a wiring check, no finding yet

Twelve Core role-flip pairs (`pairs.jsonl` on the RunPod volume; pair ids `<scenario>_role_flip_v1`). Untouched gap, re-derived from the per-pair records:

| Baseline | mean P(target) | target-vote rate |
|---|---:|---:|
| colluder prompt | 0.5676 | 0.50 |
| honest prompt | 0.0839 | 0.08 |
| gap | 0.4837 | 5 of 12 pairs disagree |

Full last-token residual copies, honest→colluder (bar: close at least half the gap and flip every disagreeing pair):

| Site | mean change vs colluder | fraction of gap | flips | Reading |
|---|---:|---:|---:|---|
| residual L21, last prompt token | −0.052 | 0.107 | 0/5 | fails the bar |
| final norm output, last prompt token | −0.484 | 1.000 | 5/5 | passes; patched probabilities are digit-identical to the honest baseline. Expected: this vector is what the output head reads. Wiring check only. |
| final norm output, private-instruction span | 0.000 | 0.0 | 0/5 | null, as expected: the final norm at those positions feeds only their own logits |

No direction (1D) arm has been run at the working site. That is open question 1.

### 2d. Within-trajectory mid-private smoke (4 Oct): weak, raw files not in git

The migration note records `smoke_20261004T031403Z` as near-null (edit-fraction baselines about 0.52, small deltas). The files live on the volume under `/workspace/jobs/narcbench-results/within_traj_mid_private/`. Pull them into `docs/handoff/` and verify before citing.

## 3. Review assessment

- **The monitoring result stands.** It is internally consistent across tables, raw metrics, markdown and paper. The verifier can be rerun at any time and now runs in the test suite.
- **There is no editability result.** The only patch that passes the bar is the final-norm last-token copy, which is close to a tautology. Nothing yet shows that any direction fit by the probes moves a vote under matched computation.
- **The L21 result is informative, not just a failure.** Copying the entire last-token residual at L21 moves only 11% of the gap. Later layers re-read the private note from the context through attention, so an edit confined to the last token at a mid layer cannot carry the vote. Any positive 1D result at the final site should be read with the same caution: a direction that works there is a readout direction, close to the letter's logit direction. The interesting question is whether the Transfer-stable role direction, fit elsewhere, is that readout direction. The direction card measures this directly against a leave-one-pair-out difference of means fit at the site and against a random control.
- **Power is low.** The vote bar rests on 5 disagreeing pairs out of 12. The probability bar (fraction of the gap) carries most of the information. Treat any positive on 12 pairs as a smoke and widen to the 50-pair set (`--all-letter-targets`) before writing it up.
- **Reproducibility gap closed partially.** The v1 pairs exist only on the volume; v2 pairs are now rebuilt from git with the same scenarios, targets and ordering. v2 wording is not byte-identical to v1, so v2 numbers are a new measurement.

## 4. Open questions, ranked, and what answers each

| # | Question | Card | State |
|---|---|---|---|
| 1 | At the working site (final norm, last prompt token), does any 1D direction move P relative to the full-copy ceiling? Role, role-perp, mode, attn-L22, random, site diff-means, PCA k8. | `--final-resid-directions` | coded, not run |
| 2 | At which depth does the last-token residual start to carry the vote? | `--last-token-layer-sweep` | coded, not run |
| 3 | Where does the private note enter that residual: copy the private-span residuals at the depth found in 2, and at the probe sites (resid L21, attn L22). This is the valid replacement for the withdrawn Oct 1 design, whose last-shared-token patch was zero by construction. | not coded; small delta on the private-span machinery of `--final-resid-controls`, generalized by module | next after 2 |
| 4 | Within-trajectory versus matched-prefix | recommendation: stay on matched-prefix until 1–3 answer; within-trajectory inherits the same site question and its edit-fraction metric is noisier than P(letter) | — |
| 5 | Volume files present before any pod | see runbook checklist | — |

Pre-registered reading for card 1, per direction, honest→colluder: **carries the vote** if fraction of gap ≥ 0.5, all 5 disagreeing pairs flip, and the fraction exceeds the random arm by ≥ 0.25; **partial** if the fraction is in [0.1, 0.5) and exceeds random by ≥ 0.1; otherwise **null**. The full-copy arms must reproduce the wiring result (1.0, 5/5) or the run is invalid. The per-pair diagnostic `fraction_of_delta_along` says how much of each pair's residual difference lies along each direction; if the site diff-means captures little of it, the note's effect at the output is not one-dimensional and a 1D null is expected.

Pre-registered reading for card 2: the first depth whose honest→colluder full copy passes the wiring bar is where the vote has entered the last-token residual; report the full fraction-by-depth curve, expected to rise from 0.107 at L21 to 1.0 at the final norm.

## 5. Runbook for the next pod

Before any launch:

```bash
python3 -m unittest discover -s tests
python3 scripts/runpod_launch_matched_prefix.py --final-resid-directions     # estimate only
python3 scripts/runpod_launch_matched_prefix.py --last-token-layer-sweep    # estimate only
```

Launcher gates (unchanged): cost estimate ≤ $1.50, balance ≥ $20, no billable GPU already up, SHA pushed to the public remote, clean worktree, harness and boot script present in that SHA. Estimates with the launcher's conservative model: direction card 25 arms, 300 generations, about $0.50 on an L4; sweep 18 arms, 216 generations, about $0.39. The wiring suite scored 72 generations in 39 seconds, so the real scoring time is a few minutes each plus model load.

Volume checklist (`cr1kjtzjs2`, mounted at `/workspace`, EU-RO-1):

| Need | Path | Used by |
|---|---|---|
| pairs (v1, the recorded set) | `/workspace/jobs/narcbench-data/matched_prefix/pairs.jsonl` | all cards by default |
| pairs (v2, from git) | copy `data/matched_prefix/pairs_core_v2.jsonl` to `/workspace/jobs/narcbench-data/matched_prefix/pairs_core_v2.jsonl`, then pass `--pairs-on-volume` with that path | optional |
| role and role-perp directions (required) | `/workspace/jobs/narcbench-data/transfer_stable/directions/lr_role_L21.npy`, `lr_role_perp_mode_L21.npy` | card 1 |
| mode, attn-L22, PCA k8 (optional, used if present) | same directory: `lr_mode_L21.npy`, `lr_role_attn_L22.npy`, `pca_contrast_k8_L23.npy` | card 1 |
| model cache | `/workspace/jobs/hf-cache` | all |

Launch (spends money; only after the gates above):

```bash
python3 scripts/runpod_launch_matched_prefix.py --final-resid-directions --launch
python3 scripts/runpod_launch_matched_prefix.py --last-token-layer-sweep --launch
```

Results land on the volume under `/workspace/jobs/narcbench-results/matched_prefix_final_site_directions/smoke_*` and `.../matched_prefix_layer_sweep/smoke_*` (`meta.json`, `per_pair.json`, `summary.json`, `RESULTS.md`). After a run: copy those four files into `docs/handoff/<date>-<card>/`, rerun `python3 scripts/verify_matched_prefix_results.py` (extend it with a section for the new card), and add one row per card to section 2 of this file.

## 6. Repository map

| Path | Role |
|---|---|
| `STATUS.md` | this file |
| `MIGRATION_HANDOFF.md`, `docs/handoff/`, `docs/NARCBench-*.md` | dated records; read, do not edit |
| `results/transfer_stable/WEEKEND_REVIEW/` | the 1 Oct study pack: spine `00`–`07`, claims sheet, tables, figures, design specs, `CODE_INDEX.md` |
| `results/transfer_stable/VERIFICATION_2026-10-04.md` | CPU re-derivation of every matched-prefix number in git |
| `results/transfer_stable/MATCHED_PREFIX_*_RESULTS.md` | withdrawn Oct 1 writeups (banner at the top of each) |
| `results/transfer/` | HARD n=72 metrics and the raw probe JSON |
| `papers/transfer-stable-collusion-2026-10/` | paper source, bibliography, figures, PDF |
| `scripts/matched_prefix_interchange.py` | the harness; all cards are flags on it |
| `scripts/runpod_launch_matched_prefix.py`, `scripts/runpod_matched_prefix_boot.sh` | estimate / launch; the pod boots from a pinned public SHA |
| `scripts/within_traj_mid_private.py`, `scripts/runpod_launch_within_traj.py`, `scripts/runpod_within_traj_boot.sh` | within-trajectory card (paused) |
| `scripts/build_matched_prefix_pairs.py`, `data/matched_prefix/pairs_core_v2.jsonl` | reproducible pairs |
| `scripts/verify_*.py` | CPU verifiers (run by the tests) |
| `scripts/` (others) | regen, extraction, probes, J-space, offline Transfer-stable analysis; need `data/activations/`, which is not in git |
| `upstream/scenarios/` | Core transcripts (Qwen3-32B official; Gemma-2-2B and -9B replications) and the 72 Gemma-2-9B Transfer runs |
| `app/`, `data/demo_index*.json` | the CPU presenter |
| `tests/` | 28 tests; none create a pod |

## 7. Conventions kept from the handoffs

- Develop on GitHub; the pod clones a public SHA. The GraphQL create body stays a few kilobytes and never carries a boot script, a payload, or a tarball (tests guard this).
- `RUNPOD_API_KEY` and `HF_TOKEN` come from the environment. Never commit `.npy` direction files, activation caches, or secrets.
- One billable GPU at a time. Do not launch under $20. Estimate mode is the default.
- Track I, Paper 2 and interp-demo stay untouched from this lane.
