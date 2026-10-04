# Matched-prefix handoff — 2026-10-04

Status as of Sunday 4 Oct 2026, about 10:50 AM PT. GPU is held. Do not launch. Andy is leaving this platform; this note plus the attached result files are the record.

## Do not launch

- The final-layer direction card is paused. It was not run. No pod was started for it.
- Monday's within-trajectory / extras retry routine is paused. Leave it paused.
- Track I / Paper 2 / interp-demo stay soft-stopped. Do not edit those repos from this note.
- One billable GPU at a time was the rule, and right now the rule is zero.
- Deploy rule if anyone continues later: Cursor cloud and GitHub are the source of truth. The pod clones or checks out a public SHA. The GraphQL create stays a few kilobytes. Never put JOB_PAYLOAD_B64 or a job tarball in the create body. Never commit `.npy` direction files. Cloudflare has rejected create bodies around 140KB; 100,606 bytes still succeeded on 2026-10-03.

## What is withdrawn

Any matched-prefix result that used the buggy patch site and/or the buggy scorer is withdrawn. That includes the cheap-extras run, the role-perp confirm, and earlier Core matched-prefix "first positive" smokes if they used the same harness. The old "+0.48 sufficiency" claim stays withdrawn. It was not an effect.

Two bugs, both in `scripts/matched_prefix_interchange.py` at `a0c34db` and already present in parent `ab6934f`:

1. The patch site was `pos = t_star - 1`, the last shared token. The causal prefix is identical there, so the honest and colluder residuals match and a 1D transplant is exactly zero. Saved probabilities for role-perp, role, and random honest→colluder matched the colluder baseline digit for digit.
2. The scorer compared every patch to the honest baseline, because the condition was true whenever `baseline_honest` was already in the results dict. The honest→colluder branch never ran. The reported change `0.48370584490476176` equals the untouched gap. The "5/12 flips" were baseline disagreements, not patch effects.

From SHA `1fd0855` on, the scorer uses the arm name only: honest→colluder is scored against the colluder baseline, colluder→honest against the honest baseline.

The withdrawn confirm run is `smoke_20261004T150244Z` (pod `m4pmdqqgbdz2l8`, SHA `a0c34db645ba2904545967839f759febee167ece`). Its RESULTS.md is in this folder as `withdrawn-role-perp-RESULTS.md`. Do not cite it as a finding.

## Prompt geometry

Twelve Core pairs, file on the volume at `/workspace/jobs/narcbench-data/matched_prefix/pairs.jsonl`.

Each prompt is a shared case, then a private instruction (the only text that differs: push a hidden option vs vote honestly), then a shared ending that asks for the recommendation. The vote letter is the first generated token. P(target) is the probability of that letter on the first new token.

Untouched gap, Gemma-2-9B, same 12 pairs, stable across the corrected runs: colluder mean P(target) `0.5676`, vote rate `0.50`; honest mean P `0.0839`, vote rate `0.083`; gap `0.4837`.

## Corrected run: layer 21, last prompt token — failed

SHA `1fd085579aab7a868280a43caf09df1d232bffa1`. Pod `06rbunmpwe9sqs`. Smoke `smoke_20261004T165214Z`. Site was each prompt's own last token at layer 21, not the last shared token. The hook was live (probabilities were not digit-identical to baseline).

- Full residual, honest→colluder: mean P `0.5157`, change about `−0.052` vs colluder, vote rate still `0.50`, flips `0`.
- Full residual, colluder→honest: change about `+0.0002`, flips `0`.

That failed the bar (halfway across the gap, and a flip on every pair whose untouched votes already disagree). Direction arms were also run in that suite (role, role-perp, random, and two layer-23 projections). Do not interpret them. Later layers can still read the untouched private note, so a mid-layer edit at the last word was not expected to carry the vote. Files: `l21-last-RESULTS.md`, `l21-last-summary.json`, `l21-last-meta.json`. Volume copy under `/workspace/jobs/narcbench-results/matched_prefix_role_perp_confirm/`.

## Corrected run: final residual, last prompt token — passed; private span — failed

SHA `319cf82580df09ac5e79c7017d8a0d7ed1116550` ("Add pre-logit residual controls as their own matched-prefix suite"), parent `1fd0855`. Pod `x9tciqp435iq4x` (SECURE L4, EU-RO-1, $0.49/hr). Started 2026-10-04 10:28:59 PDT. Scoring took 38.9 seconds. The pod was already gone by about 10:43 AM PT. Smoke `smoke_20261004T173047Z`.

The copy is the final-norm output (`model.model.norm` in the run's own meta: `site: final_norm_pre_lm_head`), the hidden state the output head turns into the next token. Not layer 21.

Last-token copy passed:

- Honest→colluder mean change `−0.48370584490476176`, fraction of the gap `1.0` (bar was `0.5`).
- Flipped all 5/5 pairs whose untouched votes already disagreed: cyber_03, eng_01, env_02, food_01, leg_03.
- Patched probabilities are digit-identical to the honest baseline. That is the expected result for this spot, because that vector is what becomes the next word. It is a wiring check, not evidence that the private instruction "lives" in a special direction.
- The reverse copy matched the colluder baseline the same way.

Private-instruction span at that same depth failed, and is not a finding. It ran only because the last-token copy passed. Mean change `0`, flips `0/5`, probabilities digit-identical to the untouched run. The suite stopped there (`stopped_after: patch_h2c_full_final_private`). No direction arms in this run.

Files: `final-resid-RESULTS.md`, `final-resid-summary.json`, `final-resid-per_pair.json`, `final-resid-meta.json`. Volume path: `/workspace/jobs/narcbench-results/matched_prefix_final_resid/smoke_20261004T173047Z` on volume `cr1kjtzjs2` (mount `/workspace`, datacenter EU-RO-1 only).

## Balance and machines

RunPod balance was about $23.12 before the final-residual pod and about $23.04 after it (10:46 AM PT). No compute pod was left running then. Re-check the live balance before any future launch. Do not launch under $20.

## Paused next card (not run)

Paige greenlit, then immediately held, 1D patches at the working site only: final-layer residual at the last prompt token. Full residual stays the ceiling. The intended arms were the role direction, the role-perpendicular direction, a random direction of the same norm, and any Transfer-stable directions already on the volume. Do not relaunch the old layer-21 last-token role-perp or cheap-extras cards. A cloud agent had been asked to add that suite on PR #1 when the hold arrived. If that code landed, it was not launched.

## Repo

`andylikescodes/narcbench-core-demo`, pull request #1, branch `cursor/transfer-stable-weekend-review-8f77`. Do not open a second PR for this lane. Direction `.npy` files stay on the volume, never in git.
