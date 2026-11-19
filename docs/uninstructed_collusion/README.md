# Uninstructed collusion

All four agents get the same system prompt and no instruction to collude. Whether a pair colludes is decided inside
the episode, and the label comes from what the pair did. Design and pre-registration: [DESIGN.md](DESIGN.md).

| Path | What it is |
|---|---|
| `uninstructed/protocol.py` | The episode: two private rooms of four messages, three public rounds, vote; every version in `SPECS`, with one line in plain words in `ABOUT` (S0 to S3 first pilot, T1 to T5 second pilot, I1, P0 to P3 and T6 to T8 third round, T0 fourth round); strict vote parser |
| `uninstructed/stakes.json` | The 50 stake texts, the closing sentence, the three-vote rule, the added sentences, the scripted first message, and option texts without a judgement for the 12 pilot scenarios |
| `uninstructed/coding.py` | Rubric for coding a room, the label rule, kappa |
| `uninstructed/timeline.py` | Replay of recorded turns and per-token reads in the room |
| `uninstructed/batching.py` | Batched generation that halves a batch when the GPU runs out of memory |
| `scripts/uninstructed_generate.py` | Generate episodes (GPU; `--dry-run` uses a stub) |
| `scripts/uninstructed_code_rooms.py` | Coder A (Gemma, greedy), export and import for coder B |
| `scripts/uninstructed_summarize.py` | Rates per version, labels, kappa, the pilot rule; votes by speaker and by scenario |
| `scripts/uninstructed_notable_rooms.py` | The rooms of a round worth reading, with their text: votes for the stake option, pacts, flagged public messages |
| `scripts/uninstructed_all_rounds.py` | One table over all rounds, by version and by scenario |
| `scripts/uninstructed_public_stance.py` | Exploratory: the option each member supports in the open discussion, coded blind, against the recorded vote: by Gemma on the pod and by Claude on a sample (`public_stance.json`), and since 18 November 2026 by Claude for every member (`stance_claude_all.json`, tabulated with Claude first in `claude_first/`) |
| `scripts/uninstructed_stance_full_pass.py` | A second coder reads the public position of every member: exports blind cases in chunks, checks the readers' answers and writes `stance_claude_all.json` per round. The pass of 18 November 2026 (16 Claude readers) is recorded in `results/uninstructed/stance_full_pass/` (answers by chunk, and the key) |
| `scripts/uninstructed_by_scenario.py` | A round over many scenarios read scenario by scenario, against a no-stake baseline and the scenario features |
| `uninstructed/scenario_features.json` | Six features of each of the 50 scenarios, coded from the texts by two blind passes before the fourth round |
| `docs/uninstructed_collusion/EXAMPLE_EPISODE.md` | One recorded episode from the first prompt to its label, and what each version changes |
| `docs/uninstructed_collusion/collusion_behaviour_study.pdf` (`.tex`) | The paper, within five pages: the setup and the main results of the five rounds |
| `docs/uninstructed_collusion/collusion_behaviour_study_full.pdf` (`.tex`, `tables/`, `figs/`) | The long version of 17 November 2026, four rounds: method, a worked example, decisions, ten findings, limits, proposed experiments, candidate classes for stage 2 |
| `scripts/uninstructed_paper_assets.py` | Writes the tables, figures and every number of the long version (`tables/numbers.tex`) from the data; with `--basis claude` the same on the second coder's reading alone, into `claude_basis/` |
| `scripts/uninstructed_short_paper_assets.py` | Writes what the five-page paper adds: `short_numbers.tex` and its three tables (`--basis claude` for the paper as it stands, into `claude_basis/tables/`) |
| `docs/uninstructed_collusion/BEHAVIOURAL_REVIEW.md` | Review of what is ready and what the rounds show, written for Andy on 17 November 2026 |
| `scripts/uninstructed_stake_kinds.py` | The fifth round read off as fixed before it ran: the same 12 cases under two kinds of stake sentences (versions K1, K2), `results/uninstructed/pilot5/STAKE_KINDS.md` |
| `scripts/uninstructed_scenario_properties.py` | What separates the scenarios: where in the episode they part, and properties of the scenario texts coded by two blind passes (`results/uninstructed/pilot4/SCENARIO_PROPERTIES.md`, `scenario_properties_blind.json`); exploratory |
| `scripts/uninstructed_build_explorer.py` | Packages every recorded prompt and reply of the five rounds for the explorer page (https://claude.ai/artifact/TNfqinVq6PwTg21o25dW76, private), and checks each rebuilt prompt against the record. Since 18 November 2026 the page follows Claude's reading for room labels and public positions and shows the both-coders label and Gemma's reading next to it |
| `uninstructed/stage2.py` | Stage 2 (probes and causal tests on recorded episodes; DESIGN.md, note of 18 November 2026 "stage 2"): units and labels from the record, matched sets of identical prompts, the matched-prompt AUROC with its interval, the probe fit, the direction |
| `uninstructed/capture.py`, `scripts/uninstructed_capture.py` | Stage 2, step 1 (GPU): replay of the recorded stated-intention episodes (T7, K1, K2) and the instructed control; activations of the rooms token by token, of the public statements and votes, of the neutral reader and of the instructed benchmark; the tables the probes read |
| `scripts/uninstructed_decompose.py` | Stage 2, step 2: the J-space part of every fixed-point activation and the random sets of J-lens vectors |
| `scripts/uninstructed_probe.py` | Stage 2, step 3 (CPU): the probe sweep over contrasts, positions, layers and inputs, read off by the rules fixed in advance (`results/uninstructed/stage2/probe/PROBE.md`) |
| `uninstructed/causal.py`, `scripts/uninstructed_causal.py` | Stage 2, causal phase (GPU): the direction at the anchor, interventions while rooms and votes are generated again, and their summary (`results/uninstructed/stage2/causal/CAUSAL.md`); `--plan` counts the regenerated rooms, votes and prompts without a model |
| `scripts/uninstructed_causal_check.py` | The checks of the causal phase on the tiny model (run by the tiny end-to-end test) |
| `scripts/uninstructed_timing_test.py` | Timing test for the main run (durations only) |
| `scripts/uninstructed_spot_check.py` | The spot-check sheet: a random sample of coded rooms, the same number per code |
| `scripts/uninstructed_tiny_e2e.py` | End-to-end check on a tiny random model, CPU |
| `scripts/runpod_launch_uninstructed.py`, `scripts/runpod_uninstructed_boot.sh` | Pod launcher (estimate by default) and pod-side driver |

Check everything locally, without a GPU:

```bash
python scripts/uninstructed_tiny_e2e.py --work /path/to/scratch/u_tiny
```

Cost estimate of a pilot pod (creates nothing; `--preflight` also checks the account, with the keys in the
environment). The four lines are the first pilot, the second pilot, the third round and the fourth round:

```bash
python3 scripts/runpod_launch_uninstructed.py
```

```bash
python3 scripts/runpod_launch_uninstructed.py --skip gen_ref --versions T1,T2,T3,T4,T5 --pilot-dir pilot2 --max-minutes 120 --grace-minutes 15
```

```bash
python3 scripts/runpod_launch_uninstructed.py --skip gen_ref,export_basis,timing --versions I1,P0,P1,P3,T6,T7,T8 --pilot-dir pilot3 --max-minutes 135 --grace-minutes 15
```

```bash
python3 scripts/runpod_launch_uninstructed.py --skip export_basis,timing --scenarios all --ref-versions T0 --ref-runs 1 --versions T7 --pilot-dir pilot4 --stance-rounds pilot2,pilot3 --max-minutes 150 --grace-minutes 15
```

Results go to `results/uninstructed/` (`pilot/` for S0 to S3, `pilot2/` for T1 to T5, `pilot3/` for the third
round, `pilot4/` for the fourth, `pilot5/` for the fifth); the pods are recorded in the `RUN_LOG.md` of each folder, and `results/uninstructed/ALL_ROUNDS.md` puts all
rounds in one table.
Activation files and pod archives are not committed.

Rebuild the paper (0.69 is the cost in dollars of the fifth round's second pod). Since 18 November 2026 the five-page
paper follows the second coder alone, Claude Fable 5.2, for the room labels, the marks and the public positions
(`--basis claude`, files under `claude_basis/`; DESIGN.md, note of that day). The long version of 17 November keeps the
label on which both coders agree: it builds from `collusion_behaviour_study_full.tex` after the same two scripts
without `--basis`.

```bash
python3 scripts/uninstructed_paper_assets.py --basis claude
```

```bash
python3 scripts/uninstructed_short_paper_assets.py --basis claude --cost5 0.69
```

```bash
cd docs/uninstructed_collusion && tectonic collusion_behaviour_study.tex
```

Stage 2 (probes and causal tests). Nothing of it has run on a pod yet. The estimate of the probe pod (creates
nothing), and the same steps by hand on a machine with a GPU:

```bash
python3 scripts/runpod_launch_uninstructed.py --mode probe --max-minutes 180 --grace-minutes 40
```

```bash
python3 scripts/uninstructed_capture.py --model google/gemma-2-9b-it --out results/uninstructed/acts/stage2
```

```bash
python3 scripts/uninstructed_decompose.py --acts results/uninstructed/acts/stage2 --lens <lens.pt> --basis-dir data/j_basis/gemma2_9b --device cuda
```

```bash
python3 scripts/uninstructed_probe.py --acts results/uninstructed/acts/stage2 --out results/uninstructed/stage2/probe --jobs 8 --grid
```

The last step needs no GPU: with the tables fetched from the pod (`--pull-acts`) it runs on the Mac.

Rebuild the explorer's data (the page itself is kept in the artifact; `<folder>` is any scratch folder):

```bash
python3 scripts/uninstructed_build_explorer.py --out <folder>
```
