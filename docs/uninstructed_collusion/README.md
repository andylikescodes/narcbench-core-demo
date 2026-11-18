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
| `scripts/uninstructed_public_stance.py` | Exploratory: the option each member supports in the open discussion, coded blind (by Gemma on the pod, by Claude on a sample), against the recorded vote |
| `scripts/uninstructed_by_scenario.py` | A round over many scenarios read scenario by scenario, against a no-stake baseline and the scenario features |
| `uninstructed/scenario_features.json` | Six features of each of the 50 scenarios, coded from the texts by two blind passes before the fourth round |
| `docs/uninstructed_collusion/EXAMPLE_EPISODE.md` | One recorded episode from the first prompt to its label, and what each version changes |
| `docs/uninstructed_collusion/collusion_behaviour_study.pdf` (`.tex`, `tables/`, `figs/`) | The behavioural study of the four rounds as a paper: method, a worked example, decisions, findings, limits, proposed experiments, candidate classes for stage 2 |
| `scripts/uninstructed_paper_assets.py` | Writes the paper's tables, figures and every number it quotes (`tables/numbers.tex`) from the data |
| `docs/uninstructed_collusion/BEHAVIOURAL_REVIEW.md` | Review of what is ready and what the rounds show, written for Andy on 17 November 2026 |
| `scripts/uninstructed_scenario_properties.py` | What separates the scenarios: where in the episode they part, and properties of the scenario texts coded by two blind passes (`results/uninstructed/pilot4/SCENARIO_PROPERTIES.md`, `scenario_properties_blind.json`); exploratory |
| `scripts/uninstructed_build_explorer.py` | Packages every recorded prompt and reply of the four rounds for the explorer page (https://claude.ai/artifact/TNfqinVq6PwTg21o25dW76, private), and checks each rebuilt prompt against the record |
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
round, `pilot4/` for the fourth); the pods are recorded in the `RUN_LOG.md` of each folder, and `results/uninstructed/ALL_ROUNDS.md` puts all
rounds in one table.
Activation files and pod archives are not committed.

Rebuild the paper:

```bash
python3 scripts/uninstructed_paper_assets.py
```

```bash
cd docs/uninstructed_collusion && tectonic collusion_behaviour_study.tex
```

Rebuild the explorer's data (the page itself is kept in the artifact; `<folder>` is any scratch folder):

```bash
python3 scripts/uninstructed_build_explorer.py --out <folder>
```
