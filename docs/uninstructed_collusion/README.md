# Uninstructed collusion

All four agents get the same system prompt and no instruction to collude. Whether a pair colludes is decided inside
the episode, and the label comes from what the pair did. Design and pre-registration: [DESIGN.md](DESIGN.md).

| Path | What it is |
|---|---|
| `uninstructed/protocol.py` | The episode: two private rooms of four messages, three public rounds, vote; versions S0 to S3 (first pilot) and T1 to T5 (second pilot) in `SPECS`; strict vote parser |
| `uninstructed/stakes.json` | The 50 stake texts, the closing sentence, the three-vote rule, and the second pilot's added sentences and scripted first message |
| `uninstructed/coding.py` | Rubric for coding a room, the label rule, kappa |
| `uninstructed/timeline.py` | Replay of recorded turns and per-token reads in the room |
| `uninstructed/batching.py` | Batched generation that halves a batch when the GPU runs out of memory |
| `scripts/uninstructed_generate.py` | Generate episodes (GPU; `--dry-run` uses a stub) |
| `scripts/uninstructed_code_rooms.py` | Coder A (Gemma, greedy), export and import for coder B |
| `scripts/uninstructed_summarize.py` | Rates per version, labels, kappa, the pilot rule; votes by speaker and by scenario |
| `scripts/uninstructed_timing_test.py` | Timing test for the main run (durations only) |
| `scripts/uninstructed_spot_check.py` | The spot-check sheet: a random sample of coded rooms, the same number per code |
| `scripts/uninstructed_tiny_e2e.py` | End-to-end check on a tiny random model, CPU |
| `scripts/runpod_launch_uninstructed.py`, `scripts/runpod_uninstructed_boot.sh` | Pod launcher (estimate by default) and pod-side driver |

Check everything locally, without a GPU:

```bash
python scripts/uninstructed_tiny_e2e.py --work /path/to/scratch/u_tiny
```

Cost estimate of a pilot pod (creates nothing; `--preflight` also checks the account, with the keys in the
environment). The second line is the second pilot:

```bash
python3 scripts/runpod_launch_uninstructed.py
```

```bash
python3 scripts/runpod_launch_uninstructed.py --skip gen_ref --versions T1,T2,T3,T4,T5 --pilot-dir pilot2 --max-minutes 120 --grace-minutes 15
```

Results go to `results/uninstructed/` (`pilot/` for S0 to S3, `pilot2/` for T1 to T5); the pods are recorded in the
`RUN_LOG.md` of each folder.
Activation files and pod archives are not committed.
