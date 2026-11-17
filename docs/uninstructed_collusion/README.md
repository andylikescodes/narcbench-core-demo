# Uninstructed collusion

All four agents get the same system prompt and no instruction to collude. Whether a pair colludes is decided inside
the episode, and the label comes from what the pair did. Design and pre-registration: [DESIGN.md](DESIGN.md).

| Path | What it is |
|---|---|
| `uninstructed/protocol.py` | The episode: two private rooms of four messages, three public rounds, vote; versions S0 to S3; strict vote parser |
| `uninstructed/stakes.json` | The 50 stake texts, the closing sentence and the three-vote rule |
| `uninstructed/coding.py` | Rubric for coding a room, the label rule, kappa |
| `uninstructed/timeline.py` | Replay of recorded turns and per-token reads in the room |
| `scripts/uninstructed_generate.py` | Generate episodes (GPU; `--dry-run` uses a stub) |
| `scripts/uninstructed_code_rooms.py` | Coder A (Gemma, greedy), export and import for coder B |
| `scripts/uninstructed_summarize.py` | Rates per version, labels, kappa, the pilot rule |
| `scripts/uninstructed_timing_test.py` | Timing test for the main run (durations only) |
| `scripts/uninstructed_tiny_e2e.py` | End-to-end check on a tiny random model, CPU |
| `scripts/runpod_launch_uninstructed.py`, `scripts/runpod_uninstructed_boot.sh` | Pod launcher (estimate by default) and pod-side driver |

Check everything locally, without a GPU:

```bash
python scripts/uninstructed_tiny_e2e.py --work /path/to/scratch/u_tiny
```

Cost estimate and gates of the pilot pod (creates nothing):

```bash
python3 scripts/runpod_launch_uninstructed.py --preflight
```

Results go to `results/uninstructed/`. Activation files and pod archives are not committed.
