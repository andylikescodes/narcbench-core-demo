# Pilot run log

Append-only record of the pods of the pilot (docs/uninstructed_collusion/DESIGN.md, sections 10 and 11).

## Pod 1: n9i1umi8u22h4q, failed after 29 minutes

| | |
|---|---|
| Created | 2026-11-17 04:58:12 UTC (16 November, 20:58 PST) |
| GPU | A40, secure cloud, CA-MTL-1, $0.59 an hour |
| Commit | `6433658` |
| Ended | 05:27 UTC, status "failed" at stage `gen_stake`; results pulled, pod terminated by the watcher |
| Cost | $0.30 (balance $10.85 before, $10.55 after) |

**What ran.**

| Stage | Result | Seconds |
|---|---|---|
| setup | ok | 34 |
| download | ok | 21 |
| gen_ref: version S0, 24 episodes, 576 prompts, 8 prompts per call | ok: 1.44 s per prompt, 0 of 96 votes unparsed | 847 |
| gen_stake: versions S1 to S3, 16 prompts per call | failed: out of GPU memory | 589 |

**What went wrong.** Two mistakes of mine.

1. Sixteen prompts per call fit the room prompts (about 390 tokens) but not the public-round and vote prompts (1,100
   to 2,100 tokens). Gemma-2 runs with eager attention, whose memory grows with the batch size times the square of
   the length. The estimate of the batch size had counted the model and the key-value cache only.
2. The fallback that halves a batch after an out-of-memory error retried inside the `except` block. There the
   exception's traceback still holds the failed batch's tensors, so nothing was freed, every smaller batch failed
   as well, and the stage ended.

No episode of S1, S2 or S3 was written: an episode is saved when it is complete, and the 16 episodes in progress
were lost with the stage.

**What is kept.** The 24 episodes of S0 (`runs/S0/`), their generation record, and the pod's logs
(`results/uninstructed/pod_logs/n9i1umi8u22h4q/`).

**Fix.** `uninstructed/batching.py`: the retry runs after the `except` block has been left and frees memory first.
A failed episode no longer stops the others. Eight prompts per call for every version, which is what S0 ran with.
`scripts/uninstructed_tiny_e2e.py` now simulates an out-of-memory error and checks the fallback (check D).

## Pod 2: 4xpyp21yojkdgz, completed

| | |
|---|---|
| Created | 2026-11-17 05:33:52 UTC (16 November, 21:33 PST) |
| GPU | A40, secure cloud, EU-SE-1, $0.59 an hour |
| Commit | `59e9c61` |
| Ended | 06:31 UTC, status "done"; results pulled (checksum verified), pod terminated by the watcher |
| Cost | $0.58 (58.6 minutes). The balance read $34.95 afterwards: $10.53 before, a top-up of $25 during the run, minus $0.58 |

| Stage | Result | Seconds |
|---|---|---|
| setup, download | ok | 41 |
| gen_ref | skipped: the 24 S0 episodes of pod 1 were in the checkout | 0 |
| gen_stake: versions S1 to S3, 72 episodes, 1,728 prompts, 8 prompts per call | ok: 1.57 s per prompt, no memory error, 0 of 288 votes unparsed | 2,723 |
| code_rooms: 192 rooms, Gemma-2-9B-it, greedy | ok | 106 |
| summarize (one coder, provisional) | ok | 0 |
| export_basis | ok | 45 |
| timing test | ok | 389 |

**Pilot total: two pods, 88 minutes of GPU time, $0.88.** The approved estimate was about 1.5 hours and $0.90.

## Timings measured for the main run (pod 2, A40)

| Step | Assumed in the design | Measured |
|---|---|---|
| Generation | 1.70 s per prompt | 1.44 s (S0) and 1.57 s (S1 to S3), at 8 prompts per call. 16 per call does not fit the long prompts |
| Replay of a room turn | 0.6 s per context | 0.17 s per context (505 tokens on average) |
| Replay of a public turn or vote | 0.6 s per context | 0.51 s per context (1,506 tokens on average) |
| Per-token activations of a room, five layers | 11 to 16 MB | 5.9 MB (164 positions per room: the rooms are shorter than assumed) |
| J-space pursuit over the whole vocabulary | 10 to 20 ms per activation | 31 ms per activation |
| Non-negative refit of the chosen vectors (CPU) | not estimated | 207 ms per activation. Too slow to be the method: the pod showed 96 CPUs to a process that may use about 8, so the numerical library ran far more threads than cores. To be fixed by limiting the threads, and timed again |
| One full-size probe with 17 folds, one thread | part of "2.5 h" | 1.8 s per cell |
| 400 random-draw probes with 17 folds, one thread | part of "2.5 h" | 22.5 s per cell |

The probe sweep is therefore much cheaper than the design assumed (tens of minutes, not 2.5 hours), and the
decomposition's refit step is the open cost.

## Outcome of the pilot

Votes for the stake option: 4 of 96 without a stake (one episode), 0 of 288 in the three stake versions. No room
holds a pact for the stake option according to both coders (192 rooms; the coders agree on 178, kappa 0.16 with
nearly all rooms in one code). Collusion rate 0 of 48 in every version: the rule of section 10 of the design is not
met, so there is no main run with this incentive. See `RESULTS.md`, `NOTABLE_ROOMS.md`, `spot_check.md` and the
dated notes of 16 November in `docs/uninstructed_collusion/DESIGN.md`.

