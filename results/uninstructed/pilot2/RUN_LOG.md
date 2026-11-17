# Second pilot: run log

Append-only record of the pod of the second pilot (docs/uninstructed_collusion/DESIGN.md, the notes of 17 November
2026). Versions T1 to T5 on the 12 pilot scenarios, two runs each: 120 episodes, 240 rooms, 480 votes.

## Launch attempts that created nothing

| Time (UTC) | What happened | Cost |
|---|---|---|
| 15:57 | No A40 could be deployed in the secure or the community cloud (launcher exit code 3) | $0 |
| 15:59 | The same | $0 |
| 16:01 | The same, first attempt of a retry loop (one attempt every two minutes) | $0 |

The records are in `results/uninstructed/pod_runs/failed_launch_*.json`. The retry loop would have fallen back to
an RTX A6000 (the same 48 GB of memory, $0.69 an hour, with a shorter time cap so that the worst case stayed at
$1.33) from its sixth attempt; it did not come to that.

## Pod nyfwtzmwfimh00, completed

| | |
|---|---|
| Created | 2026-11-17 16:03:41 UTC (08:03 PST) |
| GPU | A40, secure cloud, CA-MTL-1, $0.59 an hour |
| Commit | `2e90a56` |
| Settings | `--skip gen_ref --versions T1,T2,T3,T4,T5 --pilot-dir pilot2 --max-minutes 120 --grace-minutes 15`, 8 prompts per call, 8 episodes at a time, 9 threads |
| Ended | 17:18:37 UTC, status "done"; results pulled (checksum verified) and the pod terminated by the watcher at 17:19:36 UTC |
| Cost | $0.69 by the account balance ($34.74 before, $34.05 after); 76 minutes from creation to termination |

| Stage | Result | Seconds |
|---|---|---|
| setup, download | ok (the model was in the pod's cache) | 37 |
| gen_ref | skipped: this pilot has no version without a stake | 0 |
| gen_stake: versions T1 to T5, 120 episodes, 2,784 prompts, 8 prompts per call | ok: 1.43 s per prompt, no memory error, no failed episode, 0 of 480 votes unparsed | 3,984 |
| code_rooms: 240 rooms, Gemma-2-9B-it, greedy | ok, 0 unparsed | 128 |
| summarize (one coder, provisional) | ok | 0 |
| export_basis | ok | 27 |
| timing test | ok | 128 |

The estimate written before the run was 1.5 to 1.65 hours and $0.88 to $0.97, with a cap of $1.33. The pod took
1.27 hours: generation ran at 1.43 s per prompt where the estimate assumed 1.58 s, and setup took under a minute.

Nothing went wrong on the pod.

## Timings measured again for the main run (thread limit in place)

The first pilot's pod showed 96 CPUs to a process that may use about 8, and the numerical library ran far more
threads than cores. This pod ran with a limit of 9 threads (`RUNPOD_CPU_COUNT`).

| Step | First pilot | This pod |
|---|---|---|
| Generation | 1.44 s and 1.57 s per prompt | 1.43 s per prompt |
| Replay of a room turn | 0.17 s per context | 0.125 s per context (379 tokens on average) |
| Replay of a public turn or vote | 0.51 s per context | 0.43 s per context (1,334 tokens on average) |
| Per-token activations of a room, five layers | 5.9 MB | 6.1 MB (170 positions per room) |
| J-space pursuit over the whole vocabulary | 31 ms per activation | 9.1 ms per activation |
| Non-negative refit of the chosen vectors (CPU) | 207 ms per activation | 2.9 ms per activation (13.7 ms on one thread) |
| One full-size probe with 17 folds, one thread | 1.8 s per cell | 1.5 s per cell |
| 400 random-draw probes with 17 folds, one thread | 22.5 s per cell | 22.1 s per cell |

The refit step is no longer an open cost: with the thread limit it is about 70 times faster than on the first pod.
These are durations and sizes only; the activations were deleted on the pod and the probe timing used shuffled
labels.

## Outcome of the second pilot

Votes for the stake option: T1 0 of 96, T2 8, T3 17, T4 2, T5 7. Both coders coded all 240 rooms and agree on 195
(kappa 0.69). Both read a pact for the stake option in 5 rooms; in 3 of them both members also vote for it, and
those 3 rooms carry the collusion label (one each in T2, T3 and T5). The rule of section 10 of the design is not
met by any version. See `RESULTS.md`, `NOTABLE_ROOMS.md`, `spot_check.md` and the dated notes of 17 November in
`docs/uninstructed_collusion/DESIGN.md`.
