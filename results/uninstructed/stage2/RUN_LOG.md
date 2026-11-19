# Stage 2: run log

Append-only record of the pods of stage 2 (docs/uninstructed_collusion/DESIGN.md, the note "stage 2 (probes and causal
tests), fixed before any activation exists" of 18 November 2026 and the outcome notes after it). Stage 2 generates no
episode for the probes: it replays 244 recorded episodes (T7 of rounds 3 and 4, K1 and K2 of round 5, and the
instructed control I1), 976 members and 488 rooms.

## Pod 5sr5jjjh7f6wk8: the probes

| | |
|---|---|
| Created | 2026-11-19 06:04:59 UTC (18 November, 22:04 PST), first attempt |
| GPU | A40, secure cloud, CA-MTL-1, $0.59 an hour; 9 CPU threads, 503 GB of memory |
| Commit | `ffbb3f2` |
| Settings | `--mode probe --max-minutes 180 --grace-minutes 40 --pull-acts` |
| Ended | 08:22:40 UTC, status "done" ("stages 16 ok of 16"). The watcher fetched the results and the tables and terminated the pod at 08:25 UTC |
| Cost | $1.38 for the pod (2 h 20 min from creation to termination); $1.44 by the account balance ($29.79 before, $28.35 after), which includes the storage of the volume. Estimate $1.35, cap $2.36 |

| Stage | From (UTC) | Minutes |
|---|---|---|
| machine start | 06:05 | 4 |
| setup, download (model and lens were on the volume), unembedding export | 06:08 | 1 |
| s2_capture: replay of 1,952 room messages (0.126 s each), 3,904 public statements and votes (0.435 s), the neutral reader (732 passes, 0.19 s), the instructed benchmark (600 statements, 0.40 s), the tables | 06:09 | 44 |
| s2_decompose: the J-space part of 97,600 activations (9.3 + 3.9 ms each) | 06:54 | 22 |
| s2_probe: the sweep, 1,185 cells on 9 workers | 07:16 | 44 |
| s2_null_1 to s2_null_10: ten passes with shuffled labels, 2.3 minutes each | 08:00 | 23 |
| done, results packaged | 08:22 | |

**What came back.** `round.tgz` (0.3 MB: `probe/PROBE.md`, `probe/probe.json`, `nulls/seed1` to `seed10`,
`capture.json`, `decompose.json`, the pod's logs) and 31 table files, 2.2 GB, each checked against its checksum:
`results/uninstructed/acts/stage2/tables/` and `core/`. The tables are ignored by git and never committed. The tables of
the token clock (`grid_L*.npy`, about 1 GB) were not served, as planned; the token clock is in `probe.json`.

**After the pod, on the Mac, no GPU.** The passes with shuffled labels of the vote contrast were brought from 10 to 100
(`scripts/uninstructed_nulls_local.sh`, seeds 11 to 100, four and a half to nine minutes a pass, done at 06:24 PST on
19 November); `nulls/NULLS.md` and `nulls/nulls.json` are the summary of all 100. `explore/` holds the checks and the
two analyses added after the results (`RECHECK.md`, `READERS.md`, `after_room_without_vote_token/`), and `figures/`
the two timelines. The outcome is read in DESIGN.md, note "19 November 2026: stage 2, the probes: outcome".
