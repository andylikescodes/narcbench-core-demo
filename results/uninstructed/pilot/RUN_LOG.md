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
