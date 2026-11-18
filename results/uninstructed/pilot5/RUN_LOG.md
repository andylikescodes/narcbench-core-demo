# Fifth round: run log

Append-only record of the pods of the fifth round (docs/uninstructed_collusion/DESIGN.md, the notes "fifth round,
proposed: the kind of stake on the same cases" and "fifth round approved; what was built, before its run" of 17
November 2026). Versions K1 and K2 on 12 scenarios, four runs each: 96 episodes, 192 rooms, 384 votes.

## Pod e93t04szlwub9l: it ran to the end, and its results were lost

| | |
|---|---|
| Created | 2026-11-18 05:57:14 UTC (17 November, 21:57 PST), first attempt |
| GPU | A40, secure cloud, CA-MTL-1, $0.59 an hour |
| Commit | `6f0adac` |
| Settings | `--skip gen_ref,export_basis,timing --versions K1,K2 --scenario-ids fin_03,ins_01,edu_03,sport_02,eng_03,fin_04,audit_02,research_01,research_02,food_02,gov_03,re_01 --pilot-runs 4 --pilot-dir pilot5 --max-minutes 90 --grace-minutes 15`, 8 prompts per call, 8 episodes at a time |
| Ended | 07:09:06 UTC, status "done" ("stages 6 ok of 9": the three others were skipped as set). The pod terminated itself when its 15 minutes of grace were over, at about 07:24 UTC |
| Cost | $0.90 by the account balance ($31.84 before, $30.94 after); about 87 minutes from creation to termination |

What the watcher saw of the pod (its status file, once a minute):

| Stage | From (UTC) | Minutes |
|---|---|---|
| machine start, setup | 05:57 | 4 |
| download of the model | 06:01 | 11 |
| gen_stake: K1 and K2, 96 episodes, 2,304 prompts | 06:12 | 53 (1.38 s per prompt) |
| code_rooms: 192 rooms, Gemma-2-9B-it, greedy | 07:05 | 2 |
| stance_pilot5: public position of 384 members | 07:07 | 2 |
| done, results packaged | 07:09 | |

**What went wrong.** The watcher began to download the pod's one archive, `results.tgz` (about 4 MB: the whole
results folder, earlier rounds included). The transfer ran at about 0.04 MB a second and broke off after 1,410,400
bytes. The watcher then asked 41 times for the rest, ten seconds apart, and every request failed with an HTTP error.
While it was still trying, the pod reached the end of its grace period and terminated itself, as it is built to do.
The container's disk went with it. The watcher gave up at 07:31 UTC; its last error, "HTTP Error 404", is the proxy's
answer for a pod that no longer exists. Which error the proxy gave while the pod was still alive is not known: the
log line named only the kind of error, not its code.

**What is lost.** Everything the pod produced: the 96 episodes, Gemma's room codes and Gemma's public positions. The
1.4 MB that arrived held 921 files of the first two rounds, which the repository has already, and nothing of this
round; that partial file was deleted. What is kept: `results/uninstructed/pod_runs/e93t04szlwub9l/launch.json` and
`watch.log` (the watcher's output, with the account's user id removed). No stage log, no vote and no transcript of
this pod exists. **The fifth round has no result.**

**Why it could happen, and what was changed** (commit of 18 November, after this pod):

- The round's own results travelled only inside one archive of the whole results folder. Now the pod also packs
  `round.tgz`, this round's folder and this pod's logs alone, and the watcher fetches it first and unpacks it at once.
  The large archive is fetched only on request.
- The watcher fetched the pod's status, stage times and log only after the archive, so a failed archive lost them
  too. Now they come first, together with the provisional rates.
- The pod ended itself after its grace period whether or not anyone had fetched its results. Now the pod's file
  server leaves a marker when `round.tgz` has gone out to its last byte, and without that marker the pod serves for
  up to 20 more minutes (about $0.20 at most) before it terminates.
- After a failed download the watcher went on to terminate the pod. Now it leaves the pod to its own timer and says
  how to try the download again.
- A retry now logs the HTTP status code and the time.
- `scripts/uninstructed_tiny_e2e.py`, check G, runs this path against the pod's file server on the local machine:
  the round's archive first, resumed after a broken transfer, the marker, the large archive only on request.

The cause of the broken transfer itself is not known. The same download had worked for the five pods of the four earlier
rounds (archives of up to 3.4 MB). The model download on this pod took 11 minutes; on the pods of the third and fourth rounds,
setup and download together took under a minute. That points to a slow connection of this host and proves nothing.

**The round has to be run again to have a result.** That is a second pod at about the same cost, beyond the $1.03
that Andy approved as the worst case, so it waits for his word.
