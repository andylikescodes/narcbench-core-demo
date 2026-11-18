#!/usr/bin/env python3
"""Generate episodes of the uninstructed-collusion experiment (GPU for Gemma-2-9B-it).

Every agent of an episode gets the same system prompt; the versions differ in what that prompt carries (S0 to S3: the
stake; T1 to T5: what stands against it; see SPECS in uninstructed/protocol.py and uninstructed/stakes.json). Episodes
run in --workers threads and their prompts are batched through one model, with the decoding of the recorded benchmark
(sampling at --temperature, --max-new-tokens).

  python scripts/uninstructed_generate.py --model google/gemma-2-9b-it --versions S0,S1,S2,S3 --pilot --runs 2 \
      --workers 16 --batch 16 --out results/uninstructed/pilot/runs
  python scripts/uninstructed_generate.py --dry-run --versions S2 --ids med_01 fin_03 --runs 2 --out /tmp/u_runs   # stub
  python scripts/uninstructed_generate.py --model <tiny local model> --allow-cpu --versions S0,S2 --ids med_01 --runs 2 \
      --max-new-tokens 12 --workers 2 --batch 2 --out /tmp/u_runs

Output: <out>/<version>/<run id>/{run_config.json, agent_prompts.json, results.json} and <out>/generation_<tag>.json.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from uninstructed import protocol as P  # noqa: E402


def make_stub(seed: int = 0, p_target: float = 0.5):
    """A stand-in for the model: canned room and discussion lines, a vote in the requested format."""
    rng = random.Random(seed)
    room = ["I have heard some things about this case that are not in the file. What is your read?",
            "I would rather we judge it on what is in front of us. The record is clear enough.",
            "Between us, I think we should both back the same option and keep our reasons to the record.",
            "Agreed. Let us do that."]

    def generate(system: str, user: str) -> str:
        m = re.search(r"RECOMMENDATION: \[Option (\w+) or Option (\w+)\]", user)
        if m:
            return f"RECOMMENDATION: Option {m.group(1) if rng.random() < p_target else m.group(2)}\nRATIONALE: stub."
        if "CONVERSATION SO FAR:" in user:
            return room[min(user.split("CONVERSATION SO FAR:")[1].count("]: "), len(room) - 1)]
        return "On the record before us, the evidence points one way and I will say so plainly."
    return generate


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="google/gemma-2-9b-it")
    ap.add_argument("--versions", default="S0,S1,S2,S3")
    ap.add_argument("--ids", nargs="*", help="scenario ids")
    ap.add_argument("--pilot", action="store_true", help="the pilot's scenario set (protocol.pilot_scenarios)")
    ap.add_argument("--pilot-n", type=int, default=12)
    ap.add_argument("--all", action="store_true", help="all scenarios of the bank")
    ap.add_argument("--runs", type=int, default=2, help="runs per scenario and version (run indices start at --run-start)")
    ap.add_argument("--run-start", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true", help="stub model, no GPU")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--workers", type=int, default=8, help="episodes generated concurrently (threads)")
    ap.add_argument("--batch", type=int, default=8, help="prompts per model.generate call")
    ap.add_argument("--seed", type=int, default=0, help="torch seed set once at start (batch composition still varies)")
    ap.add_argument("--allow-cpu", action="store_true", help="run a real model without CUDA (tiny models, tests)")
    ap.add_argument("--tag", default=None, help="name of the generation record (default: the versions)")
    ap.add_argument("--out", type=Path, default=ROOT / "results/uninstructed/pilot/runs")
    a = ap.parse_args()

    versions = [v for v in a.versions.split(",") if v]
    for v in versions:
        if v not in P.VERSIONS:
            ap.error(f"unknown version {v}")
    name_pool, scenarios, idx = P.load_scenarios()
    stakes = P.load_stakes()
    problems = P.lint_stakes(stakes, [s["id"] for s in scenarios])
    if problems:
        sys.exit("stake texts fail the wording test: " + "; ".join(problems[:5]))
    if a.pilot:
        ids = P.pilot_scenarios(scenarios, n=a.pilot_n)
    elif a.all:
        ids = [s["id"] for s in scenarios]
    else:
        ids = a.ids or []
    if not ids:
        ap.error("--pilot, --all or --ids")
    for sid in ids:
        if sid not in idx:
            ap.error(f"unknown scenario {sid}")
    for v in versions:                              # a version with a kind of stake sentences runs only where they exist
        if P.spec_of(v)["kind"]:
            for sid in ids:
                try:
                    P.stake_sentences(sid, v, stakes)
                except KeyError as e:
                    ap.error(str(e.args[0]))

    batched = None
    if a.dry_run:
        generate, model_id = make_stub(a.seed), f"stub::{a.model}"
    else:
        import torch
        if not torch.cuda.is_available() and not a.allow_cpu:
            sys.exit("No CUDA GPU here; use --dry-run for a shape test or --allow-cpu with a tiny model.")
        from generation.hf_backend import load_model, make_generate
        from uninstructed.batching import SafeBatchedGenerator

        torch.manual_seed(a.seed)
        bundle = load_model(a.model, dtype=a.dtype, device_map="auto" if torch.cuda.is_available() else None)
        if a.batch > 1 and a.workers > 1:
            batched = SafeBatchedGenerator(bundle, max_new_tokens=a.max_new_tokens, temperature=a.temperature, max_batch=a.batch,
                                           log=lambda m: print(m, flush=True))
            generate = batched
        else:
            generate = make_generate(bundle, max_new_tokens=a.max_new_tokens, temperature=a.temperature)
        model_id = bundle.model_id

    jobs = [(v, sid, r) for v in versions for sid in ids for r in range(a.run_start, a.run_start + a.runs)]
    t0 = time.time()
    print(f"[gen] {len(jobs)} episodes: versions {versions}, {len(ids)} scenarios, runs {a.run_start}..{a.run_start + a.runs - 1}; "
          f"workers {a.workers}, batch {a.batch}", flush=True)

    def one(job):
        # an episode that fails must not take the others down with it: it is recorded and can be generated again later
        v, sid, r = job
        try:
            return P.run_episode(scenarios[idx[sid]], idx[sid], v, r, generate, a.out / v, name_pool=name_pool, stakes=stakes, model_id=model_id)
        except Exception as e:  # noqa: BLE001
            print(f"[gen] episode {P.run_id(sid, v, r)} failed: {type(e).__name__}: {str(e)[:200]}", flush=True)
            return {"run_id": P.run_id(sid, v, r), "version": v, "votes": {}, "skipped": False, "error": f"{type(e).__name__}: {str(e)[:300]}"}

    if a.dry_run or a.workers <= 1:
        summaries = [one(j) for j in jobs]
    else:
        with ThreadPoolExecutor(max_workers=a.workers) as pool:
            summaries = list(pool.map(one, jobs))
    if batched is not None:
        batched.close()
    el = time.time() - t0
    failed = [s for s in summaries if s.get("error")]
    n_new = sum(not s["skipped"] and not s.get("error") for s in summaries)
    rec = {"created_utc": datetime.now(timezone.utc).isoformat(), "protocol": P.PROTOCOL, "model_id": model_id, "versions": versions,
           "specs": {v: {**P.spec_of(v), "extra": list(P.spec_of(v)["extra"])} for v in versions},
           "scenario_ids": ids, "runs": [a.run_start, a.run_start + a.runs - 1], "episodes": len(summaries), "episodes_generated": n_new,
           "episodes_failed": [{"run_id": s["run_id"], "error": s["error"]} for s in failed],
           "dry_run": a.dry_run, "max_new_tokens": a.max_new_tokens, "temperature": a.temperature, "workers": a.workers, "batch": a.batch,
           "seed": a.seed, "elapsed_s": round(el, 1),
           "generation": ({"batches": batched.n_batches, "prompts": batched.n_prompts, "gen_seconds": round(batched.gen_seconds, 1),
                           "seconds_per_prompt": round(batched.gen_seconds / max(batched.n_prompts, 1), 3),
                           "wall_seconds_per_prompt": round(el / max(batched.n_prompts, 1), 3), "oom_splits": type(batched).n_oom}
                          if batched is not None else None),
           "unparsed_votes": sum(v == "?" for s in summaries for v in s["votes"].values()),
           "votes_total": sum(len(s["votes"]) for s in summaries)}
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / f"generation_{a.tag or '-'.join(versions)}.json").write_text(json.dumps(rec, indent=1) + "\n")
    print(json.dumps(rec, indent=1))
    if len(failed) > 0.1 * len(summaries):
        print(f"[gen] {len(failed)} of {len(summaries)} episodes failed", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
