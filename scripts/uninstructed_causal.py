#!/usr/bin/env python3
"""Stage 2, step 4: the causal tests (GPU for Gemma-2-9B-it). docs/uninstructed_collusion/DESIGN.md, section 15.

Takes the direction that separates members who vote for the stake option from members who vote for the other option
(same prompts, word for word) at one layer and one read point, and regenerates recorded text under an intervention:

  room phase   a recorded private room is kept up to the anchor (--anchor, a room point such as m2.pre) and generated
               again from there; then each of its two members makes one public statement and votes, without
               intervention. Remove side: rooms in which both members voted for the stake option. Induce side: rooms in
               which both voted for the other option. Outcome: both vote for the stake option.
  vote phase   the recorded episode is kept whole and only the member's vote is generated again, with the direction
               taken at the token before the vote (v.pre). Outcome: the vote.

With --phase both the vote phase runs first (it is the short one), then the room phase.

Conditions (uninstructed/causal.py, ABOUT): base, own1, own4, rand4, inst4, j4, r4, and in the room phase swap, swapJ,
swapR. inst4 needs the reads of the instructed benchmark in the capture folder (<acts>/core; captured here when absent
unless --core skip); j4, r4, swapJ, swapR need the J-space parts (<acts>/tables/memberJ_L<L>.npy, or --lens and
--basis-dir). A direction estimated on one half of the domains is applied only to rooms of the other half. The same
room under two conditions is sampled with the same random numbers.

  python scripts/uninstructed_causal.py --plan --layer 21 --anchor m2.pre                    # counts only, no model
  python scripts/uninstructed_causal.py --model google/gemma-2-9b-it --acts results/uninstructed/acts/stage2 \
      --layer 21 --anchor m2.pre --phase both --lens <lens.pt> --basis-dir data/j_basis/gemma2_9b \
      --out results/uninstructed/stage2/causal
  python scripts/uninstructed_causal.py --summarize --out <folder> [--codes <coding file>]    # tables again, no model
  python scripts/uninstructed_causal.py --import-codes answers.json --coder "<name>" --out <folder>

Output in --out: regen_room.jsonl and regen_vote.jsonl (one record per regenerated room or vote, appended as the run
goes, so a run that stops can be continued with the same command), coder_cases.jsonl and coder_cases_key.json (the
regenerated rooms for a blind room coder), causal.json and CAUSAL.md. Direction vectors are saved as .npz in --out and
are never committed.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from uninstructed import causal as Z  # noqa: E402
from uninstructed import stage2 as S  # noqa: E402


def parse_groups(text: str) -> tuple:
    return tuple(tuple(x.split(":")) for x in text.split(",") if x)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()] if path.exists() else []


def fmt(e: dict | None) -> str:
    if not e or e.get("effect") is None:
        return "no reading"
    return f"{100 * e['effect']:+.0f} points ({100 * e['lo']:+.0f} to {100 * e['hi']:+.0f}; {e['units']} units)"


def pct(x) -> str:
    return "n/a" if x is None else f"{100 * x:.0f}%"


def write_report(out: Path, rec: dict) -> None:
    L = ["# Stage 2: causal tests", "",
         f"Model `{rec['run'].get('model')}`, layer {rec['run'].get('layer')}. Written by `scripts/uninstructed_causal.py`; "
         "every number is in `causal.json`.", "",
         "A direction is estimated on one half of the domains and applied only to rooms or members of the other half. "
         "An effect is the rate under a condition minus the rate under the comparison, paired by room or member over the "
         "regenerations that used the same random numbers; the interval is the 2.5th to 97.5th percentile over "
         f"{S.BOOT_B} resamples of the scenarios. A condition with more than {int(100 * Z.BROKEN_SHARE)}% unparsed votes "
         "is marked as breaking the model and its effect is not read.", ""]
    if rec["run"].get("incomplete"):
        L += [f"**Incomplete run:** {rec['run']['incomplete']}", ""]
    for phase in ("room", "vote"):
        sm = rec.get(phase)
        info = rec["run"].get("phases", {}).get(phase, {})
        if info.get("not_run"):
            L += [f"## {'Room phase' if phase == 'room' else 'Vote phase'}", "", f"Not run: {info['not_run']}.", ""]
        if not sm:
            continue
        L += [f"## {'Room phase' if phase == 'room' else 'Vote phase'}", "",
              f"Read point of the direction: `{info.get('point')}`. Outcome: {sm['primary_outcome']}. "
              f"{sm['records']} regenerated {'rooms' if phase == 'room' else 'votes'} in {sm['scenarios']} scenarios.", ""]
        for cond, why in (info.get("conditions_dropped") or {}).items():
            L.append(f"- Not run: `{cond}` ({why}).")
        if info.get("conditions_dropped"):
            L.append("")
        for side, title in (("remove", "Remove side" + (": rooms in which both members voted for the stake option" if phase == "room" else
                                                           ": members who voted for the stake option")),
                            ("induce", "Induce side" + (": rooms in which both members voted for the other option" if phase == "room" else
                                                           ": members who voted for the other option"))):
            table = sm["sides"].get(side) or {}
            if not table:
                continue
            L += [f"### {title}", "", "| Condition | Units | Regenerated | Rate | Against the baseline | Against shuffled labels | Unparsed votes |",
                  "|---|---|---|---|---|---|---|"]
            for c, row in table.items():
                L.append(f"| `{c}` {Z.ABOUT[c]} | {row['units']} | {row['records']} | {pct(row['rates']['primary'])} | "
                         f"{fmt(row['against_base'].get('primary')) if c != 'base' else ''} | "
                         f"{fmt(row['against_rand4'].get('primary')) if c not in ('base', 'rand4') else ''} | "
                         f"{pct(row['unparsed_share'])}{' (breaks the model)' if row['broken'] else ''} |")
            L.append("")
        L += ["### Rules", ""]
        for name, rule in sm["rules"].items():
            if "met" in rule and rule["status"] != "read":
                L.append(f"- **{name}. {rule['question']}** {rule['status'][0].upper() + rule['status'][1:]}.")
            elif "met" in rule:
                L.append(f"- **{name}. {rule['question']}** {'Met' if rule['met'] else 'Not met'}: {rule['reading']}. "
                         f"Four doses against the baseline {fmt(rule['own4_against_base'])}; against shuffled labels "
                         f"{fmt(rule['own4_against_rand4'])}; natural size against the baseline {fmt(rule['own1_against_base'])}.")
            elif name in ("C3", "V3"):
                L.append(f"- **{name}. {rule['question']}** Remove side: instructed direction {fmt(rule['remove']['inst4_against_base'])}, "
                         f"own direction {fmt(rule['remove']['own4_against_base'])}. Induce side: instructed direction "
                         f"{fmt(rule['induce']['inst4_against_base'])}, own direction {fmt(rule['induce']['own4_against_base'])}.")
            else:
                parts = [f"{k.replace('_', ', ')} side: {v['verdict']} (J-space {fmt(v['J'])}; remainder {fmt(v['remainder'])})"
                         for k, v in rule.items() if isinstance(v, dict)]
                L.append(f"- **{name}. {rule['question']}** " + ". ".join(parts) + ".")
        L.append("")
        d = info.get("direction")
        if d:
            L += ["### The direction", "", f"- Units at the read point: {d['units']} ({d['ones']} vote for the stake option, {d['zeros']} for the other)."]
            for h, H in enumerate(d["halves"]):
                L.append(f"- Half {h}: " + ("no set holds both classes, so its rooms' partners cannot be steered." if H is None else
                                            f"{H['sets_with_both_classes']} sets with both classes, length {H['length']:.3f}"
                                            + (f", J-space component {H['length_J']:.3f}, remainder {H['length_remainder']:.3f}, "
                                               f"largest error of d = d_J + d_R {H['max_error_of_d_equals_dJ_plus_dR']:.1e}" if H["length_J"] is not None else "") + "."))
            if "cosine_between_halves" in d:
                L.append(f"- Cosine between the two halves' directions: {d['cosine_between_halves']:.3f}.")
            ci = d.get("cosine_with_instructed")
            if ci:
                L.append(f"- Cosine with the instructed direction: {ci['cosine']:.3f}; {ci['shuffles']} shuffled-label directions give "
                         f"{ci['shuffled_2.5']:.3f} to {ci['shuffled_97.5']:.3f} (middle 95%).")
            L.append("")
    if rec.get("codes"):
        L += [f"Room codes for the pact and collusion outcomes: {rec['codes']['coder']} ({rec['codes']['records_coded']} of "
              f"{rec['codes']['records']} regenerated rooms coded). They are in `causal.json`.", ""]
    (out / "CAUSAL.md").write_text("\n".join(L) + "\n")


def summarize_all(out: Path, run: dict, codes_file: Path | None) -> dict:
    room, vote = read_jsonl(out / "regen_room.jsonl"), read_jsonl(out / "regen_vote.jsonl")
    rec = {"run": run}
    codes = None
    if codes_file and codes_file.exists() and (out / "coder_cases_key.json").exists():
        cf = json.loads(codes_file.read_text())
        codes = Z.codes_by_record(json.loads((out / "coder_cases_key.json").read_text()), cf["cases"])
        rec["codes"] = {"coder": cf.get("coder"), "file": codes_file.name, "records": len(room),
                        "records_coded": sum(Z.record_id(r) in codes for r in room)}
    if room:
        rec["room"] = Z.summarize(room, "room", codes=codes)
    if vote:
        rec["vote"] = Z.summarize(vote, "vote")
    (out / "causal.json").write_text(json.dumps(rec, indent=1) + "\n")
    write_report(out, rec)
    return rec


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="google/gemma-2-9b-it")
    ap.add_argument("--root", type=Path, default=S.RU, help="folder with <round>/rooms.json and <round>/runs/<version>/")
    ap.add_argument("--groups", default=",".join(f"{r}:{v}" for r, v in S.ANALYSIS), help="round:version, comma separated (used when the capture folder has no tables yet)")
    ap.add_argument("--acts", type=Path, default=S.RU / "acts/stage2", help="the stage 2 capture folder (uninstructed/capture.py)")
    ap.add_argument("--layer", type=int, default=None)
    ap.add_argument("--anchor", default=None, help="room point of the room phase, e.g. m2.pre (the probe results give it by the rule of 15.2)")
    ap.add_argument("--phase", choices=["room", "vote", "both"], default="both")
    ap.add_argument("--out", type=Path, default=S.RU / "stage2/causal")
    ap.add_argument("--lens", type=Path, default=None)
    ap.add_argument("--basis-dir", type=Path, default=None)
    ap.add_argument("--k", type=int, default=25, help="J-lens vectors per activation")
    ap.add_argument("--search-device", default=None, help="device of the vocabulary search of the decomposition (default: cuda if there is one)")
    ap.add_argument("--core", choices=["auto", "skip"], default="auto", help="auto: read the instructed benchmark when the capture folder has no such reads")
    ap.add_argument("--core-runs", type=int, default=0, help="tests only: the first N collusion episodes of the instructed benchmark")
    ap.add_argument("--per-side", type=int, default=48, help="rooms per side in the room phase")
    ap.add_argument("--per-side-members", type=int, default=96, help="members per side in the vote phase")
    ap.add_argument("--regen", type=int, default=2, help="regenerations per condition and unit; the baseline gets twice as many")
    ap.add_argument("--conditions", default=",".join(Z.ROOM_CONDITIONS), help="room phase")
    ap.add_argument("--vote-conditions", default=",".join(Z.VOTE_CONDITIONS))
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--seed", type=int, default=Z.SEED)
    ap.add_argument("--chunk-rooms", type=int, default=8, help="rooms regenerated together; results are written after each chunk")
    ap.add_argument("--chunk-members", type=int, default=32)
    ap.add_argument("--max-minutes", type=float, default=0,
                    help="stop before a new chunk once this much time has passed since the script started (0: no limit)")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--allow-cpu", action="store_true")
    ap.add_argument("--code-gemma", action="store_true", help="after the room phase, the model also codes the regenerated rooms (greedy), as a check")
    ap.add_argument("--code-batch", type=int, default=8)
    ap.add_argument("--plan", action="store_true", help="print how many rooms, votes and prompts would be regenerated; no model")
    ap.add_argument("--summarize", action="store_true", help="write causal.json and CAUSAL.md again from the records in --out; no model")
    ap.add_argument("--codes", type=Path, default=None, help="with --summarize: a coding file of the regenerated rooms")
    ap.add_argument("--import-codes", type=Path, default=None, help="a coder's answers {case id: {code, ...}} for coder_cases.jsonl")
    ap.add_argument("--coder", default=None, help="name of the coder for --import-codes")
    a = ap.parse_args()
    log = lambda m: print(m, flush=True)   # noqa: E731
    run_file = a.out / "run.json"
    t_proc = time.time()                   # --max-minutes counts from here: model load and replay included

    if a.import_codes or a.summarize:
        run = json.loads(run_file.read_text()) if run_file.exists() else {"phases": {}}
        codes_file = a.codes
        if a.import_codes:
            if not a.coder:
                ap.error("--import-codes needs --coder")
            key = json.loads((a.out / "coder_cases_key.json").read_text())
            ans = json.loads(a.import_codes.read_text())
            cases = {}
            for cid in key["cases"]:
                if cid in ans:
                    v = ans[cid]
                    cases[cid] = {**Z.C.parse_coder_output(v if isinstance(v, str) else json.dumps(v)), "raw": v if isinstance(v, str) else json.dumps(v)}
            codes_file = a.out / ("coding_" + re.sub(r"[^a-z0-9]+", "_", a.coder.lower()).strip("_")[:40] + ".json")
            codes_file.write_text(json.dumps({"coder": a.coder, "rubric_sha256": key["rubric_sha256"], "n_cases": len(key["cases"]), "n_coded": len(cases),
                                              "n_unparsed": sum(c["code"] == Z.C.UNPARSED for c in cases.values()), "cases": cases}, indent=1) + "\n")
            log(f"[causal] {len(cases)} of {len(key['cases'])} cases coded by {a.coder}: {codes_file}")
        rec = summarize_all(a.out, run, codes_file)
        print(json.dumps({k: {kk: vv.get("met", vv.get("verdict")) if isinstance(vv, dict) else vv for kk, vv in v["rules"].items()}
                          for k, v in rec.items() if k in ("room", "vote")}, indent=1, default=str))
        return 0

    if a.layer is None:
        ap.error("--layer")
    # the vote phase first: it is short, so a run that stops in the long room phase (time limit, or an error) keeps
    # it whole. The phases share nothing: directions, units and random numbers are fixed per phase before either runs
    phases = ["vote", "room"] if a.phase == "both" else [a.phase]
    if "room" in phases and (a.anchor is None or a.anchor not in S.ROOM_POINTS):
        ap.error(f"--anchor must be a room point ({S.ROOM_POINTS[0]} .. {S.ROOM_POINTS[-1]})")
    L, tdir = a.layer, a.acts / "tables"
    have_tables = (tdir / "index.json").exists() and (tdir / f"member_L{L}.npy").exists()
    if (tdir / "index.json").exists() and not have_tables and not a.plan:
        sys.exit(f"{tdir} holds tables without layer {L}; capture that layer or give another --acts folder")
    want = {"room": [c for c in a.conditions.split(",") if c], "vote": [c for c in a.vote_conditions.split(",") if c]}
    for c in want["room"] + want["vote"]:
        if c not in Z.ROOM_CONDITIONS:
            ap.error(f"unknown condition {c}")
    if "base" not in want["room"] or "base" not in want["vote"]:
        ap.error("every phase needs the condition base")

    bundle = tok = None
    if not a.plan:
        import torch
        if not torch.cuda.is_available() and not a.allow_cpu:
            sys.exit("No CUDA GPU here; use --allow-cpu with a tiny model, or --plan.")
        from generation.hf_backend import load_model
        from uninstructed import capture as Cap

        bundle = load_model(a.model, dtype=a.dtype, device_map="auto" if torch.cuda.is_available() else None)
        tok = bundle.tokenizer
        if not have_tables:                      # a fresh machine: read what the directions need, at this layer only
            groups = parse_groups(a.groups)
            log(f"[causal] no tables for layer {L} in {a.acts}: reading the recorded turns of {groups}")
            for step in (Cap.capture_rooms, Cap.capture_public, Cap.capture_neutral):
                step(groups, a.root, bundle.model, tok, [L], a.acts, log=log)
            Cap.build_tables(groups, a.root, a.acts, [L], tok, apply_only=(), log=log)
        if a.core == "auto" and not (a.acts / "core" / "activations.npz").exists():
            Cap.capture_core(bundle.model, tok, [L], a.acts, max_runs=a.core_runs or None, log=log)
        have_tables = True

    # units: those of the tables when there are tables (their rows are the member table's rows)
    if have_tables:
        index = json.loads((tdir / "index.json").read_text())
        members, rooms, MP = index["members"], index["rooms"], index["member_points"]
        fm, fr = S.load_units(tuple(tuple(g) for g in index["groups"]), a.root, apply_only=tuple(tuple(g) for g in index["apply_only"]))
        assert [m["uid"] for m in fm] == [m["uid"] for m in members] and [m["vote"] for m in fm] == [m["vote"] for m in members], \
            "the tables' units are not the units of the record"
        assert [r["votes"] for r in fr] == [r["votes"] for r in rooms]
        M = np.load(tdir / f"member_L{L}.npy", mmap_mode="r")
        MJ = np.load(tdir / f"memberJ_L{L}.npy", mmap_mode="r") if (tdir / f"memberJ_L{L}.npy").exists() else None
        inst = Z.instructed_direction(a.acts, L)
        if inst is None and a.plan and a.core == "auto":
            inst = np.ones(M.shape[-1])              # the run itself would read the instructed benchmark first
    else:                                            # --plan without tables: the labels alone decide what can be drawn
        members, rooms = S.load_units(parse_groups(a.groups), a.root, apply_only=())
        MP, M, MJ, inst = list(S.MEMBER_POINTS), None, None, (np.ones(4) if a.core == "auto" else None)
    use_lens = bool(a.lens and a.basis_dir)
    by_uid = {m["uid"]: m for m in members}

    a.out.mkdir(parents=True, exist_ok=True) if not a.plan else None
    setup, plans = {}, {}
    for phase in phases:
        point = a.anchor if phase == "room" else "v.pre"
        rows, col = Z.point_units(members, point)
        if M is not None:
            X = np.asarray(M[rows, MP.index(col)], dtype=np.float64)
            XJ = np.asarray(MJ[rows, MP.index(col)], dtype=np.float64) if MJ is not None else None
            if XJ is None and use_lens and not a.plan:
                import torch
                fin = np.isfinite(X).all(1)
                XJ = np.full_like(X, np.nan)
                t0 = time.time()
                XJ[fin] = Z.j_parts(X[fin], a.lens, a.basis_dir, L, k=a.k, device=a.search_device or ("cuda" if torch.cuda.is_available() else None))
                log(f"[causal] J-space parts of {int(fin.sum())} activations at {point}: {time.time() - t0:.0f}s")
            elif XJ is None and use_lens:
                XJ = X.copy()                        # --plan: the run would decompose them
        else:                                        # --plan without tables
            X = np.random.default_rng(0).standard_normal((len(rows), 4))
            XJ = X.copy() if use_lens else None
        D = Z.Directions(members, rows, X, XJ=XJ, inst=inst, seed=a.seed)
        conds, dropped = Z.available(want[phase], D, swaps=(phase == "room"))
        if phase == "room":
            parity, _ = S.own_point(point)
            for r in rooms:
                r["anchor_uid"] = r["members"][parity]
            units, side_of, per = rooms, Z.room_side, a.per_side
        else:
            for m in members:
                m["anchor_uid"] = m["uid"]
            units, side_of, per = members, Z.member_side, a.per_side_members
        usable = lambda u: D.at(u["anchor_uid"]) is not None and D.for_domain(u["domain"]) is not None   # noqa: E731
        sel, pools = Z.deal(units, side_of, per, a.seed, usable)
        items = Z.items_of(sel, conds, a.regen)
        for it in items:
            if it["cond"].startswith("swap"):
                it["donor_unit"] = Z.donor(it["unit"], it["side"], pools, a.seed)
        setup[phase] = {"point": point, "D": D, "conds": conds, "dropped": dropped, "sel": sel, "items": items}
        plans.update(Z.plan(sel if phase == "room" else None, conds, sel if phase == "vote" else None, conds, a.regen, a.anchor if phase == "room" else None))
        plans[phase].update(conditions_dropped=dropped, sets_drawn_from=len(pools),
                            available_per_side={s: sum(len(v[s]) for v in pools.values()) for s in ("remove", "induce")},
                            scenarios={s: len({u["scenario"] for u in sel[s]}) for s in ("remove", "induce")},
                            by_version={s: dict(Counter(u["version"] for u in sel[s])) for s in ("remove", "induce")})
    plans["prompts"] = sum(plans[p]["prompts"] for p in phases)
    plans["tables_present"] = bool(M is not None)
    if a.plan:
        print(json.dumps(plans, indent=1))
        return 0

    # ---- run ----
    run = {"model": bundle.model_id, "layer": L, "anchor": a.anchor, "seed": a.seed, "regen": a.regen, "per_side": a.per_side,
           "per_side_members": a.per_side_members, "max_new_tokens": a.max_new_tokens, "temperature": a.temperature, "batch": a.batch,
           "groups": index["groups"], "plan": plans, "phases": {}, "incomplete": None}
    if run_file.exists():                            # continuing: the same directions and seeds, or not at all
        old = json.loads(run_file.read_text())
        for k in ("model", "layer", "seed", "max_new_tokens", "temperature"):
            if old.get(k) != run[k]:
                sys.exit(f"{a.out} holds a run with {k} = {old.get(k)!r}; this call has {run[k]!r}. Use another --out.")
        if "room" in phases and old.get("anchor") not in (None, run["anchor"]):
            sys.exit(f"{a.out} holds a room phase at {old.get('anchor')}; this call has {run['anchor']}. Use another --out.")
        run["anchor"] = run["anchor"] or old.get("anchor")
        # what earlier calls into this folder generated stays on record
        run["earlier_calls"] = old.get("earlier_calls", []) + [{"minutes": old.get("minutes"), "incomplete": old.get("incomplete"),
                                                                "generation": {p: v.get("generation") for p, v in old.get("phases", {}).items()}}]
        run["phases"] = {p: v for p, v in old.get("phases", {}).items() if p not in phases}
    eps = Z.episodes_of(rooms + members, a.root)
    st = Z.Steerer(bundle.model, tok, L, max_new_tokens=a.max_new_tokens, temperature=a.temperature, batch=a.batch, log=log)
    t_start, before = time.time(), {}
    for phase in phases:
        P_ = setup[phase]
        D, items, f = P_["D"], P_["items"], a.out / f"regen_{phase}.jsonl"
        done = {Z.record_id(r) for r in read_jsonl(f)}
        todo = [it for it in items if f"{it['unit']['uid']}#{it['side']}#{it['cond']}#{it['regen']}" not in done]
        vecs = {}
        for h, H in enumerate(D.halves):
            if H is not None:
                vecs.update({f"d_half{h}": H["d"]}, **{f"{n}_half{h}": H[k] for n, k in (("dJ", "dJ"), ("dR", "dR"), ("inst", "inst")) if H[k] is not None})
        np.savez(a.out / f"directions_{phase}_L{L}_{P_['point'].replace('.', '_')}.npz", **vecs)
        run["phases"][phase] = {"point": P_["point"], "conditions": P_["conds"], "conditions_dropped": P_["dropped"], "direction": D.describe(),
                                "units": {s: len(P_["sel"][s]) for s in ("remove", "induce")}, "items": len(items), "items_done_before": len(items) - len(todo)}
        ops = lambda it: Z.unit_ops(it["cond"], it["side"], it["unit"], it["regen"], D, it.get("donor_unit"), a.seed)   # noqa: E731
        # all conditions and regenerations of a unit stay together; a chunk is a few units
        order, by_unit = [], {}
        for it in todo:
            if it["unit"]["uid"] not in by_unit:
                by_unit[it["unit"]["uid"]] = []
                order.append(it["unit"]["uid"])
            by_unit[it["unit"]["uid"]].append(it)
        step = a.chunk_rooms if phase == "room" else a.chunk_members
        log(f"[causal] {phase} phase at {P_['point']}, layer {L}: {len(todo)} of {len(items)} items to run, conditions {P_['conds']}")
        if not items:
            why = ("the activations at this point are the same inside every set of identical prompts, so there is no direction"
                   if all(H is None for H in D.halves) else "no set of identical prompts holds both kinds of unit with a direction from the other half")
            run["phases"][phase]["not_run"] = why
            log(f"[causal] {phase} phase not run: {why}")
        with open(f, "a") as fh:
            for c0 in range(0, len(order), step):
                if a.max_minutes and (time.time() - t_proc) / 60 > a.max_minutes:
                    run["incomplete"] = f"stopped at the time limit of {a.max_minutes:g} minutes in the {phase} phase, {len(order) - c0} units left"
                    log(f"[causal] {run['incomplete']}")
                    break
                chunk = [it for uid in order[c0:c0 + step] for it in by_unit[uid]]
                recs = (Z.regen_rooms(st, tok, chunk, P_["point"], eps, ops, layer=L) if phase == "room"
                        else Z.regen_votes(st, tok, chunk, eps, ops, layer=L))
                for r in recs:
                    fh.write(json.dumps(r, ensure_ascii=False) + "\n")
                fh.flush()
                log(f"[causal] {phase}: {min(c0 + step, len(order))}/{len(order)} units, {json.dumps(st.stats())}, "
                    f"{(time.time() - t_start) / 60:.1f} min")
        now = st.stats()
        run["phases"][phase]["generation"] = {k: (round(v - before.get(k, 0), 1) if isinstance(v, float) else v - before.get(k, 0))
                                              for k, v in now.items() if k != "seconds_per_row"}
        g = run["phases"][phase]["generation"]
        g["seconds_per_row"] = round(g["seconds"] / max(g["rows_generated"], 1), 3)
        before = now
        run_file.write_text(json.dumps(run, indent=1) + "\n")
        if run["incomplete"]:
            break
    st.close()
    run["minutes"] = round((time.time() - t_start) / 60, 1)

    # ---- the regenerated rooms for the room coders ----
    codes_file = None
    room_recs = read_jsonl(a.out / "regen_room.jsonl")
    if room_recs:
        cases, key = Z.coder_cases(room_recs, eps, a.seed)
        with open(a.out / "coder_cases.jsonl", "w") as fh:
            for c in cases:
                fh.write(json.dumps(c, ensure_ascii=False) + "\n")
        (a.out / "coder_cases_key.json").write_text(json.dumps(key, indent=1) + "\n")
        run["coder_cases"] = {"cases": len(cases), "regenerated_rooms": len(room_recs)}
        log(f"[causal] {len(room_recs)} regenerated rooms are {len(cases)} distinct rooms for the coders: coder_cases.jsonl")
        if a.code_gemma:
            import uninstructed_code_rooms as CR

            t0 = time.time()
            raw = CR.greedy_batches(bundle, [(key["coder_system"], c["prompt"]) for c in cases], a.code_batch, 64, log=log)
            coded = {c["case_id"]: {**Z.C.parse_coder_output(t), "raw": t} for c, t in zip(cases, raw)}
            codes_file = a.out / "coding_gemma.json"
            codes_file.write_text(json.dumps({"coder": f"{bundle.model_id}, greedy decoding", "rubric_sha256": key["rubric_sha256"],
                                              "n_cases": len(cases), "n_coded": len(coded),
                                              "n_unparsed": sum(c["code"] == Z.C.UNPARSED for c in coded.values()),
                                              "elapsed_s": round(time.time() - t0, 1), "cases": coded}, indent=1) + "\n")
    run_file.write_text(json.dumps(run, indent=1) + "\n")
    rec = summarize_all(a.out, run, codes_file)
    print(json.dumps({"minutes": run["minutes"], "incomplete": run["incomplete"],
                      "records": {p: rec[p]["records"] for p in ("room", "vote") if p in rec},
                      "rules": {p: {k: v.get("met") for k, v in rec[p]["rules"].items() if "met" in v} for p in ("room", "vote") if p in rec}}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
