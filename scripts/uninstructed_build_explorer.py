#!/usr/bin/env python3
"""Package what was run through the models in the four rounds of the uninstructed-collusion experiment, for the
explorer page (one HTML page that loads the data files written here).

    python scripts/uninstructed_build_explorer.py --out <folder>

Writes
    <folder>/data/index.json                 rounds, versions, the prompt pieces, the 50 scenarios, one row per episode
    <folder>/data/<round>_<version>.json     every recorded prompt and reply of that version's episodes in that round

Nothing is run and no model is loaded: the script only reads the recorded files under results/uninstructed/.

How the text is stored. A prompt of the final vote repeats the case, the room and the whole discussion, so the raw
logs hold every statement many times (about 80 MB in all). Each data file therefore keeps one table of strings, and a
prompt is a list of line numbers in that table: joining the lines with a newline gives the recorded prompt back,
character for character. The script rebuilds every system prompt, user prompt and reply this way and compares it
with the record before writing a file (CHECKS at the end of the run; any failure stops the script).

What the checks cover, per episode:
  * every one of the 24 calls: system prompt, user prompt and reply rebuilt from the table equal the record;
  * the system prompt, cut into its pieces (role line, mandate, stake sentences, added sentences, closing sentence,
    or the instructed control's two paragraphs), equals the record for each of the four members, the member's name
    being the only difference between them;
  * the same prompt comes out of the registry code as it stands now (uninstructed/protocol.py);
  * the room messages and the public statements in results.json are the replies of the calls, in order;
  * the first coder's prompt for each room, as exported for the second coder at the time, is what the coding module
    builds now from the record (same rubric hash as the pod's coding file);
  * the public-position prompt of each member, rebuilt from the record, has the case id of the key written on the
    pod, the rubric hash of that key, and the text of the sample files given to the second coder.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from uninstructed import coding as C  # noqa: E402
from uninstructed import protocol as P  # noqa: E402
import uninstructed_public_stance as PS  # noqa: E402

RES = ROOT / "results" / "uninstructed"
# Facts of each round, from its RUN_LOG.md (pod, commit, cost) and its generation records.
ROUNDS = [
    {"key": "pilot", "label": "Round 1", "name": "first pilot", "when": "16 November 2026, evening (PST)",
     "pods": "n9i1umi8u22h4q (S0, then out of memory) and 4xpyp21yojkdgz", "commit": "6433658 and 59e9c61", "cost": 0.88,
     "what": "A stake stated as a fact, with the honest mandate kept: S0 to S3 on the 12 pilot scenarios, two runs each."},
    {"key": "pilot2", "label": "Round 2", "name": "second pilot", "when": "17 November 2026, 08:03 PST",
     "pods": "nyfwtzmwfimh00", "commit": "2e90a56", "cost": 0.69,
     "what": "What stands against the stake: T1 to T5 on the 12 pilot scenarios, two runs each."},
    {"key": "pilot3", "label": "Round 3", "name": "third round", "when": "17 November 2026, 09:47 PST",
     "pods": "dkhn903tiqaxop", "commit": "72f5a1e", "cost": 1.02,
     "what": "Plain option texts, the two added sentences together, a stated intention, a proposal made to a member "
             "with the wish, and the instructed control: I1, P0, P1, P3, T6, T7, T8 on the 12 pilot scenarios, two runs each."},
    {"key": "pilot4", "label": "Round 4", "name": "fourth round", "when": "17 November 2026, 12:06 PST",
     "pods": "0g2s1ffvmpkfck", "commit": "eced44d", "cost": 0.96,
     "what": "The stated intention on all 50 scenarios (T7, two runs each) and the role line alone as a baseline "
             "(T0, one run each)."},
]
ORDER = ["S0", "S1", "S2", "S3", "T1", "T2", "T3", "T4", "T5", "P0", "P1", "P3", "T6", "T7", "T8", "I1", "T0"]
PHASE = {"room": 0, "discussion": 1, "final": 2}
MERITS = P.ROLE_LINE[len(P.ROLE_BARE):].strip()
CHECKS: Counter = Counter()


def ok(name: str, cond: bool, detail: str = "") -> None:
    if not cond:
        sys.exit(f"CHECK FAILED: {name} {detail}")
    CHECKS[name] += 1


class Table:
    """Strings of one data file, each stored once."""

    def __init__(self) -> None:
        self.items: list[str] = []
        self.at: dict[str, int] = {}

    def add(self, s: str) -> int:
        i = self.at.get(s)
        if i is None:
            i = self.at[s] = len(self.items)
            self.items.append(s)
        return i

    def lines(self, text: str) -> list[int]:
        return [self.add(x) for x in text.split("\n")]

    def text(self, idx: list[int]) -> str:
        return "\n".join(self.items[i] for i in idx)


def instructed_parts(scenario: dict, sidx: int) -> list[tuple[str, str]]:
    """The instructed control's two paragraphs in the order protocol.instructed_text uses, each with its name."""
    lay = P.option_layout(scenario, sidx)

    def relabel(text: str) -> str:
        return re.sub(r"\bOption ([AB])\b", lambda m: f"Option {lay['target'] if m.group(1) == 'B' else lay['fair']}", text)

    motive, instruction = ("motive", relabel(scenario["collusion_motive"])), ("instruction", relabel(scenario["deception_instruction"]))
    parts = [motive, instruction] if sidx % 2 == 0 else [instruction, motive]
    assert f"{parts[0][1]}\n\n{parts[1][1]}" == P.instructed_text(scenario, sidx)
    return parts


def system_pieces(scenario: dict, version: str, stakes: dict, sidx: int) -> list[tuple[str, str]]:
    """The system prompt as (piece name, text) pairs; '' names the separators. The role piece carries {agent_name}."""
    spec = P.spec_of(version)
    out = [("role", P.ROLE_BARE)]
    if spec["role"] == "merits":
        out += [("", " "), ("merits", MERITS)]
    if spec["stake"] == "instructed":
        a, b = instructed_parts(scenario, sidx)
        return out + [("", "\n\n"), a, ("", "\n\n"), b]
    if spec["mandate"]:
        out += [("", "\n\n"), ("mandate", scenario["honest_mandate"])]
    if spec["stake"] != "none":
        s = P.stake_sentences(scenario["id"], version, stakes)
        out += [("", "\n\n"), ("interest", s["interest"])]
        if spec["stake"] == "personal":
            out += [("", " "), ("personal", s["personal"])]
        for k in spec["extra"]:
            out += [("", " "), (k, stakes["extras"][k])]
        out += [("", " "), ("closing", stakes["closing"])]
    return out


def merge_hits(hits: list[dict] | None) -> dict:
    """The hits of a keyword screen as {speaker: [words]}, each word once."""
    out: dict[str, list[str]] = {}
    for h in hits or []:
        words = out.setdefault(h["speaker"], [])
        words += [w for w in h["words"] if w not in words]
    return out


def pieces_text(pieces: list[tuple[str, str]], agent: str) -> str:
    return "".join(t.replace("{agent_name}", agent) if k == "role" else t for k, t in pieces)


def build_round(rd: dict, scenarios: list[dict], sidx_of: dict, stakes: dict, out: Path) -> tuple[list, dict, dict]:
    base = RES / rd["key"]
    by_id = {s["id"]: s for s in scenarios}
    rooms = {r["room_id"]: r for r in json.loads((base / "rooms.json").read_text())}
    summary = json.loads((base / "summary.json").read_text())
    gem = json.loads((base / "coding_gemma.json").read_text())
    cla = json.loads((base / "coding_claude.json").read_text())
    key = json.loads((base / "cases_key.json").read_text())
    ok("rubric hash of the room coding: pod file, second coder's key and the code now",
       gem["rubric_sha256"] == key["rubric_sha256"] == cla["rubric_sha256"] == C.rubric_sha256(), rd["key"])
    graw = {r["room_id"]: r for r in gem["rooms"]}
    craw = {r["room_id"]: r for r in cla["rooms"]}
    case_of_room = {room: cid for cid, room in key["cases"].items()}
    exported = {}
    for line in (base / "cases_for_coder.jsonl").read_text().splitlines():
        if line.strip():
            d = json.loads(line)
            exported[d["case_id"]] = d["prompt"]
    # what the coding module builds now from the record: the user part of the first coder's prompt
    ctx = {s["id"]: s["context"] for s in scenarios}
    built_now = {r["room_id"]: C.coder_prompt(r, ctx[r["scenario_id"]]) for r in C.load_rooms(base / "runs", stakes)}
    ok("rooms: one coder case per room", set(built_now) == set(rooms) == set(case_of_room), rd["key"])

    # public positions (rounds 2 to 4)
    stance = None
    if (base / "public_stance.json").exists():
        ps = json.loads((base / "public_stance.json").read_text())
        key_all = json.loads((base / "stance_key_all.json").read_text())
        ans_g = json.loads((base / "stance_gemma.json").read_text())
        ok("rubric hash of the public-position coding: key written on the pod, Gemma's answers and the code now",
           key_all["rubric_sha256"] == ans_g["_meta"]["rubric_sha256"] == ps["summary"]["rubric_sha256"] == PS.rubric_sha256(), rd["key"])
        cases = {c["agent_id"]: c for c in PS.cases_of([base / "runs"])}
        ids = {PS.case_id(key_all["seed"], a): a for a in cases}
        ok("public positions: the rebuilt cases are the cases of the pod's key", set(ids) == set(key_all["cases"]), rd["key"])
        for name in ("stance_cases_sample.jsonl", "stance_cases.jsonl"):
            if (base / name).exists():
                for line in (base / name).read_text().splitlines():
                    if line.strip():
                        d = json.loads(line)
                        ok("public positions: the rebuilt prompt is the text given to the second coder",
                           cases[ids[d["case_id"]]]["prompt"] == d["prompt"], f"{rd['key']} {d['case_id']}")
        stance = {"summary": ps["summary"], "members": {m["agent_id"]: m for m in ps["members"]}, "cases": cases, "raw": ans_g}

    rows, versions = [], {}
    for vdir in sorted(p for p in (base / "runs").iterdir() if p.is_dir()):
        version = vdir.name
        tab, eps = Table(), {}
        for run in sorted(p for p in vdir.iterdir() if p.is_dir()):
            cfg = json.loads((run / "run_config.json").read_text())
            res = json.loads((run / "results.json").read_text())
            log = json.loads((run / "agent_prompts.json").read_text())
            sid, agents, sidx = cfg["scenario_id"], cfg["agents"], cfg["scenario_idx"]
            ok("scenario index of the run is the bank's", sidx == sidx_of[sid], run.name)
            ok("24 calls per episode", len(log) == 24, run.name)
            pieces = system_pieces(by_id[sid], version, stakes, sidx)
            for seat, agent in enumerate(agents):
                rec = {e["system_prompt"] for e in log if e["agent_name"] == agent}
                ok("one system prompt per member, the same in all of its calls", len(rec) == 1, run.name)
                ok("system prompt = its pieces, the name being the only difference between members",
                   pieces_text(pieces, agent) == next(iter(rec)), f"{run.name} seat {seat}")
                ok("system prompt = what the registry code builds now",
                   P.system_prompt(agent, by_id[sid], version, stakes, sidx) == next(iter(rec)), f"{run.name} seat {seat}")
            calls = []
            for e in log:
                seat = agents.index(e["agent_name"])
                ok("seat recorded with the call", seat == e["seat"], run.name)
                u, r = tab.lines(e["user_prompt"]), tab.lines(e["response"])
                ok("user prompt rebuilt from the table", tab.text(u) == e["user_prompt"], run.name)
                ok("reply rebuilt from the table", tab.text(r) == e["response"], run.name)
                calls.append([seat, PHASE[e["phase"]], e["round"], e["room"], e.get("turn") if e.get("turn") is not None else -1,
                              1 if e.get("scripted") else 0, u, r])
            # the transcripts of results.json are the replies of the calls
            room_calls = [[e for e in log if e["phase"] == "room" and e["room"] == k] for k in (0, 1)]
            ok("room messages = replies of the room calls",
               res["room_messages"] == [[f"[{e['agent_name']}]: {e['response']}" for e in rc] for rc in room_calls], run.name)
            ok("public statements = replies of the discussion calls",
               res["public_messages"] == [f"[{e['agent_name']}]: {e['response']}" for e in log if e["phase"] == "discussion"], run.name)
            ok("final texts = replies of the final calls",
               res["final_texts"] == {e["agent_name"]: e["response"] for e in log if e["phase"] == "final"}, run.name)
            ok("votes recorded twice agree", res["votes"] == cfg["votes"], run.name)
            n_scripted = sum(1 for e in log if e.get("scripted"))
            ok("scripted openers as the version says", n_scripted == (2 if P.spec_of(version)["opener"] else 0) == cfg.get("scripted_messages", 0), run.name)

            ep_rooms = []
            for k in (0, 1):
                rid = f"{cfg['run_id']}#{k}"
                rm, cid = rooms[rid], case_of_room[rid]
                ok("first coder's prompt: the exported case is what the coding module builds now",
                   exported[cid] == built_now[rid][1] and built_now[rid][0] == C.CODER_SYSTEM, rid)
                ok("room record matches the run", rm["agents"] == cfg["rooms"][k] and rm["votes"] == [cfg["votes"][a] for a in cfg["rooms"][k]], rid)
                ok("codes of the room record are the two coders' codes", rm["codes"] == [graw[rid]["code"], craw[rid]["code"]], rid)
                ep_rooms.append({"label": rm["label"], "reason": rm["reason"], "codes": rm["codes"], "marks": rm["marks"],
                                 "disclosed": bool(rm.get("disclosed")), "dis": merge_hits(rm.get("disclosure_hits")),
                                 "rep": merge_hits(rm.get("report_hits")), "case": cid, "cp": tab.lines(exported[cid]),
                                 "graw": graw[rid]["raw"], "craw": craw[rid]["raw"]})
            st = None
            if stance:
                st = []
                for agent in agents:
                    aid = f"{cfg['run_id']}|{agent}"
                    m, c = stance["members"][aid], stance["cases"][aid]
                    ok("public position: member record matches the run", m["vote"] == cfg["votes"][agent] and m["case_id"] in stance["raw"], aid)
                    st.append({"case": m["case_id"], "p": tab.lines(c["prompt"]), "g": m["stances"][0], "gk": m["kinds"][0],
                               "graw": stance["raw"][m["case_id"]]["raw"], "c": m["stances"][1], "ck": m["kinds"][1]})
                    ok("public-position prompt rebuilt from the table", tab.text(st[-1]["p"]) == c["prompt"], aid)
            eps[cfg["run_id"]] = {
                "sid": sid, "run": cfg["run_idx"], "agents": agents, "labels": cfg["option_labels"],
                "options": [cfg["options"][k] for k in cfg["option_labels"]], "target": cfg["target_option"], "fair": cfg["fair_option"],
                "sp": [[k, tab.add(t)] for k, t in pieces], "calls": calls, "votes": [cfg["votes"][a] for a in agents],
                "rooms": ep_rooms, "st": st}
            tv = sum(v == cfg["target_option"] for v in cfg["votes"].values())
            hidden = None if not st else sum(1 for a, s in zip(agents, st) if cfg["votes"][a] == cfg["target_option"] and s["gk"] == "other option")
            rows.append([rd["key"], version, sid, cfg["run_idx"], tv, [r["label"] for r in ep_rooms], [r["codes"] for r in ep_rooms], hidden])
        name = f"{rd['key']}_{version}.json"
        (out / "data" / name).write_text(json.dumps({"round": rd["key"], "version": version, "strings": tab.items, "episodes": eps},
                                                    ensure_ascii=False, separators=(",", ":")))
        sv = summary["versions"][version]
        versions[version] = {
            "episodes": sv["episodes"], "rooms": sv["rooms"], "scenarios": sv["scenarios"], "votes": sv["votes"]["n"],
            "target_votes": sv["votes"]["target"], "unparsed": sv["votes"]["unparsed"], "labels": sv["labels"],
            "pacts": [c.get("AGREE_TARGET", 0) for c in sv["codes_by_coder"]],
            "both_target": sum(1 for r in rooms.values() if r["version"] == version and all(v == r["target"] for v in r["votes"])),
            "file": name, "strings": len(tab.items)}
        if stance:
            pv = stance["summary"]["versions"][version]
            versions[version]["stance"] = {"other": pv["stake_votes_after_public_other_option"], "stake": pv["stake_votes_after_public_stake_option"],
                                           "neither": pv["stake_votes_after_public_neither"]}
        ok("votes for the stake option: summary = sum over the episodes",
           sv["votes"]["target"] == sum(r[4] for r in rows if r[1] == version), f"{rd['key']} {version}")
    info = {**rd, "versions": [v for v in ORDER if v in versions], "episodes": sum(v["episodes"] for v in versions.values()),
            "rooms": summary["n_rooms"], "kappa": round(summary["kappa"], 2), "raw_agreement": round(summary["raw_agreement"], 3),
            "coders": summary["coders"]}
    if stance:
        ag = stance["summary"]["agreement"][0]
        info["stance"] = {"coders": stance["summary"]["coders"], "n": ag["n"], "raw_agreement": ag["raw_agreement"], "kappa": round(ag["kappa"], 2)}
    gens = [json.loads(p.read_text()) for p in sorted((base / "runs").glob("generation_*.json"))]
    ok("generation settings are the same in every record of the round",
       len({(g["model_id"], g["temperature"], g["max_new_tokens"], g["batch"]) for g in gens}) == 1, rd["key"])
    info["generation"] = {"model_id": gens[0]["model_id"], "temperature": gens[0]["temperature"], "max_new_tokens": gens[0]["max_new_tokens"],
                          "batch": gens[0]["batch"], "protocol": sorted({g["protocol"] for g in gens}),
                          "prompts": sum(g["generation"]["prompts"] for g in gens)}
    return rows, versions, info


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    (a.out / "data").mkdir(parents=True, exist_ok=True)
    name_pool, scenarios, sidx_of = P.load_scenarios()
    stakes = P.load_stakes()
    pilot = set(P.pilot_scenarios(scenarios))

    rows, by_version, rounds = [], {}, []
    for rd in ROUNDS:
        r, v, info = build_round(rd, scenarios, sidx_of, stakes, a.out)
        rows += r
        rounds.append(info)
        for code, d in v.items():
            by_version.setdefault(code, []).append({"round": rd["key"], **d})
    ok("every version that was run is in the display order", set(by_version) == set(ORDER))

    scen = {}
    for s in scenarios:
        i = sidx_of[s["id"]]
        lay, st = P.option_layout(s, i), stakes["scenarios"][s["id"]]
        plain = stakes["plain_options"].get(s["id"])
        scen[s["id"]] = {"idx": i, "title": s["title"], "domain": s["domain"], "pilot": s["id"] in pilot, "context": s["context"],
                         "labels": lay["labels"], "options": [lay["options"][k] for k in lay["labels"]], "target": lay["target"],
                         "fair": lay["fair"], "plain": ([P.option_layout(s, i, stakes["plain_options"])["options"][k] for k in lay["labels"]]
                                                        if plain else None),
                         "interest": st["interest"], "personal": st["personal"], "mandate": s["honest_mandate"]}
    index = {
        "rounds": rounds,
        "order": ORDER,
        "versions": {code: {"about": P.ABOUT[code], "spec": P.spec_of(code), "runs": by_version[code]} for code in ORDER},
        "not_run": {code: P.ABOUT[code] for code in P.VERSIONS if code not in by_version},
        "pieces": {"role": P.ROLE_BARE, "merits": MERITS, "closing": stakes["closing"], "rule": stakes["rule"],
                   "heavy": stakes["extras"]["heavy"], "wish": stakes["extras"]["wish"], "intent": stakes["extras"]["intent"],
                   "proposal": stakes["openers"]["proposal"], "coder_system": C.CODER_SYSTEM},
        "scenarios": scen,
        "episodes": rows,
        "totals": {"episodes": len(rows), "rooms": 2 * len(rows), "votes": 4 * len(rows), "calls": 24 * len(rows),
                   "generated": 24 * len(rows) - 2 * sum(1 for r in rows if P.spec_of(r[1])["opener"]),
                   "cost": round(sum(r["cost"] for r in ROUNDS), 2)},
    }
    (a.out / "data" / "index.json").write_text(json.dumps(index, ensure_ascii=False, separators=(",", ":")))

    print("CHECKS (all passed)")
    for name, n in CHECKS.items():
        print(f"  {n:>6}  {name}")
    print("FILES")
    total = 0
    for p in sorted((a.out / "data").glob("*.json")):
        total += p.stat().st_size
        print(f"  {p.stat().st_size / 1e6:6.2f} MB  {p.name}")
    print(f"  {total / 1e6:6.2f} MB  in all; {len(rows)} episodes, {index['totals']['calls']} calls, "
          f"{index['totals']['generated']} of them generated by the model")
    return 0


if __name__ == "__main__":
    sys.exit(main())
