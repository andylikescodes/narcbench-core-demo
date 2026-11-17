#!/usr/bin/env python3
"""Pipeline check for the uninstructed-collusion pilot on a tiny random Gemma-2 model (CPU, no downloads).

Every number this writes is meaningless as science: the model is randomly initialised. The point is that each stage
of the pilot pod runs end to end and writes files in the shapes the GPU run will produce, and that three properties
the design relies on hold:

  A. identical prompts: two runs of a scenario have the same activation at the token before the first room message;
  B. causal reads: the activation saved for a reply token equals the one from a context cut right after that token;
  C. alignment: the saved token ids of every room message decode back to the recorded message;
  D. the out-of-memory fallback retries outside the except block and keeps the order of the prompts;
  E. the second pilot's versions: system prompts as specified, the scripted first message in place and not generated;
  F. the first pilot's recorded episodes are rebuilt prompt for prompt by the current code.

  python scripts/uninstructed_tiny_e2e.py --work /path/to/scratch/u_tiny
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

LAYERS = [2, 3, 4]


def run(cmd, log):
    t0 = time.time()
    cmd = [str(c) for c in cmd]
    p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    log.append({"cmd": " ".join(cmd), "rc": p.returncode, "s": round(time.time() - t0, 1)})
    print(f"[{p.returncode}] {' '.join(cmd)[:150]} ({time.time() - t0:.0f}s)", flush=True)
    if p.returncode != 0:
        print(p.stdout[-3000:], p.stderr[-3000:])
        raise SystemExit(p.returncode)
    return p


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", type=Path, required=True)
    a = ap.parse_args()
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, Gemma2Config, Gemma2ForCausalLM

    from jspace_interp.jacobian import estimate_lens, save_lens, save_unembed_basis
    from jspace_interp.positions import gemma_chat_text
    from jspace_interp_tiny_e2e import train_tokenizer
    from uninstructed import coding as C
    from uninstructed import protocol as P
    from uninstructed import timeline as T

    torch.manual_seed(0)
    w = a.work
    w.mkdir(parents=True, exist_ok=True)
    log = []
    name_pool, scenarios, idx = P.load_scenarios()
    stakes = P.load_stakes()
    assert P.lint_stakes(stakes, [s["id"] for s in scenarios]) == [], "stake texts fail the wording test"

    # 1. tiny model and tokenizer
    texts = []
    for s in scenarios:
        texts += [s["context"], s["honest_mandate"], s["public_frame"], s["final_frame"], s["options"]["A"], s["options"]["B"]]
    texts += [x for v in stakes["scenarios"].values() for x in (v["interest"], v["personal"])] + [stakes["closing"], stakes["rule"],
                                                                                                P.ROOM_FRAME, C.RUBRIC, C.CASE] + name_pool
    texts += [v for k, v in stakes["extras"].items() if k != "_about"] + [v for k, v in stakes["openers"].items() if k != "_about"]
    tok = train_tokenizer(texts, w / "model", vocab_size=3000)
    cfg = Gemma2Config(vocab_size=len(tok), hidden_size=64, intermediate_size=128, num_hidden_layers=6, num_attention_heads=2,
                       num_key_value_heads=1, head_dim=32, max_position_embeddings=8192, sliding_window=4096,
                       pad_token_id=tok.pad_token_id, bos_token_id=tok.bos_token_id, eos_token_id=tok.eos_token_id, tie_word_embeddings=True)
    model = Gemma2ForCausalLM(cfg).eval()
    model.save_pretrained(w / "model")
    prompts = [tok(t[:2000], add_special_tokens=False)["input_ids"][:96] for t in texts[:300:30] if t]
    save_lens(estimate_lens(model, prompts, LAYERS, skip=4), w / "lens/tiny_jacobian_lens.pt")
    save_unembed_basis(model, w / "basis")
    print("tokenizer", len(tok), "params", sum(p.numel() for p in model.parameters()))

    py = sys.executable
    ids = ["med_01", "fin_03", "re_01"]           # option layouts 0, 2 and 1/3
    runs = w / "runs"
    # 2. generation (threads + batches through the real code path), two versions, two runs each
    run([py, "scripts/uninstructed_generate.py", "--model", w / "model", "--allow-cpu", "--dtype", "float32", "--versions", "S0,S2",
         "--ids", *ids, "--runs", "2", "--max-new-tokens", "12", "--workers", "3", "--batch", "2", "--out", runs], log)
    gen = json.loads((runs / "generation_S0-S2.json").read_text())
    assert gen["episodes_generated"] == 12 and gen["generation"]["prompts"] == 12 * 24, gen
    # 3. coder A through the model, coder B through the export / import round trip, summary with two coders
    run([py, "scripts/uninstructed_code_rooms.py", "--model", w / "model", "--allow-cpu", "--dtype", "float32", "--max-new-tokens", "8",
         "--batch", "4", "--runs", runs, "--out", w / "coding_a.json"], log)
    run([py, "scripts/uninstructed_code_rooms.py", "--runs", runs, "--export", w / "cases.jsonl", "--key", w / "key.json"], log)
    cases = [json.loads(ln) for ln in (w / "cases.jsonl").read_text().splitlines()]
    assert len(cases) == 24 and all("__S" not in c["case_id"] and "votes" not in c["prompt"].lower() for c in cases)
    answers = {c["case_id"]: {"code": C.CODES[i % 4], "stake": 0, "proposal": 0, "acceptance": 0} for i, c in enumerate(cases)}
    (w / "answers_b.json").write_text(json.dumps(answers))
    run([py, "scripts/uninstructed_code_rooms.py", "--runs", runs, "--import-answers", w / "answers_b.json", "--key", w / "key.json",
         "--coder", "test coder B", "--out", w / "coding_b.json"], log)
    run([py, "scripts/uninstructed_summarize.py", "--runs", runs, "--coding", w / "coding_a.json", "--coding", w / "coding_b.json",
         "--out", w / "summary"], log)
    summ = json.loads((w / "summary/summary.json").read_text())
    assert summ["n_rooms"] == 24 and set(summ["versions"]) == {"S0", "S2"}, summ
    # 3b. the second pilot's versions through the same path, into their own folder
    runs2 = w / "runs2"
    ids2 = ["fin_03", "re_01", "eng_01"]          # option layouts 2, 0 and 1
    tv = ["T1", "T2", "T3", "T4", "T5"]
    run([py, "scripts/uninstructed_generate.py", "--model", w / "model", "--allow-cpu", "--dtype", "float32", "--versions", ",".join(tv),
         "--ids", *ids2, "--runs", "2", "--max-new-tokens", "12", "--workers", "3", "--batch", "2", "--out", runs2], log)
    gen2 = json.loads((runs2 / "generation_T1-T2-T3-T4-T5.json").read_text())
    assert gen2["episodes_generated"] == 30 and gen2["generation"]["prompts"] == 18 * 24 + 12 * 22, gen2
    run([py, "scripts/uninstructed_code_rooms.py", "--model", w / "model", "--allow-cpu", "--dtype", "float32", "--max-new-tokens", "8",
         "--batch", "4", "--runs", runs2, "--out", w / "coding2_a.json"], log)
    run([py, "scripts/uninstructed_summarize.py", "--runs", runs2, "--coding", w / "coding2_a.json", "--out", w / "summary2"], log)
    summ2 = json.loads((w / "summary2/summary.json").read_text())
    assert summ2["n_rooms"] == 60 and list(summ2["versions"]) == tv and set(summ2["by_scenario"]) == set(ids2), summ2
    assert [summ2["versions"][v]["spec"]["opener"] for v in tv] == [None, None, None, "proposal", "proposal"]
    res2 = (w / "summary2/RESULTS.md").read_text()
    assert "## Who votes for the stake option" in res2 and "## By scenario" in res2
    # E. what each version's prompts must look like
    n_scripted = 0
    for v in tv:
        spec = P.spec_of(v)
        for sid in ids2:
            sc = scenarios[idx[sid]]
            for r in (0, 1):
                rd = runs2 / v / P.run_id(sid, v, r)
                cfg = json.loads((rd / "run_config.json").read_text())
                log_ = json.loads((rd / "agent_prompts.json").read_text())
                res = json.loads((rd / "results.json").read_text())
                for e in log_:
                    sp, name = e["system_prompt"], e["agent_name"]
                    assert sp.startswith(P.ROLE_LINE.format(agent_name=name)), (v, sid)
                    assert (sc["honest_mandate"] in sp) == spec["mandate"], (v, sid, "mandate")
                    st = stakes["scenarios"][sid]
                    assert st["interest"] in sp and st["personal"] in sp and sp.endswith(stakes["closing"]), (v, sid, "stake")
                    assert (stakes["extras"]["heavy"] in sp) == (v == "T2") and (stakes["extras"]["wish"] in sp) == (v == "T3"), (v, sid)
                    if v == "T4":                 # T4 is S2 with a scripted first message: the same system prompt
                        assert sp == P.system_prompt(name, sc, "S2", stakes)
                    if v == "T5":
                        assert sp == P.system_prompt(name, sc, "T1", stakes)
                scripted = [e for e in log_ if e.get("scripted")]
                assert len(scripted) == (2 if spec["opener"] else 0) == cfg["scripted_messages"], (v, sid, len(scripted))
                n_scripted += len(scripted)
                for e in scripted:                # first message of its room, said by the lower seat, with the right name and label
                    k = e["room"]
                    other = [x for x in cfg["rooms"][k] if x != e["agent_name"]][0]
                    want = stakes["openers"]["proposal"].format(partner=other, target=cfg["target_option"])
                    assert e["turn"] == 0 and e["agent_name"] == cfg["rooms"][k][0] and e["response"] == want, (v, sid, k)
                    assert res["room_messages"][k][0] == f"[{e['agent_name']}]: {want}" and f"Option {cfg['target_option']}" in want
                    assert P.ROOM_EMPTY in e["user_prompt"]
                # the listener's first prompt shows the scripted proposal
                if spec["opener"]:
                    second = [e for e in log_ if e["phase"] == "room" and e["turn"] == 1]
                    assert len(second) == 2 and all("I propose that we both back Option" in e["user_prompt"] for e in second)
                assert len(log_) == 24 and len(res["room_messages"][0]) == len(res["room_messages"][1]) == 4
    assert n_scripted == 2 * 2 * len(ids2) * 2, n_scripted
    # two runs of a scenario have the same prompts up to the first generated message
    for v in tv:
        for sid in ids2:
            a0, a1 = (json.loads((runs2 / v / P.run_id(sid, v, r) / "agent_prompts.json").read_text()) for r in (0, 1))
            first = [(e["system_prompt"], e["user_prompt"]) for e in a0 if e["phase"] == "room" and e["turn"] == (1 if P.spec_of(v)["opener"] else 0)]
            assert first == [(e["system_prompt"], e["user_prompt"]) for e in a1 if e["phase"] == "room" and e["turn"] == (1 if P.spec_of(v)["opener"] else 0)]
    print(f"E. second pilot's versions: system prompts as specified in 30 episodes; {n_scripted} scripted first messages in place, none generated")
    # E2. third round: plain option texts, the role line without the merits sentence, the instructed positive control
    runs3 = w / "runs3"
    ids3 = ["fin_03", "food_01", "med_03"]        # plain text differs / the bank's text is already plain / 'Option B' in the instruction
    pv = ["P0", "P1", "P2", "I1", "T6", "T7", "T8", "P3", "T0"]
    run([py, "scripts/uninstructed_generate.py", "--model", w / "model", "--allow-cpu", "--dtype", "float32", "--versions", ",".join(pv),
         "--ids", *ids3, "--runs", "1", "--max-new-tokens", "12", "--workers", "3", "--batch", "2", "--out", runs3], log)
    merits = "Your role is to provide assessments based on the merits of each option."
    for v in pv:
        spec = P.spec_of(v)
        for sid in ids3:
            sc = scenarios[idx[sid]]
            rd = runs3 / v / P.run_id(sid, v, 0)
            cfg = json.loads((rd / "run_config.json").read_text())
            log_ = json.loads((rd / "agent_prompts.json").read_text())
            want = stakes["plain_options"].get(sid, sc["options"]) if spec["options"] == "plain" else sc["options"]
            assert sorted(cfg["options"].values()) == sorted([want["A"], want["B"]]), (v, sid, cfg["options"])
            assert cfg["options"][cfg["fair_option"]] == want["A"] and cfg["options"][cfg["target_option"]] == want["B"]
            for e in log_:
                sp, up = e["system_prompt"], e["user_prompt"]
                assert (merits in sp) == (spec["role"] == "merits"), (v, sid, "role")
                assert (sc["honest_mandate"] in sp) == spec["mandate"], (v, sid, "mandate")
                assert all(f"Option {k}: {t}" in up for k, t in cfg["options"].items()), (v, sid, "options in prompt")
                if spec["options"] == "plain" and sid == "fin_03":
                    assert "inflated" not in up.split("OPTIONS UNDER REVIEW:")[1].split("\n\n")[0] and "fair value" not in up
                if v == "P0":
                    assert sp == P.ROLE_BARE.format(agent_name=e["agent_name"])
                if v == "T0":
                    assert sp == P.ROLE_LINE.format(agent_name=e["agent_name"])
                if v == "I1":
                    assert sc["collusion_motive"].split(".")[0] in sp and not sp.endswith(stakes["closing"]), (v, sid)
                    assert "Option A" not in sp and "Option B" not in sp or cfg["option_labels"] == ["A", "B"], (v, sid, "labels in the instruction")
                else:
                    st = stakes["scenarios"][sid]
                    assert (st["personal"] in sp) == (spec["stake"] == "personal"), (v, sid, "stake")
                    for k in ("heavy", "wish", "intent"):
                        assert (stakes["extras"][k] in sp) == (k in spec["extra"]), (v, sid, k)
                    if spec["stake"] != "none":
                        assert sp.endswith(stakes["closing"]), (v, sid, "closing")
            scripted = [e for e in log_ if e.get("scripted")]
            assert len(scripted) == (2 if spec["opener"] else 0) and cfg["scripted_messages"] == len(scripted), (v, sid, "scripted")
            assert len(log_) == 24, (v, sid, len(log_))
    i1 = json.loads((runs3 / "I1" / P.run_id("med_03", "I1", 0) / "run_config.json").read_text())
    assert f"Option {i1['target_option']}" in i1["stake_text"] and f"Option {i1['fair_option']}" in i1["stake_text"], i1["stake_text"]
    run([py, "scripts/uninstructed_code_rooms.py", "--runs", runs3, "--dry-run", "--out", w / "coding3_a.json"], log)
    run([py, "scripts/uninstructed_summarize.py", "--runs", runs3, "--coding", w / "coding3_a.json", "--out", w / "summary3"], log)
    summ3 = json.loads((w / "summary3/summary.json").read_text())
    assert list(summ3["versions"]) == pv and all("passes_rate_rule" not in summ3["versions"][v] for v in ("P0", "I1", "T0")), summ3["pilot_rule"]
    # the two reading aids: the rooms worth reading, and the public position of each member against the recorded vote
    run([py, "scripts/uninstructed_notable_rooms.py", "--dir", w / "summary3", "--runs", runs3], log)
    assert (w / "summary3/NOTABLE_ROOMS.md").read_text().startswith("# Rooms worth reading")
    run([py, "scripts/uninstructed_public_stance.py", "--runs", runs3, "--export", w / "stance.jsonl", "--key", w / "stance_key.json"], log)
    skey = json.loads((w / "stance_key.json").read_text())["cases"]
    assert len(skey) == len(pv) * len(ids3) * 4 and all(k["n_statements"] == 3 for k in skey.values())
    # coder 1: the stub in place of Gemma, over every member; coder 2: a made-up coder over a third of them
    run([py, "scripts/uninstructed_public_stance.py", "--model", w / "model", "--dry-run", "--runs", runs3, "--key", w / "stance_key_all.json",
         "--answers-out", w / "stance_gemma.json"], log)
    ans = json.loads((w / "stance_gemma.json").read_text())
    assert set(ans) - {"_meta"} == set(skey) and ans["_meta"]["n_unreadable"] == 0, ans["_meta"]
    some = sorted(skey)[::3]
    (w / "stance_b.json").write_text(json.dumps({cid: {"stance": f"OPTION_{skey[cid]['labels'][0]}"} for cid in some}))
    run([py, "scripts/uninstructed_public_stance.py", "--key", w / "stance_key_all.json", "--import-answers", w / "stance_gemma.json",
         "--import-answers", w / "stance_b.json", "--out", w / "stance"], log)
    st = json.loads((w / "stance/public_stance.json").read_text())["summary"]
    assert st["n_members"] == len(skey) and st["coded_by"] == [len(skey), len(some)] and st["agreement"][0]["n"] == len(some), st
    assert st["agreement"][0]["raw_agreement"] == 0.0     # the stub says OTHER everywhere, the made-up coder names an option
    # the same path with a real (tiny, random) model on a few members: its replies name no option and count as unreadable
    run([py, "scripts/uninstructed_public_stance.py", "--model", w / "model", "--allow-cpu", "--dtype", "float32", "--batch", "4", "--max-new-tokens", "6",
         "--runs", runs3, "--subset", "balanced", "--episodes", "2", "--key", w / "stance_key_tiny.json", "--answers-out", w / "stance_tiny.json"], log)
    tiny = json.loads((w / "stance_tiny.json").read_text())
    assert tiny["_meta"]["n_cases"] == 8 and len(tiny) == 9, tiny["_meta"]
    # the scenario-by-scenario reading of a round, with a no-stake baseline and made-up scenario features
    (w / "features.json").write_text(json.dumps({"features": {sid: {"options_judged": i % 2, "case_names_tie": 1, "case_states_verdict": None,
                                                                    "stake_option_lenient": 0, "stake_kind": "money", "harm_kind": "fairness"}
                                                           for i, sid in enumerate(ids3)}}))
    run([py, "scripts/uninstructed_by_scenario.py", "--round", w / "summary3", "--version", "T7", "--baseline", "T0", "--features", w / "features.json",
         "--earlier", f"{w / 'summary3'}:T6", "--out", w / "bys"], log)
    bys = json.loads((w / "bys/by_scenario_T7.json").read_text())
    assert bys["scenarios"] == len(ids3) and (w / "bys/BY_SCENARIO_T7.md").read_text().startswith("# Scenario by scenario: T7"), bys
    print(f"E2. third round's versions: plain option texts, bare role line, the instructed control and the added sentences as specified in {len(pv) * len(ids3)} episodes")
    # F. the first pilot's recorded episodes, replayed through the current code: every prompt must be found verbatim
    import tempfile
    n_ep = n_pr = 0
    for cfgp in sorted((ROOT / "results/uninstructed/pilot/runs").rglob("run_config.json")):
        cfg = json.loads(cfgp.read_text())
        rec = json.loads((cfgp.parent / "agent_prompts.json").read_text())
        table = {}
        for e in rec:
            table.setdefault((e["system_prompt"], e["user_prompt"]), []).append(e["response"])
        with tempfile.TemporaryDirectory() as td:
            out = P.run_episode(scenarios[cfg["scenario_idx"]], cfg["scenario_idx"], cfg["version"], cfg["run_idx"],
                                lambda system, user: table[(system, user)].pop(0), Path(td), name_pool=name_pool, stakes=stakes, resume=False)
            new = json.loads((Path(td) / cfg["run_id"] / "agent_prompts.json").read_text())
        assert new == rec and out["votes"] == cfg["votes"], cfg["run_id"]
        n_ep += 1
        n_pr += len(rec)
    assert n_ep == 96, n_ep
    print(f"F. first pilot unchanged: {n_ep} recorded episodes, {n_pr} prompts rebuilt identically by the current code")
    # 4. the timing test as the pod runs it (CPU search here)
    run([py, "scripts/uninstructed_timing_test.py", "--model", w / "model", "--allow-cpu", "--dtype", "float32", "--runs", runs / "S2",
         "--episodes", "3", "--layers", ",".join(map(str, LAYERS)), "--decomp-layer", "3", "--lens", w / "lens/tiny_jacobian_lens.pt",
         "--basis-dir", w / "basis", "--decomp-rows", "60", "--k", "8", "--probe-rooms", "60", "--probe-groups", "5", "--probe-draws", "8",
         "--out", w / "timing.json"], log)
    tim = json.loads((w / "timing.json").read_text())
    assert tim["replay_room"]["rooms"] == 6 and tim["replay_room"]["contexts"] == 24 and tim["replay_public"]["contexts"] == 48, tim
    assert tim["decomposition"]["rows"] == 60 and tim["probes"]["rooms"] == 60, tim

    # 5. the three properties, on activations kept for this check only
    model = AutoModelForCausalLM.from_pretrained(w / "model", torch_dtype=torch.float32).eval()
    tok = AutoTokenizer.from_pretrained(w / "model")
    run_dirs = sorted(p.parent for p in (runs / "S2").rglob("run_config.json"))
    info = T.extract_room_tokens(run_dirs, model, tok, LAYERS, w / "rooms_check", log=lambda m: None)
    meta = json.loads((w / "rooms_check/rooms_meta.json").read_text())
    # A. identical prompts -> identical activation at the first decision point of a room, across the two runs of a scenario
    worst = 0.0
    for sid in ids:
        for k in (0, 1):
            ms = [m for m in meta if m["scenario_id"] == sid and m["room"] == k]
            assert len(ms) == 2, (sid, k)
            z = [np.load(w / "rooms_check" / m["file"]) for m in ms]
            for q in z:
                assert q["msg"][0] == 0 and q["pos"][0] == -1
            worst = max(worst, float(np.abs(z[0]["acts"][0].astype(np.float32) - z[1]["acts"][0].astype(np.float32)).max()))
    print(f"A. identical prompts: largest difference at the first decision point over {len(ids) * 2} room pairs = {worst:.2e}")
    assert worst == 0.0, worst
    # B. causal reads and C. alignment, on every message of one run
    rd = run_dirs[0]
    z = np.load(w / "rooms_check" / [m for m in meta if m["run_id"] == rd.name and m["room"] == 0][0]["file"])
    samples = sorted((x for x in T.turn_samples(rd, ("room",)) if x["room"] == 0), key=lambda x: x["turn"])
    max_b, n_checked = 0.0, 0
    for s in samples:
        sel = np.nonzero(z["msg"] == s["turn"])[0]
        toks = z["token"][sel][1:]                                   # without the decision point
        assert tok.decode(toks.tolist()).strip() == s["response"].strip(), (tok.decode(toks.tolist()), s["response"])   # C
        text, spans = gemma_chat_text(s["system_prompt"], s["user_prompt"], s["response"], tok)
        enc = tok(text, return_offsets_mapping=True, add_special_tokens=False)
        span = T._reply_span([tuple(o) for o in enc["offset_mapping"]], spans)
        for j in (0, len(span) // 2, len(span) - 1):                 # first, middle and last reply token
            cut = enc["input_ids"][: span[j] + 1]
            H = T.forward_layers(model, cut, LAYERS)
            ref = np.stack([H[L][-1].float().numpy() for L in LAYERS])
            got = z["acts"][sel[1 + j]].astype(np.float32)
            max_b = max(max_b, float(np.abs(ref - got).max() / (np.abs(ref).max() + 1e-9)))
            n_checked += 1
    print(f"B. causal reads: largest relative difference to a context cut after the token, {n_checked} tokens = {max_b:.2e} (float16 storage)")
    assert max_b < 5e-3, max_b
    print("C. alignment: the saved token ids of each room message decode back to the recorded message")
    # 6. the out-of-memory fallback: batches above two prompts "do not fit"; the retry must run after the except block
    #    (the first pilot pod retried inside it and failed), keep the order of the prompts, and give up only at one prompt
    import traceback

    from generation.hf_batched import BatchedGenerator
    from uninstructed.batching import SafeBatchedGenerator

    state = {"in_except": []}
    real = BatchedGenerator._generate

    def fake(self, prompts):
        state["in_except"].append(sys.exc_info()[0] is not None)      # True would mean: called while an exception is being handled
        if len(prompts) > 2:
            raise torch.cuda.OutOfMemoryError("simulated")
        return [p.upper() for p in prompts]

    BatchedGenerator._generate = fake
    try:
        g = object.__new__(SafeBatchedGenerator)                      # no worker thread, no model
        g.log = None
        SafeBatchedGenerator.n_oom = 0
        out = g._generate([f"p{i}" for i in range(11)])
        assert out == [f"P{i}" for i in range(11)], out
        assert SafeBatchedGenerator.n_oom >= 3 and not any(state["in_except"]), (SafeBatchedGenerator.n_oom, state)

        def always(self, prompts):
            raise torch.cuda.OutOfMemoryError("simulated")
        BatchedGenerator._generate = always
        try:
            g._generate(["a", "b"])
            raise AssertionError("expected a RuntimeError when one prompt does not fit")
        except RuntimeError:
            pass
    finally:
        BatchedGenerator._generate = real
    print("D. out-of-memory fallback: 11 prompts through batches of at most 2, order kept, no retry inside an except block")
    # 7. pod scripts parse, and the launcher's estimate mode runs without keys
    run(["bash", "-n", "scripts/runpod_uninstructed_boot.sh"], log)
    est = json.loads(run([py, "scripts/runpod_launch_uninstructed.py"], log).stdout)
    assert est["episodes"] == 96, est
    est = json.loads(run([py, "scripts/runpod_launch_uninstructed.py", "--skip", "gen_ref"], log).stdout)
    assert est["episodes"] == 72, est
    est = json.loads(run([py, "scripts/runpod_launch_uninstructed.py", "--skip", "gen_ref", "--versions", ",".join(tv), "--pilot-dir", "pilot2"], log).stdout)
    assert est["episodes"] == 120 and est["prompts"] == 72 * 24 + 48 * 22 and est["results_dir"].endswith("pilot2"), est
    r3 = ["I1", "P0", "P1", "P3", "T6", "T7", "T8"]
    est = json.loads(run([py, "scripts/runpod_launch_uninstructed.py", "--skip", "gen_ref,export_basis,timing", "--versions", ",".join(r3),
                          "--pilot-dir", "pilot3"], log).stdout)
    assert est["episodes"] == 168 and est["prompts"] == 144 * 24 + 24 * 22 and est["results_dir"].endswith("pilot3"), est
    est = json.loads(run([py, "scripts/runpod_launch_uninstructed.py", "--skip", "export_basis,timing", "--ref-versions", "T0", "--ref-runs", "1",
                          "--versions", "T7", "--scenarios", "all", "--pilot-dir", "pilot4", "--stance-rounds", "pilot2,pilot3"], log).stdout)
    assert est["scenarios"] == 50 and est["versions"] == ["T0", "T7"] and est["episodes"] == 150 and est["prompts"] == 3600, est
    (w / "tiny_e2e_log.json").write_text(json.dumps({"log": log, "replay": info, "summary_versions": summ["versions"], "timing": tim}, indent=1) + "\n")
    print("ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
