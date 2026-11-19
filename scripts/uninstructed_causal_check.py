#!/usr/bin/env python3
"""Checks of the causal half of stage 2 on a tiny random model (CPU). Nothing here is a result.

Run by scripts/uninstructed_tiny_e2e.py (check I), or by hand:

  python scripts/uninstructed_causal_check.py --work <scratch folder> --model <tiny model> \\
      --lens <tiny lens .pt> --basis <tiny basis folder>

Checks (DESIGN.md 15.9 plus the parts added on 18 November 2026):
  i     an intervention of size zero reproduces the unsteered output token for token
  ii    swapping a room's activation with itself changes nothing; a swap with a donor puts the donor's activation there
  iii   a large addition changes the output, and only the anchor position of the prompt pass and the generated
        positions are edited
  iv    projecting out leaves the component along the direction at the fixed value at every edited position
  v     the same row seed gives the same continuation whatever else is in the batch
  vi    the prompt rebuild reproduces every recorded prompt of the test round and of the recorded analysis episodes
  vii   with a forced prefix the regenerated message starts with the recorded prefix; messages before the anchor are kept
  viii  d = d_J + d_R
  ix    both phases run end to end through the command line; --plan matches what ran; a second call finds nothing to do
  plus  the row sampler never takes a token of probability zero and follows the probabilities; the summary recovers a
        planted effect and marks a condition with many unparsed votes as broken; the recorded episodes with Gemma's own
        tokenizer when it is cached on the machine (reported as NOT CHECKED otherwise)
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path


def build_round(repo: Path, work: Path, model: Path, layers: str) -> None:
    """A test round for the causal checks: 8 scenarios of 8 domains x 4 runs of version T7, written by a stub whose texts
    DIFFER between runs and whose votes follow what was said in the room, then read by the tiny model.

    The stub of scripts/uninstructed_generate.py --dry-run says the same words in every run, so two runs of a scenario
    have exactly the same activation at every read point and every direction is zero. Here the opener proposes or not,
    the answerer accepts or not, filler words vary, and a member votes for the stake option with probability 0.9 after an
    accepted proposal, 0.45 after a declined one and 0.1 otherwise."""
    import random

    from uninstructed import protocol as P

    root, acts = work / "causal_root", work / "causal_acts"
    if (acts / "tables/index.json").exists():
        return
    subprocess.run(["rm", "-rf", str(root), str(acts)], check=True)
    name_pool, scenarios, idx = P.load_scenarios()
    stakes = P.load_stakes()
    filler = ["on balance", "frankly", "as the file shows", "in my reading", "for the record", "to be plain", "all told", "at this point"]

    def make_generate(target: str, fair: str, rng: random.Random):
        fill = lambda: " and ".join(rng.sample(filler, 2))   # noqa: E731

        def generate(system: str, user: str) -> str:
            if "RECOMMENDATION: [" in user:
                room = user.split("BEFORE THE MEETING")[1].split("FULL DISCUSSION HISTORY")[0]
                p = 0.9 if "Agreed" in room else 0.45 if "I propose" in room else 0.1
                return f"RECOMMENDATION: Option {target if rng.random() < p else fair}" + "\n" + f"RATIONALE: {fill()}."
            if "CONVERSATION SO FAR:" in user:
                hist = user.split("CONVERSATION SO FAR:")[1]
                n = hist.count("]: ")
                if n == 0:
                    return f"I propose that we both back the sponsor's side, {fill()}." if rng.random() < 0.6 else f"What is your read of the case, {fill()}?"
                if n == 1:
                    if "I propose" in hist:
                        return f"Agreed, we back it together, {fill()}." if rng.random() < 0.65 else f"No, I will judge it on the record, {fill()}."
                    return f"I would keep to the evidence, {fill()}."
                return f"Then we know where we stand, {fill()}." if n == 2 else f"Understood, {fill()}."
            return f"The evidence before us should decide this, {fill()}."
        return generate

    for sid in P.pilot_scenarios(scenarios)[:8]:
        for r in range(4):
            lay = P.option_layout(scenarios[idx[sid]], idx[sid])
            P.run_episode(scenarios[idx[sid]], idx[sid], "T7", r, make_generate(lay["target"], lay["fair"], random.Random(f"{sid}:{r}")),
                          root / "r1/runs/T7", name_pool=name_pool, stakes=stakes, model_id="stub with varied texts")
    py = sys.executable
    for cmd in ([py, "scripts/uninstructed_code_rooms.py", "--runs", str(root / "r1/runs"), "--dry-run", "--out", str(root / "r1/coding_a.json")],
                [py, "scripts/uninstructed_summarize.py", "--runs", str(root / "r1/runs"), "--coding", str(root / "r1/coding_a.json"), "--out", str(root / "r1")],
                [py, "scripts/uninstructed_capture.py", "--model", str(model), "--allow-cpu", "--dtype", "float32", "--layers", layers, "--root", str(root),
                 "--groups", "r1:T7", "--apply-only", "", "--core-runs", "3", "--out", str(acts)]):
        p = subprocess.run(cmd, cwd=repo, capture_output=True, text=True)
        assert p.returncode == 0, (cmd, p.stdout[-1500:], p.stderr[-1500:])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--lens", type=Path, required=True)
    ap.add_argument("--basis", type=Path, required=True)
    ap.add_argument("--layer", type=int, default=3)
    a = ap.parse_args()
    sys.path.insert(0, str(a.repo))
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from uninstructed import causal as Z
    from uninstructed import stage2 as S

    build_round(a.repo, a.work, a.model, "2,3,4")
    root, acts, L = a.work / "causal_root", a.work / "causal_acts", a.layer
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.float32).eval()
    tok = AutoTokenizer.from_pretrained(a.model)
    index = json.loads((acts / "tables/index.json").read_text())
    members, rooms, MP = index["members"], index["rooms"], index["member_points"]
    M = np.load(acts / f"tables/member_L{L}.npy").astype(np.float64)
    d = model.config.hidden_size
    rel = lambda x, y: float(np.abs(np.asarray(x) - np.asarray(y)).max() / (np.abs(np.asarray(y)).max() + 1e-12))   # noqa: E731

    # vi. the prompt rebuild
    r1 = Z.check_rebuild(tuple(tuple(g) for g in index["groups"]), root)
    r2 = Z.check_rebuild(S.ANALYSIS, S.RU)
    print(f"vi.   prompt rebuild from the recorded history: {r1['prompts']} prompts of {r1['episodes']} test episodes and "
          f"{r2['prompts']} prompts of {r2['episodes']} recorded analysis episodes reproduced verbatim (room, public and vote prompts, "
          "system prompts, and the speaking order of the first public round)")
    assert r2["episodes"] == 220 and r2["prompts"] == 220 * 24

    # directions at m2.pre, with J-space parts from the tiny lens
    anchor = "m2.pre"
    rows, col = Z.point_units(members, anchor)
    X = M[rows, MP.index(col)]
    XJ = Z.j_parts(X, a.lens, a.basis, L, k=25)
    inst = Z.instructed_direction(acts, L)
    D = Z.Directions(members, rows, X, XJ=XJ, inst=inst)
    desc = D.describe(n_shuffles=50)
    errs = [H["split_error"] for H in D.halves if H is not None]
    print(f"viii. d = d_J + d_R: largest relative error over the halves = {max(errs):.1e}; halves have "
          f"{[None if H is None else H['sets'] for H in D.halves]} sets with both classes; cosine between the halves "
          f"{desc.get('cosine_between_halves', float('nan')):.3f}; instructed direction present: {inst is not None}")
    assert max(errs) < 1e-9 and all(H is not None for H in D.halves)
    parity, _ = S.own_point(anchor)
    for r in rooms:
        r["anchor_uid"] = r["members"][parity]
    usable = lambda u: D.at(u["anchor_uid"]) is not None and D.for_domain(u["domain"]) is not None   # noqa: E731
    sel, pools = Z.deal(rooms, Z.room_side, 4, Z.SEED, usable)
    assert sel["remove"] and sel["induce"], "the test round has no set of rooms with both kinds"
    for side in sel:                                 # drawn from sets that hold both kinds; never the instructed control
        for u in sel[side]:
            assert Z.room_side(u) == side and u["version"] != "I1" and pools[u["set"]]["remove"] and pools[u["set"]]["induce"]
    eps = Z.episodes_of(rooms, root)
    unit = sel["remove"][0]
    ep = eps[(unit["round"], unit["run_id"])]
    turns = ep.room_turns(unit["room"])
    n_same = sum(Z.anchor_prompt(tok, eps[(r["round"], r["run_id"])].room_turns(r["room"])[1], "pre")[0]
                 == Z.gen_prompt_ids(tok, eps[(r["round"], r["run_id"])].room_turns(r["room"])[1]["system_prompt"],
                                     eps[(r["round"], r["run_id"])].room_turns(r["room"])[1]["user_prompt"]) for r in rooms)
    print(f"      the capture's tokens up to a message's decision point equal the generation prompt's tokens in {n_same} of {len(rooms)} rooms")

    st = Z.Steerer(model, tok, L, max_new_tokens=12, batch=4)
    sample = (sel["remove"] + sel["induce"])[:6]
    jobs = []
    for u in sample:
        e = eps[(u["round"], u["run_id"])].room_turns(u["room"])[1]
        jobs.append({"ids": Z.anchor_prompt(tok, e, "pre")[0], "seed": Z.row_seed(u["uid"], 0, "m2")})
    with_op = lambda op: [{**j, "op": op} for j in jobs]   # noqa: E731

    # i. size zero
    base = st.run(with_op(None))
    z_add, z_first = st.run(with_op({"add": np.zeros(d)})), st.run(with_op({"add_first": np.zeros(d)}))
    mixed = st.run([{**j, "op": ({"add": np.zeros(d)} if i % 2 else None)} for i, j in enumerate(jobs)])
    print(f"i.    size zero: {len(jobs)} rows x 12 tokens identical to the unsteered output under add 0 ({z_add == base}), "
          f"add-at-the-anchor 0 ({z_first == base}) and in a batch that mixes steered and unsteered rows ({mixed == base})")
    assert z_add == base and z_first == base and mixed == base

    # ii. swap with itself; swap with a donor
    _, first_self = Z.unit_ops("swap", "remove", unit, 0, D, unit)
    j0 = {"ids": Z.anchor_prompt(tok, turns[1], "pre")[0], "seed": Z.row_seed(unit["uid"], 0, "m2")}
    self_swap = st.run([{**j0, "op": first_self}])
    plain = st.run([{**j0, "op": None}])
    don = Z.donor(unit, "remove", pools)
    assert Z.room_side(don) == "induce" and don["set"] == unit["set"] and don["uid"] != unit["uid"]
    _, first = Z.unit_ops("swap", "remove", unit, 0, D, don)
    st.trace = []
    swapped = st.run([{**j0, "op": first}])
    tr = st.trace
    st.trace = None
    e_before, e_after = rel(tr[0]["before"][0], D.at(unit["anchor_uid"])), rel(tr[0]["after"][0], D.at(don["anchor_uid"]))
    later = max(float(np.abs(t["after"] - t["before"]).max()) for t in tr[1:])
    print(f"ii.   swap: with itself the output is unchanged ({self_swap == plain}; largest entry of the added vector "
          f"{float(np.abs(first_self['add_first']).max()):.1e}); with a donor of the other kind and the same prompts the state at the anchor "
          f"was the recipient's captured activation (relative difference {e_before:.1e}) and became the donor's ({e_after:.1e}); "
          f"no later position was edited (largest change {later:.1e}); output changed: {swapped != plain}")
    assert self_swap == plain and e_before < 5e-3 and e_after < 5e-3 and later == 0.0
    _, fJ = Z.unit_ops("swapJ", "remove", unit, 0, D, don)
    _, fR = Z.unit_ops("swapR", "remove", unit, 0, D, don)
    print(f"      J-space swap plus remainder swap equals the full swap: {rel(fJ['add_first'] + fR['add_first'], first['add_first']):.1e}")
    assert rel(fJ["add_first"] + fR["add_first"], first["add_first"]) < 1e-9

    # iii. a large addition, and where the hook edits
    big = np.random.default_rng(0).standard_normal(d) * 40
    st.trace = []
    steered = st.run([{**j0, "op": {"add": big}}])
    tr = st.trace
    st.trace = None
    pos0 = np.nonzero(tr[0]["changed"][0])[0].tolist()
    print(f"iii.  a large addition changes the output ({steered != plain}); passes: the prompt ({tr[0]['seq']} tokens, edited positions "
          f"{pos0}), then {len(tr) - 1} passes of one token each, every one edited ({all(t['seq'] == 1 and t['changed'].all() for t in tr[1:])}); "
          f"the edit is the vector itself (largest deviation {max(float(np.abs(t['after'][0] - t['before'][0] - big).max()) for t in tr):.1e})")
    assert steered != plain and pos0 == [tr[0]["seq"] - 1] and all(t["seq"] == 1 and t["changed"].all() for t in tr[1:])
    assert len(tr) == len(steered[0]) and max(float(np.abs(t["after"][0] - t["before"][0] - big).max()) for t in tr) < 1e-3

    # iv. projecting out
    H = D.for_domain(unit["domain"])
    steer, _ = Z.unit_ops("own1", "remove", unit, 0, D)
    st.trace = []
    st.run([{**j0, "op": steer}, {**jobs[1], "op": None}])
    tr = st.trace
    st.trace = None
    comp = [float(t["after"][0] @ H["u"]) for t in tr]
    off = max(float(np.abs((t["after"][0] - t["before"][0]) - ((t["after"][0] - t["before"][0]) @ H["u"]) * H["u"]).max()) for t in tr)
    other = max(float(np.abs(t["after"][1] - t["before"][1]).max()) for t in tr)
    print(f"iv.   project out: the component along the direction is {H['c0']:.4f} (the class-0 mean) at every edited position "
          f"(largest deviation {max(abs(c - H['c0']) for c in comp):.1e} over {len(comp)} positions); nothing else of the state moves "
          f"({off:.1e}); the unsteered row of the same batch is untouched ({other:.1e})")
    assert max(abs(c - H["c0"]) for c in comp) < 1e-3 and off < 1e-4 and other == 0.0
    add1, _ = Z.unit_ops("own1", "induce", sel["induce"][0], 0, D)
    a4, _ = Z.unit_ops("own4", "remove", unit, 0, D)
    r4a, _ = Z.unit_ops("rand4", "remove", unit, 0, D)
    r4b, _ = Z.unit_ops("rand4", "remove", unit, 1, D)
    i4, _ = Z.unit_ops("inst4", "remove", unit, 0, D)
    nrm = np.linalg.norm
    print(f"      doses: one dose has the length of d ({nrm(add1['add']):.3f} = {D.for_domain(sel['induce'][0]['domain'])['norm']:.3f}); four doses "
          f"subtract on the remove side (cosine with d {float(a4['add'] @ H['d'] / nrm(a4['add']) / H['norm']):+.2f}, length ratio {nrm(a4['add']) / H['norm']:.2f}); "
          f"shuffled-label and instructed directions have the same length ({nrm(r4a['add']) / H['norm']:.2f}, {nrm(i4['add']) / H['norm']:.2f}); "
          f"a fresh shuffle per regeneration (cosine between two {float(r4a['add'] @ r4b['add'] / nrm(r4a['add']) / nrm(r4b['add'])):+.2f})")
    assert abs(nrm(a4["add"]) / H["norm"] - 4) < 1e-9 and abs(nrm(r4a["add"]) / H["norm"] - 4) < 1e-9 and abs(nrm(i4["add"]) / H["norm"] - 4) < 1e-9
    # a direction is never applied to the half it came from
    for u in sel["remove"] + sel["induce"]:
        Hu = D.for_domain(u["domain"])
        assert not Hu["mask"][D.pos[u["anchor_uid"]]], "a room was given the direction of its own half"
    print("      halves: no chosen room lies in the half that estimated the direction applied to it")

    # v. the same seed in other company
    st2 = Z.Steerer(model, tok, L, max_new_tokens=12, batch=8)
    alone = [st2.run([{**j, "op": None}])[0] for j in jobs]
    others = []
    for u in rooms[:40:5]:                           # other scenarios, other lengths: padding
        e = eps[(u["round"], u["run_id"])].final_entry(u["agents"][0])
        others.append({"ids": Z.anchor_prompt(tok, e, "pre")[0], "seed": 5, "op": {"add": big}})
    lens = sorted({len(o["ids"]) for o in others} | {len(j["ids"]) for j in jobs})
    n_diff = 0
    for i, j in enumerate(jobs):
        got = st2.run(others[:5] + [{**j, "op": None}] + others[5:7])[5]
        n_diff += got != alone[i]
    print(f"v.    the same row seed in other company: {len(jobs) - n_diff} of {len(jobs)} rows give the same 12 tokens when the row shares a "
          f"batch with 7 steered rows of other lengths (input lengths from {lens[0]} to {lens[-1]} tokens, so the row is left-padded); "
          f"{n_diff} rows differ")
    # the out-of-memory fallback: batches above two rows "do not fit"; the halves are retried after the except block has been left
    mix = with_op(None) + [{**jobs[0], "op": {"add": big}}, {**jobs[1], "op": {"add_first": big}}]
    ref = st2.run(mix)
    real, seen = Z.Steerer._generate, {"in_except": [], "sizes": []}

    def tight(self, js):
        seen["in_except"].append(sys.exc_info()[0] is not None)
        seen["sizes"].append(len(js))
        if len(js) > 2:
            raise torch.cuda.OutOfMemoryError("simulated")
        return real(self, js)
    Z.Steerer._generate = tight
    try:
        got = st2.run(mix)
    finally:
        Z.Steerer._generate = real
    print(f"      out of memory, simulated above two rows: {len(mix)} rows through batches of {sorted(set(seen['sizes']))} rows, {st2.n_oom} splits, "
          f"the same tokens as in one batch ({got == ref}), no retry while an exception was being handled ({not any(seen['in_except'])})")
    assert got == ref and st2.n_oom >= 2 and not any(seen["in_except"])
    st2.close()

    # vii. a forced prefix
    recs = Z.regen_rooms(st, tok, [{"unit": u, "side": "remove", "cond": "base", "regen": 0} for u in sample[:3]], "m2.q2", eps,
                         lambda it: (None, None), layer=L)
    ok_prefix = ok_kept = 0
    for u, r in zip(sample[:3], recs):
        t = eps[(u["round"], u["run_id"])].room_turns(u["room"])
        ids, prefix = Z.anchor_prompt(tok, t[1], "q2")
        ok_prefix += r["messages"][1].startswith(f"[{t[1]['agent_name']}]: " + Z.decode_reply(tok, prefix)) and r["forced_reply_tokens"] == len(prefix) > 0
        ok_kept += r["messages"][0] == f"[{t[0]['agent_name']}]: {t[0]['response']}" and r["messages_kept"] == 1 and len(r["messages"]) == 4
    print(f"vii.  anchor inside a message (m2.q2): the regenerated second message starts with the recorded prefix in {ok_prefix} of 3 rooms; "
          f"the first message is the recorded one and four messages result in {ok_kept} of 3; two statements and two votes follow "
          f"({all(len(r['statements']) == 2 and len(r['vote_texts']) == 2 for r in recs)})")
    assert ok_prefix == 3 and ok_kept == 3
    again = Z.regen_rooms(st, tok, [{"unit": u, "side": "remove", "cond": "base", "regen": 0} for u in sample[:3]], "m2.q2", eps,
                          lambda it: (None, None), layer=L)
    other_regen = Z.regen_rooms(st, tok, [{"unit": u, "side": "remove", "cond": "base", "regen": 1} for u in sample[:3]], "m2.q2", eps,
                                lambda it: (None, None), layer=L)
    print(f"      the same regeneration index gives the same room again ({[r['messages'] for r in again] == [r['messages'] for r in recs]}); "
          f"another index gives another room ({[r['messages'] for r in other_regen] != [r['messages'] for r in recs]})")
    assert [r["messages"] for r in again] == [r["messages"] for r in recs] and [r["messages"] for r in other_regen] != [r["messages"] for r in recs]
    st.close()

    # the row sampler by itself
    probs = torch.tensor([[0.0, 0.5, 0.0, 0.3, 0.2, 0.0], [0.0, 0.0, 1.0, 0.0, 0.0, 0.0]])
    cnt = Counter()
    with Z.seeded_sampling([1, 2]) as sm:
        for _ in range(4000):
            t = torch.multinomial(probs, num_samples=1)
            cnt[int(t[0])] += 1
            assert int(t[1]) == 2
        assert sm.calls == 4000
        assert torch.multinomial(torch.ones(3), 2).shape == (2,)          # any other call goes to the original function
    assert torch.multinomial.__name__ == "multinomial"
    print(f"      row sampler: 4,000 draws from (0, .5, 0, .3, .2, 0) give {[cnt[i] for i in range(6)]}; a token of probability zero is never taken; "
          "torch.multinomial is restored afterwards")
    assert cnt[0] == cnt[2] == cnt[5] == 0 and abs(cnt[1] / 4000 - 0.5) < 0.03 and abs(cnt[3] / 4000 - 0.3) < 0.03
    seen = []
    orig = torch.multinomial
    def spy(p, num_samples=1, replacement=False, **kw):   # noqa: E306
        seen.append(int((p > 0).sum(-1).max()))
        return orig(p, num_samples, replacement, **kw)
    torch.multinomial = spy
    try:
        ids = jobs[0]["ids"]
        with torch.inference_mode():
            model.generate(input_ids=torch.tensor([ids]), attention_mask=torch.ones(1, len(ids), dtype=torch.long), max_new_tokens=6,
                           pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id, do_sample=True, temperature=0.7)
    finally:
        torch.multinomial = orig
    print(f"      what generate hands to the draw with the recorded runs' arguments: at most {max(seen)} tokens of non-zero probability per row "
          f"(vocabulary {len(tok)}), i.e. this transformers version ({__import__('transformers').__version__}) applies its default top-k")

    # the summary on made-up records: a planted effect of the own direction and none of the shuffled one; a broken condition
    rng = np.random.default_rng(3)
    fake = []
    for si in range(12):
        for ui in range(4):
            for cond, p in (("base", 0.8), ("own4", 0.3), ("rand4", 0.8), ("own1", 0.8), ("inst4", 0.8)):
                for g in range(4 if cond == "base" else 2):
                    u = np.random.default_rng([si, ui, g]).random()        # the same random number under every condition
                    both = bool(u < p)
                    unp = 2 if (cond == "inst4" and rng.random() < 0.5) else 0
                    fake.append({"phase": "room", "uid": f"r|s{si}|room{ui}", "side": "remove", "cond": cond, "regen": g, "scenario": f"s{si}",
                                 "both_stake": both and not unp, "stake_votes": [both, both], "unparsed": unp, "disclosure_flag": False})
    sm = Z.summarize(fake, "room")
    c1, t = sm["rules"]["C1"], sm["sides"]["remove"]
    print(f"      summary on made-up records (planted: own direction at four doses 80% -> 30%, shuffled and natural size none): C1 met = {c1['met']}; "
          f"own4 against the baseline {c1['own4_against_base']['effect']:+.2f} ({c1['own4_against_base']['lo']:+.2f} to {c1['own4_against_base']['hi']:+.2f}), "
          f"against shuffled {c1['own4_against_rand4']['effect']:+.2f}; natural size alone = {c1['natural_size_alone']}; the condition with half its votes "
          f"unparsed is marked broken ({t['inst4']['broken']}) and not read ({sm['rules']['C3']['remove']['inst4_against_base'] is None})")
    assert c1["met"] and not c1["natural_size_alone"] and t["inst4"]["broken"] and sm["rules"]["C3"]["remove"]["inst4_against_base"] is None
    assert abs(c1["own4_against_base"]["effect"] + 0.5) < 0.12 and abs(t["rand4"]["against_base"]["primary"]["effect"]) < 1e-12

    # the recorded episodes themselves, with Gemma's own tokenizer if it is cached here, and canned replies in place of a model
    real_tok = None
    for snap in sorted(Path.home().glob(".cache/huggingface/hub/models--google--gemma-2-9b-it/snapshots/*")):
        try:
            real_tok = AutoTokenizer.from_pretrained(snap)
        except Exception:  # noqa: BLE001
            real_tok = None
    if real_tok is None:
        print("      recorded episodes with Gemma's tokenizer: NOT CHECKED (the tokenizer is not cached on this machine)")
    else:
        import re

        mem_r, rooms_r = S.load_units(S.ANALYSIS, S.RU, apply_only=())
        eps_r = Z.episodes_of(rooms_r, S.RU)
        n_turns = n_same = n_pref = 0
        for r in rooms_r:
            for e in eps_r[(r["round"], r["run_id"])].room_turns(r["room"]):
                ids, first, _ = Z.recorded_turn(real_tok, e)                 # asserts that the reply's tokens are contiguous
                n_turns += 1
                n_same += ids[:first] == Z.gen_prompt_ids(real_tok, e["system_prompt"], e["user_prompt"])
                n_pref += e["response"].strip().startswith(Z.decode_reply(real_tok, Z.anchor_prompt(real_tok, e, "q2")[1]))
        n_v = sum(Z.anchor_prompt(real_tok, eps_r[(m["round"], m["run_id"])].final_entry(m["agent"]), "pre")[0]
                  == Z.gen_prompt_ids(real_tok, *[eps_r[(m["round"], m["run_id"])].final_entry(m["agent"])[k] for k in ("system_prompt", "user_prompt")])
                  for m in mem_r)
        print(f"      recorded episodes with Gemma's tokenizer: in {n_same} of {n_turns} room turns and {n_v} of {len(mem_r)} votes the capture's tokens "
              f"up to the decision point are the generation prompt's tokens; the forced prefix at q2 decodes to the start of the recorded message in "
              f"{n_pref} of {n_turns}")
        assert n_same == n_turns == 1760 and n_v == len(mem_r) == 880 and n_pref == n_turns

        class Canned:                                     # stands in for the Steerer: no model, replies that parse
            max_new_tokens = 256

            def run(self, js):
                out = []
                for j in js:
                    m = re.search(r"RECOMMENDATION: \[Option (\w+) or Option (\w+)\]", real_tok.decode(j["ids"]))
                    reply = (f"RECOMMENDATION: Option {m.group(1)}" + "\n" + "RATIONALE: canned." if m else
                             " and that is my view." if j.get("prefix") else "No, I will judge it on the record." if j.get("op") else "Agreed.")
                    out.append(real_tok(reply, add_special_tokens=False)["input_ids"])
                return out

        rows_r, _ = Z.point_units(mem_r, "m2.q2")
        Xr = np.random.default_rng(1).standard_normal((len(rows_r), 8))
        Dr = Z.Directions(mem_r, rows_r, Xr, XJ=Xr / 2, inst=np.ones(8))
        for r in rooms_r:
            r["anchor_uid"] = r["members"][1]
        sel_r, pools_r = Z.deal(rooms_r, Z.room_side, 2, Z.SEED, lambda u: Dr.for_domain(u["domain"]) is not None)
        items = Z.items_of(sel_r, Z.ROOM_CONDITIONS, 1)
        for it in items:
            if it["cond"].startswith("swap"):
                it["donor_unit"] = Z.donor(it["unit"], it["side"], pools_r)
        recs_r = Z.regen_rooms(Canned(), real_tok, items, "m2.q2", eps_r, lambda it: Z.unit_ops(it["cond"], it["side"], it["unit"], it["regen"], Dr, it.get("donor_unit")),
                               layer=21)
        ok = 0
        for r in recs_r:
            t = eps_r[(r["uid"].split("|")[0], r["uid"].split("|")[1])].room_turns(r["room"])
            ok += (r["messages"][0] == f"[{t[0]['agent_name']}]: {t[0]['response']}" and r["unparsed"] == 0 and len(r["messages"]) == 4
                   and r["messages"][1].startswith(f"[{t[1]['agent_name']}]: " + Z.decode_reply(real_tok, Z.anchor_prompt(real_tok, t[1], "q2")[1])))
        cases_r, key_r = Z.coder_cases(recs_r, eps_r)
        sm_r = Z.summarize(recs_r, "room")
        for m in mem_r:
            m["anchor_uid"] = m["uid"]
        sel_m, _ = Z.deal(mem_r, Z.member_side, 3, Z.SEED)
        rows_v, _ = Z.point_units(mem_r, "v.pre")
        Dv = Z.Directions(mem_r, rows_v, np.random.default_rng(2).standard_normal((len(rows_v), 8)))
        recs_v = Z.regen_votes(Canned(), real_tok, Z.items_of(sel_m, ("base", "own1", "own4", "rand4"), 1), eps_r,
                               lambda it: Z.unit_ops(it["cond"], it["side"], it["unit"], it["regen"], Dv), layer=21)
        print(f"      the same with canned replies in place of the model (4 recorded rooms x 10 conditions, anchor m2.q2; 6 members x 4 conditions): "
              f"{ok} of {len(recs_r)} regenerated rooms keep the first recorded message, start the second with the recorded prefix and end in two "
              f"parsed votes; {sum(r['unparsed'] == 0 for r in recs_v)} of {len(recs_v)} regenerated votes parse; {len(cases_r)} coder cases; "
              f"rules read: {sorted(sm_r['rules'])}")
        assert ok == len(recs_r) == 4 * 11 and all(r["unparsed"] == 0 for r in recs_v) and len(recs_v) == 6 * 5
        assert all("Message 4 [" in c["prompt"] and "It is rumoured that" in c["prompt"] for c in cases_r)

    # ix. both phases through the command line
    out = a.work / "causal_run"
    subprocess.run(["rm", "-rf", str(out)], check=True)
    py, script = sys.executable, str(a.repo / "scripts/uninstructed_causal.py")
    common = ["--root", str(root), "--acts", str(acts), "--layer", str(L), "--anchor", "m2.pre", "--phase", "both", "--per-side", "3",
              "--per-side-members", "5", "--regen", "1", "--lens", str(a.lens), "--basis-dir", str(a.basis), "--out", str(out)]
    p = subprocess.run([py, script, "--plan"] + common, cwd=a.repo, capture_output=True, text=True)
    assert p.returncode == 0, p.stderr[-2000:]
    plan = json.loads(p.stdout)
    p = subprocess.run([py, script, "--model", str(a.model), "--allow-cpu", "--dtype", "float32", "--max-new-tokens", "12", "--batch", "4",
                        "--chunk-rooms", "2", "--chunk-members", "4", "--code-gemma"] + common, cwd=a.repo, capture_output=True, text=True)
    assert p.returncode == 0, (p.stdout[-3000:], p.stderr[-3000:])
    room = [json.loads(ln) for ln in (out / "regen_room.jsonl").read_text().splitlines()]
    vote = [json.loads(ln) for ln in (out / "regen_vote.jsonl").read_text().splitlines()]
    got_room, got_vote = dict(Counter(r["cond"] for r in room)), dict(Counter(r["cond"] for r in vote))
    print(f"ix.   command line, both phases on the test round: {len(room)} regenerated rooms {got_room} and {len(vote)} regenerated votes {got_vote}; "
          f"--plan said {plan['room']['regenerated_rooms']} rooms and {plan['vote']['regenerated_votes']} votes, {plan['prompts']} prompts at most")
    assert got_room == plan["room"]["regenerated_rooms_by_condition"] and got_vote == plan["vote"]["regenerated_votes_by_condition"]
    assert set(got_room) == set(Z.ROOM_CONDITIONS) and set(got_vote) == set(Z.VOTE_CONDITIONS)
    run = json.loads((out / "run.json").read_text())
    assert list(run["phases"]) == ["vote", "room"], "the short vote phase runs first, so that a run stopped in the room phase keeps it"
    gen = [run["phases"][ph]["generation"] for ph in ("room", "vote")]
    rows_run, reused = sum(g["rows_generated"] for g in gen), sum(g["rows_reused"] for g in gen)
    print(f"      model rows generated {rows_run} (room phase {gen[0]['rows_generated']}, vote phase {gen[1]['rows_generated']}); reused because "
          f"input and seed were identical: {reused}; together {rows_run + reused}, the plan's {plan['prompts']} prompts")
    assert rows_run + reused == plan["prompts"]
    res = json.loads((out / "causal.json").read_text())
    assert (out / "CAUSAL.md").read_text().startswith("# Stage 2: causal tests") and set(res["room"]["rules"]) == {"C1", "C2", "C3", "C4"}
    assert set(res["vote"]["rules"]) == {"V1", "V2", "V3"} and res["codes"]["records_coded"] == len(room)
    key = json.loads((out / "coder_cases_key.json").read_text())
    cases = [json.loads(ln) for ln in (out / "coder_cases.jsonl").read_text().splitlines()]
    assert sorted(x for v in key["cases"].values() for x in v) == sorted(Z.record_id(r) for r in room) and len(cases) == len(key["cases"])
    assert all(c["prompt"].startswith("You are coding a private conversation") and "own4" not in c["prompt"] and "votes for" not in c["prompt"].lower()
               for c in cases)
    base_same = sum(r["messages"] == b["messages"] for r in room for b in room
                    if r["cond"] == "own1" and b["cond"] == "base" and r["uid"] == b["uid"] and r["regen"] == b["regen"])
    n_own1 = sum(r["cond"] == "own1" for r in room)
    print(f"      coder export: {len(cases)} distinct rooms stand for the {len(room)} regenerated rooms, in the case format of the recorded rooms' "
          f"coders, without condition or votes; the model's own coding pass ran on them ({res['codes']['coder']}); under the natural-size "
          f"condition {base_same} of {n_own1} regenerated rooms are word for word the baseline's room of the same regeneration")
    # a made-up second coder through the import path, then the summary again
    (out / "answers_b.json").write_text(json.dumps({c["case_id"]: {"code": "AGREE_TARGET" if i % 2 else "NO_PACT", "stake": 1, "proposal": 2, "acceptance": 0}
                                                    for i, c in enumerate(cases)}))
    p = subprocess.run([py, script, "--import-codes", str(out / "answers_b.json"), "--coder", "made-up coder B", "--out", str(out)],
                       cwd=a.repo, capture_output=True, text=True)
    assert p.returncode == 0, p.stderr[-2000:]
    res2 = json.loads((out / "causal.json").read_text())
    assert res2["codes"]["coder"] == "made-up coder B" and "pact" in res2["room"]["sides"]["remove"]["base"]["rates"]
    p = subprocess.run([py, script, "--model", str(a.model), "--allow-cpu", "--dtype", "float32", "--max-new-tokens", "12", "--batch", "4"] + common,
                       cwd=a.repo, capture_output=True, text=True)
    assert p.returncode == 0 and "0 of" in p.stdout, (p.stdout[-1500:], p.stderr[-1500:])
    again = [json.loads(ln) for ln in (out / "regen_room.jsonl").read_text().splitlines()]
    print(f"      a coder's answers go back in through --import-codes (pact and collusion outcomes appear); the same call again finds nothing left "
          f"to run ({len(again)} records, as before: {len(again) == len(room)}); unparsed votes on the random model: "
          f"{sum(r['unparsed'] for r in room)} of {2 * len(room)} (so every condition is marked broken here, as it must be)")
    assert len(again) == len(room)
    p = subprocess.run([py, script, "--model", str(a.model), "--allow-cpu", "--dtype", "float32", "--max-new-tokens", "12", "--layer", str(L + 1),
                        "--anchor", "m2.pre", "--root", str(root), "--acts", str(acts), "--out", str(out)], cwd=a.repo, capture_output=True, text=True)
    assert p.returncode != 0 and "Use another --out" in (p.stderr + p.stdout), (p.stdout[-800:], p.stderr[-800:])
    print("      a call with another layer into the same folder is refused")
    # a fresh machine: no tables in the capture folder, so the script reads what it needs at this layer only
    acts2, out2 = a.work / "causal_acts_fresh", a.work / "causal_run_fresh"
    subprocess.run(["rm", "-rf", str(acts2), str(out2)], check=True)
    tiny = ["--model", str(a.model), "--allow-cpu", "--dtype", "float32", "--max-new-tokens", "12", "--batch", "4"]
    p = subprocess.run([py, script] + tiny + ["--root", str(root), "--groups", "r1:T7", "--acts", str(acts2), "--layer", str(L), "--phase", "vote",
                                              "--per-side-members", "3", "--regen", "1", "--core-runs", "3", "--out", str(out2)],
                       cwd=a.repo, capture_output=True, text=True)
    assert p.returncode == 0, (p.stdout[-2500:], p.stderr[-2500:])
    idx2 = json.loads((acts2 / "tables/index.json").read_text())
    run2 = json.loads((out2 / "run.json").read_text())
    v2 = [json.loads(ln) for ln in (out2 / "regen_vote.jsonl").read_text().splitlines()]
    print(f"      without tables the script captures layer {idx2['layers']} itself (and the instructed benchmark): {len(v2)} regenerated votes under "
          f"{run2['phases']['vote']['conditions']}; not run for want of a lens: {sorted(run2['phases']['vote']['conditions_dropped'])}")
    assert idx2["layers"] == [L] and run2["phases"]["vote"]["conditions"] == ["base", "own1", "own4", "rand4", "inst4"]
    assert sorted(run2["phases"]["vote"]["conditions_dropped"]) == ["j4", "r4"] and len(v2) == 6 * (2 + 4)
    # the same members get the same baseline vote from the full capture and from the one-layer capture
    b1 = {(r["uid"], r["regen"]): r["vote_text"] for r in vote if r["cond"] == "base"}
    b2 = {(r["uid"], r["regen"]): r["vote_text"] for r in v2 if r["cond"] == "base"}
    shared = set(b1) & set(b2)
    print(f"      baseline votes that both runs regenerated: {len(shared)}; word for word the same in {sum(b1[k] == b2[k] for k in shared)}")
    assert all(b1[k] == b2[k] for k in shared)
    # the time limit: nothing starts after it, the report says so, and the next call finishes the work
    out3 = a.work / "causal_run_cap"
    subprocess.run(["rm", "-rf", str(out3)], check=True)
    capped = [py, script] + tiny + ["--root", str(root), "--acts", str(acts), "--layer", str(L), "--anchor", "m3.t0", "--phase", "room", "--per-side", "2",
                                    "--regen", "1", "--conditions", "base,own4,rand4", "--chunk-rooms", "1", "--out", str(out3)]
    p = subprocess.run(capped + ["--max-minutes", "0.000001"], cwd=a.repo, capture_output=True, text=True)
    assert p.returncode == 0, (p.stdout[-2000:], p.stderr[-2000:])
    r3 = json.loads((out3 / "run.json").read_text())
    assert r3["incomplete"] and "**Incomplete run:**" in (out3 / "CAUSAL.md").read_text() and not (out3 / "regen_room.jsonl").read_text()
    p = subprocess.run(capped, cwd=a.repo, capture_output=True, text=True)
    assert p.returncode == 0, (p.stdout[-2000:], p.stderr[-2000:])
    r3b = json.loads((out3 / "run.json").read_text())
    rec3 = [json.loads(ln) for ln in (out3 / "regen_room.jsonl").read_text().splitlines()]
    print(f"      time limit: the first call stops before any room ('{r3['incomplete']}'), the second finishes: {len(rec3)} regenerated rooms at the "
          f"anchor m3.t0 (two recorded messages kept: {all(r['messages_kept'] == 2 and r['forced_reply_tokens'] == 1 for r in rec3)}), "
          f"incomplete = {r3b['incomplete']}, the earlier call is on record ({len(r3b['earlier_calls'])})")
    assert r3b["incomplete"] is None and len(rec3) == 4 * (2 + 1 + 1) and len(r3b["earlier_calls"]) == 1
    assert all(r["messages_kept"] == 2 and r["forced_reply_tokens"] == 1 for r in rec3)
    # an anchor before anything was said: every set has one activation there, so there is no direction and nothing runs
    out4 = a.work / "causal_run_first"
    subprocess.run(["rm", "-rf", str(out4)], check=True)
    p = subprocess.run([py, script] + tiny + ["--root", str(root), "--acts", str(acts), "--layer", str(L), "--anchor", "m1.pre", "--phase", "room",
                                              "--per-side", "2", "--regen", "1", "--out", str(out4)], cwd=a.repo, capture_output=True, text=True)
    assert p.returncode == 0, (p.stdout[-2000:], p.stderr[-2000:])
    r4 = json.loads((out4 / "run.json").read_text())
    print(f"      anchor m1.pre (before the first message): not run, because {r4['phases']['room']['not_run']}")
    assert "no direction" in r4["phases"]["room"]["not_run"] and "Not run:" in (out4 / "CAUSAL.md").read_text()
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
