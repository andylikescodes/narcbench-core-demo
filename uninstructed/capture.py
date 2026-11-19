"""Activation reads for stage 2 (docs/uninstructed_collusion/DESIGN.md, section 7 and the dated note "stage 2").

Every recorded turn is passed through the model once more in the context in which it was written, and the residual
stream is read at the requested layers. Nothing is generated. Under <out>/:

  rooms/<group>/<run id>__room<k>.npz   every token of every room message plus the token before it (timeline.py)
  public/<group>.npz, .json             each public statement and each vote at three positions: the token just before the
                                        reply starts (pre), its last token (end) and the last token of the sequence
                                        (last: the read position of the instructed study); acts [n turns, 3, layers, d]
  neutral/<group>.npz, .json            the neutral reader: the same model reads the room's words alone, and the public
                                        discussion's words alone, without system prompt, scenario or the other channel.
                                        room [n rooms, 4 messages, 6 fixed points, layers, d];
                                        pub  [n episodes, 12 statements, (pre, end), layers, d]
  core/activations.npz, metadata.json   the instructed benchmark: public statements of its collusion episodes at `last`
  tables/                               what the probes read: one array per layer over units and fixed points, the words
                                        said up to each point, and index.json (units with their labels)

<group> is <round folder>_<version>: run ids repeat between rounds 3 and 4, so files are kept per round.
All arrays are float16; a point that does not exist (a message without tokens) is nan.
"""
from __future__ import annotations

import gzip
import json
import re
import time
from pathlib import Path

import numpy as np

from jspace_interp.positions import gemma_chat_text

from . import stage2 as S
from . import timeline as T

PUB_POS = ("pre", "end", "last")


def _tokens_in(offsets, a0: int, a1: int) -> list[int]:
    """Tokens that overlap the character span [a0, a1)."""
    return [i for i, (a, b) in enumerate(offsets) if b > a and b > a0 and a < a1]


def _prompt_log(run_dir: Path) -> list[dict]:
    return json.loads((Path(run_dir) / "agent_prompts.json").read_text())


def capture_rooms(groups, root: Path, model, tok, layers: list[int], out: Path, *, log=print) -> dict:
    info = {}
    for rnd, ver in groups:
        g = S.group_name(rnd, ver)
        dirs = S.run_dirs(root, rnd, ver)
        dest = Path(out) / "rooms" / g
        meta = dest / "rooms_meta.json"
        if meta.exists() and len(json.loads(meta.read_text())) == 2 * len(dirs):
            log(f"[capture] rooms of {g}: already there")
            continue
        info[g] = T.extract_room_tokens(dirs, model, tok, layers, dest, log=lambda m: None)
        log(f"[capture] rooms of {g}: {info[g]['rooms']} rooms, {info[g]['contexts']} contexts, {info[g]['seconds_per_context']} s per context")
    return info


def capture_public(groups, root: Path, model, tok, layers: list[int], out: Path, *, log=print) -> dict:
    """The public statements and the votes: pre, end and last of each turn."""
    info = {}
    (Path(out) / "public").mkdir(parents=True, exist_ok=True)
    for rnd, ver in groups:
        g = S.group_name(rnd, ver)
        f = Path(out) / "public" / f"{g}.npz"
        if f.exists():
            log(f"[capture] public turns of {g}: already there")
            continue
        rows, A, t0 = [], [], time.time()
        for rd in S.run_dirs(root, rnd, ver):
            for s in T.turn_samples(rd, ("discussion", "final")):
                text, spans = gemma_chat_text(s["system_prompt"], s["user_prompt"], s["response"], tok)
                enc = tok(text, return_offsets_mapping=True, add_special_tokens=False)
                ids, offsets = enc["input_ids"], [tuple(o) for o in enc["offset_mapping"]]
                span = T._reply_span(offsets, spans)
                first = span[0] if span else len(ids) - 2
                idx = [first - 1, span[-1] if span else first - 1, len(ids) - 1]
                H = T.forward_layers(model, ids, layers)
                A.append(np.stack([H[L][idx].float().cpu().numpy().astype(np.float16) for L in layers], axis=1))   # [3, layers, d]
                rows.append({"run_id": s["run_id"], "seat": s["seat"], "agent_name": s["agent_name"], "phase": s["phase"],
                             "round": s["round"], "n_reply_tokens": len(span), "n_tokens": len(ids)})
        np.savez(f, acts=np.stack(A))
        f.with_suffix(".json").write_text(json.dumps(rows) + "\n")
        el = time.time() - t0
        info[g] = {"contexts": len(rows), "seconds": round(el, 1), "seconds_per_context": round(el / max(len(rows), 1), 3)}
        log(f"[capture] public turns of {g}: {len(rows)} contexts, {info[g]['seconds_per_context']} s per context")
    return info


def reader_tokens(header: str, lines: list[tuple[str, str]], tok) -> tuple[list[int], list[list[int]]]:
    """The neutral reader's input: a user turn that holds the header and one line '[Name]: words' per message, nothing
    else. Returns the token ids and, for every message, the tokens of its words."""
    parts, spans, pos = [header + "\n"], [], len(header) + 1
    for name, body in lines:
        prefix = f"[{name}]: "
        spans.append((pos + len(prefix), pos + len(prefix) + len(body)))
        parts.append(prefix + body + "\n")
        pos += len(prefix) + len(body) + 1
    content = "".join(parts).strip()
    text, sp = gemma_chat_text("", content, None, tok)
    c0 = sp["content"][0]
    assert c0 >= 0, "the reader's text was not found in its chat layout"
    enc = tok(text, return_offsets_mapping=True, add_special_tokens=False)
    offsets = [tuple(o) for o in enc["offset_mapping"]]
    return enc["input_ids"], [_tokens_in(offsets, c0 + a, c0 + b) if b > a else [] for a, b in spans]


def capture_neutral(groups, root: Path, model, tok, layers: list[int], out: Path, *, log=print) -> dict:
    """The neutral reader of section 8: one pass over each room's transcript and one over each episode's public
    discussion. The model is causal, so the read at a token has seen the words before it and nothing else."""
    info = {}
    (Path(out) / "neutral").mkdir(parents=True, exist_ok=True)
    for rnd, ver in groups:
        g = S.group_name(rnd, ver)
        f = Path(out) / "neutral" / f"{g}.npz"
        if f.exists():
            log(f"[capture] neutral reader of {g}: already there")
            continue
        room_a, pub_a, meta, t0, n_ctx = [], [], {"rooms": [], "episodes": []}, time.time(), 0
        for rd in S.run_dirs(root, rnd, ver):
            entries = _prompt_log(rd)
            run_id = json.loads((rd / "run_config.json").read_text())["run_id"]
            for k in (0, 1):
                turns = sorted((e for e in entries if e["phase"] == "room" and e["room"] == k), key=lambda e: e["turn"])
                ids, toks = reader_tokens("CONVERSATION:", [(e["agent_name"], e["response"]) for e in turns], tok)
                H = T.forward_layers(model, ids, layers)
                arr = np.full((len(turns), len(S.FIXED), len(layers), H[layers[0]].shape[-1]), np.nan, dtype=np.float16)
                for m, tk in enumerate(toks):
                    for fi, (name, j) in enumerate(S.fixed_index(len(tk)).items()):
                        if not tk:
                            continue                      # a message without words: no point of it exists for the reader
                        p = tk[0] - 1 if j < 0 else tk[j]
                        arr[m, S.FIXED.index(name)] = np.stack([H[L][p].float().cpu().numpy() for L in layers]).astype(np.float16)
                room_a.append(arr)
                meta["rooms"].append({"run_id": run_id, "room": k})
                n_ctx += 1
            disc = [e for e in entries if e["phase"] == "discussion"]
            ids, toks = reader_tokens("DISCUSSION:", [(e["agent_name"], e["response"]) for e in disc], tok)
            H = T.forward_layers(model, ids, layers)
            arr = np.full((len(disc), 2, len(layers), H[layers[0]].shape[-1]), np.nan, dtype=np.float16)
            for m, tk in enumerate(toks):
                if tk:
                    for c, p in enumerate((tk[0] - 1, tk[-1])):
                        arr[m, c] = np.stack([H[L][p].float().cpu().numpy() for L in layers]).astype(np.float16)
            pub_a.append(arr)
            meta["episodes"].append({"run_id": run_id, "statements": [{"seat": e["seat"], "round": e["round"]} for e in disc]})
            n_ctx += 1
        np.savez(f, room=np.stack(room_a), pub=np.stack(pub_a))
        f.with_suffix(".json").write_text(json.dumps(meta) + "\n")
        el = time.time() - t0
        info[g] = {"contexts": n_ctx, "seconds": round(el, 1), "seconds_per_context": round(el / max(n_ctx, 1), 3)}
        log(f"[capture] neutral reader of {g}: {n_ctx} contexts, {info[g]['seconds_per_context']} s per context")
    return info


def capture_core(model, tok, layers: list[int], out: Path, *, max_runs: int | None = None, log=print) -> dict:
    """The instructed benchmark (recorded Core episodes): the public statements of its collusion episodes, read at the
    last token as in the instructed study. They give the instructed probe of Q6 and the instructed direction of 15.3."""
    from jspace_interp.agents import load_rows, runs_dir_for
    from jspace_interp.contexts import apply_condition
    from jspace_interp.extract import extract_rows

    dest = Path(out) / "core"
    if (dest / "activations.npz").exists():
        log("[capture] instructed benchmark: already there")
        return {}
    rows = [r for r in load_rows("core") if r.get("mode") == "collusion" and r.get("channel") == "public" and r.get("phase") == "discussion"]
    if max_runs:
        keep = sorted({r["run_id"] for r in rows})[:max_runs]
        rows = [r for r in rows if r["run_id"] in keep]
    t0 = time.time()
    extract_rows(apply_condition(rows, "original", runs_dir_for("core"), "core"), model, tok, layers, dest, positions=("last",),
                 log=lambda m: None)
    el = time.time() - t0
    log(f"[capture] instructed benchmark: {len(rows)} public statements of {len({r['run_id'] for r in rows})} collusion episodes, "
        f"{el / max(len(rows), 1):.3f} s per context")
    return {"contexts": len(rows), "seconds": round(el, 1), "seconds_per_context": round(el / max(len(rows), 1), 3)}


def build_tables(groups, root: Path, out: Path, layers: list[int], tok, *, apply_only=S.APPLY_ONLY, grid_step: int = 4,
                 grid_n: int = 60, log=print) -> dict:
    """Assemble what the probes read from the per-turn files: for every layer

      member_L<L>.npy        [members, MEMBER_POINTS, d]   a member's own two room messages (six fixed points each), then
                                                           before and at the end of its three public statements and its vote
      member_last_L<L>.npy   [members, d]                  mean over the three public statements at `last` (Q6)
      neutral_room_L<L>.npy  [rooms, ROOM_POINTS, d]       the neutral reader at the same points of the room
      neutral_pub_L<L>.npy   [members, PUBLIC_POINTS, d]   the neutral reader of the public discussion (the vote's two
                                                           points read the end of the discussion)
      grid_L<L>.npy          [rooms, grid_n, d]            every grid_step-th position of the room in order (the token
                                                           clock); grid_seat.npy says who is writing there (-1: past the end)
    and texts.json.gz with the words said up to each point (message bodies only, no names), once as they are and once
    with every word marked as the unit's own or someone else's (the keys ending in _who), and index.json."""
    out = Path(out)
    tdir = out / "tables"
    tdir.mkdir(parents=True, exist_ok=True)
    members, rooms = S.load_units(groups, root, apply_only=apply_only)
    mi, ri = {m["uid"]: i for i, m in enumerate(members)}, {r["uid"]: i for i, r in enumerate(rooms)}
    nL, MP, RP, PP = len(layers), S.MEMBER_POINTS, S.ROOM_POINTS, S.PUBLIC_POINTS
    A = {}                                              # allocated when the first activation shows the width

    def alloc(d):
        if not A:
            A.update(member=np.full((len(members), len(MP), nL, d), np.nan, np.float16), last=np.zeros((len(members), nL, d), np.float32),
                     nroom=np.full((len(rooms), len(RP), nL, d), np.nan, np.float16), npub=np.full((len(members), len(PP), nL, d), np.nan, np.float16),
                     grid=np.full((len(rooms), grid_n, nL, d), np.nan, np.float16))
    grid_seat = np.full((len(rooms), grid_n), -1, np.int8)
    texts = {"room": [[""] * len(RP) for _ in rooms], "public": [[""] * len(PP) for _ in members], "grid": [[""] * grid_n for _ in rooms],
             "room_who": [[""] * len(RP) for _ in rooms], "public_who": [[""] * len(PP) for _ in members], "grid_who": [[""] * grid_n for _ in rooms]}
    n_last = np.zeros(len(members), int)
    dec = lambda toks: tok.decode([int(t) for t in toks], skip_special_tokens=True).strip() if len(toks) else ""   # noqa: E731
    join = lambda parts: "\n".join(p for p in parts if p)                                                            # noqa: E731

    def who(parts) -> str:
        """The same words with every word marked as the unit's own (own_) or someone else's (oth_): parts = [(text, own?)]."""
        return "\n".join(" ".join(("own_" if own else "oth_") + w for w in re.findall(r"\w+", t)) for t, own in parts if t)

    for rnd, ver in groups:
        g = S.group_name(rnd, ver)
        logs = {json.loads((rd / "run_config.json").read_text())["run_id"]: _prompt_log(rd) for rd in S.run_dirs(root, rnd, ver)}
        # rooms: the speaker side, token by token
        for meta in json.loads((out / "rooms" / g / "rooms_meta.json").read_text()):
            z = np.load(out / "rooms" / g / meta["file"])
            acts, msg, pos, token, speaker = z["acts"], z["msg"], z["pos"], z["token"], z["speaker"]
            alloc(acts.shape[-1])
            r = ri[f"{rnd}|{meta['run_id']}|room{meta['room']}"]
            bodies = [e["response"].strip() for e in sorted((e for e in logs[meta["run_id"]] if e["phase"] == "room" and e["room"] == meta["room"]),
                                                             key=lambda e: e["turn"])]
            for k in range(4):
                sel = np.nonzero(msg == k)[0]
                if len(sel) == 0:
                    continue
                assert pos[sel[0]] == -1, "a room message starts with its decision point"
                m = mi[f"{rnd}|{meta['run_id']}|s{int(speaker[sel[0]])}"]
                assert members[m]["room"] == meta["room"] and (members[m]["seat"] % 2) == k % 2, "who writes which room message"
                for name, j in S.fixed_index(len(sel) - 1).items():
                    A["member"][m, MP.index(f"o{k // 2 + 1}.{name}")] = acts[sel[0] + 1 + j]
                    part = dec(token[sel[1:2 + j]]) if j >= 0 else ""
                    texts["room"][r][RP.index(f"m{k + 1}.{name}")] = join(bodies[:k] + [part])
                    texts["room_who"][r][RP.index(f"m{k + 1}.{name}")] = who([(bodies[q], q % 2 == k % 2) for q in range(k)] + [(part, True)])
            for gi in range(min(grid_n, (len(msg) + grid_step - 1) // grid_step)):
                p = gi * grid_step
                A["grid"][r, gi] = acts[p]
                grid_seat[r, gi] = speaker[p]
                k, j = int(msg[p]), int(pos[p])
                part = dec(token[np.nonzero(msg == k)[0][1:2 + j]]) if j >= 0 else ""
                texts["grid"][r][gi] = join(bodies[:k] + [part])
                texts["grid_who"][r][gi] = who([(bodies[q], q % 2 == k % 2) for q in range(k)] + [(part, True)])
        # public statements and votes
        z = np.load(out / "public" / f"{g}.npz")["acts"]
        for i, row in enumerate(json.loads((out / "public" / f"{g}.json").read_text())):
            m = mi[f"{rnd}|{row['run_id']}|s{row['seat']}"]
            name = f"p{row['round']}" if row["phase"] == "discussion" else "v"
            A["member"][m, MP.index(f"{name}.pre")], A["member"][m, MP.index(f"{name}.end")] = z[i, 0], z[i, 1]
            if row["phase"] == "discussion":
                A["last"][m] += z[i, 2].astype(np.float32)
                n_last[m] += 1
        for run_id, entries in logs.items():
            disc = [e for e in entries if e["phase"] == "discussion"]
            said = [e["response"].strip() for e in disc]
            for e in entries:
                if e["phase"] not in ("discussion", "final"):
                    continue
                m = mi[f"{rnd}|{run_id}|s{e['seat']}"]
                room_turns = sorted((x for x in entries if x["phase"] == "room" and x["room"] == e["room"]), key=lambda x: x["turn"])
                own_room = [x["response"].strip() for x in room_turns]
                if e["phase"] == "discussion":
                    j = next(q for q, x in enumerate(disc) if x is e)
                    name, before, own = f"p{e['round']}", said[:j], said[j]
                else:
                    j = len(disc)
                    name, before, own = "v", said, e["response"].strip()
                texts["public"][m][PP.index(f"{name}.pre")] = join(own_room + before)
                texts["public"][m][PP.index(f"{name}.end")] = join(own_room + before + [own])
                tagged = [(x["response"].strip(), x["seat"] == e["seat"]) for x in room_turns] + [(x["response"].strip(), x["seat"] == e["seat"]) for x in disc[:j]]
                texts["public_who"][m][PP.index(f"{name}.pre")] = who(tagged)
                texts["public_who"][m][PP.index(f"{name}.end")] = who(tagged + [(own, True)])
        # the neutral reader
        nz = np.load(out / "neutral" / f"{g}.npz")
        nmeta = json.loads((out / "neutral" / f"{g}.json").read_text())
        for i, row in enumerate(nmeta["rooms"]):
            A["nroom"][ri[f"{rnd}|{row['run_id']}|room{row['room']}"]] = nz["room"][i].reshape(len(RP), nL, -1)
        for i, ep in enumerate(nmeta["episodes"]):
            for j, st in enumerate(ep["statements"]):
                m = mi[f"{rnd}|{ep['run_id']}|s{st['seat']}"]
                A["npub"][m, PP.index(f"p{st['round']}.pre")], A["npub"][m, PP.index(f"p{st['round']}.end")] = nz["pub"][i, j, 0], nz["pub"][i, j, 1]
            for seat in range(4):
                m = mi[f"{rnd}|{ep['run_id']}|s{seat}"]
                A["npub"][m, PP.index("v.pre")] = A["npub"][m, PP.index("v.end")] = nz["pub"][i, len(ep["statements"]) - 1, 1]
        log(f"[tables] {g}: {sum(1 for r in rooms if r['group'] == g)} rooms, {sum(1 for m in members if m['group'] == g)} members")

    assert (n_last == 3).all(), "every member has three public statements"
    files = {}
    for li, L in enumerate(layers):
        for name, arr in (("member", A["member"][:, :, li]), ("member_last", (A["last"][:, li] / 3).astype(np.float16)),
                          ("neutral_room", A["nroom"][:, :, li]), ("neutral_pub", A["npub"][:, :, li]), ("grid", A["grid"][:, :, li])):
            np.save(tdir / f"{name}_L{L}.npy", np.ascontiguousarray(arr))
            files[f"{name}_L{L}.npy"] = list(arr.shape)
    np.save(tdir / "grid_seat.npy", grid_seat)
    with gzip.open(tdir / "texts.json.gz", "wt") as fh:
        json.dump(texts, fh)
    index = {"layers": layers, "d": int(A["member"].shape[-1]), "member_points": list(MP), "room_points": list(RP), "public_points": list(PP),
             "grid": {"step": grid_step, "n": grid_n}, "groups": [list(x) for x in groups], "apply_only": [list(x) for x in apply_only],
             "members": members, "rooms": rooms, "files": files,
             "missing_points": int(np.isnan(A["member"][:, :, 0, 0].astype(np.float32)).sum())}
    (tdir / "index.json").write_text(json.dumps(index) + "\n")
    return {"members": len(members), "rooms": len(rooms), "files": len(files), "missing_points": index["missing_points"]}
