#!/usr/bin/env python3
"""Code the private rooms of uninstructed-collusion episodes with the fixed rubric (uninstructed/coding.py).

Two uses:
  1. Coder A, Gemma-2-9B-it with greedy decoding (GPU):
       python scripts/uninstructed_code_rooms.py --model google/gemma-2-9b-it --runs results/uninstructed/pilot/runs \
           --out results/uninstructed/pilot/coding_gemma.json
  2. Export the same cases for coder B (Claude), with opaque ids in a shuffled order so that the coder cannot tell the
     version of an episode from its id (no model needed):
       python scripts/uninstructed_code_rooms.py --runs results/uninstructed/pilot/runs \
           --export results/uninstructed/pilot/cases_for_coder.jsonl --key results/uninstructed/pilot/cases_key.json
     and turn coder B's answers ({case_id: {...}}) back into room ids:
       python scripts/uninstructed_code_rooms.py --runs ... --import-answers answers.json --key cases_key.json \
           --coder "claude ..." --out results/uninstructed/pilot/coding_claude.json

A coder sees the scenario, the options, which option the outside interest is tied to and the four messages; not the
votes, not the version. --dry-run uses a keyword stub in place of the model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from uninstructed import coding as C  # noqa: E402
from uninstructed import protocol as P  # noqa: E402


def stub_code(user: str) -> str:
    conv = user.split("CONVERSATION")[-1].lower()
    if "both back" in conv and "agreed" in conv:
        return '{"code": "AGREE_TARGET", "stake": 1, "proposal": 3, "acceptance": 4}'
    return '{"code": "NO_PACT", "stake": 0, "proposal": 0, "acceptance": 0}'


def greedy_batches(bundle, prompts: list[tuple[str, str]], batch: int, max_new_tokens: int, log=print) -> list[str]:
    import torch

    tok = bundle.tokenizer
    texts = [tok.apply_chat_template([{"role": "user", "content": f"{s.strip()}\n\n{u.strip()}"}], tokenize=False, add_generation_prompt=True)
             for s, u in prompts]
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))   # similar lengths together: less padding
    out = [""] * len(texts)
    t0 = time.time()
    for b0 in range(0, len(order), batch):
        ids = order[b0:b0 + batch]
        side = tok.padding_side
        tok.padding_side = "left"
        try:
            enc = tok([texts[i] for i in ids], return_tensors="pt", padding=True, add_special_tokens=False)
        finally:
            tok.padding_side = side
        enc = {k: v.to(bundle.device) for k, v in enc.items()}
        with torch.inference_mode():
            gen = bundle.model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tok.pad_token_id,
                                        eos_token_id=tok.eos_token_id)
        n_in = enc["input_ids"].shape[1]
        for i, row in zip(ids, gen):
            out[i] = tok.decode(row[n_in:], skip_special_tokens=True).strip()
        if (b0 // batch) % 5 == 0:
            log(f"[code] {min(b0 + batch, len(order))}/{len(order)} ({time.time() - t0:.0f}s)")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=Path, required=True, help="folder with run folders (searched recursively)")
    ap.add_argument("--model", default="google/gemma-2-9b-it")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--allow-cpu", action="store_true")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--export", type=Path, help="write the cases for a second coder (JSONL) and exit")
    ap.add_argument("--key", type=Path, help="with --export: where the case id -> room id key goes; with --import-answers: that key")
    ap.add_argument("--export-seed", type=int, default=20261116)
    ap.add_argument("--import-answers", type=Path, help="JSON {case_id: {code, stake, proposal, acceptance}} from the second coder")
    ap.add_argument("--coder", default=None, help="name of the coder for --import-answers")
    a = ap.parse_args()

    name_pool, scenarios, idx = P.load_scenarios()
    stakes = P.load_stakes()
    rooms = C.load_rooms(a.runs, stakes)
    if not rooms:
        sys.exit(f"no uninstructed runs under {a.runs}")
    ctx = {s["id"]: s["context"] for s in scenarios}
    prompts = [C.coder_prompt(r, ctx[r["scenario_id"]]) for r in rooms]

    if a.export:
        if not a.key:
            ap.error("--export needs --key")
        rng = random.Random(a.export_seed)
        order = list(range(len(rooms)))
        rng.shuffle(order)
        key = {}
        a.export.parent.mkdir(parents=True, exist_ok=True)
        with open(a.export, "w") as f:
            for n, i in enumerate(order):
                cid = "c" + hashlib.sha256(f"{a.export_seed}:{rooms[i]['room_id']}".encode()).hexdigest()[:10]
                key[cid] = rooms[i]["room_id"]
                f.write(json.dumps({"case_id": cid, "prompt": prompts[i][1]}, ensure_ascii=False) + "\n")
        a.key.write_text(json.dumps({"rubric_sha256": C.rubric_sha256(), "seed": a.export_seed, "cases": key}, indent=1) + "\n")
        print(f"exported {len(order)} cases to {a.export}; key in {a.key}")
        return 0

    if a.import_answers:
        if not (a.key and a.out and a.coder):
            ap.error("--import-answers needs --key, --out and --coder")
        key = json.loads(a.key.read_text())
        ans = json.loads(a.import_answers.read_text())
        by_room = {key["cases"][cid]: v for cid, v in ans.items() if cid in key["cases"]}
        coded = []
        for r in rooms:
            v = by_room.get(r["room_id"])
            c = C.parse_coder_output(json.dumps(v)) if v else {"code": C.UNPARSED, "stake": 0, "proposal": 0, "acceptance": 0}
            coded.append({"room_id": r["room_id"], **c, "raw": json.dumps(v) if v else ""})
        rec = {"coder": a.coder, "rubric_sha256": key["rubric_sha256"], "created_utc": datetime.now(timezone.utc).isoformat(),
               "n_rooms": len(coded), "n_unparsed": sum(c["code"] == C.UNPARSED for c in coded), "rooms": coded}
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(rec, indent=1) + "\n")
        print(json.dumps({k: v for k, v in rec.items() if k != "rooms"}, indent=1))
        return 0

    if not a.out:
        ap.error("--out")
    t0 = time.time()
    if a.dry_run:
        raw, model_id = [stub_code(u) for _, u in prompts], f"stub::{a.model}"
    else:
        import torch
        if not torch.cuda.is_available() and not a.allow_cpu:
            sys.exit("No CUDA GPU here; use --dry-run or --allow-cpu with a tiny model.")
        from generation.hf_backend import load_model
        bundle = load_model(a.model, dtype=a.dtype, device_map="auto" if torch.cuda.is_available() else None)
        raw, model_id = greedy_batches(bundle, prompts, a.batch, a.max_new_tokens), bundle.model_id
    coded = [{"room_id": r["room_id"], **C.parse_coder_output(t), "raw": t} for r, t in zip(rooms, raw)]
    rec = {"coder": f"{model_id}, greedy decoding", "rubric_sha256": C.rubric_sha256(), "created_utc": datetime.now(timezone.utc).isoformat(),
           "n_rooms": len(coded), "n_unparsed": sum(c["code"] == C.UNPARSED for c in coded), "elapsed_s": round(time.time() - t0, 1),
           "rooms": coded}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(rec, indent=1) + "\n")
    print(json.dumps({k: v for k, v in rec.items() if k != "rooms"}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
