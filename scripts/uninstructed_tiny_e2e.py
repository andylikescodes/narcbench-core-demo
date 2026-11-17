#!/usr/bin/env python3
"""Pipeline check for the uninstructed-collusion pilot on a tiny random Gemma-2 model (CPU, no downloads).

Every number this writes is meaningless as science: the model is randomly initialised. The point is that each stage
of the pilot pod runs end to end and writes files in the shapes the GPU run will produce, and that three properties
the design relies on hold:

  A. identical prompts: two runs of a scenario have the same activation at the token before the first room message;
  B. causal reads: the activation saved for a reply token equals the one from a context cut right after that token;
  C. alignment: the saved token ids of every room message decode back to the recorded message.

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
    # 6. pod scripts parse, and the launcher's estimate mode runs without keys
    run(["bash", "-n", "scripts/runpod_uninstructed_boot.sh"], log)
    est = json.loads(run([py, "scripts/runpod_launch_uninstructed.py"], log).stdout)
    assert est["episodes"] == 96, est
    (w / "tiny_e2e_log.json").write_text(json.dumps({"log": log, "replay": info, "summary_versions": summ["versions"], "timing": tim}, indent=1) + "\n")
    print("ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
