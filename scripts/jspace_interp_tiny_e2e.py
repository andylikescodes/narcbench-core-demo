"""Pipeline check for jspace-interp on a tiny random Gemma-2 model (CPU, no downloads).

Every number this writes is meaningless as science: the model is randomly initialised and the
tokenizer is trained here on the transcripts. The point is that each stage runs end to end on the
real NARCBench transcripts and writes files in the shapes the GPU run will produce:

  1. train a BPE tokenizer (Metaspace, Gemma special tokens) on the transcripts
  2. build a tiny Gemma2ForCausalLM (random weights), save model + tokenizer
  3. extract activations for all four conditions on a subset of Core and Transfer runs
  4. estimate a Jacobian lens on a few prompts and save the unembedding basis
  5. run the Experiment 1 + 2 analysis (k sweep, positions, random-subspace controls)
  6. run Experiment 3 steering on a few final-vote prefixes (prefix-cached tail passes, checked against full passes)
  7. dry-run the matched-vocabulary generator (stub model)
  8. generate warned episodes with the tiny model itself (threads + batched generate), extract the warned split for the
     original and prompt_only conditions, and re-run the analysis so the matched-vocabulary readings are exercised

  python scripts/jspace_interp_tiny_e2e.py --work /tmp/jspace_tiny
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

SPECIAL = ["<pad>", "<eos>", "<bos>", "<unk>", "<start_of_turn>", "<end_of_turn>"]
GEMMA_TEMPLATE = (
    "{{ bos_token }}{% for message in messages %}{% if message['role'] == 'assistant' %}{% set role = 'model' %}{% else %}{% set role = message['role'] %}{% endif %}{{ '<start_of_turn>' + role + '\\n' + message['content'] | trim + '<end_of_turn>\\n' }}{% endfor %}{% if add_generation_prompt %}{{ '<start_of_turn>model\\n' }}{% endif %}"
)
LAYERS = [2, 3, 4]


def train_tokenizer(texts, out: Path, vocab_size=4000):
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast

    tk = Tokenizer(models.BPE(unk_token="<unk>"))
    tk.pre_tokenizer = pre_tokenizers.Metaspace(replacement="▁", prepend_scheme="first")
    tk.decoder = decoders.Metaspace(replacement="▁", prepend_scheme="first")
    tr = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=SPECIAL, min_frequency=2)
    tk.train_from_iterator(texts, tr)
    fast = PreTrainedTokenizerFast(tokenizer_object=tk, pad_token="<pad>", eos_token="<eos>", bos_token="<bos>",
                                   unk_token="<unk>", additional_special_tokens=["<start_of_turn>", "<end_of_turn>"])
    fast.chat_template = GEMMA_TEMPLATE  # the generation backends call apply_chat_template
    out.mkdir(parents=True, exist_ok=True)
    fast.save_pretrained(out)
    tk.save(str(out / "tokenizer.json"))
    return fast


def run(cmd, log):
    t0 = time.time()
    cmd = [str(c) for c in cmd]
    p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    log.append({"cmd": " ".join(str(c) for c in cmd), "rc": p.returncode, "s": round(time.time() - t0, 1),
                "tail": (p.stdout + p.stderr)[-1500:]})
    print(f"[{p.returncode}] {' '.join(str(c) for c in cmd)[:140]} ({time.time() - t0:.0f}s)", flush=True)
    if p.returncode != 0:
        print(p.stdout[-3000:], p.stderr[-3000:])
        raise SystemExit(p.returncode)
    return p


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--core-rows", type=int, default=340)
    ap.add_argument("--xfer-rows", type=int, default=260)
    ap.add_argument("--max-tokens", type=int, default=768)
    args = ap.parse_args()
    import numpy as np
    import torch
    from transformers import Gemma2Config, Gemma2ForCausalLM

    from jspace_interp.agents import load_rows
    from jspace_interp.jacobian import estimate_lens, save_lens, save_unembed_basis

    torch.manual_seed(0)
    w = args.work
    w.mkdir(parents=True, exist_ok=True)
    log = []
    texts = []
    for split in ("core", "transfer"):
        for r in load_rows(split):
            texts += [r.get("system_prompt") or "", r.get("user_prompt") or "", r.get("response") or ""]
    tok = train_tokenizer(texts, w / "model")
    cfg = Gemma2Config(vocab_size=len(tok), hidden_size=64, intermediate_size=128, num_hidden_layers=6,
                       num_attention_heads=2, num_key_value_heads=1, head_dim=32, max_position_embeddings=8192,
                       sliding_window=4096, pad_token_id=tok.pad_token_id, bos_token_id=tok.bos_token_id,
                       eos_token_id=tok.eos_token_id, tie_word_embeddings=True)
    model = Gemma2ForCausalLM(cfg).eval()
    model.save_pretrained(w / "model")
    print("tokenizer", len(tok), "params", sum(p.numel() for p in model.parameters()))

    # 4. lens + basis (a few prompts, short)
    prompts = [tok(t[:2000], add_special_tokens=False)["input_ids"][:96] for t in texts[:400:40] if t]
    lens = estimate_lens(model, prompts, LAYERS, skip=4)
    save_lens(lens, w / "lens/tiny_jacobian_lens.pt")
    save_unembed_basis(model, w / "basis")

    py = sys.executable
    lay = ",".join(map(str, LAYERS))
    run([py, "scripts/jspace_interp_extract.py", "--model", w / "model", "--splits", "core", "--layers", lay,
         "--max-rows", args.core_rows, "--max-tokens", args.max_tokens, "--device", "cpu", "--dtype", "float32",
         "--out", w / "acts"], log)
    run([py, "scripts/jspace_interp_extract.py", "--model", w / "model", "--splits", "transfer", "--layers", lay,
         "--max-rows", args.xfer_rows, "--max-tokens", args.max_tokens, "--device", "cpu", "--dtype", "float32",
         "--out", w / "acts_t"], log)
    (w / "acts/transfer").unlink(missing_ok=True) if (w / "acts/transfer").is_symlink() else None
    if not (w / "acts/transfer").exists():
        (w / "acts/transfer").symlink_to(w / "acts_t/transfer")
    run([py, "scripts/jspace_interp_analyze.py", "--root", w / "acts", "--lens", w / "lens/tiny_jacobian_lens.pt",
         "--basis-dir", w / "basis", "--tokenizer", w / "model/tokenizer.json", "--layers", lay, "--ks", "4,8,25",
         "--positions", "last,prevote", "--conditions", "original,swap,prompt_only",
         "--n-random", "5", "--n-random-lodo", "1", "--boot", "50", "--candidates", "256",
         "--out", w / "analysis"], log)
    run([py, "scripts/jspace_interp_steer.py", "--model", w / "model", "--lens", w / "lens/tiny_jacobian_lens.pt",
         "--basis-dir", w / "basis", "--layer", "3", "--condition", "swap", "--n-random", "5", "--max-episodes", "6",
         "--device", "cpu", "--dtype", "float32", "--out", w / "steer"], log)
    run([py, "scripts/jspace_interp_gen_matched_vocab.py", "--dry-run", "--ids", "med_01", "fin_01",
         "--out", w / "matched_vocab_dry"], log)
    steer = json.loads((w / "steer/steer_core_swap_L3.json").read_text())
    chk = steer["cache_check"]
    print("steer prefix cache:", steer["prefix_cache"], chk)
    if steer["prefix_cache"] and chk["rows"] and chk["max_abs_diff_y0"] > 1e-3:
        raise SystemExit(f"cached and full passes disagree on y0: {chk}")

    # 8. matched-vocabulary control end to end: warned episodes from the tiny model, extraction, analysis
    sys.path.insert(0, str(ROOT / "scripts"))
    from jspace_interp_extract import subsample

    core_ids = []
    for r in subsample(load_rows("core"), args.core_rows):
        if r["scenario_id"] not in core_ids:
            core_ids.append(r["scenario_id"])
    warned_ids = core_ids[:6]
    run([py, "scripts/jspace_interp_gen_matched_vocab.py", "--model", w / "model", "--allow-cpu", "--dtype", "float32",
         "--ids", *warned_ids, "--max-new-tokens", "8", "--workers", "3", "--batch", "2", "--out", w / "matched_vocab_tiny"], log)
    meta = json.loads((w / "matched_vocab_tiny/batch_meta.json").read_text())
    print("warned episodes:", meta["n"], "generation:", meta["generation"])
    run([py, "scripts/jspace_interp_extract.py", "--model", w / "model", "--splits", "warned", "--warned-runs", w / "matched_vocab_tiny",
         "--conditions", "original", "prompt_only", "--layers", lay, "--max-tokens", args.max_tokens, "--device", "cpu",
         "--dtype", "float32", "--out", w / "acts"], log)
    run([py, "scripts/jspace_interp_analyze.py", "--root", w / "acts", "--lens", w / "lens/tiny_jacobian_lens.pt",
         "--basis-dir", w / "basis", "--tokenizer", w / "model/tokenizer.json", "--layers", lay, "--ks", "4,25",
         "--positions", "last", "--conditions", "original,prompt_only",
         "--n-random", "4", "--n-random-lodo", "1", "--n-random-mv", "2", "--boot", "50", "--candidates", "256",
         "--out", w / "analysis_mv"], log)
    res = json.loads((w / "analysis_mv/sweep_results.json").read_text())
    cell = res["cells"][0]
    mv = cell["k25"]["J25_fill"].get("matched_vocab")
    if not mv or "auroc_colluder_vs_warned" not in mv or "episode" not in mv:
        raise SystemExit(f"matched-vocabulary readings missing from the sweep: {list(cell['k25']['J25_fill'])}")
    if "matched_vocab" not in cell["k25"]["randsub"] or not res["role_dirs"]:
        raise SystemExit("random-draw matched-vocabulary spread or role directions missing")
    e1 = [e for e in res["exp1"] if e["condition"] == "original"][0]
    if "matched_vocab_concepts" not in e1["core"]:
        raise SystemExit("matched-vocabulary concept readings missing from Experiment 1")
    print("matched-vocab readings (tiny model, meaningless):", {k: v for k, v in mv.items() if not isinstance(v, dict)})
    (w / "PIPELINE_CHECK_ONLY.txt").write_text("Random tiny model: no number here means anything about Gemma.\n")
    (w / "e2e_log.json").write_text(json.dumps(log, indent=1) + "\n")
    print("ok", w)


if __name__ == "__main__":
    main()
