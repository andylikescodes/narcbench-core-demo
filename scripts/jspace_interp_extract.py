"""Extract residual activations for the jspace-interp experiments (GPU for Gemma-2-9B-it).

Reads the saved NARCBench transcripts, rewrites each agent's context for one of four conditions
(original, swap, nosys, prompt_only; see jspace_interp/contexts.py), and saves activations at
several read positions for each condition into  <out>/<split>/<condition>/.

Splits: core, transfer, and warned (the matched-vocabulary control episodes written by
scripts/jspace_interp_gen_matched_vocab.py; --warned-runs points at them). --export-basis writes the unembedding
basis (unembed_fp16.npy, unembed_row_norms_fp32.npy, final_norm_weight_fp32.npy) and tokenizer.json from the loaded
model, which the analysis needs.

  # real run (needs a GPU and HF_TOKEN for the gated Gemma repo)
  python scripts/jspace_interp_extract.py --model google/gemma-2-9b-it --out results/jspace_interp/acts \
      --export-basis data/j_basis/gemma2_9b
  python scripts/jspace_interp_extract.py --model google/gemma-2-9b-it --splits warned \
      --conditions original prompt_only --out results/jspace_interp/acts

  # pipeline check on a tiny local model (CPU)
  python scripts/jspace_interp_extract.py --model /path/to/tiny --max-rows 40 --max-tokens 256 --device cpu
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from jspace_interp.agents import SPLITS, load_rows, runs_dir_for  # noqa: E402
from jspace_interp.contexts import CONDITIONS, apply_condition, residual_leak_report  # noqa: E402
from jspace_interp.extract import extract_rows  # noqa: E402
from jspace_interp.positions import POSITIONS  # noqa: E402


def subsample(rows, max_rows):
    """Keep whole runs so episodes stay complete."""
    if not max_rows:
        return rows
    out, runs = [], []
    for r in rows:
        if r["run_id"] not in runs:
            runs.append(r["run_id"])
    keep = set()
    n = 0
    by_run = {}
    for r in rows:
        by_run.setdefault(r["run_id"], []).append(r)
    # alternate modes so both collusion and control runs appear
    runs.sort(key=lambda x: (0 if "collusion" in x else 1, x))
    col = [x for x in runs if "collusion" in x]
    ctl = [x for x in runs if "collusion" not in x]
    order = [x for pair in zip(col, ctl) for x in pair] + col[len(ctl):] + ctl[len(col):]
    for rid in order:
        if n >= max_rows:
            break
        keep.add(rid)
        n += len(by_run[rid])
    return [r for r in rows if r["run_id"] in keep]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="google/gemma-2-9b-it")
    ap.add_argument("--splits", nargs="+", default=["core", "transfer"], choices=list(SPLITS))
    ap.add_argument("--warned-runs", type=Path, default=None,
                    help="run directory of the matched-vocabulary episodes (default results/jspace_interp/matched_vocab_runs)")
    ap.add_argument("--export-basis", type=Path, default=None,
                    help="also write the unembedding basis + tokenizer.json for the analysis into this directory")
    ap.add_argument("--conditions", nargs="+", default=list(CONDITIONS))
    ap.add_argument("--layers", default="19,20,21,22,23")
    ap.add_argument("--positions", nargs="+", default=list(POSITIONS))
    ap.add_argument("--out", type=Path, default=ROOT / "results/jspace_interp/acts")
    ap.add_argument("--max-rows", type=int, default=0, help="smoke tests only (keeps whole runs)")
    ap.add_argument("--max-tokens", type=int, default=0, help="smoke tests only (left-truncates)")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if args.device == "cuda" and not torch.cuda.is_available():
        sys.exit("No CUDA GPU here. Use --device cpu with a tiny model for a pipeline check.")
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=getattr(torch, args.dtype),
                                                 attn_implementation="eager").to(args.device).eval()
    layers = [int(x) for x in args.layers.split(",")]
    if args.export_basis:
        from jspace_interp.jacobian import save_unembed_basis

        save_unembed_basis(model, args.export_basis)
        tok.save_pretrained(args.export_basis / "tokenizer")
        src = args.export_basis / "tokenizer" / "tokenizer.json"
        if src.exists():
            src.replace(args.export_basis / "tokenizer.json")
        print(json.dumps({"export_basis": str(args.export_basis), "vocab": int(model.get_output_embeddings().weight.shape[0])}))
    report = {}
    for split in args.splits:
        runs_dir = runs_dir_for(split, args.warned_runs)
        rows = subsample(load_rows(split, args.warned_runs), args.max_rows)
        for cond in args.conditions:
            samples = apply_condition(rows, cond, runs_dir, split)
            leak = residual_leak_report(samples)
            out = extract_rows(samples, model, tok, layers, args.out / split / cond, positions=args.positions,
                               max_tokens=args.max_tokens or None)
            report[f"{split}/{cond}"] = {"n_rows": len(samples), "leak": leak, "out": str(out)}
            print(json.dumps({f"{split}/{cond}": report[f"{split}/{cond}"]}, default=str))
    rep_p = args.out / "extract_report.json"
    old = json.loads(rep_p.read_text()) if rep_p.exists() else {}
    old.update(report)
    rep_p.write_text(json.dumps(old, indent=1, default=str) + "\n")


if __name__ == "__main__":
    main()
