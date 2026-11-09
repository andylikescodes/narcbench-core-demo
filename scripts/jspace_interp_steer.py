"""Run Experiment 3 (J-space steering of the final vote). GPU for Gemma-2-9B-it.

  python scripts/jspace_interp_steer.py --model google/gemma-2-9b-it --lens <jacobian_lens.pt> \
      --basis-dir <dir with unembed_fp16.npy etc.> --layer 22 --condition swap \
      [--role-direction results/jspace_interp/sweep/role_dir_L22.npy] [--hj-norm 41.2]

Directions: the four frozen concept groups (jspace_interp/concepts.py), the optional J-space role
direction from the analysis, and --n-random Haar directions. See jspace_interp/steer.py for the bars.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from jspace_interp.agents import CORE_RUNS, XFER_RUNS, load_rows  # noqa: E402
from jspace_interp.concepts import GROUPS, group_token_ids  # noqa: E402
from jspace_interp.contexts import apply_condition  # noqa: E402
from jspace_interp.nov5_jspace_decompose import load_lens, norm_scale  # noqa: E402
from jspace_interp.steer import ALPHAS, concept_directions, final_rows, run_steering, summarize  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="google/gemma-2-9b-it")
    ap.add_argument("--lens", type=Path, required=True)
    ap.add_argument("--basis-dir", type=Path, required=True)
    ap.add_argument("--split", default="core", choices=["core", "transfer"])
    ap.add_argument("--condition", default="swap", choices=["original", "swap"])
    ap.add_argument("--layer", type=int, default=22)
    ap.add_argument("--role-direction", type=Path, default=None)
    ap.add_argument("--hj-norm", type=float, default=None)
    ap.add_argument("--n-random", type=int, default=20)
    ap.add_argument("--random-kind", choices=["gaussian", "lens"], default="gaussian",
                    help="null directions: unit Gaussian vectors (pre-registered), or 'lens': unit mean J-lens atom of a random set of "
                         "vocabulary tokens, size-matched to the concept groups, which shares the lens atoms' common component")
    ap.add_argument("--out-suffix", default="", help="appended to the output file name (e.g. _lensnull)")
    ap.add_argument("--alphas", default=",".join(str(a) for a in ALPHAS))
    ap.add_argument("--max-episodes", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-cache", action="store_true", help="full forward passes instead of re-running only the tail on the cached prefix")
    ap.add_argument("--signs", choices=["role", "both"], default="role",
                    help="role: +u on honest agents and -u on colluders (what the bars use); both: both signs for every row")
    ap.add_argument("--out", type=Path, default=ROOT / "results/jspace_interp/steer")
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=getattr(torch, args.dtype),
                                                 attn_implementation="eager").to(args.device).eval()
    Js, lens_info = load_lens(args.lens, [args.layer])
    J = Js[args.layer]
    W = np.load(args.basis_dir / "unembed_fp16.npy").astype(np.float32)
    s = norm_scale(np.load(args.basis_dir / "final_norm_weight_fp32.npy"), "gemma")

    rows = load_rows(args.split)
    rows = apply_condition(rows, args.condition, CORE_RUNS if args.split == "core" else XFER_RUNS, args.split)
    fr = final_rows(rows)
    if args.max_episodes:
        keep = sorted({r["run_id"] for r in fr})[: args.max_episodes]
        fr = [r for r in fr if r["run_id"] in keep]

    vocab = tok.get_vocab()
    gids = group_token_ids(vocab)
    dirs = concept_directions(J, W, s, gids)
    if args.role_direction:
        u = np.load(args.role_direction).astype(np.float32)
        dirs["jspace_role_probe"] = u / np.linalg.norm(u)
    rng = np.random.default_rng(args.seed)
    d = J.shape[0]
    random_tokens = {}
    if args.random_kind == "lens":
        sizes = [len(v) for v in gids.values() if v]
        for i in range(args.n_random):
            ids = [int(x) for x in rng.choice(W.shape[0], size=sizes[i % len(sizes)], replace=False)]
            random_tokens[f"random:{i}"] = ids
            dirs[f"random:{i}"] = concept_directions(J, W, s, {"r": ids})["concept:r"]
    else:
        for i in range(args.n_random):
            v = rng.standard_normal(d).astype(np.float32)
            dirs[f"random:{i}"] = v / np.linalg.norm(v)

    res = run_steering(fr, model, tok, args.layer, dirs, alphas=[float(a) for a in args.alphas.split(",")],
                       scale=args.hj_norm, use_cache=not args.no_cache, signs=args.signs)
    res["summary"] = summarize(res)
    res["lens"] = lens_info
    res["config"] = {k: str(v) for k, v in vars(args).items()}
    res["concept_groups"] = {g: len(v) for g, v in gids.items()}
    res["random_kind"] = args.random_kind
    if random_tokens:
        res["random_tokens"] = random_tokens
        unit = {k: v / np.linalg.norm(v) for k, v in dirs.items()}
        res["random_cosine_with_concepts"] = {r: {c: float(unit[r] @ unit[c]) for c in dirs if c.startswith("concept:")} for r in random_tokens}
    args.out.mkdir(parents=True, exist_ok=True)
    out = args.out / f"steer_{args.split}_{args.condition}_L{args.layer}{args.out_suffix}.json"
    out.write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res["summary"]["passes_all_bars"], indent=1))
    print("prefix cache:", res["prefix_cache"], "check:", res["cache_check"], "rows:", len(res["records"]))
    print("wrote", out)


if __name__ == "__main__":
    main()
