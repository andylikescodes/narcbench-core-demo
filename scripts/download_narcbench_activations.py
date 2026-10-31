#!/usr/bin/env python3
"""Download the official NARCBench activations (the paper's own extraction) for the J-space analysis.

Dataset: https://huggingface.co/datasets/aaronrose227/narcbench (MIT). Layout per model and tier:
activations_gen.npz (keys layer_<L>, float32) + metadata_gen.json (rows aligned). Sizes: 14 to 210 MB per split.

  python3 scripts/download_narcbench_activations.py                       # qwen3_32b, deepseek_r1_32b, gpt_oss_20b; core + transfer
  python3 scripts/download_narcbench_activations.py --models qwen3_32b --tiers core transfer stego

Files land under data/activations/official/<model>/<tier>/ (gitignored). Unembeddings are fetched separately
with scripts/fetch_unembed_rows.py. Then:

  python3 scripts/jspace_separability.py --core-acts data/activations/official/qwen3_32b/core \\
      --transfer-acts data/activations/official/qwen3_32b/transfer --unembed data/j_basis/qwen3_32b_awq \\
      --layers 26-30 --out results/jspace/qwen3_32b --tag qwen3_32b
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parents[1]
REPO = "aaronrose227/narcbench"
LAYERS = {"qwen3_32b": "26-30", "deepseek_r1_32b": "26-30", "gpt_oss_20b": "10-14", "llama_3_1_70b": "32-37"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["qwen3_32b", "deepseek_r1_32b", "gpt_oss_20b"])
    ap.add_argument("--tiers", nargs="+", default=["core", "transfer"])
    ap.add_argument("--dest", type=Path, default=ROOT / "data/activations/official")
    args = ap.parse_args()
    dest = args.dest if args.dest.is_absolute() else ROOT / args.dest
    dest.mkdir(parents=True, exist_ok=True)
    patterns = [f"activations/{m}/{t}/*" for m in args.models for t in args.tiers]
    snap = Path(snapshot_download(REPO, repo_type="dataset", allow_patterns=patterns))
    for m in args.models:
        for t in args.tiers:
            src = snap / "activations" / m / t
            if not src.exists():
                print(f"  missing on the hub: {m}/{t}")
                continue
            out = dest / m / t
            out.mkdir(parents=True, exist_ok=True)
            for f in ("activations_gen.npz", "metadata_gen.json"):
                shutil.copy2(src / f, out / f)
            print(f"  {out}  (layers {LAYERS.get(m, '?')})")
    print("done; unembeddings: scripts/fetch_unembed_rows.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
