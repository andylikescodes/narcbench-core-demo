"""Convert the Nov 5 Gemma caches (paper layout) into the jspace-interp extraction layout.

The Nov 5 caches are last-token residuals for the original transcripts:
  <dir>/activations_gen.npz  keys layer_<L>   and   <dir>/metadata_gen.json
This writes  <out>/<split>/original/activations.npz  (keys last__layer_<L>)  and  metadata.json,
dropping all-zero rows exactly as scripts/jspace_separability.py does, so that
scripts/jspace_interp_analyze.py can run Experiments 1 and 2 on them (CPU, no GPU needed):

  python scripts/jspace_interp_convert_paper_cache.py --core <core_cache_dir> --transfer <transfer_cache_dir> \
      --out data/jspace_interp/gemma2_9b --layers 19-23
  python scripts/jspace_interp_analyze.py --root data/jspace_interp/gemma2_9b --lens <lens.pt> \
      --basis-dir <dir with unembed_fp16.npy and final_norm_weight_fp32.npy> --tokenizer <tokenizer.json> \
      --layers 19-23 --ks 8,25,50,100 --positions last --conditions original --n-random 200 \
      --out results/jspace_interp/gemma2_9b_last_original

Only the 'last' position and the 'original' condition exist in these caches; the other positions and the
swap / nosys / prompt_only conditions still need a fresh extraction (GPU).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_layers(spec: str) -> list[int]:
    out = []
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        elif part.strip():
            out.append(int(part))
    return out


def convert(src: Path, dst: Path, layers: list[int]) -> dict:
    meta = json.loads((src / "metadata_gen.json").read_text())
    npz = np.load(src / "activations_gen.npz")
    avail = sorted(int(k.split("_")[1]) for k in npz.files if k.startswith("layer_"))
    use = [L for L in layers if L in avail]
    if not use:
        raise SystemExit(f"{src}: none of {layers} in cache (has {avail})")
    n = len(meta)
    keep = np.abs(npz[f"layer_{use[0]}"][:n]).sum(axis=1) > 0
    dst.mkdir(parents=True, exist_ok=True)
    np.savez(dst / "activations.npz",
             **{f"last__layer_{L}": np.asarray(npz[f"layer_{L}"][:n][keep], dtype=np.float16) for L in use})
    out_meta = []
    for m, k in zip(meta, keep):
        if not k:
            continue
        r = dict(m)
        r["condition"] = "original"
        r["context_changed"] = False
        r.setdefault("positions", {})
        r.setdefault("fallback", {})
        r["source_cache"] = "nov5_paper_layout_last_token"
        out_meta.append(r)
    (dst / "metadata.json").write_text(json.dumps(out_meta, indent=1, default=str) + "\n")
    return {"src": str(src), "rows_in": n, "rows_kept": int(keep.sum()), "layers": use}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--core", type=Path, required=True)
    ap.add_argument("--transfer", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--layers", default="19-23")
    args = ap.parse_args()
    layers = parse_layers(args.layers)
    rep = {"core": convert(args.core, args.out / "core" / "original", layers),
           "transfer": convert(args.transfer, args.out / "transfer" / "original", layers)}
    (args.out / "convert_report.json").write_text(json.dumps(rep, indent=1) + "\n")
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
