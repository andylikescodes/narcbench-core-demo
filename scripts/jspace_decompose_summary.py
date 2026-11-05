#!/usr/bin/env python3
"""Summarise one or more jspace_decompose.py runs (the paper-faithful J-space split) for the run log and the paper.

Reads <variant dir>/decomp_tokens.npz, decomp_meta.json and j_only/metadata_gen.json and writes decomp_summary.json +
decomp_summary.md with, per split and layer: FVE_J (mean, median, quartiles), the shuffled-control FVE, pursuit steps and
filled atoms, the share of chosen atoms among the readout's top-25, FVE_J by (mode, phase, role), and the most frequently
chosen tokens (decoded when a tokenizer.json is given) with their mean coefficient and the share of activations choosing them.

    python3 scripts/jspace_decompose_summary.py \
        --variant core=data/activations/gemma2_9b/core/20261001T012639Z/variants/jpaper_nnomp_k25 \
        --variant transfer=data/activations/gemma2_9b/transfer/RUNPOD/variants/jpaper_nnomp_k25 \
        --tokenizer <hf cache>/tokenizer.json --out results/jspace/gemma2_9b_jpaper_nnomp_k25
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def decoder(tokenizer_path: Path | None):
    if tokenizer_path is None:
        return lambda i: str(i)
    try:
        from tokenizers import Tokenizer

        tok = Tokenizer.from_file(str(tokenizer_path))
        return lambda i: tok.decode([int(i)], skip_special_tokens=False)
    except Exception as e:  # noqa: BLE001
        print(f"tokenizer not usable ({type(e).__name__}: {e}); token ids only", file=sys.stderr)
        return lambda i: str(i)


def q(x: np.ndarray) -> dict:
    return {"mean": float(np.mean(x)), "median": float(np.median(x)), "q25": float(np.percentile(x, 25)),
            "q75": float(np.percentile(x, 75)), "min": float(np.min(x)), "max": float(np.max(x))}


def summarise(label: str, vdir: Path, dec, top_n: int) -> dict:
    meta = json.loads((vdir / "decomp_meta.json").read_text())
    rows = json.loads((vdir / "j_only" / "metadata_gen.json").read_text())
    tk = np.load(vdir / "decomp_tokens.npz")
    layers = meta["settings"]["layers"]
    k = meta["settings"]["k"]
    out = {"label": label, "dir": str(vdir), "settings": meta["settings"], "lens": meta.get("lens"), "n_rows": len(rows), "layers": {}}
    base = None
    try:
        base = np.load(vdir.parents[1] / "activations_gen.npz") if (vdir.parents[1] / "activations_gen.npz").exists() else None
    except Exception:  # noqa: BLE001
        base = None
    sh = np.load(vdir / "shuffled" / "activations_gen.npz") if (vdir / "shuffled" / "activations_gen.npz").exists() else None
    groups = [(str(m.get("mode", "?")), str(m.get("phase", "?")), str(m.get("role", "?"))) for m in rows]
    for L in layers:
        toks = tk[f"tokens_L{L}"]; coefs = tk[f"coefs_L{L}"]; steps = tk[f"steps_L{L}"]; filled = tk[f"filled_L{L}"]
        fve = tk[f"fve_j_L{L}"]; read = tk[f"readout_top_L{L}"]
        n = toks.shape[0]
        n_tok = steps + filled
        entry = {"fve_j": q(fve), "steps": q(steps.astype(np.float64)), "frac_early_stop": float((steps < k).mean()),
                 "filled": q(filled.astype(np.float64)), "frac_fve_j_below_0_10": float((fve < 0.10).mean()),
                 "stats_from_run": meta["layers"][str(L)] if "layers" in meta else None}
        overlaps = [len(set(toks[i, : n_tok[i]].tolist()) & set(read[i].tolist())) / max(int(n_tok[i]), 1) for i in range(n)]
        entry["frac_chosen_in_readout_top25"] = float(np.mean(overlaps))
        if sh is not None and base is not None:
            H = base[f"layer_{L}"][:n]; S = sh[f"layer_{L}"][:n]
            entry["fve_shuffled"] = q(np.einsum("ij,ij->i", S, S) / (np.einsum("ij,ij->i", H, H) + 1e-12))
        by = defaultdict(list)
        for g, f in zip(groups, fve):
            by[g].append(float(f))
        entry["fve_j_by_mode_phase_role"] = {" / ".join(g): {"n": len(v), "mean": float(np.mean(v)), "median": float(np.median(v))}
                                              for g, v in sorted(by.items())}
        cnt: Counter = Counter(); csum: defaultdict = defaultdict(float); first: Counter = Counter()
        for i in range(n):
            for j in range(int(n_tok[i])):
                t = int(toks[i, j]); cnt[t] += 1; csum[t] += float(coefs[i, j])
            if n_tok[i] > 0:
                first[int(toks[i, 0])] += 1
        entry["top_tokens"] = [{"id": t, "text": dec(t), "share_of_activations": c / n, "mean_coef": csum[t] / c} for t, c in cnt.most_common(top_n)]
        entry["top_first_tokens"] = [{"id": t, "text": dec(t), "share": c / n} for t, c in first.most_common(10)]
        rc: Counter = Counter(int(t) for t in read[:, 0])
        entry["top_readout_first"] = [{"id": t, "text": dec(t), "share": c / n} for t, c in rc.most_common(10)]
        out["layers"][str(L)] = entry
    return out


def md(summaries: list[dict], out_path: Path) -> None:
    L = ["# J-space decomposition summary (paper-faithful split)", "",
         f"Written {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())} by `scripts/jspace_decompose_summary.py`.", ""]
    for s in summaries:
        st = s["settings"]
        L += [f"## {s['label']} — {s['n_rows']} activations, k = {st['k']}, method {st['method']}, select {st['select']}, "
              f"candidates {st['candidates']}, fill {st['fill']}", "", f"`{s['dir']}`", "",
              "| layer | FVE_J mean | median | q25–q75 | share FVE_J < 0.10 | shuffled FVE mean | steps mean | early stop | filled mean | chosen ∩ readout top-25 |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for Ls, e in s["layers"].items():
            f = e["fve_j"]; shf = e.get("fve_shuffled", {}).get("mean")
            L.append(f"| {Ls} | {f['mean']:.4f} | {f['median']:.4f} | {f['q25']:.4f}–{f['q75']:.4f} | {e['frac_fve_j_below_0_10']:.2f} | "
                     f"{'—' if shf is None else f'{shf:.4f}'} | {e['steps']['mean']:.1f} | {e['frac_early_stop']:.2f} | {e['filled']['mean']:.1f} | "
                     f"{e['frac_chosen_in_readout_top25']:.2f} |")
        L.append("")
        for Ls, e in s["layers"].items():
            L += [f"### {s['label']} layer {Ls}: FVE_J by mode / phase / role", "", "| group | n | mean | median |", "|---|---|---|---|"]
            for g, v in e["fve_j_by_mode_phase_role"].items():
                L.append(f"| {g} | {v['n']} | {v['mean']:.4f} | {v['median']:.4f} |")
            L += ["", f"Most chosen atoms (share of activations choosing the token; mean nonnegative coefficient):", ""]
            L.append(", ".join(f"`{t['text']!r}` {t['share_of_activations']:.2f} ({t['mean_coef']:.1f})" for t in e["top_tokens"][:20]))
            L += ["", "First atom chosen: " + ", ".join(f"`{t['text']!r}` {t['share']:.2f}" for t in e["top_first_tokens"][:8]),
                  "", "Readout top-1 token: " + ", ".join(f"`{t['text']!r}` {t['share']:.2f}" for t in e["top_readout_first"][:8]), ""]
    out_path.write_text("\n".join(L) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", action="append", required=True, metavar="LABEL=DIR", help="a jspace_decompose.py --out-root (repeatable)")
    ap.add_argument("--tokenizer", type=Path, default=None, help="tokenizer.json for decoding token ids")
    ap.add_argument("--top", type=int, default=40)
    ap.add_argument("--out", type=Path, required=True, help="directory for decomp_summary.{json,md}")
    a = ap.parse_args()
    dec = decoder(a.tokenizer)
    summaries = []
    for spec in a.variant:
        label, _, d = spec.partition("=")
        vdir = Path(d) if Path(d).is_absolute() else ROOT / d
        summaries.append(summarise(label, vdir, dec, a.top))
    out = a.out if a.out.is_absolute() else ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "decomp_summary.json").write_text(json.dumps(summaries, indent=1) + "\n")
    md(summaries, out / "decomp_summary.md")
    for s in summaries:
        for Ls, e in s["layers"].items():
            print(f"[{s['label']} L{Ls}] FVE_J mean {e['fve_j']['mean']:.4f} median {e['fve_j']['median']:.4f}; shuffled "
                  f"{e.get('fve_shuffled', {}).get('mean', float('nan')):.4f}; early stop {e['frac_early_stop']:.2f}; "
                  f"top atoms {[t['text'] for t in e['top_tokens'][:6]]}")
    print(f"wrote {out/'decomp_summary.json'} and {out/'decomp_summary.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
