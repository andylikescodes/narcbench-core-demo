#!/usr/bin/env python3
"""Project activations into J-only / complement under multiple scientifically meaningful defs.

Variants (document each in results):
  wu_topk_k{K}     — Stage2 proxy: top-k unit W_U rows by ⟨Ŵ,h⟩ (J≈I logit-lens)
  jac_topk_k{K}    — true J-lens: scores ⟨W_U[i], J_ℓ h⟩; basis rows = (W_U J)_active
                     = W_active @ J  (requires J_layer_*.npy)
  wu_fixed_k{K}    — fixed global top-k tokens by mean |score| over train acts (not per-row)

  python3 scripts/project_j_variants.py \\
    --acts-dir data/activations/gemma2_9b/core/20261001T012639Z \\
    --variant wu_topk --k 25 --layers 19-23 \\
    --out-root data/activations/gemma2_9b/core/20261001T012639Z/variants/wu_topk_k25
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
# reuse helpers
sys.path.insert(0, str(ROOT / "scripts"))
from project_j_complement import topk_indices_batched, project_one, project_layer  # noqa: E402


def parse_layers(spec: str, avail: list[int]) -> list[int]:
    if "-" in spec and "," not in spec:
        a, b = spec.split("-", 1)
        layers = list(range(int(a), int(b) + 1))
    else:
        layers = [int(x) for x in spec.split(",") if x.strip()]
    return [L for L in layers if L in avail] or avail


def scores_with_J(acts: np.ndarray, W: np.ndarray, norms: np.ndarray, J: np.ndarray,
                  *, k: int, vocab_chunk: int = 8192, batch: int = 32) -> np.ndarray:
    """Top-k indices by s_i = ⟨Ŵ_i, J h⟩ / equivalent to ⟨(W J)_i unit?, h⟩.
    We use unit W rows dotted with (J h) — matches ⟨W_i, Jh⟩ / ||W_i||.
    """
    n, d = acts.shape
    # transform acts: H_j = acts @ J.T  because (J h) = h @ J.T if h is row vector
    # acts [n,d], J [d,d] maps h_col -> J @ h_col; for rows: (J @ h.T).T = h @ J.T
    Hj = (acts.astype(np.float32) @ J.T.astype(np.float32)).astype(np.float32)
    return topk_indices_batched(Hj, W, norms, k=k, vocab_chunk=vocab_chunk, batch=batch)


def basis_rows_jac(W_rows: np.ndarray, J: np.ndarray) -> np.ndarray:
    """J-lens active vectors in residual space: (W_U J) rows = W_rows @ J."""
    return (W_rows.astype(np.float32) @ J.astype(np.float32)).astype(np.float32)


def project_layer_jac(acts, W, norms, idx, J):
    n, d = acts.shape
    j_out = np.zeros((n, d), dtype=np.float32)
    c_out = np.zeros((n, d), dtype=np.float32)
    fve = []
    for i in range(n):
        rows = basis_rows_jac(W[idx[i]].astype(np.float32), J)
        hj, hc = project_one(acts[i], rows)
        j_out[i] = hj
        c_out[i] = hc
        num = float(np.dot(hj, hj))
        den = float(np.dot(acts[i], acts[i])) + 1e-12
        fve.append(num / den)
        if (i + 1) % 200 == 0 or i + 1 == n:
            print(f"  projected {i+1}/{n}", flush=True)
    stats = {
        "mean_fve_j": float(np.mean(fve)),
        "median_fve_j": float(np.median(fve)),
        "mean_j_norm": float(np.mean(np.linalg.norm(j_out, axis=1))),
        "mean_c_norm": float(np.mean(np.linalg.norm(c_out, axis=1))),
    }
    return j_out, c_out, stats


def fixed_topk_indices(acts, W, norms, k, vocab_chunk=8192, batch=64):
    """Global top-k tokens by mean absolute unit-row score across samples."""
    n, d = acts.shape
    V = W.shape[0]
    acc = np.zeros(V, dtype=np.float64)
    cnt = 0
    for i0 in range(0, n, batch):
        i1 = min(n, i0 + batch)
        H = acts[i0:i1].astype(np.float32)
        for v0 in range(0, V, vocab_chunk):
            v1 = min(V, v0 + vocab_chunk)
            Wc = W[v0:v1].astype(np.float32)
            nc = norms[v0:v1].astype(np.float32)
            Wc = Wc / np.maximum(nc[:, None], 1e-12)
            sc = H @ Wc.T
            acc[v0:v1] += np.mean(np.abs(sc), axis=0)
        cnt += 1
        print(f"  scored batch {i0}:{i1}", flush=True)
    acc /= max(cnt, 1)
    idx = np.argpartition(acc, -k)[-k:]
    idx = idx[np.argsort(acc[idx])[::-1]].astype(np.int32)
    return np.broadcast_to(idx, (n, k)).copy()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--acts-dir", type=Path, required=True)
    ap.add_argument("--basis-dir", type=Path, default=ROOT / "data/j_basis/gemma2_9b")
    ap.add_argument("--jac-dir", type=Path, default=None,
                    help="Dir with J_layer_{L}_fp16.npy for jac_topk variant")
    ap.add_argument("--variant", choices=["wu_topk", "jac_topk", "wu_fixed"], required=True)
    ap.add_argument("--k", type=int, default=25)
    ap.add_argument("--layers", default="19-23")
    ap.add_argument("--out-root", type=Path, required=True)
    ap.add_argument("--vocab-chunk", type=int, default=8192)
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args()

    acts_dir = args.acts_dir if args.acts_dir.is_absolute() else ROOT / args.acts_dir
    basis_dir = args.basis_dir if args.basis_dir.is_absolute() else ROOT / args.basis_dir
    out_root = args.out_root if args.out_root.is_absolute() else ROOT / args.out_root
    jac_dir = None
    if args.jac_dir:
        jac_dir = args.jac_dir if args.jac_dir.is_absolute() else ROOT / args.jac_dir

    meta_p = acts_dir / "metadata_gen.json"
    npz_p = acts_dir / "activations_gen.npz"
    W = np.load(basis_dir / "unembed_fp16.npy", mmap_mode="r")
    norms_p = basis_dir / "unembed_row_norms_fp32.npy"
    norms = np.load(norms_p) if norms_p.exists() else np.linalg.norm(W.astype(np.float32), axis=1).astype(np.float32)

    meta = json.loads(meta_p.read_text())
    npz = np.load(npz_p)
    avail = sorted(int(k.split("_")[1]) for k in npz.files if k.startswith("layer_"))
    layers = parse_layers(args.layers, avail)

    j_dir = out_root / "j_only"
    c_dir = out_root / "complement"
    j_dir.mkdir(parents=True, exist_ok=True)
    c_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(meta_p, j_dir / "metadata_gen.json")
    shutil.copy2(meta_p, c_dir / "metadata_gen.json")

    defs = {
        "wu_topk": (
            f"Per-activation orthogonal proj onto span of top-k={args.k} unit W_U rows "
            f"by s_i=⟨Ŵ_i,h⟩. Logit-lens / J≈I proxy for Gurnee J-lens."
        ),
        "jac_topk": (
            f"Per-activation orthogonal proj onto span of top-k={args.k} true J-lens vectors "
            f"(rows of W_U J_ℓ) selected by s_i=⟨Ŵ_i, J_ℓ h⟩. J_ℓ from randomized sketch estimate."
        ),
        "wu_fixed": (
            f"Orthogonal proj onto FIXED global top-k={args.k} unit W_U rows "
            f"(mean |score| over all acts), same basis for every sample."
        ),
    }

    j_arrays, c_arrays, layer_stats = {}, {}, {}
    for L in layers:
        key = f"layer_{L}"
        acts = npz[key][: len(meta)].astype(np.float32)
        print(f"[L{L}] variant={args.variant} k={args.k} acts={acts.shape}", flush=True)
        if args.variant == "wu_topk":
            idx = topk_indices_batched(acts, W, norms, k=args.k, vocab_chunk=args.vocab_chunk, batch=args.batch)
            j_arr, c_arr, stats = project_layer(acts, W, norms, idx)
        elif args.variant == "wu_fixed":
            idx = fixed_topk_indices(acts, W, norms, args.k, args.vocab_chunk, args.batch)
            j_arr, c_arr, stats = project_layer(acts, W, norms, idx)
        else:
            if jac_dir is None:
                print("BLOCKED: --jac-dir required for jac_topk", file=sys.stderr)
                return 2
            jp = jac_dir / f"J_layer_{L}_fp16.npy"
            if not jp.exists():
                # try nearest available layer
                cands = sorted(jac_dir.glob("J_layer_*_fp16.npy"))
                if not cands:
                    print(f"BLOCKED: no Jacobians in {jac_dir}", file=sys.stderr)
                    return 2
                # map to closest
                avail_j = [int(x.name.split("_")[2]) for x in cands]
                nearest = min(avail_j, key=lambda x: abs(x - L))
                jp = jac_dir / f"J_layer_{nearest}_fp16.npy"
                print(f"  WARN: missing J_layer_{L}; using {nearest}", flush=True)
            J = np.load(jp).astype(np.float32)
            idx = scores_with_J(acts, W, norms, J, k=args.k, vocab_chunk=args.vocab_chunk, batch=args.batch)
            j_arr, c_arr, stats = project_layer_jac(acts, W, norms, idx, J)
            stats["J_path"] = str(jp)
        np.save(j_dir / f"topk_indices_layer_{L}.npy", idx)
        j_arrays[key] = j_arr
        c_arrays[key] = c_arr
        layer_stats[key] = stats
        print(f"  FVE_J mean={stats['mean_fve_j']:.4f}", flush=True)

    np.savez_compressed(j_dir / "activations_gen.npz", **j_arrays)
    np.savez_compressed(c_dir / "activations_gen.npz", **c_arrays)
    definition = {
        "variant": args.variant,
        "k": args.k,
        "layers": layers,
        "definition": defs[args.variant],
        "layer_stats": layer_stats,
        "source_acts": str(acts_dir),
        "jac_dir": str(jac_dir) if jac_dir else None,
    }
    (out_root / "projection_meta.json").write_text(json.dumps(definition, indent=2) + "\n")
    print(json.dumps({"out_root": str(out_root), "variant": args.variant, "k": args.k}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
