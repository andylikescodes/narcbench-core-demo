#!/usr/bin/env python3
"""Project residual activations into J-space vs complement (CPU).

Definition used (documented also in results/stage2/J_SPACE_DEFINITION.md):

  J-lens dictionary = rows of tied unembedding W_U for google/gemma-2-9b-it
  (model.embed_tokens.weight). This is the logit-lens / late-path approximation
  to Gurnee–Lindsey J-lens vectors (arXiv:2607.15495) when the layer→logit
  Jacobian is dominated by the final unembed map. Full per-layer Jacobians
  are NOT estimated here (would need GPU forwards); mid-layer band 19–23 uses
  this fixed W_U frame.

  For each residual h ∈ R^d:
    1. Score unit unembed rows: s_i = ⟨W_i / ||W_i||, h⟩
    2. Take top-k indices by s_i (default k=25, Gurnee elbow)
    3. Form orthonormal basis Q for those k rows (QR on R^{d×k})
    4. h_J = Q Qᵀ h   (orthogonal projection onto span of active J-lens vectors)
    5. h_C = h − h_J  (complement; Yoo & Skapars arXiv:2609.02893 style)

  Outputs ambient R^d tensors so Stage-1 probe protocol is unchanged.

  python3 scripts/project_j_complement.py \\
    --acts-dir data/activations/gemma2_9b/core/20261001T012639Z \\
    --basis-dir data/j_basis/gemma2_9b \\
    --out-root data/activations/gemma2_9b/core/20261001T012639Z \\
    --k 25
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def topk_indices_batched(
    acts: np.ndarray,
    W: np.ndarray,
    norms: np.ndarray,
    *,
    k: int,
    vocab_chunk: int = 8192,
    batch: int = 64,
) -> np.ndarray:
    """Return [n, k] indices of top-k unit-row scores s_i = ⟨Ŵ_i, h⟩."""
    n, d = acts.shape
    V = W.shape[0]
    assert W.shape[1] == d
    out = np.zeros((n, k), dtype=np.int32)
    for i0 in range(0, n, batch):
        i1 = min(n, i0 + batch)
        H = acts[i0:i1].astype(np.float32, copy=False)  # [b, d]
        b = H.shape[0]
        best_val = np.full((b, k), -np.inf, dtype=np.float32)
        best_idx = np.zeros((b, k), dtype=np.int32)
        for v0 in range(0, V, vocab_chunk):
            v1 = min(V, v0 + vocab_chunk)
            Wc = W[v0:v1].astype(np.float32)  # [c, d]
            nc = norms[v0:v1].astype(np.float32)
            # unit rows
            Wc = Wc / np.maximum(nc[:, None], 1e-12)
            sc = H @ Wc.T  # [b, c]
            # merge chunk into running top-k
            if v1 - v0 >= k:
                part_idx = np.argpartition(sc, -k, axis=1)[:, -k:]
                part_val = np.take_along_axis(sc, part_idx, axis=1)
            else:
                part_idx = np.broadcast_to(
                    np.arange(v1 - v0, dtype=np.int32), (b, v1 - v0)
                ).copy()
                part_val = sc
            part_idx = part_idx + v0
            cand_val = np.concatenate([best_val, part_val], axis=1)
            cand_idx = np.concatenate([best_idx, part_idx], axis=1)
            keep = np.argpartition(cand_val, -k, axis=1)[:, -k:]
            best_val = np.take_along_axis(cand_val, keep, axis=1)
            best_idx = np.take_along_axis(cand_idx, keep, axis=1)
        # sort each row descending
        order = np.argsort(best_val, axis=1)[:, ::-1]
        out[i0:i1] = np.take_along_axis(best_idx, order, axis=1)
        print(f"  scored samples {i0}:{i1}/{n}", flush=True)
    return out


def project_one(h: np.ndarray, basis_rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Orthogonal proj of h onto span of basis_rows [k,d]."""
    # QR on columns = rows^T
    B = basis_rows.astype(np.float64)
    # drop near-zero / dependent rows via QR with column pivoting on B.T
    Q, R = np.linalg.qr(B.T, mode="reduced")  # Q: [d, k]
    # drop columns with tiny R diagonal
    diag = np.abs(np.diag(R)) if R.ndim == 2 else np.array([abs(R)])
    rank = int(np.sum(diag > 1e-8))
    if rank == 0:
        z = np.zeros_like(h, dtype=np.float32)
        return z, h.astype(np.float32)
    Q = Q[:, :rank]
    hj = (Q @ (Q.T @ h.astype(np.float64))).astype(np.float32)
    hc = (h.astype(np.float32) - hj).astype(np.float32)
    return hj, hc


def project_layer(
    acts: np.ndarray,
    W: np.ndarray,
    norms: np.ndarray,
    idx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict]:
    n, d = acts.shape
    k = idx.shape[1]
    j_out = np.zeros((n, d), dtype=np.float32)
    c_out = np.zeros((n, d), dtype=np.float32)
    fve = []
    for i in range(n):
        rows = W[idx[i]].astype(np.float32)
        hj, hc = project_one(acts[i], rows)
        j_out[i] = hj
        c_out[i] = hc
        num = float(np.dot(hj, hj))
        den = float(np.dot(acts[i], acts[i])) + 1e-12
        fve.append(num / den)
        if (i + 1) % 200 == 0 or i + 1 == n:
            print(f"  projected {i+1}/{n}", flush=True)
    stats = {
        "k": k,
        "mean_fve_j": float(np.mean(fve)),
        "median_fve_j": float(np.median(fve)),
        "mean_resid_norm": float(np.mean(np.linalg.norm(acts, axis=1))),
        "mean_j_norm": float(np.mean(np.linalg.norm(j_out, axis=1))),
        "mean_c_norm": float(np.mean(np.linalg.norm(c_out, axis=1))),
    }
    return j_out, c_out, stats


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--acts-dir", type=Path, required=True)
    ap.add_argument("--basis-dir", type=Path, default=ROOT / "data/j_basis/gemma2_9b")
    ap.add_argument("--out-root", type=Path, default=None,
                    help="Parent for j_only/ and complement/ dirs (default: sibling of acts-dir)")
    ap.add_argument("--k", type=int, default=25)
    ap.add_argument("--layers", default="19-23")
    ap.add_argument("--vocab-chunk", type=int, default=8192)
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args()

    acts_dir = args.acts_dir if args.acts_dir.is_absolute() else ROOT / args.acts_dir
    basis_dir = args.basis_dir if args.basis_dir.is_absolute() else ROOT / args.basis_dir
    out_root = args.out_root
    if out_root is None:
        out_root = acts_dir
    else:
        out_root = out_root if out_root.is_absolute() else ROOT / out_root

    meta_p = acts_dir / "metadata_gen.json"
    npz_p = acts_dir / "activations_gen.npz"
    if not meta_p.exists() or not npz_p.exists():
        print(f"BLOCKED: missing {meta_p} or {npz_p}", file=sys.stderr)
        return 2

    W_path = basis_dir / "unembed_fp16.npy"
    N_path = basis_dir / "unembed_row_norms_fp32.npy"
    if not W_path.exists():
        print(f"BLOCKED: missing J dictionary {W_path}", file=sys.stderr)
        return 2
    print(f"[load] W from {W_path}", flush=True)
    W = np.load(W_path, mmap_mode="r")  # float16 [V,d]
    if N_path.exists():
        norms = np.load(N_path)
    else:
        print("[norms] computing row norms...", flush=True)
        norms = np.linalg.norm(W.astype(np.float32), axis=1).astype(np.float32)
        np.save(N_path, norms)

    meta = json.loads(meta_p.read_text())
    npz = np.load(npz_p)
    if "-" in args.layers and "," not in args.layers:
        a, b = args.layers.split("-", 1)
        layers = list(range(int(a), int(b) + 1))
    else:
        layers = [int(x) for x in args.layers.split(",") if x.strip()]
    avail = sorted(int(k.split("_")[1]) for k in npz.files if k.startswith("layer_"))
    layers = [L for L in layers if L in avail] or avail

    j_dir = out_root / "j_only"
    c_dir = out_root / "complement"
    j_dir.mkdir(parents=True, exist_ok=True)
    c_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(meta_p, j_dir / "metadata_gen.json")
    shutil.copy2(meta_p, c_dir / "metadata_gen.json")

    j_arrays = {}
    c_arrays = {}
    layer_stats = {}
    for L in layers:
        key = f"layer_{L}"
        acts = npz[key][: len(meta)].astype(np.float32)
        print(f"[layer {L}] acts {acts.shape}; selecting top-{args.k}", flush=True)
        idx = topk_indices_batched(
            acts, W, norms, k=args.k, vocab_chunk=args.vocab_chunk, batch=args.batch
        )
        np.save(j_dir / f"topk_indices_layer_{L}.npy", idx)
        print(f"[layer {L}] projecting", flush=True)
        j_arr, c_arr, stats = project_layer(acts, W, norms, idx)
        j_arrays[key] = j_arr
        c_arrays[key] = c_arr
        layer_stats[key] = stats
        print(f"[layer {L}] FVE_J mean={stats['mean_fve_j']:.4f} median={stats['median_fve_j']:.4f}", flush=True)

    np.savez_compressed(j_dir / "activations_gen.npz", **j_arrays)
    np.savez_compressed(c_dir / "activations_gen.npz", **c_arrays)

    definition = {
        "j_space_definition": (
            "Per-activation orthogonal projection onto the span of the top-k=25 "
            "unit unembedding rows (tied W_U = embed_tokens for gemma-2-9b-it), "
            "ranked by s_i = ⟨Ŵ_i, h⟩. Complement = residual orthogonal to that span. "
            "W_U is a logit-lens proxy for Gurnee/Lindsey J-lens vectors "
            "(arXiv:2607.15495); subspace-vs-complement split follows Yoo & Skapars "
            "(arXiv:2609.02893). NOT full layer Jacobians; NOT nonnegative cone NMF "
            "(linear span of active frame for probe-tractable orthogonal complement)."
        ),
        "k": args.k,
        "basis_dir": str(basis_dir),
        "source_acts": str(acts_dir),
        "layers": layers,
        "layer_stats": layer_stats,
        "gpu_needed": False,
        "notes": [
            "Full unembed span is ~full R^d; sparsity k≪d is essential (Gurnee).",
            "Nonnegative cone projection omitted; QR span projection gives h_J ⊥ h_C.",
        ],
    }
    (j_dir / "projection_meta.json").write_text(json.dumps(definition, indent=2) + "\n")
    (c_dir / "projection_meta.json").write_text(json.dumps(definition, indent=2) + "\n")
    (out_root / "projection_meta.json").write_text(json.dumps(definition, indent=2) + "\n")
    print(json.dumps({"j_dir": str(j_dir), "c_dir": str(c_dir), "layers": layers}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
