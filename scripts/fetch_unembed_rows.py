#!/usr/bin/env python3
"""Fetch only the lm_head (unembedding) tensor of a Hugging Face model, by HTTP range requests on its safetensors shard.

The J-space analysis needs W_U and nothing else from the model, so a 1.2 to 1.6 GB tensor is read out of a
3 to 9 GB shard without downloading the shard. Used for the official NARCBench models (Qwen3-32B-AWQ,
DeepSeek-R1-Distill-Qwen-32B, GPT-OSS-20B); Llama-3.1-70B is gated.

Writes the repo's J-basis layout:
  <out>/unembed_fp16.npy            [V, d] float16 (lossless for BF16 values in the fp16 range)
  <out>/unembed_row_norms_fp32.npy  [V]    float32
  <out>/unembed_meta.json

  python3 scripts/fetch_unembed_rows.py Qwen/Qwen3-32B-AWQ data/j_basis/qwen3_32b_awq
  python3 scripts/fetch_unembed_rows.py openai/gpt-oss-20b data/j_basis/gpt_oss_20b
  python3 scripts/fetch_unembed_rows.py deepseek-ai/DeepSeek-R1-Distill-Qwen-32B data/j_basis/deepseek_r1_32b

Needs `requests` and `huggingface_hub`. Resumable. Output belongs under data/j_basis/ (not committed).
"""
from __future__ import annotations

import argparse
import json
import os
import struct
import sys
import time
from pathlib import Path

import numpy as np
import requests
from huggingface_hub import hf_hub_download, hf_hub_url

CHUNK = 1 << 22  # 4 MiB


def find_shard(repo: str, tensor: str, local: Path) -> str:
    try:
        idx_p = hf_hub_download(repo, "model.safetensors.index.json", local_dir=str(local))
        wm = json.load(open(idx_p))["weight_map"]
        if tensor in wm:
            return wm[tensor]
        cands = [k for k in wm if tensor.split(".")[0] in k]
        raise SystemExit(f"{tensor} not in index; candidates {cands[:10]}")
    except Exception as e:  # single-file model
        if "index" in str(e) or "404" in str(e) or "Entry" in str(e):
            return "model.safetensors"
        raise


def auth() -> dict:
    """Bearer header from HF_TOKEN when set (gated repos such as google/gemma-2-9b-it); never printed."""
    tok = os.environ.get("HF_TOKEN")
    return {"Authorization": f"Bearer {tok}"} if tok else {}


def get_header(url: str) -> tuple[int, dict]:
    r = requests.get(url, headers={**auth(), "Range": "bytes=0-7"}, allow_redirects=True, timeout=60)
    r.raise_for_status()
    n = struct.unpack("<Q", r.content[:8])[0]
    r = requests.get(url, headers={**auth(), "Range": f"bytes=8-{8 + n - 1}"}, allow_redirects=True, timeout=120)
    r.raise_for_status()
    return n, json.loads(r.content[:n].decode("utf-8"))


def stream_range(url: str, start: int, end_inclusive: int, dest: Path) -> None:
    """Resumable download of bytes [start, end_inclusive] into dest."""
    have = dest.stat().st_size if dest.exists() else 0
    total = end_inclusive - start + 1
    if have >= total:
        print(f"  already have {have} bytes", flush=True)
        return
    headers = {**auth(), "Range": f"bytes={start + have}-{end_inclusive}"}
    t0, done = time.time(), have
    with requests.get(url, headers=headers, stream=True, allow_redirects=True, timeout=120) as r:
        r.raise_for_status()
        if r.status_code != 206:
            raise SystemExit(f"server did not honour Range (status {r.status_code})")
        with open(dest, "ab") as f:
            for chunk in r.iter_content(CHUNK):
                f.write(chunk)
                done += len(chunk)
                if done % (CHUNK * 32) < CHUNK:
                    el = time.time() - t0
                    rate = (done - have) / max(el, 1e-6) / 1e6
                    print(f"  {done/1e6:,.0f}/{total/1e6:,.0f} MB  {rate:.1f} MB/s  eta {(total-done)/max(rate,1e-6)/1e6:.0f}s", flush=True)
    if dest.stat().st_size != total:
        raise SystemExit(f"size mismatch {dest.stat().st_size} != {total}")


def convert_vector(raw: Path, dtype: str, shape: list[int], out: Path, name: str) -> None:
    """A 1-D tensor (e.g. the final norm weight model.norm.weight) -> <out>/<name>_fp32.npy."""
    (n,) = shape
    b = np.fromfile(raw, dtype=np.uint8)
    if dtype == "BF16":
        f32 = (b.view("<u2").astype(np.uint32) << 16).view(np.float32)
    elif dtype == "F16":
        f32 = b.view("<f2").astype(np.float32)
    else:
        f32 = b.view("<f4")
    assert f32.shape[0] == n, (f32.shape, n)
    np.save(out / f"{name}_fp32.npy", f32.astype(np.float32))
    print(f"  wrote {out / (name + '_fp32.npy')} {f32.shape}; mean {f32.mean():.4f} min {f32.min():.4f} max {f32.max():.4f}", flush=True)


def convert(raw: Path, dtype: str, shape: list[int], out: Path, rows_per_block: int = 8192) -> None:
    V, d = shape
    itemsize = {"BF16": 2, "F16": 2, "F32": 4}[dtype]
    mm = np.memmap(raw, dtype=np.uint8, mode="r")
    assert mm.shape[0] == V * d * itemsize, (mm.shape, V, d, itemsize)
    W = np.lib.format.open_memmap(out / "unembed_fp16.npy", mode="w+", dtype=np.float16, shape=(V, d))
    norms = np.zeros(V, dtype=np.float32)
    for r0 in range(0, V, rows_per_block):
        r1 = min(V, r0 + rows_per_block)
        b = np.asarray(mm[r0 * d * itemsize : r1 * d * itemsize])
        if dtype == "BF16":
            u16 = b.view("<u2").reshape(r1 - r0, d)
            f32 = (u16.astype(np.uint32) << 16).view(np.float32)
        elif dtype == "F16":
            f32 = b.view("<f2").reshape(r1 - r0, d).astype(np.float32)
        else:
            f32 = b.view("<f4").reshape(r1 - r0, d)
        W[r0:r1] = f32.astype(np.float16)
        norms[r0:r1] = np.linalg.norm(f32, axis=1)
    W.flush()
    np.save(out / "unembed_row_norms_fp32.npy", norms)
    print(f"  wrote {out/'unembed_fp16.npy'} {W.shape}; norms mean {norms.mean():.4f} min {norms.min():.4f} max {norms.max():.4f}; "
          f"max|w| {float(np.abs(W[:1000]).max()):.4f} (first 1000 rows)", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("repo")
    ap.add_argument("out", type=Path)
    ap.add_argument("--tensor", default="lm_head.weight",
                    help="tied-embedding models (Gemma): model.embed_tokens.weight; 1-D tensors such as model.norm.weight are "
                         "written as <out>/<name>_fp32.npy")
    ap.add_argument("--name", default=None, help="file stem for a 1-D tensor (default final_norm_weight for *norm.weight)")
    ap.add_argument("--revision", default="main")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    shard = find_shard(args.repo, args.tensor, args.out / "_index")
    url = hf_hub_url(args.repo, shard, revision=args.revision)
    print(f"[{args.repo}] tensor {args.tensor} in {shard}", flush=True)
    n, header = get_header(url)
    if args.tensor not in header:
        raise SystemExit(f"{args.tensor} not in header; keys sample {list(header)[:8]}")
    info = header[args.tensor]
    start, end = info["data_offsets"]
    abs_start, abs_end = 8 + n + start, 8 + n + end - 1
    print(f"  dtype {info['dtype']} shape {info['shape']} bytes {(end-start)/1e9:.3f} GB", flush=True)
    raw = args.out / f"_raw_{args.tensor}.bin"
    stream_range(url, abs_start, abs_end, raw)
    meta = {"repo": args.repo, "revision": args.revision, "shard": shard, "tensor": args.tensor,
            "dtype_source": info["dtype"], "shape": info["shape"], "data_offsets": [start, end],
            "fetched_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    if len(info["shape"]) == 1:
        name = args.name or ("final_norm_weight" if args.tensor.endswith("norm.weight") else args.tensor.replace(".", "_"))
        convert_vector(raw, info["dtype"], info["shape"], args.out, name)
        meta["stored_as"] = "float32"
        (args.out / f"{name}_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    else:
        convert(raw, info["dtype"], info["shape"], args.out)
        meta["stored_as"] = "float16 (lossless for BF16 values within fp16 range)"
        (args.out / "unembed_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    raw.unlink()
    print("  done", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
