"""A small exact Jacobian-lens estimator, for tiny test models only (Gemma-2-9B uses the published lens).

J_l = mean over prompts and source positions t of sum_{t' >= t} d h_final[t'] / d h_l[t]
(the jlens definition: summed over target positions t' >= t, averaged over source positions,
then over prompts; positions before ``skip`` and the last position are excluded). h_final is the
last block's output before the final norm. Saved in the jlens format
{"J": {layer: fp16 [d, d]}, "n_prompts", "source_layers", "d_model"} so that
``nov5_jspace_decompose.load_lens`` reads it.
"""
from __future__ import annotations

from pathlib import Path


def estimate_lens(model, prompts_ids: list[list[int]], layers: list[int], skip: int = 4):
    import torch

    backbone = model.model if hasattr(model, "model") else model
    mods = backbone.layers
    n_layers = len(mods)
    d = model.config.hidden_size
    acc = {L: torch.zeros(d, d, dtype=torch.float64) for L in layers}
    n = 0
    for ids in prompts_ids:
        T = len(ids)
        valid = list(range(skip, T - 1))
        if not valid:
            continue
        for L in layers:
            store = {}

            def grab(_m, _i, out, L=L):
                h = out[0] if isinstance(out, (tuple, list)) else out
                h.retain_grad()
                store["h"] = h

            def final(_m, _i, out):
                store["f"] = out[0] if isinstance(out, (tuple, list)) else out

            h1 = mods[L].register_forward_hook(grab)
            h2 = mods[n_layers - 1].register_forward_hook(final)
            try:
                with torch.enable_grad():
                    backbone(input_ids=torch.tensor([ids]), use_cache=False)
                    f = store["f"][0]  # [T, d]
                    hl = store["h"]
                    rows = []
                    for i in range(d):
                        g = torch.autograd.grad(f[valid, i].sum(), hl, retain_graph=True)[0][0]  # [T, d]
                        rows.append(g[valid].sum(0))
                    Jsum = torch.stack(rows)  # [d_out, d_in] summed over t and t' >= t
            finally:
                h1.remove()
                h2.remove()
            acc[L] += Jsum.double() / len(valid)
        n += 1
    J = {L: (acc[L] / max(n, 1)).to(torch.float16) for L in layers}
    return {"J": J, "n_prompts": n, "source_layers": layers, "d_model": d}


def save_lens(lens: dict, path: Path):
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(lens, str(path))


def save_unembed_basis(model, basis_dir: Path):
    """Write unembed_fp16.npy, unembed_row_norms_fp32.npy, final_norm_weight_fp32.npy (nov5 layout)."""
    import numpy as np

    basis_dir.mkdir(parents=True, exist_ok=True)
    W = model.get_output_embeddings().weight.detach().float().cpu().numpy()
    np.save(basis_dir / "unembed_fp16.npy", W.astype(np.float16))
    np.save(basis_dir / "unembed_row_norms_fp32.npy", np.linalg.norm(W, axis=1).astype(np.float32))
    backbone = model.model if hasattr(model, "model") else model
    np.save(basis_dir / "final_norm_weight_fp32.npy", backbone.norm.weight.detach().float().cpu().numpy())
