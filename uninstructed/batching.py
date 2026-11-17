"""Batched generation that survives a batch too large for GPU memory.

The first pilot pod (n9i1umi8u22h4q, 17 November 2026 UTC) ran the stake versions at 16 prompts per call. That fits
the short room prompts but not the public-round prompts of about 2,000 tokens, because Gemma-2 uses eager attention,
whose memory grows with batch size times the square of the length. The fallback of that run retried inside the
except block, where the exception's traceback still holds the failed batch's tensors, so nothing was freed and every
smaller batch failed too. Here the retry happens after the except block has been left.
"""
from __future__ import annotations

import gc

import torch

from generation.hf_batched import BatchedGenerator


class SafeBatchedGenerator(BatchedGenerator):
    """Halves a batch that does not fit in GPU memory instead of failing the episodes in it."""

    n_oom = 0

    def _generate(self, prompts: list[str]) -> list[str]:
        try:
            return super()._generate(prompts)
        except torch.cuda.OutOfMemoryError:
            pass                      # fall through: the retry must not run while the traceback is alive
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if len(prompts) == 1:
            raise RuntimeError("a single prompt does not fit in GPU memory")
        type(self).n_oom += 1
        if self.log:
            self.log(f"[gen] out of memory at batch {len(prompts)}; freed and splitting in two")
        h = len(prompts) // 2
        return self._generate(prompts[:h]) + self._generate(prompts[h:])
