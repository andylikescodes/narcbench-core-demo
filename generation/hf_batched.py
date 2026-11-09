"""Thread-safe batched generation on one HF model: concurrent generate(system, user) calls are grouped into batches.

The Core protocol is sequential inside an episode but independent across episodes, so running several episodes in
threads and batching their pending prompts (left padding, one model.generate per batch) raises GPU throughput several
times over one prompt at a time. Decoding settings are those of generation.hf_backend.make_generate (sampling at the
given temperature, max_new_tokens, the tokenizer's eos). Every prompt is tokenized with the same chat layout as the
single-prompt backend.
"""
from __future__ import annotations

import threading
import time
from typing import Any

import torch


class BatchedGenerator:
    def __init__(self, bundle, *, max_new_tokens: int = 256, temperature: float = 0.7, max_batch: int = 8,
                 wait_s: float = 0.05, log=None):
        self.bundle = bundle
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.max_batch = max_batch
        self.wait_s = wait_s
        self.log = log
        self._lock = threading.Lock()
        self._cv = threading.Condition(self._lock)
        self._queue: list[dict] = []
        self._stop = False
        self.n_batches = 0
        self.n_prompts = 0
        self.gen_seconds = 0.0
        self._worker = threading.Thread(target=self._loop, daemon=True)
        self._worker.start()

    def _prompt(self, system: str, user: str) -> str:
        messages = [{"role": "user", "content": f"{system.strip()}\n\n{user.strip()}"}]
        return self.bundle.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def __call__(self, system: str, user: str) -> str:
        req = {"prompt": self._prompt(system, user), "done": threading.Event(), "text": None, "error": None}
        with self._cv:
            self._queue.append(req)
            self._cv.notify()
        req["done"].wait()
        if req["error"] is not None:
            raise req["error"]
        return req["text"]

    def close(self):
        with self._cv:
            self._stop = True
            self._cv.notify_all()

    def _loop(self):
        while True:
            with self._cv:
                while not self._queue and not self._stop:
                    self._cv.wait()
                if self._stop and not self._queue:
                    return
                # let other threads' prompts arrive for a moment, then take up to max_batch
                deadline = time.time() + self.wait_s
                while len(self._queue) < self.max_batch and time.time() < deadline:
                    self._cv.wait(timeout=max(deadline - time.time(), 0.001))
                batch = self._queue[: self.max_batch]
                del self._queue[: self.max_batch]
            try:
                t0 = time.time()
                texts = self._generate([b["prompt"] for b in batch])
                self.gen_seconds += time.time() - t0
                self.n_batches += 1
                self.n_prompts += len(batch)
                if self.log and (self.n_batches % 10 == 0 or self.n_batches == 1):
                    self.log(f"[gen] batches {self.n_batches} prompts {self.n_prompts} "
                             f"({self.gen_seconds / max(self.n_prompts, 1):.1f}s per prompt amortised)")
                for b, t in zip(batch, texts):
                    b["text"] = t
            except Exception as e:  # noqa: BLE001
                for b in batch:
                    b["error"] = e
            finally:
                for b in batch:
                    b["done"].set()

    def _generate(self, prompts: list[str]) -> list[str]:
        tok = self.bundle.tokenizer
        side = tok.padding_side
        tok.padding_side = "left"
        try:
            inputs = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False)
        finally:
            tok.padding_side = side
        inputs = {k: v.to(self.bundle.device) for k, v in inputs.items()}
        gen_kwargs: dict[str, Any] = {"max_new_tokens": self.max_new_tokens, "pad_token_id": tok.pad_token_id,
                                      "eos_token_id": tok.eos_token_id}
        if self.temperature and self.temperature > 0:
            gen_kwargs.update(do_sample=True, temperature=self.temperature)
        else:
            gen_kwargs["do_sample"] = False
        with torch.inference_mode():
            out = self.bundle.model.generate(**inputs, **gen_kwargs)
        n_in = inputs["input_ids"].shape[1]
        return [tok.decode(row[n_in:], skip_special_tokens=True).strip() for row in out]
