"""HF transformers chat backend (Gemma-2-IT). Fold system into first user turn."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

DEFAULT_MODEL = "google/gemma-2-2b-it"


@dataclass
class HFBundle:
    model: Any
    tokenizer: Any
    device: torch.device
    model_id: str


def load_model(
    model_id: str = DEFAULT_MODEL,
    dtype: str = "bfloat16",
    device_map: str | None = "auto",
) -> HFBundle:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch_dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }.get(dtype, torch.bfloat16)

    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    kwargs: dict[str, Any] = {"torch_dtype": torch_dtype, "low_cpu_mem_usage": True}
    if device_map is not None:
        kwargs["device_map"] = device_map
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    model.eval()
    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return HFBundle(model=model, tokenizer=tok, device=device, model_id=model_id)


def make_generate(
    bundle: HFBundle,
    *,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
) -> "Callable[[str, str], str]":
    def generate(system: str, user: str) -> str:
        messages = [
            {"role": "user", "content": f"{system.strip()}\n\n{user.strip()}"}
        ]
        prompt = bundle.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = bundle.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(bundle.device) for k, v in inputs.items()}
        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "pad_token_id": bundle.tokenizer.pad_token_id,
            "eos_token_id": bundle.tokenizer.eos_token_id,
        }
        if temperature and temperature > 0:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = temperature
        else:
            gen_kwargs["do_sample"] = False
        with torch.inference_mode():
            out = bundle.model.generate(**inputs, **gen_kwargs)
        new_tokens = out[0, inputs["input_ids"].shape[-1] :]
        return bundle.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    return generate
