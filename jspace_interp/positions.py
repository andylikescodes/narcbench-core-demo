"""Chat text construction and read positions inside an agent turn.

Positions (token indices into the full sequence):
  last        final token (the newline after <end_of_turn>), the position every Nov 5 number used
  eot         the <end_of_turn> token that closes the reply
  prevote     last reply token before the vote line ("RECOMMENDATION:"); Transfer and turns without
              a vote line fall back to the last reply token (flagged in ``fallback``)
  frac25/50/75  the reply token at 25%, 50%, 75% of the reply span
  reply_mean  mean over all reply tokens (returned as a span)
  prompt_end  last token of the system prompt text (used by the prompt_only condition)
"""
from __future__ import annotations

POSITIONS = ("last", "eot", "prevote", "frac25", "frac50", "frac75", "reply_mean")
VOTE_MARKERS = ("RECOMMENDATION:", "Recommendation:", "RECOMMENDATION")


def gemma_chat_text(system: str, user: str, response: str | None, tokenizer=None) -> tuple[str, dict]:
    """Gemma-2-IT layout (system folded into the user turn), with character spans of the parts.

    Uses ``tokenizer.apply_chat_template`` when available and checks it agrees with the manual layout.
    """
    content = f"{system}\n\n{user}" if system and user else (system or user)
    if response is None:
        text = f"<bos><start_of_turn>user\n{content}<end_of_turn>\n<start_of_turn>model\n"
    else:
        text = (f"<bos><start_of_turn>user\n{content}<end_of_turn>\n<start_of_turn>model\n"
                f"{response}<end_of_turn>\n")
    if tokenizer is not None and getattr(tokenizer, "chat_template", None):
        msgs = [{"role": "user", "content": content}]
        if response is not None:
            msgs.append({"role": "assistant", "content": response})
        tmpl = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=response is None)
        if tmpl != text:
            text = tmpl  # trust the model's own template; spans recomputed below
    spans = {}
    c0 = text.find(content)
    spans["content"] = (c0, c0 + len(content))
    spans["system"] = (c0, c0 + len(system)) if system else (c0, c0)
    if response is not None:
        r0 = text.rfind(response) if response else text.rfind("<start_of_turn>model\n") + len("<start_of_turn>model\n")
        spans["response"] = (r0, r0 + len(response))
    return text, spans


def token_positions(offsets: list[tuple[int, int]], spans: dict, response: str | None,
                    eot_token_indices: list[int]) -> tuple[dict, dict]:
    """Map character spans to token positions. ``offsets`` = tokenizer offset_mapping for the text."""
    n = len(offsets)
    pos, fallback = {"last": n - 1}, {}
    if "response" in spans and response is not None:
        r0, r1 = spans["response"]
        span = [i for i, (a, b) in enumerate(offsets) if b > a and a >= r0 and b <= r1 + 1]
        if not span:  # empty reply
            span = [max(0, n - 3)]
            fallback["reply_span"] = True
        after = [i for i in eot_token_indices if i > span[-1]]
        pos["eot"] = after[0] if after else n - 2
        fallback["eot"] = not after
        cut = -1
        for m in VOTE_MARKERS:
            j = response.find(m)
            if j >= 0:
                cut = r0 + j
                break
        if cut >= 0:
            pre = [i for i in span if offsets[i][1] <= cut]
            pos["prevote"] = pre[-1] if pre else span[0]
            fallback["prevote"] = not pre
        else:
            pos["prevote"] = span[-1]
            fallback["prevote"] = True
        for f in (25, 50, 75):
            pos[f"frac{f}"] = span[int(round((f / 100) * (len(span) - 1)))]
        pos["reply_mean"] = (span[0], span[-1] + 1)
    s0, s1 = spans.get("system", (0, 0))
    if s1 > s0:
        sys_toks = [i for i, (a, b) in enumerate(offsets) if b > a and a >= s0 and b <= s1 + 1]
        if sys_toks:
            pos["prompt_end"] = sys_toks[-1]
    return pos, fallback
