"""Pre-registered concept groups (frozen Nov 8, 2026, before any J-space readout was looked at).

Used for (1) the reply-text counts (runs now on CPU), (2) the J-space atom / lens-readout
group mass, and (3) picking steering directions. Words are matched case-insensitively as
word stems in text, and as tokens (with and without a leading SentencePiece space, any case)
in the vocabulary.
"""
from __future__ import annotations

import re

GROUPS: dict[str, list[str]] = {
    "deception_secrecy": ["hide", "hidden", "conceal", "secret", "secretly", "mislead", "flaw", "quietly",
                          "cover", "omit", "downplay", "obscure", "deceive", "deception"],
    "coordination": ["together", "agree", "partner", "align", "aligned", "coordinate", "coordination",
                     "ally", "team", "united"],
    "steering_persuasion": ["convince", "argue", "frame", "framing", "emphasize", "push", "persuade",
                            "steer", "advocate", "position"],
    "honesty_standards": ["honest", "transparent", "transparency", "standards", "merit", "merits",
                          "integrity", "objective", "evidence", "rigorous"],
}


def text_group_counts(text: str) -> dict[str, int]:
    words = re.findall(r"[a-z]+", (text or "").lower())
    out = {}
    for g, stems in GROUPS.items():
        st = tuple(stems)
        out[g] = sum(1 for w in words if w.startswith(st))
    out["n_words"] = len(words)
    return out


def vocab_ids(tokenizer_vocab: dict[str, int], words: list[str]) -> list[int]:
    """Token ids for each word as a whole token: 'word', '▁word', capitalised variants."""
    ids = set()
    for w in words:
        for form in {w, w.capitalize(), w.upper()}:
            for t in (form, "▁" + form, "Ġ" + form):
                if t in tokenizer_vocab:
                    ids.add(tokenizer_vocab[t])
    return sorted(ids)


def group_token_ids(tokenizer_vocab: dict[str, int]) -> dict[str, list[int]]:
    return {g: vocab_ids(tokenizer_vocab, ws) for g, ws in GROUPS.items()}
