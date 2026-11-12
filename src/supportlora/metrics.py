"""Text-similarity metric used to compare generated answers with the reference answers."""

from __future__ import annotations

import re

_WORD = re.compile(r"[a-z0-9']+")


def tokenize(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def lcs_length(a: list[str], b: list[str]) -> int:
    """Length of the longest common subsequence (O(len(a)*len(b)) time, O(len(b)) memory)."""
    previous = [0] * (len(b) + 1)
    for x in a:
        current = [0]
        for j, y in enumerate(b, start=1):
            current.append(previous[j - 1] + 1 if x == y else max(previous[j], current[j - 1]))
        previous = current
    return previous[-1]


def rouge_l(prediction: str, reference: str) -> float:
    """ROUGE-L F1 on lower-cased words: 1.0 = same word sequence, 0.0 = nothing in common."""
    pred, ref = tokenize(prediction), tokenize(reference)
    if not pred or not ref:
        return 0.0
    lcs = lcs_length(pred, ref)
    if lcs == 0:
        return 0.0
    precision, recall = lcs / len(pred), lcs / len(ref)
    return 2 * precision * recall / (precision + recall)
