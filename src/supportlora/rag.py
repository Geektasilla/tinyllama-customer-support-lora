"""Retrieval over the shop policies: split Markdown into sections and rank them with BM25.

Pure Python, no model or internet needed, so retrieval quality is unit-tested in CI.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

POLICY_DIR = Path("policies")

_WORD = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset(
    "a an and are as at be by can could do does for from have how i if in is it its me my of on or our please "
    "the there this to was we what when where which will with would you your u ya pls".split()
)
# Customers rarely use the words of the policy. A small, explicit synonym list closes the most common gaps.
_SYNONYMS = {
    "reimbursement": "refund",
    "money": "refund",
    "repay": "refund",
    "parcel": "package",
    "delivery": "shipping",
    "deliver": "shipping",
    "ship": "shipping",
    "shipped": "shipping",
    "track": "tracking",
    "back": "return",
    "login": "password",
    "remove": "delete",
    "close": "delete",
    "erase": "delete",
    "bill": "invoice",
    "receipt": "invoice",
    "pay": "payment",
    "card": "payment",
    "double": "twice",
    "contact": "support",
    "reach": "support",
    "call": "support",
    "talk": "support",
    "speak": "support",
    "human": "support",
    "agent": "support",
    "much": "cost",
    "price": "cost",
    "fee": "cost",
    "expensive": "cost",
    "phone": "support",
    "hours": "open",
    "abroad": "eu",
    "austria": "eu",
    "france": "eu",
    "italy": "eu",
    "netherlands": "eu",
    "spain": "eu",
}


def _stem(word: str) -> str:
    for suffix in ("ations", "ation", "ing", "ies", "ed", "es", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[: -len(suffix)] + ("y" if suffix == "ies" else "")
    return word


def tokenize(text: str) -> list[str]:
    words = [w for w in _WORD.findall(text.lower()) if w not in _STOPWORDS]
    # Look the synonym up for the word and for its stem, so plurals ("deliveries") are mapped too.
    return [_stem(_SYNONYMS.get(w) or _SYNONYMS.get(_stem(w)) or w) for w in words]


@dataclass(frozen=True)
class Chunk:
    id: str  # "<file>#<section-slug>"
    title: str
    text: str

    def as_context(self) -> str:
        return f"[{self.title}]\n{self.text}"


def _slug(title: str) -> str:
    return "-".join(_WORD.findall(title.lower()))


def split_sections(name: str, markdown: str) -> list[Chunk]:
    """One chunk per '## ' section; the '# ' document title is prepended to each section title."""
    doc_title, chunks, title, lines = name, [], None, []
    for line in markdown.splitlines():
        if line.startswith("# "):
            doc_title = line[2:].strip()
        elif line.startswith("## "):
            if title is not None:
                chunks.append((title, lines))
            title, lines = line[3:].strip(), []
        elif title is not None:
            lines.append(line)
    if title is not None:
        chunks.append((title, lines))
    return [
        Chunk(f"{name}#{_slug(t)}", f"{doc_title} — {t}", "\n".join(body).strip())
        for t, body in chunks
        if "\n".join(body).strip()
    ]


def load_policies(directory: str | Path = POLICY_DIR) -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in sorted(Path(directory).glob("*.md")):
        if path.name.lower() != "readme.md":
            chunks += split_sections(path.stem, path.read_text(encoding="utf-8"))
    return chunks


class BM25Retriever:
    """Okapi BM25. Section titles count twice, because they summarise the section."""

    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75) -> None:
        if not chunks:
            raise ValueError("No policy chunks to search")
        self.chunks = chunks
        self.k1, self.b = k1, b
        self.docs = [Counter(tokenize(c.title) * 2 + tokenize(c.text)) for c in chunks]
        self.lengths = [sum(d.values()) for d in self.docs]
        self.avg_length = sum(self.lengths) / len(self.lengths)
        df = Counter(term for d in self.docs for term in d)
        n = len(chunks)
        self.idf = {term: math.log(1 + (n - f + 0.5) / (f + 0.5)) for term, f in df.items()}

    def scores(self, query: str) -> list[float]:
        terms = tokenize(query)
        result = []
        for doc, length in zip(self.docs, self.lengths, strict=True):
            score = 0.0
            for term in terms:
                tf = doc.get(term, 0)
                if tf:
                    norm = tf + self.k1 * (1 - self.b + self.b * length / self.avg_length)
                    score += self.idf[term] * tf * (self.k1 + 1) / norm
            result.append(score)
        return result

    def search(self, query: str, k: int = 2, min_score: float = 1.0) -> list[tuple[Chunk, float]]:
        """Top-k sections. Sections below min_score are dropped, so off-topic questions get no context."""
        ranked = sorted(zip(self.chunks, self.scores(query), strict=True), key=lambda pair: pair[1], reverse=True)
        return [(chunk, round(score, 2)) for chunk, score in ranked[:k] if score >= min_score]


NO_MATCH_CONTEXT = "[No matching policy section]\nThe shop policy does not cover this question."


def build_context(results: list[tuple[Chunk, float]]) -> str:
    """Policy text for the prompt. If nothing matched, say so explicitly instead of giving the model no signal."""
    if not results:
        return NO_MATCH_CONTEXT
    return "\n\n".join(chunk.as_context() for chunk, _ in results)
