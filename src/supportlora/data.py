"""Loading, cleaning and splitting customer-support conversations. Pure Python + pandas, no ML dependencies."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

SYSTEM_PROMPT = (
    "You are a polite and helpful customer support assistant of an online shop. "
    "Answer the customer's request clearly and step by step. "
    "Never invent order numbers, links, phone numbers or other specific data - use placeholders instead."
)

# Typical UTF-8 text that was decoded as Windows-1252 somewhere on the way ("customer’s" -> "customerâ€™s").
_MOJIBAKE_MARKERS = ("â€", "Ã", "Â")
_SPACES = re.compile(r"[ \t]+")
_BLANK_LINES = re.compile(r"\n{3,}")


@dataclass(frozen=True)
class Example:
    question: str
    answer: str
    intent: str = ""


def fix_mojibake(text: str) -> str:
    """Repair UTF-8 text that was mis-decoded as cp1252. Leaves correct text untouched."""
    if not any(marker in text for marker in _MOJIBAKE_MARKERS):
        return text
    try:
        return text.encode("cp1252").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return text


def normalize(text: str) -> str:
    text = fix_mojibake(str(text)).replace("\r\n", "\n").replace("\xa0", " ")
    text = _SPACES.sub(" ", text)
    text = _BLANK_LINES.sub("\n\n", text)
    return "\n".join(line.strip() for line in text.split("\n")).strip()


def clean(
    df: pd.DataFrame, question_col: str = "instruction", answer_col: str = "response", intent_col: str = "intent"
) -> list[Example]:
    """Normalize text, drop empty rows and duplicate questions (case-insensitive)."""
    intents = df[intent_col] if intent_col in df else [""] * len(df)
    examples: list[Example] = []
    seen: set[str] = set()
    for question, answer, intent in zip(df[question_col], df[answer_col], intents, strict=True):
        if pd.isna(question) or pd.isna(answer):
            continue
        q, a = normalize(question), normalize(answer)
        key = q.lower()
        if not q or not a or key in seen:
            continue
        seen.add(key)
        examples.append(Example(q, a, "" if pd.isna(intent) else str(intent)))
    return examples


def sample_per_intent(
    examples: list[Example], train_per_intent: int, test_per_intent: int, seed: int = 42
) -> tuple[list[Example], list[Example]]:
    """Balanced, deterministic subset: the same number of train and test examples for every intent.

    Train and test never share a question, and every intent is represented in the test set.
    The test set is interleaved (intent A, B, C, ..., A, B, C) so its first N rows cover N different intents.
    """
    if train_per_intent < 1 or test_per_intent < 1:
        raise ValueError("Need at least one train and one test example per intent")
    frame = pd.DataFrame([asdict(e) for e in examples])
    train: list[dict] = []
    test_rounds: list[list[dict]] = [[] for _ in range(test_per_intent)]
    for _, group in frame.groupby("intent", sort=True):
        shuffled = group.sample(frac=1, random_state=seed).to_dict("records")
        for i in range(test_per_intent):
            test_rounds[i].append(shuffled[i])
        train += shuffled[test_per_intent : test_per_intent + train_per_intent]
    order = pd.Series(range(len(train))).sample(frac=1, random_state=seed).tolist()
    return [Example(**train[i]) for i in order], [Example(**row) for rnd in test_rounds for row in rnd]


def save(examples: list[Example], path: str | Path) -> None:
    pd.DataFrame([asdict(e) for e in examples]).to_csv(path, index=False, encoding="utf-8")


def load(path: str | Path) -> list[Example]:
    return clean(pd.read_csv(path, encoding="utf-8"), "question", "answer", "intent")


def to_messages(question: str, answer: str | None = None) -> list[dict[str, str]]:
    """Chat messages in the format expected by tokenizer.apply_chat_template()."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": question}]
    if answer is not None:
        messages.append({"role": "assistant", "content": answer})
    return messages
