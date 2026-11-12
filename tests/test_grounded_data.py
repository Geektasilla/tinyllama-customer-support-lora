"""Quality gates for the policy-grounded training data (train_grounded.json)."""

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRAIN = json.loads((ROOT / "policies" / "train_grounded.json").read_text(encoding="utf-8"))
EVAL = json.loads((ROOT / "policies" / "eval_questions.json").read_text(encoding="utf-8"))
POLICY_TEXT = "\n".join(p.read_text(encoding="utf-8") for p in (ROOT / "policies").glob("*.md"))

_NUMBER = re.compile(r"\d+(?:[.:,]\d+)?")


def normalize(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


def test_examples_are_complete():
    assert len(TRAIN) >= 60
    assert all(item["question"].strip() and item["answer"].strip() for item in TRAIN)


def test_no_evaluation_question_leaks_into_training():
    eval_questions = {normalize(q["question"]) for q in EVAL}
    assert not [t["question"] for t in TRAIN if normalize(t["question"]) in eval_questions]


def test_training_questions_are_unique():
    questions = [normalize(t["question"]) for t in TRAIN]
    assert len(questions) == len(set(questions))


def test_every_number_in_an_answer_exists_in_the_policy():
    """Reference answers must not invent facts: every time, price or duration comes from policies/*.md
    or from the customer's own question (e.g. "I got my order 40 days ago")."""
    invented = {
        (item["question"], number)
        for item in TRAIN
        for number in _NUMBER.findall(item["answer"])
        if number not in POLICY_TEXT and number not in item["question"]
    }
    assert invented == set()


def test_answers_use_real_facts_not_placeholders():
    assert not [t["question"] for t in TRAIN if "{{" in t["answer"]]


def test_out_of_scope_questions_are_forwarded_not_answered():
    forwarded = [t for t in TRAIN if "forward your question" in t["answer"]]
    assert len(forwarded) >= 5
