import json
from pathlib import Path

import pytest

from supportlora.data import SYSTEM_PROMPT, system_prompt, to_messages
from supportlora.rag import (
    NO_MATCH_CONTEXT,
    BM25Retriever,
    Chunk,
    build_context,
    load_policies,
    split_sections,
    tokenize,
)
from supportlora.rag_eval import contains_fact, retrieval_report

ROOT = Path(__file__).resolve().parents[1]
POLICIES = load_policies(ROOT / "policies")
QUESTIONS = json.loads((ROOT / "policies" / "eval_questions.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def retriever() -> BM25Retriever:
    return BM25Retriever(POLICIES)


def test_split_sections_creates_one_chunk_per_heading():
    markdown = "# Returns\n\nintro text\n\n## Returning an item\nWithin 30 days.\n\n## Refunds\nIn 5 days.\n"

    chunks = split_sections("returns", markdown)

    assert [c.id for c in chunks] == ["returns#returning-an-item", "returns#refunds"]
    assert chunks[0].title == "Returns — Returning an item"
    assert chunks[0].text == "Within 30 days."


def test_empty_sections_are_skipped():
    assert split_sections("x", "# X\n## Empty\n\n## Full\ntext") == [Chunk("x#full", "X — Full", "text")]


def test_tokenize_removes_stopwords_stems_and_maps_synonyms():
    assert tokenize("How do I get my money back?") == ["get", "refund", "return"]
    # Plurals are mapped too (regression: "deliveries" used to bypass the synonym list).
    assert tokenize("Deliveries to Austria") == tokenize("shipping to EU")


def test_all_policy_files_are_loaded():
    assert len(POLICIES) >= 15
    assert len({c.id for c in POLICIES}) == len(POLICIES)
    assert all(c.text for c in POLICIES)


def test_obvious_question_finds_the_right_section(retriever):
    [(best, _), *_] = retriever.search("I was charged twice")

    assert best.id == "payments#charged-twice"


def test_off_topic_question_gets_no_context(retriever):
    assert retriever.search("what's the weather like today?") == []
    assert build_context([]) == NO_MATCH_CONTEXT


def test_eval_questions_point_to_existing_sections():
    ids = {c.id for c in POLICIES}
    assert all(q["section"] in ids for q in QUESTIONS)
    assert all(q["facts"] for q in QUESTIONS)


def test_retrieval_quality_on_held_out_questions(retriever):
    """Regression guard for the retriever: measured 14/16 at top-1 and 16/16 at top-2."""
    report = retrieval_report(retriever, QUESTIONS, k=2)

    assert report["hit_at_2_pct"] == 100
    assert report["hit_at_1_pct"] >= 85


def test_context_is_added_to_the_system_prompt_only_when_present():
    context = build_context([(POLICIES[0], 3.0)])

    assert system_prompt() == SYSTEM_PROMPT
    assert POLICIES[0].text in system_prompt(context)
    # Regression: the style prompt says "use placeholders", which contradicts "take facts from the policy".
    assert "placeholder" not in system_prompt(context)
    assert to_messages("q", context=context)[0]["content"] == system_prompt(context)


@pytest.mark.parametrize(
    ("answer", "facts", "expected"),
    [
        ("You can return items within 30 days of delivery.", ["30 days"], True),
        ("Standard shipping costs €4,90.", ["4.90", "4,90"], True),
        ("Returns are possible within two weeks.", ["30 days"], False),
        ("WE ACCEPT PAYPAL", ["PayPal"], True),
    ],
)
def test_contains_fact(answer, facts, expected):
    assert contains_fact(answer, facts) is expected
