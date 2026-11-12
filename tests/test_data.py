from collections import Counter

import pandas as pd
import pytest

from supportlora.data import (
    SYSTEM_PROMPT,
    Example,
    clean,
    fix_mojibake,
    load,
    normalize,
    sample_per_intent,
    save,
    to_messages,
)


def test_fix_mojibake_repairs_cp1252_decoded_utf8():
    assert fix_mojibake("the customerâ€™s order") == "the customer’s order"
    assert fix_mojibake("refund â€“ 5 days") == "refund – 5 days"


def test_fix_mojibake_leaves_correct_text_alone():
    assert fix_mojibake("Plain text, café, customer’s") == "Plain text, café, customer’s"


def test_normalize_collapses_whitespace_but_keeps_paragraphs():
    assert normalize("  Hello\xa0  world \r\n\r\n\r\n\r\nNext   line  ") == "Hello world\n\nNext line"


def test_clean_drops_empty_rows_and_duplicate_questions():
    df = pd.DataFrame(
        {
            "instruction": ["Cancel my order", "cancel my order  ", None, "Where is my package?", "Empty answer"],
            "response": ["Sure.", "Duplicate.", "No question.", "On its way.", "   "],
            "intent": ["cancel_order", "cancel_order", "x", "track_order", "x"],
        }
    )

    assert clean(df) == [
        Example("Cancel my order", "Sure.", "cancel_order"),
        Example("Where is my package?", "On its way.", "track_order"),
    ]


@pytest.fixture
def examples() -> list[Example]:
    return [Example(f"q{intent}-{i}", f"a{intent}-{i}", f"intent_{intent}") for intent in range(5) for i in range(20)]


def test_sample_per_intent_is_balanced(examples):
    train, test = sample_per_intent(examples, train_per_intent=12, test_per_intent=3)

    assert Counter(e.intent for e in train) == {f"intent_{i}": 12 for i in range(5)}
    assert Counter(e.intent for e in test) == {f"intent_{i}": 3 for i in range(5)}


def test_train_and_test_do_not_overlap(examples):
    train, test = sample_per_intent(examples, 12, 3)

    assert not {e.question for e in train} & {e.question for e in test}


def test_first_test_rows_cover_different_intents(examples):
    _, test = sample_per_intent(examples, 12, 3)

    assert len({e.intent for e in test[:5]}) == 5


def test_sampling_is_deterministic(examples):
    assert sample_per_intent(examples, 12, 3, seed=1) == sample_per_intent(examples, 12, 3, seed=1)
    assert sample_per_intent(examples, 12, 3, seed=1) != sample_per_intent(examples, 12, 3, seed=2)


@pytest.mark.parametrize(("train", "test"), [(0, 3), (12, 0)])
def test_sampling_rejects_empty_splits(examples, train, test):
    with pytest.raises(ValueError):
        sample_per_intent(examples, train, test)


def test_save_and_load_round_trip(tmp_path, examples):
    path = tmp_path / "train.csv"

    save(examples[:10], path)

    assert load(path) == examples[:10]


def test_to_messages_uses_chat_roles():
    messages = to_messages("Cancel my order", "Sure, here is how.")

    assert [m["role"] for m in messages] == ["system", "user", "assistant"]
    assert messages[0]["content"] == SYSTEM_PROMPT
    assert to_messages("q")[-1]["role"] == "user"
