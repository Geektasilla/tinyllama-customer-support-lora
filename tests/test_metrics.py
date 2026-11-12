import pytest

from supportlora.metrics import lcs_length, rouge_l, tokenize


def test_tokenize_lowercases_and_drops_punctuation():
    assert tokenize("Hello, World! It's 2026.") == ["hello", "world", "it's", "2026"]


@pytest.mark.parametrize(
    ("a", "b", "expected"),
    [("a b c d", "a c d", 3), ("a b c", "x y z", 0), ("", "a b", 0), ("a b c", "a b c", 3)],
)
def test_lcs_length(a, b, expected):
    assert lcs_length(a.split(), b.split()) == expected


def test_rouge_l_identical_texts():
    assert rouge_l("Seek help from a professional.", "seek help from a professional") == 1.0


def test_rouge_l_unrelated_texts():
    assert rouge_l("cats and dogs", "quantum physics") == 0.0


def test_rouge_l_partial_overlap():
    # LCS = "talk to a doctor" (4 words); precision 4/5, recall 4/6
    score = rouge_l("please talk to a doctor", "you should talk to a doctor")
    assert score == pytest.approx(2 * (4 / 5) * (4 / 6) / (4 / 5 + 4 / 6))


def test_rouge_l_empty_prediction():
    assert rouge_l("", "reference") == 0.0
