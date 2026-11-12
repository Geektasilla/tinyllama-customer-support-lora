import pytest

from supportlora.features import IGNORE_INDEX, build_example, pad_batch

EOS, PAD = 2, 0
PROMPT = [1, 10, 11, 12]
ANSWER = [20, 21, 22]


def test_prompt_is_masked_and_answer_is_learned():
    example = build_example(PROMPT, [*PROMPT, *ANSWER, EOS], max_length=100, eos_id=EOS)

    assert example["labels"] == [IGNORE_INDEX] * 4 + [20, 21, 22, EOS]
    assert example["attention_mask"] == [1] * 8


def test_end_of_answer_token_is_always_learned():
    """Regression: with pad == eos the old collator masked every EOS, so the model never learned to stop."""
    example = build_example(PROMPT, [*PROMPT, *ANSWER], max_length=100, eos_id=EOS)

    assert example["input_ids"][-1] == EOS
    assert example["labels"][-1] == EOS


def test_long_answer_is_truncated_but_still_ends_with_eos():
    example = build_example(PROMPT, [*PROMPT, *range(100, 200), EOS], max_length=10, eos_id=EOS)

    assert len(example["input_ids"]) == 10
    assert example["input_ids"][-1] == EOS
    assert example["labels"][-1] == EOS
    assert example["labels"][:4] == [IGNORE_INDEX] * 4


def test_prompt_must_be_a_prefix():
    with pytest.raises(ValueError):
        build_example([1, 2, 3], [1, 9, 3, 4], max_length=100, eos_id=EOS)


def test_pad_batch_pads_to_longest_and_ignores_padding_in_loss():
    short = build_example(PROMPT, [*PROMPT, 20, EOS], max_length=100, eos_id=EOS)
    long = build_example(PROMPT, [*PROMPT, *ANSWER, EOS], max_length=100, eos_id=EOS)

    batch = pad_batch([short, long], pad_id=PAD)

    assert [len(row) for row in batch["input_ids"]] == [8, 8]
    assert batch["input_ids"][0][-2:] == [PAD, PAD]
    assert batch["labels"][0][-2:] == [IGNORE_INDEX, IGNORE_INDEX]
    assert batch["attention_mask"][0] == [1] * 6 + [0, 0]
    assert PAD != EOS
