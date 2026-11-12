"""Turning chat examples into training features. Pure functions on token-id lists, so they are unit-tested."""

from __future__ import annotations

IGNORE_INDEX = -100  # label value ignored by the loss


def build_example(prompt_ids: list[int], full_ids: list[int], max_length: int, eos_id: int) -> dict[str, list[int]]:
    """Mask the prompt so the loss is computed only on the assistant answer.

    The original script trained on the whole text and used EOS as the padding token; the collator then masked every
    EOS, so the model never learned where an answer ends. Here the answer always ends with EOS and EOS keeps its label.
    """
    if full_ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError("The full conversation must start with the prompt tokens")
    if not full_ids or full_ids[-1] != eos_id:
        full_ids = [*full_ids, eos_id]

    labels = [IGNORE_INDEX] * len(prompt_ids) + full_ids[len(prompt_ids) :]
    if len(full_ids) > max_length:
        # Keep the beginning of the answer and close it with EOS instead of cutting it mid-way without an ending.
        full_ids = [*full_ids[: max_length - 1], eos_id]
        labels = [*labels[: max_length - 1], eos_id]
    return {"input_ids": full_ids, "labels": labels, "attention_mask": [1] * len(full_ids)}


def pad_batch(examples: list[dict[str, list[int]]], pad_id: int) -> dict[str, list[list[int]]]:
    """Right-pad a batch to its longest example (dynamic padding instead of a fixed 512)."""
    width = max(len(e["input_ids"]) for e in examples)
    batch: dict[str, list[list[int]]] = {"input_ids": [], "labels": [], "attention_mask": []}
    for e in examples:
        gap = width - len(e["input_ids"])
        batch["input_ids"].append(e["input_ids"] + [pad_id] * gap)
        batch["labels"].append(e["labels"] + [IGNORE_INDEX] * gap)
        batch["attention_mask"].append(e["attention_mask"] + [0] * gap)
    return batch
