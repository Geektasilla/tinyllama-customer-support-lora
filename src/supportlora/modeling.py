"""Model, tokenizer and dataset helpers shared by training and evaluation."""

from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase

from supportlora.data import Example, to_messages
from supportlora.features import build_example, pad_batch

BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


def load_tokenizer(name: str = BASE_MODEL) -> PreTrainedTokenizerBase:
    tokenizer = AutoTokenizer.from_pretrained(name)
    if tokenizer.pad_token is None or tokenizer.pad_token_id == tokenizer.eos_token_id:
        # A separate padding token, so padding never hides the real end-of-answer token.
        tokenizer.pad_token = tokenizer.unk_token
    return tokenizer


def load_base_model(name: str = BASE_MODEL) -> torch.nn.Module:
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype)
    return model.to("cuda" if torch.cuda.is_available() else "cpu")


def prompt_text(tokenizer: PreTrainedTokenizerBase, question: str) -> str:
    return tokenizer.apply_chat_template(to_messages(question), tokenize=False, add_generation_prompt=True)


def encode(tokenizer: PreTrainedTokenizerBase, pair: Example, max_length: int) -> dict[str, list[int]]:
    prompt = prompt_text(tokenizer, pair.question)
    full = tokenizer.apply_chat_template(to_messages(pair.question, pair.answer), tokenize=False)
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(full, add_special_tokens=False)["input_ids"]
    return build_example(prompt_ids, full_ids, max_length=max_length, eos_id=tokenizer.eos_token_id)


class Collator:
    def __init__(self, pad_id: int) -> None:
        self.pad_id = pad_id

    def __call__(self, examples: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        return {key: torch.tensor(value) for key, value in pad_batch(examples, self.pad_id).items()}
