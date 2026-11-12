"""Compare the base model with the fine-tuned adapter on the held-out test questions.

python -m supportlora.evaluate --adapter adapter
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean

import torch
from peft import PeftModel

from supportlora.data import Example, load
from supportlora.metrics import rouge_l, tokenize
from supportlora.modeling import Collator, encode, load_base_model, load_tokenizer, prompt_text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--test", default="data/test.csv")
    parser.add_argument("--adapter", default="adapter")
    parser.add_argument("--samples", type=int, default=8, help="test questions to generate answers for")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


@torch.no_grad()
def answer_loss(model, tokenizer, pairs: list[Example], max_length: int) -> float:
    """Mean cross-entropy on the answer tokens only."""
    collate = Collator(tokenizer.pad_token_id)
    losses = []
    for pair in pairs:
        batch = {k: v.to(model.device) for k, v in collate([encode(tokenizer, pair, max_length)]).items()}
        losses.append(model(**batch).loss.item())
    return mean(losses)


@torch.no_grad()
def generate(model, tokenizer, question: str, max_new_tokens: int) -> tuple[str, bool]:
    """Greedy answer. Returns the text and whether the model ended the answer by itself (EOS)."""
    inputs = tokenizer(prompt_text(tokenizer, question), return_tensors="pt", add_special_tokens=False).to(model.device)
    output = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        repetition_penalty=1.1,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )[0][inputs["input_ids"].shape[1] :]
    finished = tokenizer.eos_token_id in output.tolist()
    return tokenizer.decode(output, skip_special_tokens=True).strip(), finished


def evaluate(model, tokenizer, test: list[Example], args: argparse.Namespace) -> tuple[dict, list[str]]:
    loss = answer_loss(model, tokenizer, test, args.max_length)
    answers = [generate(model, tokenizer, p.question, args.max_new_tokens) for p in test[: args.samples]]
    metrics = {
        "answer_loss": round(loss, 3),
        "perplexity": round(math.exp(loss), 2),
        "rouge_l": round(mean(rouge_l(a, p.answer) for (a, _), p in zip(answers, test, strict=False)), 3),
        "finished_answers_pct": round(100 * mean(done for _, done in answers)),
        "avg_answer_words": round(mean(len(tokenize(a)) for a, _ in answers)),
        # The dataset uses placeholders like {{Order Number}} instead of real data; a model that adopted the style
        # writes placeholders instead of inventing order numbers, links or phone numbers.
        "uses_placeholders_pct": round(100 * mean("{{" in a for a, _ in answers)),
    }
    return metrics, [a for a, _ in answers]


def main() -> None:
    args = parse_args()
    test = load(args.test)
    tokenizer = load_tokenizer()

    model = load_base_model()
    model.eval()
    print("Evaluating the base model...")
    base_metrics, base_answers = evaluate(model, tokenizer, test, args)

    model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()
    print("Evaluating the fine-tuned model...")
    tuned_metrics, tuned_answers = evaluate(model, tokenizer, test, args)

    Path("results").mkdir(exist_ok=True)
    report = {
        "test_questions": len(test),
        "generated_samples": args.samples,
        "base": base_metrics,
        "fine_tuned": tuned_metrics,
    }
    Path("results/metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    lines = ["# Sample answers on held-out test questions", ""]
    for pair, base, tuned in zip(test, base_answers, tuned_answers, strict=False):
        lines += [f"## {pair.question}", "", f"*Intent: `{pair.intent}`*", "", "**Base TinyLlama:**", "", base, ""]
        lines += ["**Fine-tuned (LoRA):**", "", tuned, ""]
        lines += ["**Reference answer from the dataset:**", "", pair.answer, "", "---", ""]
    Path("results/samples.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
