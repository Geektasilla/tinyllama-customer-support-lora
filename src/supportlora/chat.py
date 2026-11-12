"""Talk to the model in the terminal.

python -m supportlora.chat               # fine-tuned model (needs ./adapter)
python -m supportlora.chat --base        # original TinyLlama, for comparison
"""

from __future__ import annotations

import argparse
from pathlib import Path

from supportlora.evaluate import generate
from supportlora.modeling import load_base_model, load_tokenizer

EXAMPLES = [
    "I want to cancel my order",
    "how can i get a refund??",
    "where is my package",
    "I forgot my password",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--adapter", default="adapter")
    parser.add_argument("--base", action="store_true", help="use the original model without the adapter")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    args = parser.parse_args()

    tokenizer = load_tokenizer()
    model = load_base_model()
    if not args.base:
        if not Path(args.adapter).exists():
            raise SystemExit(f"No adapter in '{args.adapter}'. Train it first: python -m supportlora.train")
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    print("Customer support assistant" + (" (base model)" if args.base else " (fine-tuned)"))
    print("Try for example: " + " | ".join(EXAMPLES))
    print("Type 'exit' to quit.\n")
    while True:
        question = input("You: ").strip()
        if question.lower() in {"exit", "quit"}:
            break
        if question:
            answer, _ = generate(model, tokenizer, question, args.max_new_tokens)
            print(f"\nAssistant: {answer}\n")


if __name__ == "__main__":
    main()
