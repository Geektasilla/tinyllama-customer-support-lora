"""Talk to the model in the terminal.

python -m supportlora.chat               # fine-tuned model + shop policy (RAG)
python -m supportlora.chat --no-policy   # fine-tuned model without the policy
python -m supportlora.chat --base        # original TinyLlama + shop policy, for comparison
"""

from __future__ import annotations

import argparse
from pathlib import Path

from supportlora.evaluate import generate
from supportlora.modeling import default_adapter, load_base_model, load_tokenizer
from supportlora.rag import BM25Retriever, build_context, load_policies

EXAMPLES = [
    "how many days do i have to return something?",
    "what does shipping cost?",
    "i was charged twice!!",
    "I forgot my password",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--adapter", default=default_adapter(), help="default: adapter_rag if trained, else adapter")
    parser.add_argument("--base", action="store_true", help="use the original model without the adapter")
    parser.add_argument("--no-policy", action="store_true", help="do not retrieve shop policy sections")
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
    retriever = None if args.no_policy else BM25Retriever(load_policies())

    name = "base model" if args.base else "fine-tuned"
    print(f"Kiwi Market support assistant ({name}, shop policy {'off' if args.no_policy else 'on'})")
    print("Try for example: " + " | ".join(EXAMPLES))
    print("Type 'exit' to quit.\n")
    while True:
        question = input("You: ").strip()
        if question.lower() in {"exit", "quit"}:
            break
        if not question:
            continue
        results = retriever.search(question) if retriever else []
        if retriever:
            print("  (policy: " + (", ".join(c.title for c, _ in results) or "no matching section") + ")")
        answer, _ = generate(
            model, tokenizer, question, args.max_new_tokens, context=build_context(results) if retriever else None
        )
        print(f"\nAssistant: {answer}\n")


if __name__ == "__main__":
    main()
