"""Does retrieval (RAG) make the answers factually correct for the fictional shop?

For every question in policies/eval_questions.json we know the policy section that answers it and the fact that
must appear in a correct answer (e.g. "30 days"). Set-ups (all except `v1` get the retrieved policy sections):

    base_rag   original TinyLlama
    v1         adapter trained on customer-support style only, no policy in the prompt
    v1_rag     the same adapter
    v2_rag     adapter trained with retrieved policy in the prompt (RAG-aware fine-tuning)

    python -m supportlora.rag_eval                                  # base_rag, v1_rag, v2_rag (~45 min on CPU)
    python -m supportlora.rag_eval --setups v2_rag
"""

from __future__ import annotations

import argparse
import json
from contextlib import nullcontext
from pathlib import Path
from statistics import mean

from supportlora.rag import BM25Retriever, build_context, load_policies

# setup -> (adapter name or None for the base model, use retrieval)
SETUPS = {"base_rag": (None, True), "v1": ("v1", False), "v1_rag": ("v1", True), "v2_rag": ("v2", True)}


def contains_fact(answer: str, facts: list[str]) -> bool:
    """True if the answer contains at least one accepted spelling of the expected fact."""
    text = answer.lower().replace(" ", " ").replace("\xa0", " ")
    return any(fact.lower() in text for fact in facts)


def retrieval_report(retriever: BM25Retriever, questions: list[dict], k: int) -> dict:
    hits_1, hits_k = [], []
    for q in questions:
        ids = [chunk.id for chunk, _ in retriever.search(q["question"], k=k)]
        hits_1.append(bool(ids) and ids[0] == q["section"])
        hits_k.append(q["section"] in ids)
    return {"hit_at_1_pct": round(100 * mean(hits_1)), f"hit_at_{k}_pct": round(100 * mean(hits_k))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--questions", default="policies/eval_questions.json")
    parser.add_argument("--v1", default="adapter", help="style-only adapter")
    parser.add_argument("--v2", default="adapter_rag", help="RAG-aware adapter")
    parser.add_argument("--setups", nargs="+", choices=list(SETUPS), default=["base_rag", "v1_rag", "v2_rag"])
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=200)
    args = parser.parse_args()

    from peft import PeftModel

    from supportlora.evaluate import generate
    from supportlora.modeling import load_base_model, load_tokenizer

    questions = json.loads(Path(args.questions).read_text(encoding="utf-8"))
    retriever = BM25Retriever(load_policies())
    report: dict = {"questions": len(questions), "retrieval": retrieval_report(retriever, questions, args.top_k)}
    print("Retrieval:", report["retrieval"])

    needed = {SETUPS[s][0] for s in args.setups} - {None}
    paths = {"v1": args.v1, "v2": args.v2}
    how_to_train = {
        "v1": "python -m supportlora.train",
        "v2": "'9. Train v2' (python -m supportlora.train --grounded ...)",
    }
    for name in sorted(needed | {"v1"}):
        if not Path(paths[name], "adapter_config.json").exists():
            raise SystemExit(f"Adapter {name} not found in '{paths[name]}'. Train it first: {how_to_train[name]}")

    tokenizer = load_tokenizer()
    model = load_base_model()
    for name in sorted(needed):
        if isinstance(model, PeftModel):
            model.load_adapter(paths[name], adapter_name=name)
        else:
            model = PeftModel.from_pretrained(model, paths[name], adapter_name=name)
    if not needed:
        model = PeftModel.from_pretrained(model, args.v1, adapter_name="v1")
    model.eval()

    answers: dict[str, list[str]] = {}
    for setup in args.setups:
        print(f"\n== {setup} ==")
        adapter, use_rag = SETUPS[setup]
        if adapter:
            model.set_adapter(adapter)
        adapter_off = model.disable_adapter() if adapter is None else nullcontext()
        results = []
        with adapter_off:
            for i, q in enumerate(questions, start=1):
                context = build_context(retriever.search(q["question"], k=args.top_k)) if use_rag else None
                answer, _ = generate(model, tokenizer, q["question"], args.max_new_tokens, context=context)
                correct = contains_fact(answer, q["facts"])
                results.append(correct)
                answers.setdefault(setup, []).append(answer)
                print(f"[{i}/{len(questions)}] {'OK ' if correct else 'MISS'} {q['question']}")
        report[setup] = {
            "fact_accuracy_pct": round(100 * mean(results)),
            "correct": sum(results),
            # dataset-style placeholders like {{Order Number}} instead of the real fact
            "answers_with_placeholders": sum("{{" in a for a in answers[setup]),
        }
        print(setup, report[setup])

    Path("results").mkdir(exist_ok=True)
    Path("results/rag_metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    lines = ["# RAG evaluation: answers for the fictional shop Kiwi Market", ""]
    for i, q in enumerate(questions):
        lines += [
            f"## {q['question']}",
            "",
            f"*Expected fact: {' / '.join(q['facts'])} — section `{q['section']}`*",
            "",
        ]
        for setup in args.setups:
            answer = answers[setup][i]
            mark = "✅" if contains_fact(answer, q["facts"]) else "❌"
            lines += [f"**{setup}** {mark}", "", answer, ""]
        lines += ["---", ""]
    Path("results/rag_samples.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n" + json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
