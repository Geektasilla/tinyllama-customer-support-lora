"""Chat window in the browser (Gradio) for the fictional shop Kiwi Market.

The fine-tuned assistant answers with facts from the shop policies (RAG); optionally the base model answers the
same request side by side.

    python -m supportlora.app            # opens http://localhost:7860
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Iterator
from pathlib import Path
from threading import Thread

import gradio as gr
import torch
from peft import PeftModel
from transformers import TextIteratorStreamer

from supportlora.modeling import default_adapter, load_base_model, load_tokenizer, prompt_text
from supportlora.rag import BM25Retriever, build_context, load_policies

TITLE = "🤖 Kiwi Market Support Assistant — TinyLlama + LoRA + RAG"
INTRO = (
    "TinyLlama-1.1B fine-tuned with LoRA on 324 customer-support conversations (trained on a laptop CPU). "
    "With *shop policy* switched on, the most relevant sections of the policies of **Kiwi Market — a fictional shop** "
    "are retrieved and given to the model. Ask in English; answers are generated on the CPU (~30–60 s)."
)
EXAMPLES = [
    "how many days do i have to return something?",
    "what does shipping cost?",
    "i was charged twice!!",
    "can I still cancel my order?",
    "do you deliver to Austria?",
    "I forgot my password",
]
REFRESH_SECONDS = 0.4  # update the page a few times per second instead of on every token -> no flickering
# Stop browsers from auto-translating the page: live translation of streamed text garbles and flickers it.
HEAD = '<meta name="google" content="notranslate">'
JS = "() => { document.documentElement.setAttribute('translate', 'no'); document.documentElement.lang = 'en'; }"


class Assistant:
    def __init__(self, adapter: str, max_new_tokens: int) -> None:
        if not Path(adapter).exists():
            raise SystemExit(f"No adapter in '{adapter}'. Train it first: python -m supportlora.train")
        self.tokenizer = load_tokenizer()
        self.model = PeftModel.from_pretrained(load_base_model(), adapter)
        self.model.eval()
        self.retriever = BM25Retriever(load_policies())
        self.max_new_tokens = max_new_tokens

    def stream(self, question: str, fine_tuned: bool, context: str | None) -> Iterator[str]:
        """Yield the growing answer while it is being generated."""
        prompt = prompt_text(self.tokenizer, question, context)
        inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(self.model.device)
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        kwargs = dict(
            **inputs,
            streamer=streamer,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            repetition_penalty=1.1,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        def run() -> None:
            with torch.no_grad():
                if fine_tuned:
                    self.model.generate(**kwargs)
                else:
                    with self.model.disable_adapter():  # the same weights without the LoRA update = base model
                        self.model.generate(**kwargs)

        Thread(target=run, daemon=True).start()
        text, last_update = "", 0.0
        for chunk in streamer:
            text += chunk
            if time.monotonic() - last_update >= REFRESH_SECONDS:
                last_update = time.monotonic()
                yield text
        yield text


def sources_markdown(results) -> str:
    if not results:
        return "*No matching policy section — the model answers without shop facts.*"
    lines = ["**Policy sections given to the model:**"]
    lines += [f"- {chunk.title} *(score {score})*" for chunk, score in results]
    return "\n".join(lines)


def build_ui(assistant: Assistant) -> gr.Blocks:
    def answer(question: str, use_policy: bool, compare: bool) -> Iterator[tuple[str, str, str]]:
        question = question.strip()
        if not question:
            yield "Please type a question.", "", ""
            return
        results = assistant.retriever.search(question) if use_policy else []
        context = build_context(results) if use_policy else None
        sources = sources_markdown(results) if use_policy else "*Shop policy is off — no retrieval.*"
        header = f"**Request:** {question}\n\n"
        base_text = "⏳ *Waiting for the fine-tuned answer…*" if compare else "*Comparison is off.*"

        # Clear both columns immediately, so an old answer is never shown under a new question.
        yield header + "⏳ *Generating…*", base_text, sources
        tuned = ""
        for tuned in assistant.stream(question, fine_tuned=True, context=context):
            yield header + tuned, base_text, sources
        if compare:
            for base in assistant.stream(question, fine_tuned=False, context=context):
                yield header + tuned, header + base, sources

    with gr.Blocks(title="Kiwi Market Support Assistant") as demo:
        gr.Markdown(f"# {TITLE}\n{INTRO}")
        question = gr.Textbox(label="Customer request", placeholder="e.g. how many days do i have to return something?")
        with gr.Row():
            ask = gr.Button("Ask", variant="primary", scale=2)
            use_policy = gr.Checkbox(label="Use shop policy (RAG)", value=True, scale=1)
            compare = gr.Checkbox(label="Also show the base model (before fine-tuning)", value=False, scale=1)
        gr.Examples(EXAMPLES, inputs=question)
        sources_out = gr.Markdown()
        with gr.Row():
            with gr.Column():
                gr.Markdown("### ✅ Fine-tuned model")
                tuned_out = gr.Markdown()
            with gr.Column():
                gr.Markdown("### Base TinyLlama")
                base_out = gr.Markdown()
        inputs, outputs = [question, use_policy, compare], [tuned_out, base_out, sources_out]
        ask.click(answer, inputs, outputs, concurrency_limit=1)
        question.submit(answer, inputs, outputs, concurrency_limit=1)
        gr.Markdown(
            "<small>Demo of a fine-tuning + RAG project. Kiwi Market and its policies are fictional "
            "(see `policies/`).</small>"
        )
    return demo


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--adapter", default=default_adapter(), help="default: adapter_rag if trained, else adapter")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    print("Loading the model (10-20 s)...")
    demo = build_ui(Assistant(args.adapter, args.max_new_tokens))
    demo.queue().launch(server_name="127.0.0.1", server_port=args.port, inbrowser=not args.no_browser, head=HEAD, js=JS)


if __name__ == "__main__":
    main()
