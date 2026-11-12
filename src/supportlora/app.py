"""Chat window in the browser (Gradio): the fine-tuned assistant, optionally side by side with the base model.

python -m supportlora.app            # opens http://localhost:7860
"""

from __future__ import annotations

import argparse
from collections.abc import Iterator
from pathlib import Path
from threading import Thread

import gradio as gr
import torch
from peft import PeftModel
from transformers import TextIteratorStreamer

from supportlora.chat import EXAMPLES
from supportlora.modeling import load_base_model, load_tokenizer, prompt_text

TITLE = "🤖 Customer-Support Assistant — TinyLlama + LoRA"
INTRO = (
    "TinyLlama-1.1B fine-tuned with LoRA on 324 customer-support conversations (trained on a laptop CPU). "
    "Ask something an online-shop customer would ask, in English. "
    "Answers are generated on the CPU, so give it ~30–60 seconds."
)


class Assistant:
    def __init__(self, adapter: str, max_new_tokens: int) -> None:
        if not Path(adapter).exists():
            raise SystemExit(f"No adapter in '{adapter}'. Train it first: python -m supportlora.train")
        self.tokenizer = load_tokenizer()
        self.model = PeftModel.from_pretrained(load_base_model(), adapter)
        self.model.eval()
        self.max_new_tokens = max_new_tokens

    def stream(self, question: str, fine_tuned: bool) -> Iterator[str]:
        """Yield the answer as it is being generated."""
        inputs = self.tokenizer(prompt_text(self.tokenizer, question), return_tensors="pt", add_special_tokens=False)
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        kwargs = dict(
            **inputs.to(self.model.device),
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
        text = ""
        for chunk in streamer:
            text += chunk
            yield text


def build_ui(assistant: Assistant) -> gr.Blocks:
    def answer(question: str, compare: bool) -> Iterator[tuple[str, str]]:
        question = question.strip()
        if not question:
            yield "Please type a question.", ""
            return
        base_note = "*Waiting for the fine-tuned answer…*" if compare else "*Comparison is off.*"
        tuned = ""
        for tuned in assistant.stream(question, fine_tuned=True):
            yield tuned, base_note
        if compare:
            for base in assistant.stream(question, fine_tuned=False):
                yield tuned, base

    with gr.Blocks(title="Customer-Support Assistant") as demo:
        gr.Markdown(f"# {TITLE}\n{INTRO}")
        question = gr.Textbox(label="Customer request", placeholder="e.g. I want to cancel my order", lines=2)
        with gr.Row():
            ask = gr.Button("Ask", variant="primary")
            compare = gr.Checkbox(label="Also show the answer of the base model (before fine-tuning)", value=True)
        gr.Examples(EXAMPLES + ["can u change the delivery adress pls", "I was charged twice!!"], inputs=question)
        with gr.Row():
            with gr.Column():
                gr.Markdown("### ✅ Fine-tuned model")
                tuned_out = gr.Markdown()
            with gr.Column():
                gr.Markdown("### Base TinyLlama")
                base_out = gr.Markdown()
        ask.click(answer, [question, compare], [tuned_out, base_out], concurrency_limit=1)
        question.submit(answer, [question, compare], [tuned_out, base_out], concurrency_limit=1)
        gr.Markdown(
            "<small>Demo of a fine-tuning project. The assistant knows the *style* of good support, "
            "not the policies of a real shop.</small>"
        )
    return demo


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--adapter", default="adapter")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    print("Loading the model (10-20 s)...")
    demo = build_ui(Assistant(args.adapter, args.max_new_tokens))
    demo.queue().launch(server_name="127.0.0.1", server_port=args.port, inbrowser=not args.no_browser)


if __name__ == "__main__":
    main()
