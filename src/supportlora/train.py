"""Fine-tune TinyLlama with LoRA as a customer-support assistant.

python -m supportlora.train                 # full run (2 epochs)
python -m supportlora.train --max-steps 2   # quick smoke test
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import Trainer, TrainingArguments, set_seed

from supportlora.data import load
from supportlora.modeling import Collator, encode, load_base_model, load_tokenizer

LORA = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train", default="data/train.csv")
    parser.add_argument("--test", default="data/test.csv")
    parser.add_argument("--output", default="adapter")
    parser.add_argument("--epochs", type=float, default=2)
    parser.add_argument("--max-steps", type=int, default=-1, help="override epochs (smoke tests)")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    torch.set_num_threads(max(1, torch.get_num_threads()))

    train_pairs, test_pairs = load(args.train), load(args.test)
    print(f"train: {len(train_pairs)} pairs, test: {len(test_pairs)} pairs")

    tokenizer = load_tokenizer()
    train_set = [encode(tokenizer, p, args.max_length) for p in train_pairs]
    test_set = [encode(tokenizer, p, args.max_length) for p in test_pairs]

    model = get_peft_model(load_base_model(), LORA)
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir="checkpoints",
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=3,
        logging_steps=2,
        eval_strategy="epoch" if args.max_steps < 0 else "no",
        save_strategy="no",
        fp16=torch.cuda.is_available(),
        report_to="none",
        dataloader_pin_memory=torch.cuda.is_available(),
        seed=args.seed,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_set,
        eval_dataset=test_set,
        data_collator=Collator(tokenizer.pad_token_id),
    )

    started = time.perf_counter()
    trainer.train()
    minutes = (time.perf_counter() - started) / 60

    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)

    Path("results").mkdir(exist_ok=True)
    log = {
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "train_minutes": round(minutes, 1),
        "train_pairs": len(train_pairs),
        "test_pairs": len(test_pairs),
        "args": vars(args),
        "history": trainer.state.log_history,
    }
    Path("results/training_log.json").write_text(json.dumps(log, indent=2), encoding="utf-8")
    print(f"Done in {minutes:.1f} min. Adapter saved to {args.output}/")


if __name__ == "__main__":
    main()
