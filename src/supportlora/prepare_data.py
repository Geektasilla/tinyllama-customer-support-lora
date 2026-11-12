"""Download the Bitext customer-support dataset and build a balanced train/test subset.

python -m supportlora.prepare_data                     # 12 train + 3 test examples per intent
python -m supportlora.prepare_data --train-per-intent 50
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

DATASET = "bitext/Bitext-customer-support-llm-chatbot-training-dataset"  # license: CDLA-Sharing-1.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train-per-intent", type=int, default=12)
    parser.add_argument("--test-per-intent", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="data")
    args = parser.parse_args()

    from datasets import load_dataset  # imported here so the rest of the package works without `datasets`

    from supportlora.data import clean, sample_per_intent, save

    frame = load_dataset(DATASET, split="train").to_pandas()
    examples = clean(frame)
    train, test = sample_per_intent(examples, args.train_per_intent, args.test_per_intent, seed=args.seed)

    out = Path(args.out)
    out.mkdir(exist_ok=True)
    save(train, out / "train.csv")
    save(test, out / "test.csv")
    print(f"{len(frame)} rows -> {len(examples)} unique questions -> train {len(train)}, test {len(test)}")
    print(f"intents: {len(Counter(e.intent for e in test))}")


if __name__ == "__main__":
    main()
