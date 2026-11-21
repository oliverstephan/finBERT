"""Create train/validation/test splits from the Financial PhraseBank.

Usage::

    python -m finbert.training.datasets --data-path FinancialPhraseBank-v1.0/Sentences_50Agree.txt
"""

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def split_phrasebank(data_path: str | Path, out_dir: str | Path = "data/sentiment_data") -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(data_path, sep=".@", names=["text", "label"], engine="python", encoding="latin-1")
    train, test = train_test_split(data, test_size=0.2, random_state=0)
    train, valid = train_test_split(train, test_size=0.1, random_state=0)
    train.to_csv(out / "train.csv", sep="\t")
    test.to_csv(out / "test.csv", sep="\t")
    valid.to_csv(out / "validation.csv", sep="\t")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Split Financial PhraseBank into train/validation/test")
    parser.add_argument("--data-path", required=True, help="Path to Sentences_50Agree.txt (or similar)")
    parser.add_argument("--out-dir", default="data/sentiment_data")
    args = parser.parse_args()
    print(f"Wrote splits to {split_phrasebank(args.data_path, args.out_dir)}")


if __name__ == "__main__":
    main()
