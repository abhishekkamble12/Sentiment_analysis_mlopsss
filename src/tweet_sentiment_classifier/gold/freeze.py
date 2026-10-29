"""Freeze the gold set: split it into gold_test and gold_train_pool and record fingerprints.

    python -m tweet_sentiment_classifier.gold.freeze --gold data/gold/gold.csv

gold_test is for final evaluation only. Never tune, select or early-stop on it; use gold_train_pool for that.
"""
import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from tweet_sentiment_classifier.evaluation.metrics import fingerprint


def freeze(gold_csv: Path, train_csv: Path, val_csv: Path, out_dir: Path, test_n: int = 1000, seed: int = 42) -> dict:
    gold = pd.read_csv(gold_csv)
    seen = set(pd.read_csv(train_csv)["text"].astype(str)) | set(pd.read_csv(val_csv)["text"].astype(str))
    leaked = gold["text"].astype(str).isin(seen)
    if leaked.any():
        raise ValueError(f"{int(leaked.sum())} gold tweets also appear in train/val; gold must come from the test split")
    if len(gold) < test_n + 50:
        raise ValueError(f"only {len(gold)} gold tweets; need at least {test_n + 50} for gold_test + a usable pool")

    gold_test, pool = train_test_split(gold, train_size=test_n, stratify=gold["sentiment"], random_state=seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    gold_test.to_csv(out_dir / "gold_test.csv", index=False)
    pool.to_csv(out_dir / "gold_train_pool.csv", index=False)

    def summary(df):
        return {"n": int(len(df)), "fingerprint": fingerprint(df["text"]),
                "label_counts": {str(int(k)): int(v) for k, v in df["sentiment"].value_counts().sort_index().items()}}

    manifest = {"seed": seed, "source": str(gold_csv), "gold_test": summary(gold_test),
                "gold_train_pool": summary(pool),
                "rule": "gold_test is for final evaluation only; tune and select on gold_train_pool"}
    (out_dir / "gold_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gold", type=Path, default=Path("data/gold/gold.csv"))
    parser.add_argument("--train", type=Path, default=Path("artifacts/data_transformation/train.csv"))
    parser.add_argument("--val", type=Path, default=Path("artifacts/data_transformation/val.csv"))
    parser.add_argument("--out", type=Path, default=Path("data/gold"))
    parser.add_argument("--test-n", type=int, default=1000)
    args = parser.parse_args(argv)
    m = freeze(args.gold, args.train, args.val, args.out, args.test_n)
    print(json.dumps(m, indent=2))
    print("\ngold_test must not be used for tuning, model selection or early stopping.")


if __name__ == "__main__":
    main()
