"""Out-of-domain test sets in the project's CSV format (text, sentiment in {-1, 0, 1}).

    python -m tweet_sentiment_classifier.experiments.ood tweeteval --out data/ood/tweeteval_sentiment_test.csv

Needs network and the `datasets` package (requirements/exp.txt). The label mapping is checked against the
dataset's own label names at runtime instead of being assumed. Texts are cleaned with the same function as the
training data so the comparison is fair.
"""
import argparse
from pathlib import Path
from typing import Sequence

import pandas as pd

from tweet_sentiment_classifier.components.data_transformation import clean_text

NAME_TO_LABEL = {"negative": -1, "neutral": 0, "positive": 1}


def to_project_format(texts: Sequence[str], label_ids: Sequence[int], label_names: Sequence[str]) -> pd.DataFrame:
    names = [n.lower() for n in label_names]
    if set(names) != set(NAME_TO_LABEL):
        raise ValueError(f"unexpected label names {label_names}; refusing to guess a mapping")
    mapping = {i: NAME_TO_LABEL[n] for i, n in enumerate(names)}
    df = pd.DataFrame({"text": [clean_text(t) for t in texts], "sentiment": [mapping[int(i)] for i in label_ids]})
    return df[df["text"] != ""].reset_index(drop=True)


def load_tweeteval(split: str = "test") -> pd.DataFrame:
    from datasets import load_dataset
    ds = load_dataset("cardiffnlp/tweet_eval", "sentiment", split=split)
    return to_project_format(ds["text"], ds["label"], ds.features["label"].names)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset", choices=["tweeteval"])
    parser.add_argument("--split", default="test")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    df = load_tweeteval(args.split)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"wrote {len(df)} rows to {args.out}; label counts {df['sentiment'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
