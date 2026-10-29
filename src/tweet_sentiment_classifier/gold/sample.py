"""Draw tweets for human annotation from the TEST split only.

    python -m tweet_sentiment_classifier.gold.sample --pilot                  # ~100 tweets, pilot round
    python -m tweet_sentiment_classifier.gold.sample --n-per-class 500 --exclude data/gold/pilot_private.csv

Sampling is class-balanced on the silver label so that rare classes get enough human labels. That is NOT the
natural class mix, so every row carries `sampling_weight = natural share / sampled share` of its silver class;
pass it to `evaluation.run --weights-column sampling_weight` to estimate natural-mix scores.

Outputs (in --out, default data/gold, gitignored because it contains tweet text):
  <prefix>to_annotate.csv   id,text                 (what annotators see; no model or silver label)
  <prefix>private.csv       id,row_index,text,silver_label,sampling_weight
  <prefix>manifest.json     seed, counts, source fingerprint
"""
import argparse
import json
from pathlib import Path
from typing import Optional

import pandas as pd

from tweet_sentiment_classifier.evaluation.metrics import fingerprint


def draw_sample(test_csv: Path, out_dir: Path, n_per_class: int = 500, seed: int = 42, prefix: str = "",
                exclude_private: Optional[Path] = None) -> pd.DataFrame:
    test = pd.read_csv(test_csv).dropna(subset=["text"]).reset_index(drop=True)
    test["row_index"] = test.index
    natural_share = test["sentiment"].value_counts(normalize=True)

    pool = test
    if exclude_private is not None:
        used = set(pd.read_csv(exclude_private)["row_index"])
        pool = test[~test["row_index"].isin(used)]

    parts = []
    for label, group in pool.groupby("sentiment"):
        if len(group) < n_per_class:
            raise ValueError(f"only {len(group)} tweets with silver label {label}, need {n_per_class}")
        parts.append(group.sample(n=n_per_class, random_state=seed))
    sample = pd.concat(parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)

    sampled_share = sample["sentiment"].value_counts(normalize=True)
    sample["sampling_weight"] = sample["sentiment"].map(lambda c: natural_share[c] / sampled_share[c])
    sample["id"] = [f"{prefix}g{seed}-{i:05d}" for i in range(len(sample))]
    sample["silver_label"] = sample["sentiment"].astype(int)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sample[["id", "text"]].to_csv(out_dir / f"{prefix}to_annotate.csv", index=False)
    sample[["id", "row_index", "text", "silver_label", "sampling_weight"]].to_csv(out_dir / f"{prefix}private.csv", index=False)
    manifest = {
        "source": str(test_csv), "source_fingerprint": fingerprint(test["text"]), "seed": seed,
        "n_per_class": n_per_class, "n": len(sample), "excluded_from": str(exclude_private) if exclude_private else None,
        "natural_share": {str(int(k)): float(v) for k, v in natural_share.sort_index().items()},
        "note": "drawn from the test split only; class-balanced on the silver label; silver label hidden from annotators",
    }
    (out_dir / f"{prefix}manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return sample


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--test", type=Path, default=Path("artifacts/data_transformation/test.csv"))
    parser.add_argument("--out", type=Path, default=Path("data/gold"))
    parser.add_argument("--n-per-class", type=int, default=500)
    parser.add_argument("--pilot", action="store_true", help="34 per class (~100 tweets), files prefixed pilot_")
    parser.add_argument("--exclude", type=Path, default=None, help="a *private.csv whose rows must not be re-drawn")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    n, prefix = (34, "pilot_") if args.pilot else (args.n_per_class, "")
    sample = draw_sample(args.test, args.out, n, args.seed, prefix, args.exclude)
    print(f"wrote {len(sample)} tweets to {args.out / (prefix + 'to_annotate.csv')}")


if __name__ == "__main__":
    main()
