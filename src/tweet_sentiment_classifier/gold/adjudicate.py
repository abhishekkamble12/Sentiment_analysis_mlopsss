"""Combine annotators into one gold label per tweet.

    python -m tweet_sentiment_classifier.gold.adjudicate --primary ann_a.csv ann_b.csv --tiebreak ann_c.csv \
        --private data/gold/private.csv --out data/gold/gold.csv

Rules: the two primary annotators agree -> that label ("agreed"). They disagree -> the tie-break annotator's
label decides if it matches one of them ("adjudicated"); otherwise, or without a tie-break label, the tweet is
"unresolved" and excluded. Tweets labelled by fewer than two primary annotators are "single" and excluded.
Raw annotator files are never modified.
"""
import argparse
from pathlib import Path
from typing import Optional

import pandas as pd

from tweet_sentiment_classifier.gold.agreement import load_annotations


def adjudicate(first: pd.DataFrame, second: pd.DataFrame, tiebreak: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    merged = first.rename(columns={"label": "a", "flags": "flags_a"}).merge(
        second.rename(columns={"label": "b", "flags": "flags_b"}), on="id", how="outer")
    if tiebreak is not None:
        merged = merged.merge(tiebreak.rename(columns={"label": "c", "flags": "flags_c"}), on="id", how="left")
    else:
        merged["c"], merged["flags_c"] = None, ""

    rows = []
    for _, r in merged.iterrows():
        a, b, c = r["a"], r["b"], r["c"]
        if pd.isna(a) or pd.isna(b):
            status, label = "single", None
        elif a == b:
            status, label = "agreed", int(a)
        elif pd.notna(c) and c in (a, b):
            status, label = "adjudicated", int(c)
        else:
            status, label = "unresolved", None
        flags = sorted({f for col in ("flags_a", "flags_b", "flags_c")
                        for f in str(r.get(col) or "").split(";") if f.strip() and f.strip() != "nan"})
        n = sum(pd.notna(x) for x in (a, b, c))
        rows.append({"id": r["id"], "sentiment": label, "status": status, "n_annotators": int(n), "flags": ";".join(flags)})
    return pd.DataFrame(rows)


def build_gold(first: Path, second: Path, private: Path, out: Path, tiebreak: Optional[Path] = None) -> pd.DataFrame:
    result = adjudicate(load_annotations(first), load_annotations(second),
                        load_annotations(tiebreak) if tiebreak else None)
    meta = pd.read_csv(private, dtype={"id": str})
    gold = result.merge(meta, on="id", how="left")
    if gold["text"].isna().any():
        raise ValueError("some annotated ids are not in the private sample file")
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    gold.to_csv(out.with_name(out.stem + "_all_statuses.csv"), index=False)
    kept = gold[gold["status"].isin(["agreed", "adjudicated"])].copy()
    kept["sentiment"] = kept["sentiment"].astype(int)
    kept.to_csv(out, index=False)
    return kept


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--primary", type=Path, nargs=2, required=True)
    parser.add_argument("--tiebreak", type=Path, default=None)
    parser.add_argument("--private", type=Path, default=Path("data/gold/private.csv"))
    parser.add_argument("--out", type=Path, default=Path("data/gold/gold.csv"))
    args = parser.parse_args(argv)
    kept = build_gold(args.primary[0], args.primary[1], args.private, args.out, args.tiebreak)
    print(f"{len(kept)} gold tweets written to {args.out} (all statuses in *_all_statuses.csv)")


if __name__ == "__main__":
    main()
