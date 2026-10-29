"""Inter-annotator agreement for the gold set.

    python -m tweet_sentiment_classifier.gold.agreement data/gold/ann_a.csv data/gold/ann_b.csv [ann_c.csv]

Each annotator file has columns `id,label` and optionally `flags` (semicolon-separated: sarcasm, code_mixed,
stance_differs, unclear). Labels may be -1/0/1 or negative/neutral/positive. A blank label means "skipped".
"""
import argparse
import itertools
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

LABEL_NAMES = {"negative": -1, "neutral": 0, "positive": 1, "-1": -1, "0": 0, "1": 1, "-1.0": -1, "0.0": 0, "1.0": 1}
FLAGS = ("sarcasm", "code_mixed", "stance_differs", "unclear")
ALPHA_THRESHOLD = 0.6


def normalise_label(value):
    if value is None or (isinstance(value, float) and np.isnan(value)) or str(value).strip() == "":
        return None
    key = str(value).strip().lower()
    if key not in LABEL_NAMES:
        raise ValueError(f"unknown label {value!r}")
    return LABEL_NAMES[key]


def load_annotations(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"id": str})
    if "id" not in df or "label" not in df:
        raise ValueError(f"{path} needs columns id,label")
    if df["id"].duplicated().any():
        raise ValueError(f"{path} has duplicate ids")
    df["label"] = df["label"].map(normalise_label)
    if "flags" not in df:
        df["flags"] = ""
    df["flags"] = df["flags"].fillna("").astype(str)
    return df[["id", "label", "flags"]]


def krippendorff_alpha_nominal(units: Dict[str, List[int]]) -> float:
    """Krippendorff's alpha for nominal data via the coincidence matrix.

    `units` maps an item to the labels it received; items with fewer than two labels are not pairable and
    are ignored. Returns nan when fewer than two values are pairable.
    """
    values = sorted({v for labels in units.values() for v in labels})
    index = {v: i for i, v in enumerate(values)}
    o = np.zeros((len(values), len(values)))
    for labels in units.values():
        m = len(labels)
        if m < 2:
            continue
        for a, b in itertools.permutations(range(m), 2):
            o[index[labels[a]], index[labels[b]]] += 1.0 / (m - 1)
    n_c = o.sum(1)
    n = n_c.sum()
    if n < 2:
        return float("nan")
    observed = o.sum() - np.trace(o)
    expected = (n * n - (n_c * n_c).sum()) / (n - 1)
    if expected == 0:
        return 1.0 if observed == 0 else 0.0
    return float(1.0 - observed / expected)


def agreement_report(annotations: Sequence[pd.DataFrame], names: Sequence[str]) -> Dict:
    merged = None
    for name, df in zip(names, annotations):
        part = df.rename(columns={"label": name, "flags": f"{name}__flags"})
        merged = part if merged is None else merged.merge(part, on="id", how="outer")

    units = {}
    for _, row in merged.iterrows():
        labels = [int(row[n]) for n in names if pd.notna(row[n])]
        units[row["id"]] = labels

    pairwise = {}
    for a, b in itertools.combinations(names, 2):
        both = merged[[a, b]].dropna()
        pairwise[f"{a}|{b}"] = {
            "n_items": int(len(both)),
            "percent_agreement": float((both[a] == both[b]).mean()) if len(both) else float("nan"),
            "cohen_kappa": float(cohen_kappa_score(both[a].astype(int), both[b].astype(int))) if len(both) else float("nan"),
        }

    flag_rates = {}
    for flag in FLAGS:
        hits = Counter()
        for n in names:
            hits[n] = int(merged[f"{n}__flags"].fillna("").str.contains(flag).sum())
        flag_rates[flag] = dict(hits)

    alpha = krippendorff_alpha_nominal(units)
    return {
        "n_items": int(len(merged)),
        "n_annotators": len(names),
        "labels_per_annotator": {n: int(merged[n].notna().sum()) for n in names},
        "krippendorff_alpha": alpha,
        "pairwise": pairwise,
        "flag_counts": flag_rates,
        "alpha_ok": bool(alpha >= ALPHA_THRESHOLD) if not np.isnan(alpha) else False,
    }


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("files", type=Path, nargs="+")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    if len(args.files) < 2:
        parser.error("need at least two annotator files")
    names = [f.stem for f in args.files]
    report = agreement_report([load_annotations(f) for f in args.files], names)
    print(json.dumps(report, indent=2))
    if not report["alpha_ok"]:
        print(f"\nWARNING: Krippendorff's alpha {report['krippendorff_alpha']:.3f} < {ALPHA_THRESHOLD}: "
              "revise the guidelines and re-run the pilot once before labelling the full set.")
    if args.out:
        args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
