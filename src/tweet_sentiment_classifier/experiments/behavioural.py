"""CheckList-style behavioural tests (Ribeiro et al., ACL 2020).

    python -m tweet_sentiment_classifier.experiments.behavioural generate --out artifacts/experiments/behavioural/cases.csv
    # a human reviews cases.csv: fix `expected`, set reviewed=1 (or delete bad rows)
    python -m tweet_sentiment_classifier.experiments.behavioural run --cases .../cases.csv --models LogisticRegression LinearSVC

Two test types:
  MFT  minimum functionality: each case has an expected label
  INV  invariance: cases sharing a `group` must all receive the same prediction (e.g. swapping politician names)
The expected labels below are DRAFTS written by the tool author; only reviewed rows are scored.
"""
import argparse
import itertools
import json
from pathlib import Path
from typing import Dict, List, Sequence

import pandas as pd

PEOPLE = ["modi", "rahul", "kejriwal", "mamata"]
TOPICS = ["the economy", "the new scheme", "the budget", "the election campaign"]

# (capability, test type, template, expected label or None for INV)
TEMPLATES = [
    ("negation", "MFT", "{topic} is not good", -1),
    ("negation", "MFT", "{topic} is not bad at all", 1),
    ("negation", "MFT", "i do not like {topic}", -1),
    ("negation", "MFT", "i do not hate {topic}", 0),
    ("contrast", "MFT", "{topic} looked promising but it failed badly", -1),
    ("contrast", "MFT", "{topic} had problems but overall it is great", 1),
    ("intensifier", "MFT", "{topic} is absolutely terrible", -1),
    ("intensifier", "MFT", "{topic} is extremely good", 1),
    ("domain_polarity", "MFT", "unemployment is rising under {topic}", -1),
    ("domain_polarity", "MFT", "prices have fallen thanks to {topic}", 1),
    ("domain_polarity", "MFT", "the economy collapsed after {topic}", -1),
    ("neutral_fact", "MFT", "{person} will speak about {topic} tomorrow", 0),
    ("neutral_fact", "MFT", "voting on {topic} starts at 7am", 0),
    ("sarcasm", "MFT", "great another power cut thanks to {topic}", -1),
    ("sarcasm", "MFT", "wow {topic} fixed everything said nobody", -1),
    ("emoji", "MFT", "{topic} 😍😍", 1),
    ("emoji", "MFT", "{topic} 😡", -1),
    ("entity_swap", "INV", "{person} gave a great speech on {topic}", None),
    ("entity_swap", "INV", "{person} has failed on {topic}", None),
    ("entity_swap", "INV", "{person} spoke about {topic} today", None),
]


def generate_cases() -> pd.DataFrame:
    rows: List[Dict] = []
    for t_id, (capability, kind, template, expected) in enumerate(TEMPLATES):
        for topic in TOPICS:
            people = PEOPLE if "{person}" in template else [None]
            for person in people:
                text = template.format(topic=topic, person=person)
                group = f"t{t_id}-{topic}" if kind == "INV" else ""
                rows.append({"capability": capability, "type": kind, "text": text, "expected": expected,
                             "group": group, "reviewed": 0})
    return pd.DataFrame(rows).drop_duplicates(subset=["text"]).reset_index(drop=True)


def score_cases(cases: pd.DataFrame, predict) -> pd.DataFrame:
    """Failure rate per capability. Only reviewed rows count."""
    cases = cases[cases["reviewed"].astype(int) == 1].copy()
    if cases.empty:
        raise ValueError("no reviewed cases: a human must review the expected labels first")
    cases["pred"] = predict(cases["text"].astype(str).tolist())
    out = []
    for (cap, kind), grp in cases.groupby(["capability", "type"]):
        if kind == "MFT":
            fails = int((grp["pred"] != grp["expected"].astype(int)).sum())
            n = len(grp)
        else:
            groups = grp.groupby("group")["pred"].nunique()
            fails, n = int((groups > 1).sum()), int(len(groups))
        out.append({"capability": cap, "type": kind, "n": n, "failures": fails, "failure_rate": fails / n if n else 0.0})
    return pd.DataFrame(out)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate")
    g.add_argument("--out", type=Path, default=Path("artifacts/experiments/behavioural/cases.csv"))
    r = sub.add_parser("run")
    r.add_argument("--cases", type=Path, required=True)
    r.add_argument("--train", type=Path, default=Path("artifacts/data_transformation/train.csv"))
    r.add_argument("--models", nargs="+", default=["LogisticRegression", "LinearSVC"])
    r.add_argument("--params", type=Path, default=Path("params.yaml"))
    args = parser.parse_args(argv)

    if args.cmd == "generate":
        args.out.parent.mkdir(parents=True, exist_ok=True)
        generate_cases().to_csv(args.out, index=False)
        print(f"wrote {args.out}; review `expected`, then set reviewed=1")
        return

    import yaml
    from tweet_sentiment_classifier.evaluation.run import TfidfModel
    params = yaml.safe_load(args.params.read_text(encoding="utf-8")) if args.params.exists() else {}
    train = pd.read_csv(args.train).dropna(subset=["text"])
    cases = pd.read_csv(args.cases)
    report = {}
    for name in args.models:
        model = TfidfModel(name, params.get(name, {}), 42, int(params.get("max_features", 15000))).fit(
            train["text"].astype(str).tolist(), train["sentiment"].astype(int).to_numpy())
        table = score_cases(cases, model.predict)
        report[name] = table.to_dict(orient="records")
        print(f"\n{name}\n{table.to_string(index=False)}")
    out = args.cases.with_name("results.json")
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
