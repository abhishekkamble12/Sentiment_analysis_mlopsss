"""Confident learning on the silver training labels (Northcutt et al., JAIR 2021).

    python -m tweet_sentiment_classifier.experiments.cleanlab_run --eval val=artifacts/data_transformation/val.csv \
        --eval gold=data/gold/gold_test.csv --out artifacts/experiments/cleanlab

Out-of-fold probabilities from the tuned LogisticRegression flag likely label errors; variants:
  baseline (no change), drop (remove flagged rows), relabel (flagged rows -> model's predicted class).
Each variant is retrained and scored on every --eval set. Needs `cleanlab` (requirements/exp.txt).
"""
import argparse
import json
from pathlib import Path
from typing import Callable, Dict, List

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_predict

from tweet_sentiment_classifier.evaluation.metrics import bootstrap_ci, evaluate
from tweet_sentiment_classifier.models.factory import build_classifier


def cleanlab_finder(labels: np.ndarray, pred_probs: np.ndarray) -> np.ndarray:
    from cleanlab.filter import find_label_issues
    return find_label_issues(labels=labels, pred_probs=pred_probs, filter_by="prune_by_noise_rate")


def run_cleanlab(train_csv: Path, eval_sets: Dict[str, Path], out_dir: Path, lr_params: Dict = None, seed: int = 42,
                 max_features: int = 15000, folds: int = 5,
                 find_issues: Callable[[np.ndarray, np.ndarray], np.ndarray] = cleanlab_finder) -> Dict:
    lr_params = lr_params or {"C": 4.0, "max_iter": 1000}
    train = pd.read_csv(train_csv).dropna(subset=["text"]).reset_index(drop=True)
    texts = train["text"].astype(str)
    y = train["sentiment"].astype(int).to_numpy()
    idx = y + 1  # cleanlab wants 0..K-1

    vec = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2))
    X = vec.fit_transform(texts)
    probs = cross_val_predict(build_classifier("LogisticRegression", lr_params, seed), X, y, cv=folds,
                              method="predict_proba")
    issues = np.asarray(find_issues(idx, probs), dtype=bool)
    predicted = probs.argmax(1) - 1

    variants = {
        "baseline": (np.ones(len(y), bool), y),
        "drop": (~issues, y),
        "relabel": (np.ones(len(y), bool), np.where(issues, predicted, y)),
    }
    result: Dict = {"n_train": int(len(y)), "n_flagged": int(issues.sum()), "flagged_share": float(issues.mean()),
                    "variants": {}}
    if "vader_disagrees" in train:  # optional precomputed column, see docs
        result["flagged_where_vader_disagrees"] = int((issues & train["vader_disagrees"].astype(bool)).sum())

    evals = {name: pd.read_csv(p).dropna(subset=["text"]) for name, p in eval_sets.items()}
    for vname, (keep, labels) in variants.items():
        clf = build_classifier("LogisticRegression", lr_params, seed).fit(X[keep], labels[keep])
        scores = {}
        for ename, edf in evals.items():
            ey = edf["sentiment"].astype(int).to_numpy()
            ep = clf.predict(vec.transform(edf["text"].astype(str)))
            m = evaluate(ey, ep)
            scores[ename] = {"macro_f1": m["macro_f1"], "accuracy": m["accuracy"],
                             "macro_f1_ci": bootstrap_ci(ey, ep, n_boot=500, seed=seed)}
        result["variants"][vname] = {"n_used": int(keep.sum()), "n_changed": int((labels != y).sum()), "scores": scores}

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    flagged = train.loc[issues, ["text", "sentiment"]].assign(model_prediction=predicted[issues])
    flagged.to_csv(out_dir / "flagged.csv", index=False)
    return result


def _parse_evals(items: List[str]) -> Dict[str, Path]:
    out = {}
    for item in items:
        name, _, path = item.partition("=")
        if not path:
            raise ValueError(f"--eval expects name=path, got {item!r}")
        out[name] = Path(path)
    return out


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train", type=Path, default=Path("artifacts/data_transformation/train.csv"))
    parser.add_argument("--eval", action="append", required=True, help="name=path (repeatable)")
    parser.add_argument("--out", type=Path, default=Path("artifacts/experiments/cleanlab"))
    args = parser.parse_args(argv)
    result = run_cleanlab(args.train, _parse_evals(args.eval), args.out)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
