"""Hyperparameter search on the VALIDATION split only.

    python -m tweet_sentiment_classifier.evaluation.tune --train .../train.csv --val .../val.csv

There is deliberately no test-set argument: the test file must never influence a choice. The winner per
model is the one with the highest validation macro-F1; ties keep the earlier (smaller / simpler) setting.
"""
import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from tweet_sentiment_classifier.evaluation.metrics import evaluate, expected_calibration_error
from tweet_sentiment_classifier.models.factory import build_classifier

# Candidate settings, ordered from simplest / most regularised to least.
GRIDS: Dict[str, List[Dict[str, Any]]] = {
    "LogisticRegression": [{"C": c, "max_iter": 1000} for c in (1.0, 4.0, 16.0, 64.0)],
    # first run's winners were at the grid edge (LinearSVC C=1.0, NB alpha=0.03), so the grids extend past them
    "LinearSVC": [{"C": c} for c in (0.3, 1.0, 3.0, 10.0)],
    "MultinomialNB": [{"alpha": a} for a in (0.3, 0.1, 0.03, 0.01)],
    "RidgeClassifier": [{"alpha": a} for a in (3.0, 1.0, 0.3)],
}


def tune(train_csv: Path, val_csv: Path, out_path: Path, grids: Dict[str, List[Dict[str, Any]]] = None,
         seed: int = 42, max_features: int = 15000, progress=print) -> Dict[str, Any]:
    grids = grids or GRIDS
    train = pd.read_csv(train_csv).dropna(subset=["text"])
    val = pd.read_csv(val_csv).dropna(subset=["text"])
    y_train = train["sentiment"].astype(int).to_numpy()
    y_val = val["sentiment"].astype(int).to_numpy()

    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2))
    x_train = vectorizer.fit_transform(train["text"].astype(str))
    x_val = vectorizer.transform(val["text"].astype(str))

    trials: Dict[str, List[Dict[str, Any]]] = {}
    best: Dict[str, Dict[str, Any]] = {}
    for name, candidates in grids.items():
        trials[name] = []
        for params in candidates:
            start = time.perf_counter()
            clf = build_classifier(name, params, seed)
            clf.fit(x_train, y_train + 1 if name == "XGBoost" else y_train)
            seconds = time.perf_counter() - start
            proba = clf.predict_proba(x_val)
            classes = np.asarray(clf.classes_) - (1 if name == "XGBoost" else 0)
            pred = classes[proba.argmax(1)]
            m = evaluate(y_val, pred, proba=proba)
            trial = {"params": params, "val_macro_f1": m["macro_f1"], "val_accuracy": m["accuracy"],
                     "val_ece": m["ece"], "fit_seconds": round(seconds, 1)}
            if hasattr(clf, "n_iter_"):  # LogisticRegression: did it converge before max_iter?
                trial["converged"] = bool(int(np.max(clf.n_iter_)) < params.get("max_iter", 100))
            trials[name].append(trial)
            progress(f"{name:<19}{str(params):<32}val macroF1={m['macro_f1']:.4f} ece={m['ece']:.3f} ({seconds:.0f}s)")
        winner = max(trials[name], key=lambda t: t["val_macro_f1"])  # max() keeps the first of equal scores
        best[name] = {"params": winner["params"], "val_macro_f1": winner["val_macro_f1"]}

    result = {"selection_metric": "val_macro_f1", "n_train": len(train), "n_val": len(val),
              "seed": seed, "max_features": max_features, "best": best, "trials": trials,
              "note": "chosen on the validation split only; the test split was not read"}
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main(argv: Sequence[str] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--val", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("artifacts/evaluation/tuning.json"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    result = tune(args.train, args.val, args.out, seed=args.seed)
    print("\nbest per model (validation macro-F1):")
    for name, b in result["best"].items():
        print(f"  {name:<19}{b['params']}  {b['val_macro_f1']:.4f}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
