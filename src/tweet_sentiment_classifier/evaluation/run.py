"""Evaluate every model on one test set in one run:

    python -m tweet_sentiment_classifier.evaluation.run --train artifacts/data_transformation/train.csv \
        --test artifacts/data_transformation/test.csv --out artifacts/evaluation

Every number in the output comes from this single invocation: same train/test files, same seed, one
run_id, one results table. Models are fit here from scratch (nothing is loaded from older pickles).
"""
import argparse
import datetime as dt
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from tweet_sentiment_classifier.evaluation.metrics import (
    bootstrap_ci, evaluate, fingerprint, measure_latency, save_run,
)
from tweet_sentiment_classifier.models.factory import build_classifier
from tweet_sentiment_classifier.models.lexicon import MajorityClassifier, TextBlobClassifier, VaderClassifier

DEFAULT_MODELS = ("Majority", "TextBlob", "VADER", "LogisticRegression", "LinearSVC", "MultinomialNB", "RidgeClassifier")


class TfidfModel:
    """TF-IDF + classifier with a single-text predict, so every model is timed the same way."""

    def __init__(self, name: str, hyperparams: Dict[str, Any], seed: int, max_features: int):
        self.name = name
        self.vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2))
        self.classifier = build_classifier(name, hyperparams, seed)

    def fit(self, texts: Sequence[str], labels: Sequence[int]) -> "TfidfModel":
        y = np.asarray(labels).astype(int)
        self.classifier.fit(self.vectorizer.fit_transform(texts), y + 1 if self.name == "XGBoost" else y)
        return self

    def _classes(self) -> np.ndarray:
        classes = np.asarray(self.classifier.classes_)
        return classes - 1 if self.name == "XGBoost" else classes

    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        return self.classifier.predict_proba(self.vectorizer.transform(texts))

    def predict(self, texts: Sequence[str]) -> np.ndarray:
        return self._classes()[self.predict_proba(texts).argmax(1)]

    def predict_one(self, text: str) -> int:
        return int(self.predict([text])[0])


def _evaluate_model(model, name: str, texts: Sequence[str], y_true: np.ndarray, seed: int, n_boot: int) -> Dict[str, Any]:
    proba = model.predict_proba(texts) if hasattr(model, "predict_proba") else None
    pred = model.predict(texts)
    result = evaluate(y_true, pred, proba=proba)
    result["macro_f1_ci"] = bootstrap_ci(y_true, pred, "macro_f1", n_boot=n_boot, seed=seed)
    if hasattr(model, "predict_one"):
        result["latency"] = measure_latency(model.predict_one, texts)
    return result


def run(train_csv: Path, test_csv: Path, out_dir: Path, models: Sequence[str] = DEFAULT_MODELS,
        params: Dict[str, Any] = None, seed: int = 42, max_features: int = 15000, n_boot: int = 1000,
        label_source: str = "silver (TextBlob-generated)") -> Dict[str, Any]:
    params = params or {}
    train = pd.read_csv(train_csv).dropna(subset=["text"])
    test = pd.read_csv(test_csv).dropna(subset=["text"])
    x_train, y_train = train["text"].astype(str).tolist(), train["sentiment"].astype(int).to_numpy()
    x_test, y_test = test["text"].astype(str).tolist(), test["sentiment"].astype(int).to_numpy()

    split_hash = fingerprint(x_test)
    overlap = len(set(x_test) & set(x_train))
    run_id = dt.datetime.now().strftime("%Y%m%d-%H%M%S") + f"-{split_hash[:6]}"

    results: Dict[str, Dict[str, Any]] = {}
    for name in models:
        if name == "Majority":
            model = MajorityClassifier().fit(x_train, y_train)
        elif name == "TextBlob":
            model = TextBlobClassifier()
        elif name == "VADER":
            model = VaderClassifier()
        else:
            model = TfidfModel(name, params.get(name, {}), seed, max_features).fit(x_train, y_train)
        results[name] = _evaluate_model(model, name, x_test, y_test, seed, n_boot)

    split_info = {
        "train_csv": str(train_csv), "test_csv": str(test_csv), "n_train": len(x_train), "n_test": len(x_test),
        "test_fingerprint": split_hash, "label_source": label_source, "seed": seed, "max_features": max_features,
        "n_bootstrap": n_boot,
        # a non-zero value means some test texts were also seen in training (inflates scores)
        "test_texts_also_in_train": overlap,
        "test_texts_also_in_train_pct": round(100 * overlap / max(len(x_test), 1), 2),
    }
    save_run(out_dir, run_id, split_info, results)
    return {"run_id": run_id, "split": split_info, "results": results}


def _format(summary: Dict[str, Any]) -> str:
    lines = [f"run {summary['run_id']}  n_test={summary['split']['n_test']}  "
             f"overlap_with_train={summary['split']['test_texts_also_in_train_pct']}%  labels={summary['split']['label_source']}",
             f"{'model':<20}{'acc':>8}{'macroF1':>9}{'95% CI':>18}{'kappa':>8}{'ECE':>7}{'p50 ms':>9}"]
    for name, r in summary["results"].items():
        lo, hi = r["macro_f1_ci"]
        ece = f"{r['ece']:.3f}" if "ece" in r else "-"
        p50 = f"{r['latency']['p50_ms']:.2f}" if "latency" in r else "-"
        lines.append(f"{name:<20}{r['accuracy']:>8.4f}{r['macro_f1']:>9.4f}{f'[{lo:.4f}, {hi:.4f}]':>18}"
                     f"{r['kappa']:>8.3f}{ece:>7}{p50:>9}")
    return "\n".join(lines)


def main(argv: List[str] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("artifacts/evaluation"))
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument("--params", type=Path, default=Path("params.yaml"), help="hyperparameters (YAML)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--label-source", default="silver (TextBlob-generated)")
    args = parser.parse_args(argv)

    import yaml
    params = yaml.safe_load(args.params.read_text(encoding="utf-8")) if args.params.exists() else {}
    summary = run(args.train, args.test, args.out, args.models, params=params, seed=args.seed,
                  max_features=int(params.get("max_features", 15000)), label_source=args.label_source)
    print(_format(summary))
    print(f"\nwrote {args.out / 'results.csv'} and results.json")


if __name__ == "__main__":
    main()
