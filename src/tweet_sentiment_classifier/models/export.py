"""Train the TF-IDF models on the training split and write the artifacts the app serves.

    python -m tweet_sentiment_classifier.models.export --train artifacts/data_transformation/train.csv

One shared TF-IDF vectorizer is fitted once and every classifier is trained on its features, so the single
`preprocessor.pkl` the app loads is correct for every model file.
"""
import argparse
from pathlib import Path
from typing import Any, Dict, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from tweet_sentiment_classifier.models.factory import build_classifier

DEFAULT_MODELS = ("LogisticRegression", "LinearSVC", "MultinomialNB", "RidgeClassifier")


def export_models(train_csv: Path, models: Sequence[str] = DEFAULT_MODELS, params: Dict[str, Any] = None,
                  model_dir: Path = Path("artifacts/model_trainer"),
                  preprocessor_path: Path = Path("artifacts/data_transformation/preprocessor.pkl"),
                  seed: int = 42, max_features: int = 15000) -> Dict[str, Path]:
    params = params or {}
    train = pd.read_csv(train_csv).dropna(subset=["text"])
    texts = train["text"].astype(str)
    y = train["sentiment"].astype(int).to_numpy()

    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2))
    features = vectorizer.fit_transform(texts)
    Path(preprocessor_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(vectorizer, preprocessor_path)

    Path(model_dir).mkdir(parents=True, exist_ok=True)
    written = {}
    for name in models:
        clf = build_classifier(name, params.get(name, {}), seed)
        clf.fit(features, y + 1 if name == "XGBoost" else y)
        if not hasattr(clf, "predict_proba"):
            raise RuntimeError(f"{name} has no predict_proba; the app cannot serve it")
        path = Path(model_dir) / f"model_{name}.pkl"
        joblib.dump(clf, path)
        written[name] = path
        print(f"wrote {path}")
    return written


def main(argv: Sequence[str] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train", type=Path, default=Path("artifacts/data_transformation/train.csv"))
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument("--params", type=Path, default=Path("params.yaml"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    import yaml
    params = yaml.safe_load(args.params.read_text(encoding="utf-8")) if args.params.exists() else {}
    export_models(args.train, args.models, params, seed=args.seed, max_features=int(params.get("max_features", 15000)))


if __name__ == "__main__":
    main()
