"""Choose ABSTAIN_THRESHOLD on the validation split (the test split is not read).

    python -m tweet_sentiment_classifier.evaluation.abstain --val .../val.csv [--target-accuracy 0.97]

The threshold is the lowest confidence at which the predictions kept (confidence >= threshold) reach the
target accuracy, i.e. the one that keeps the most predictions. Labels are the silver ones the model was
trained on, so "accuracy" here is agreement with TextBlob, not with people.
"""
import argparse
import json
from pathlib import Path
from typing import Any, Dict, Sequence

import joblib
import numpy as np
import pandas as pd


def choose_threshold(confidence: np.ndarray, correct: np.ndarray, target_accuracy: float) -> Dict[str, Any]:
    confidence, correct = np.asarray(confidence, float), np.asarray(correct, bool)
    order = np.argsort(-confidence, kind="stable")
    kept_accuracy = np.cumsum(correct[order]) / np.arange(1, len(order) + 1)
    ok = np.flatnonzero(kept_accuracy >= target_accuracy)
    if len(ok) == 0:
        return {"threshold": None, "target_accuracy": target_accuracy, "coverage": 0.0, "kept_accuracy": None}
    threshold = float(confidence[order][ok[-1]])
    kept = confidence >= threshold  # ties at the threshold are kept, so recompute on the real mask
    return {"threshold": round(threshold, 4), "target_accuracy": target_accuracy,
            "coverage": round(float(kept.mean()), 4), "kept_accuracy": round(float(correct[kept].mean()), 4),
            "overall_accuracy": round(float(correct.mean()), 4), "n": int(len(correct))}


def main(argv: Sequence[str] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--val", type=Path, required=True, help="validation CSV with text and sentiment columns")
    parser.add_argument("--model", type=Path, default=Path("artifacts/best_model/best_model.pkl"))
    parser.add_argument("--preprocessor", type=Path, default=Path("artifacts/data_transformation/preprocessor.pkl"))
    parser.add_argument("--target-accuracy", type=float, default=0.97)
    parser.add_argument("--out", type=Path, default=Path("artifacts/evaluation/abstain.json"))
    args = parser.parse_args(argv)

    val = pd.read_csv(args.val).dropna(subset=["text", "sentiment"])
    model, vectorizer = joblib.load(args.model), joblib.load(args.preprocessor)
    proba = model.predict_proba(vectorizer.transform(val["text"].astype(str)))
    pred = np.asarray(model.classes_)[proba.argmax(axis=1)]
    result = choose_threshold(proba.max(axis=1), pred == val["sentiment"].astype(int).to_numpy(),
                              args.target_accuracy)
    result.update({"split": "validation", "labels": "silver (TextBlob-generated)", "model": str(args.model)})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
