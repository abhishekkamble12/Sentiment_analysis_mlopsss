"""Deployed models come from the clean split, and the selector never picks a lexicon baseline."""
import json

import joblib
import numpy as np
import pandas as pd
import pytest

from tweet_sentiment_classifier.components.best_model_selector import BestModelSelector
from tweet_sentiment_classifier.config.config_entity import BestModelConfig
from tweet_sentiment_classifier.models.export import export_models

WORDS = {-1: "awful corrupt failure angry hate".split(), 0: "report meeting today city date".split(),
         1: "great happy success proud love".split()}


def write_train(path, n=90):
    rng = np.random.default_rng(0)
    ys = [(-1, 0, 1)[i % 3] for i in range(n)]
    texts = [" ".join(rng.choice(WORDS[y], 3)) + f" t{i}" for i, y in enumerate(ys)]
    pd.DataFrame({"text": texts, "sentiment": [float(y) for y in ys]}).to_csv(path, index=False)


def test_export_writes_servable_models_sharing_one_vectorizer(tmp_path):
    write_train(tmp_path / "train.csv")
    written = export_models(tmp_path / "train.csv", params={"LogisticRegression": {"C": 4.0, "max_iter": 500}},
                            model_dir=tmp_path / "models", preprocessor_path=tmp_path / "pre.pkl")
    vec = joblib.load(tmp_path / "pre.pkl")
    X = vec.transform(["great happy success", "awful corrupt hate"])
    assert set(written) == {"LogisticRegression", "LinearSVC", "MultinomialNB", "RidgeClassifier"}
    for name, path in written.items():
        model = joblib.load(path)
        proba = model.predict_proba(X)                       # every model fits the shared vectorizer
        assert list(model.classes_[proba.argmax(1)]) == [1, -1], name
    assert joblib.load(written["LogisticRegression"]).C == 4.0  # params are applied


COLUMNS = ["model", "n", "accuracy", "macro_f1", "macro_f1_ci_low", "macro_f1_ci_high", "weighted_f1", "kappa",
           "ece", "f1_negative", "f1_neutral", "f1_positive", "latency_p50_ms", "latency_p95_ms"]


def results(tmp_path, rows):
    p = tmp_path / "results.csv"
    pd.DataFrame([dict(zip(COLUMNS, r)) for r in rows]).to_csv(p, index=False)
    (tmp_path / "results.json").write_text(json.dumps({"run_id": "r-42"}))
    return p


def selector(tmp_path, results_csv):
    cfg = BestModelConfig(root_dir=tmp_path, model_path=tmp_path / "best" / "best_model.pkl",
                          metadata_path=tmp_path / "best" / "meta.json", metrics_dir=tmp_path)
    (tmp_path / "models").mkdir(exist_ok=True)
    for name in ("LinearSVC", "LogisticRegression"):
        (tmp_path / "models" / f"model_{name}.pkl").write_bytes(name.encode())
    return BestModelSelector(cfg, results_csv=results_csv, model_dir=tmp_path / "models"), cfg


def row(name, f1, lat):
    return [name, 100, f1, f1, f1 - 0.01, f1 + 0.01, f1, f1, 0.05, f1, f1, f1, lat, lat]


def test_selector_ignores_baselines_even_when_they_score_higher(tmp_path):
    csv = results(tmp_path, [row("TextBlob", 0.999, 0.3), row("Majority", 0.2, None),
                             row("LinearSVC", 0.93, 4.0), row("LogisticRegression", 0.92, 1.0)])
    sel, cfg = selector(tmp_path, csv)
    assert sel.select_best_model() == "LinearSVC"
    meta = json.loads(cfg.metadata_path.read_text())
    assert meta["tier_1_fast_fallback"] == "LinearSVC" and meta["run_id"] == "r-42"
    assert cfg.model_path.read_bytes() == b"LinearSVC"


def test_selector_breaks_ties_on_latency(tmp_path):
    csv = results(tmp_path, [row("LinearSVC", 0.93, 4.0), row("LogisticRegression", 0.93, 1.0)])
    sel, _ = selector(tmp_path, csv)
    assert sel.select_best_model() == "LogisticRegression"


def test_selector_fails_clearly_without_trained_models_or_results(tmp_path):
    sel, _ = selector(tmp_path, results(tmp_path, [row("TextBlob", 0.99, 0.3)]))
    with pytest.raises(ValueError, match="only baselines"):
        sel.select_best_model()
    with pytest.raises(FileNotFoundError):
        selector(tmp_path, tmp_path / "missing.csv")[0].select_best_model()
