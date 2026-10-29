"""Evaluation module and baselines: metrics are checked against scikit-learn, not against themselves."""
import json
import random
import sys
import time
import types

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score, precision_recall_fscore_support

from tweet_sentiment_classifier.evaluation import run as runner
from tweet_sentiment_classifier.evaluation.metrics import (
    bootstrap_ci, evaluate, expected_calibration_error, fingerprint, measure_latency, save_run,
)
from tweet_sentiment_classifier.models.factory import SKLEARN_MODELS, build_classifier
from tweet_sentiment_classifier.models.lexicon import MajorityClassifier, TextBlobClassifier, VaderClassifier


def random_labels(n=500, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.choice([-1, 0, 1], size=n, p=[0.2, 0.35, 0.45])
    noisy = np.where(rng.random(n) < 0.3, rng.choice([-1, 0, 1], size=n), y)
    return y, noisy


# ---- metrics vs scikit-learn ------------------------------------------------------------------

def test_metrics_match_scikit_learn():
    y, p = random_labels()
    m = evaluate(y, p)
    assert m["n"] == 500
    assert m["accuracy"] == pytest.approx(accuracy_score(y, p))
    assert m["macro_f1"] == pytest.approx(f1_score(y, p, average="macro"))
    assert m["weighted_f1"] == pytest.approx(f1_score(y, p, average="weighted"))
    assert m["kappa"] == pytest.approx(cohen_kappa_score(y, p))
    prec, rec, f1, sup = precision_recall_fscore_support(y, p, labels=[-1, 0, 1], zero_division=0)
    for i, name in enumerate(["negative", "neutral", "positive"]):
        pc = m["per_class"][name]
        assert (pc["precision"], pc["recall"], pc["f1"], pc["support"]) == pytest.approx((prec[i], rec[i], f1[i], sup[i]))


def test_class_that_is_never_predicted_scores_zero_not_nan():
    m = evaluate([-1, 0, 1, 1], [0, 0, 1, 1])
    assert m["per_class"]["negative"]["f1"] == 0.0 and not np.isnan(m["macro_f1"])


def test_float_labels_from_csv_are_accepted_and_bad_labels_rejected():
    assert evaluate([-1.0, 0.0, 1.0], [-1, 0, 1])["accuracy"] == 1.0
    with pytest.raises(ValueError, match="outside"):
        evaluate([0, 1, 2], [0, 1, 1])
    with pytest.raises(ValueError, match="integers"):
        evaluate([0.5], [0])
    with pytest.raises(ValueError, match="rows"):
        evaluate([0, 1], [0])


# ---- calibration ------------------------------------------------------------------------------

def test_ece_is_near_zero_when_confidence_matches_accuracy_and_large_when_overconfident():
    n = 1000
    y = np.zeros(n, dtype=int)
    proba = np.tile([0.1, 0.8, 0.1], (n, 1))                # always predicts class 0 with 0.8 confidence
    wrong = np.arange(n) % 5 == 0                            # 80% correct -> calibrated
    y_cal = np.where(wrong, 1, 0)
    assert expected_calibration_error(y_cal, proba) == pytest.approx(0.0, abs=1e-9)
    over = np.tile([0.005, 0.99, 0.005], (n, 1))
    y_half = np.where(np.arange(n) % 2 == 0, 0, 1)           # 50% correct at 99% confidence
    assert expected_calibration_error(y_half, over) == pytest.approx(0.49, abs=0.01)


def test_ece_weights_each_confidence_bin_by_its_share_of_samples():
    # half the samples: confidence 0.9, always right (gap 0.1); half: confidence 0.6, right 60% (gap 0)
    proba = np.vstack([np.tile([0.05, 0.9, 0.05], (100, 1)), np.tile([0.2, 0.6, 0.2], (100, 1))])
    y = np.concatenate([np.zeros(100, dtype=int), np.where(np.arange(100) < 60, 0, 1)])
    assert expected_calibration_error(y, proba) == pytest.approx(0.5 * 0.1 + 0.5 * 0.0)


# ---- bootstrap --------------------------------------------------------------------------------

def test_bootstrap_ci_brackets_the_estimate_is_seeded_and_shrinks_with_more_data():
    y, p = random_labels(2000)
    point = f1_score(y, p, average="macro")
    lo, hi = bootstrap_ci(y, p, "macro_f1", n_boot=300, seed=1)
    assert lo < point < hi
    assert bootstrap_ci(y, p, "macro_f1", n_boot=300, seed=1) == (lo, hi)        # reproducible
    lo_s, hi_s = bootstrap_ci(y[:200], p[:200], "macro_f1", n_boot=300, seed=1)
    assert (hi_s - lo_s) > (hi - lo)                                              # less data -> wider


def test_bootstrap_perfect_predictions_and_unsupported_metric():
    y = [-1, 0, 1] * 50
    assert bootstrap_ci(y, y, "accuracy", n_boot=50) == (1.0, 1.0)
    with pytest.raises(ValueError):
        bootstrap_ci(y, y, "auc")


# ---- latency / fingerprint / saving --------------------------------------------------------------

def test_measure_latency_times_single_predictions():
    out = measure_latency(lambda t: time.sleep(0.002), ["x"] * 300, n=40, warmup=2)
    assert out["n"] == 40 and out["p50_ms"] >= 1.5 and out["p95_ms"] >= out["p50_ms"]


def test_fingerprint_is_stable_and_order_sensitive():
    assert fingerprint(["a", "b"]) == fingerprint(["a", "b"])
    assert fingerprint(["a", "b"]) != fingerprint(["b", "a"])


def fake_result(n=10, acc=0.5):
    m = evaluate([0] * n, [0] * n)
    m["macro_f1_ci"] = (0.1, 0.9)
    return m


def test_save_run_writes_both_files_and_rebuilds_from_scratch(tmp_path):
    save_run(tmp_path, "r1", {"k": 1}, {"A": fake_result(), "B": fake_result()})
    assert len(pd.read_csv(tmp_path / "results.csv")) == 2
    save_run(tmp_path, "r2", {"k": 1}, {"C": fake_result()})                      # not appended to the old table
    table = pd.read_csv(tmp_path / "results.csv")
    assert list(table["model"]) == ["C"]
    assert json.loads((tmp_path / "results.json").read_text())["run_id"] == "r2"


def test_save_run_refuses_models_scored_on_different_rows(tmp_path):
    with pytest.raises(ValueError, match="different numbers of rows"):
        save_run(tmp_path, "r", {}, {"A": fake_result(10), "B": fake_result(11)})


# ---- baselines --------------------------------------------------------------------------------------

def test_majority_classifier():
    m = MajorityClassifier().fit(["a", "b", "c", "d"], [1, 1, 0, -1])
    assert m.predict(["x", "y", "z"]).tolist() == [1, 1, 1]


@pytest.fixture
def stub_lexicons(monkeypatch):
    polarity = {"good": 0.7, "bad": -0.4, "flat": 0.0}

    class TextBlob:
        def __init__(self, text):
            self.sentiment = types.SimpleNamespace(polarity=polarity.get(text, 0.0))

    compound = {"good": 0.05, "bad": -0.05, "meh": 0.049, "awful": -0.049, "great": 0.9}

    class SentimentIntensityAnalyzer:
        def polarity_scores(self, text):
            return {"compound": compound.get(text, 0.0)}

    pkg = types.ModuleType("vaderSentiment")
    sub = types.ModuleType("vaderSentiment.vaderSentiment")
    sub.SentimentIntensityAnalyzer = SentimentIntensityAnalyzer
    tb = types.ModuleType("textblob")
    tb.TextBlob = TextBlob
    for name, mod in (("textblob", tb), ("vaderSentiment", pkg), ("vaderSentiment.vaderSentiment", sub)):
        monkeypatch.setitem(sys.modules, name, mod)


def test_textblob_baseline_is_sign_of_polarity_with_exact_zero_neutral(stub_lexicons):
    assert TextBlobClassifier().predict(["good", "bad", "flat"]).tolist() == [1, -1, 0]


def test_vader_baseline_uses_the_conventional_005_band(stub_lexicons):
    # inclusive at +/-0.05, neutral just inside it
    assert VaderClassifier().predict(["good", "bad", "meh", "awful", "great"]).tolist() == [1, -1, 0, 0, 1]


def test_real_lexicon_libraries_agree_on_obvious_sentences():
    pytest.importorskip("textblob")
    pytest.importorskip("vaderSentiment")
    texts = ["i love this wonderful excellent day", "this is terrible awful and horrible"]
    assert TextBlobClassifier().predict(texts).tolist() == [1, -1]
    assert VaderClassifier().predict(texts).tolist() == [1, -1]


# ---- factory ------------------------------------------------------------------------------------------

@pytest.mark.parametrize("name", [m for m in SKLEARN_MODELS if m != "XGBoost"])
def test_every_factory_model_trains_and_exposes_probabilities(name):
    from sklearn.feature_extraction.text import TfidfVectorizer
    texts, labels = toy_corpus(n=36)
    X = TfidfVectorizer().fit_transform(texts)
    clf = build_classifier(name, {"max_iter": 500}, 42).fit(X, labels)
    proba = clf.predict_proba(X)
    assert proba.shape == (len(texts), 3) and np.allclose(proba.sum(1), 1.0)


def test_unknown_classifier_rejected():
    with pytest.raises(ValueError):
        build_classifier("Nope", {}, 0)


# ---- end-to-end runner -----------------------------------------------------------------------------------

WORDS = {-1: "terrible awful corrupt failure angry disaster worst hate".split(),
         0: "meeting report today scheduled announced minister city date".split(),
         1: "great excellent happy success proud wonderful best love".split()}


def toy_corpus(seed=0, n=300):
    rng = random.Random(seed)
    texts, labels = [], []
    for i in range(n):
        label = (-1, 0, 1)[i % 3]
        texts.append(" ".join(rng.sample(WORDS[label], 4)) + f" tok{seed}x{i}")
        labels.append(label)
    return texts, labels


def write_csv(path, texts, labels):
    pd.DataFrame({"text": texts, "sentiment": [float(x) for x in labels]}).to_csv(path, index=False)


def test_runner_produces_one_comparable_table(tmp_path):
    tr_x, tr_y = toy_corpus(seed=1)
    te_x, te_y = toy_corpus(seed=2, n=90)
    te_x[0] = tr_x[0]                                           # plant one train/test overlap
    write_csv(tmp_path / "train.csv", tr_x, tr_y)
    write_csv(tmp_path / "test.csv", te_x, te_y)

    summary = runner.run(tmp_path / "train.csv", tmp_path / "test.csv", tmp_path / "out",
                         models=("Majority", "LogisticRegression", "LinearSVC", "RidgeClassifier"), n_boot=100)

    table = pd.read_csv(tmp_path / "out" / "results.csv")
    assert list(table["model"]) == ["Majority", "LogisticRegression", "LinearSVC", "RidgeClassifier"]
    assert set(table["n"]) == {90}                                           # every model scored on the same rows
    assert summary["split"]["test_texts_also_in_train"] == 1                  # leakage is measured and reported
    assert summary["split"]["test_fingerprint"] == fingerprint(te_x)
    res = summary["results"]
    assert res["Majority"]["accuracy"] == pytest.approx(1 / 3)
    assert res["LogisticRegression"]["macro_f1"] > 0.9
    for name in ("LogisticRegression", "LinearSVC", "RidgeClassifier"):
        lo, hi = res[name]["macro_f1_ci"]
        assert lo <= res[name]["macro_f1"] <= hi
        assert 0.0 <= res[name]["ece"] <= 1.0 and res[name]["latency"]["p50_ms"] > 0
    assert "ece" not in res["Majority"] and "latency" not in res["Majority"]


def test_runner_cli_prints_the_table(tmp_path, capsys):
    tr_x, tr_y = toy_corpus(seed=1)
    te_x, te_y = toy_corpus(seed=2, n=60)
    write_csv(tmp_path / "train.csv", tr_x, tr_y)
    write_csv(tmp_path / "test.csv", te_x, te_y)
    runner.main(["--train", str(tmp_path / "train.csv"), "--test", str(tmp_path / "test.csv"),
                 "--out", str(tmp_path / "o"), "--models", "Majority", "LogisticRegression",
                 "--params", str(tmp_path / "missing.yaml")])
    out = capsys.readouterr().out
    assert "macroF1" in out and "LogisticRegression" in out and "overlap_with_train" in out
    assert (tmp_path / "o" / "results.csv").exists()
