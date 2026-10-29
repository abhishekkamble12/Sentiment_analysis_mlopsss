"""Every fast-tier model type must serve a label in {-1,0,1} with a real probability (audit B3)."""
import numpy as np
import pandas as pd
import pytest
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from conftest import StubProbaModel
from tweet_sentiment_classifier.config.config_entity import ModelTrainerConfig

NEG = ["terrible awful failure", "awful corrupt disaster", "horrible failure angry", "bad disaster corrupt"]
NEU = ["meeting scheduled tomorrow", "report published today", "minister visits city", "election date announced"]
POS = ["great excellent win", "wonderful success happy", "excellent progress proud", "happy great success"]


def corpus(reps=3):
    texts = (NEG + NEU + POS) * reps
    labels = ([-1] * len(NEG) + [0] * len(NEU) + [1] * len(POS)) * reps
    return texts, labels


def fitted(model):
    texts, labels = corpus()
    vec = TfidfVectorizer()
    return model.fit(vec.fit_transform(texts), labels), vec


MODELS = {
    "LogisticRegression": lambda: LogisticRegression(max_iter=500),
    "MultinomialNB": lambda: MultinomialNB(),
    "LinearSVC": lambda: CalibratedClassifierCV(LinearSVC(), cv=3),
    "RidgeClassifier": lambda: CalibratedClassifierCV(RidgeClassifier(), cv=3),
}


@pytest.mark.parametrize("name", MODELS)
def test_each_model_type_serves_label_and_probability(make_pipeline, name):
    model, vec = fitted(MODELS[name]())
    p = make_pipeline("local", model)
    p.vectorizer = vec
    for text, expected in [("great excellent success", 1), ("terrible awful disaster", -1)]:
        label, conf = p._predict_fast(text)
        assert label == expected
        assert 0.0 < conf <= 1.0


def test_uncalibrated_margin_model_is_rejected_not_given_a_made_up_confidence(make_pipeline):
    model, vec = fitted(LinearSVC())
    p = make_pipeline("local", model)
    p.vectorizer = vec
    with pytest.raises(RuntimeError, match="retrain"):
        p._predict_fast("great excellent success")


def test_xgboost_style_zero_based_classes_are_mapped_back(make_pipeline):
    # XGBoost is trained on labels shifted to 0..2; serving must map back to -1..1.
    p = make_pipeline("local", StubProbaModel([0.1, 0.1, 0.8], classes=(0, 1, 2)))
    assert p._predict_fast("x")[0] == 1
    p = make_pipeline("local", StubProbaModel([0.8, 0.1, 0.1], classes=(0, 1, 2)))
    assert p._predict_fast("x")[0] == -1


@pytest.mark.parametrize("name", ["LinearSVC", "RidgeClassifier"])
def test_trainer_produces_servable_margin_models(make_pipeline, tmp_path, monkeypatch, name):
    from tweet_sentiment_classifier.components.model_trainer import ModelTrainer
    import joblib

    monkeypatch.chdir(tmp_path)  # the trainer writes the vectorizer to a relative artifacts/ path
    texts, labels = corpus()
    train_csv = tmp_path / "train.csv"
    pd.DataFrame({"text": texts, "sentiment": labels}).to_csv(train_csv, index=False)
    out = tmp_path / f"model_{name}.pkl"

    ModelTrainer(ModelTrainerConfig(
        root_dir=tmp_path, trained_model_path=out, train_data_path=train_csv, test_data_path=train_csv,
        model_name=name, random_state=42, max_features=1000, hyperparams={},
    )).train()

    p = make_pipeline("local", joblib.load(out))
    p.vectorizer = joblib.load(tmp_path / "artifacts" / "data_transformation" / "preprocessor.pkl")
    label, conf = p._predict_fast("great excellent success")
    assert label == 1 and 0.0 < conf <= 1.0
    assert np.isfinite(conf)
