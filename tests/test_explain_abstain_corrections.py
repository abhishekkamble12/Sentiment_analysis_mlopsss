"""Phase 2: per-term explanations, abstaining on low confidence, and user label corrections."""
import asyncio

import numpy as np
import pytest
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from conftest import StubProbaModel
from tweet_sentiment_classifier.components.data_transformation import clean_text
from tweet_sentiment_classifier.entity.models import LabelCorrection
from tweet_sentiment_classifier.evaluation.abstain import choose_threshold
from tweet_sentiment_classifier.pipeline.explain import LinearExplainer, find_spans

TEXTS = ["great speech", "great work well done", "love this great idea", "worst policy ever",
         "terrible awful policy", "worst idea terrible", "the meeting is today", "policy meeting at noon",
         "today is monday"] * 3
LABELS = [1, 1, 1, -1, -1, -1, 0, 0, 0] * 3


@pytest.fixture(scope="module")
def vectorizer():
    return TfidfVectorizer(ngram_range=(1, 2)).fit(TEXTS)


def test_contributions_sum_to_the_linear_score(vectorizer):
    model = LogisticRegression().fit(vectorizer.transform(TEXTS), LABELS)
    e = LinearExplainer(model, vectorizer)
    x = vectorizer.transform(["worst speech ever"])
    for label in (-1, 0, 1):
        _, contrib = e.contributions(x, label)
        c = list(model.classes_).index(label)
        assert contrib.sum() + model.intercept_[c] == pytest.approx(model.decision_function(x)[0, c])
    assert e.explain("worst speech ever", "worst speech ever", -1)["method"] == "linear_contributions"


def test_calibrated_svc_explanation_points_at_the_right_words(vectorizer):
    model = CalibratedClassifierCV(LinearSVC(), cv=3).fit(vectorizer.transform(TEXTS), LABELS)
    text = "Worst policy ever, but the speech was GREAT!"
    out = LinearExplainer(model, vectorizer).explain(text, clean_text(text), -1)
    assert out["method"] == "linear_contributions_approx"
    assert out["supporting"][0]["term"] == "worst"
    assert "great" in [t["term"] for t in out["opposing"]]
    great = next(t for t in out["opposing"] if t["term"] == "great")
    assert [text[s:e] for s, e in great["spans"]] == ["GREAT"]


def test_spans_map_back_through_punctuation_case_and_urls():
    text = "Don't go https://t.co/x — WORST, policy!"
    assert [text[s:e] for s, e in find_spans(text, "dont")] == ["Don't"]
    assert [text[s:e] for s, e in find_spans(text, "worst policy")] == ["WORST, policy"]
    assert find_spans(text, "go") == [[6, 8]]
    assert find_spans("policymaker", "policy") == []  # whole tokens only


def test_no_explanation_for_a_non_linear_model(vectorizer):
    forest = RandomForestClassifier(n_estimators=3).fit(vectorizer.transform(TEXTS), LABELS)
    assert LinearExplainer.maybe(forest, vectorizer) is None


def run(pipeline, text="some tweet"):
    return asyncio.run(pipeline.predict(text))


def test_pipeline_returns_contributions_only_for_the_explained_local_decision(make_pipeline):
    class StubExplainer:
        def explain(self, text, cleaned, label):
            return {"label": label}

    p = make_pipeline("local", StubProbaModel([0.1, 0.2, 0.7]))
    assert run(p)["contributions"] is None  # no explainer: null, never a fake one
    p.explainer = StubExplainer()
    assert run(p)["contributions"] == {"label": 1}

    llm = make_pipeline("llm", StubProbaModel([0.1, 0.2, 0.7]),
                        llm={"label": -1, "explanation": "x", "source": "llm:groq/x"})
    llm.explainer = StubExplainer()
    assert run(llm)["contributions"] is None


def test_abstains_below_threshold_only(make_pipeline):
    p = make_pipeline("local", StubProbaModel([0.1, 0.2, 0.7]))
    p.abstain_threshold = 0.8
    r = run(p)
    assert r["prediction"] == "uncertain" and r["confidence"] == pytest.approx(0.7)
    assert "abstained_low_confidence" in r["warnings"]
    p.abstain_threshold = 0.6
    assert run(p)["prediction"] == "positive"


def test_hybrid_llm_decision_is_never_abstained(make_pipeline):
    p = make_pipeline("hybrid", StubProbaModel([0.1, 0.2, 0.7]), threshold=0.75,
                      llm={"label": -1, "explanation": "x", "source": "llm:groq/x"})
    p.abstain_threshold = 0.8
    assert run(p)["prediction"] == "negative"


def test_choose_threshold_keeps_the_most_predictions_that_reach_the_target():
    conf = np.array([0.99, 0.95, 0.9, 0.8, 0.7, 0.6])
    correct = np.array([1, 1, 1, 0, 1, 0], bool)
    r = choose_threshold(conf, correct, 0.75)
    assert r["threshold"] == 0.7 and r["coverage"] == 0.8333 and r["kept_accuracy"] == 0.8
    assert choose_threshold(conf, np.zeros(6, bool), 0.5)["threshold"] is None


def test_correction_is_stored_and_validated(env):
    client, Session = env
    ok = client.post("/api/v1/corrections", json={"text": "not bad at all", "predicted": "negative",
                                                  "corrected": "positive", "decision_source": "local:linearsvc"})
    assert ok.status_code == 201 and ok.json()["status"] == "stored"
    with Session() as db:
        row = db.get(LabelCorrection, ok.json()["id"])
        assert (row.corrected, row.reviewed) == ("positive", False)

    bad = [{"text": "x", "predicted": "negative", "corrected": "angry"},
           {"text": "", "predicted": "negative", "corrected": "positive"},
           {"text": "x" * 5000, "predicted": "negative", "corrected": "positive"},
           {"text": "x", "predicted": "positive", "corrected": "positive"}]
    assert [client.post("/api/v1/corrections", json=b).status_code for b in bad] == [422, 422, 422, 400]
