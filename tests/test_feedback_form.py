"""Tests for the Google Form feedback UI and webhook endpoint."""
import pytest
from conftest import LLM, LOCAL, FakePipeline
from tweet_sentiment_classifier.entity.models import PredictionLog


def test_feedback_page_renders(env):
    client, _ = env
    res = client.get("/feedback")
    assert res.status_code == 200
    assert "Google Form" in res.text
    assert "/api/form-submission" in res.text


def test_form_submission_success(env):
    client, Session = env
    payload = {
        "feedback_text": "The new update is fantastic and makes everything so much faster!",
        "respondent": "test_user@example.com"
    }
    res = client.post("/api/form-submission", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "success"
    assert data["sentiment"] == "positive"
    assert data["confidence"] == 0.9
    assert data["decision_source"] == "local:logisticregression"
    assert data["respondent"] == "test_user@example.com"

    # Verify database persistence
    with Session() as db:
        log = db.query(PredictionLog).one()
        assert log.prediction == "positive"
        assert log.model_used == "local:logisticregression"
        assert log.confidence == 0.9


def test_form_submission_empty_rejected(env):
    client, _ = env
    res = client.post("/api/form-submission", json={"feedback_text": "   "})
    assert res.status_code == 400
    assert "cannot be empty" in res.json()["detail"].lower()


def test_form_submission_overlong_rejected(env):
    client, _ = env
    overlong = "a" * 2005
    res = client.post("/api/form-submission", json={"feedback_text": overlong})
    assert res.status_code == 413
    assert "too long" in res.json()["detail"].lower()


def test_form_submission_with_llm_null_confidence(env, app_module, monkeypatch):
    client, Session = env
    monkeypatch.setattr(app_module, "pipeline", FakePipeline(default=LLM))

    payload = {"feedback_text": "Terrible service and completely broken."}
    res = client.post("/api/form-submission", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["sentiment"] == "negative"
    assert data["confidence"] is None
    assert data["decision_source"].startswith("llm:")

    with Session() as db:
        log = db.query(PredictionLog).one()
        assert log.confidence is None
        assert log.prediction == "negative"
