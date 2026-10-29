"""Tests for the JSON endpoints the React frontend consumes (added without breaking the legacy ones)."""
import io
import json

import pytest

from conftest import LOCAL
from tweet_sentiment_classifier.entity.models import PredictionLog


# ---- /api/predict --------------------------------------------------------------------------

def test_api_predict_uses_json_body_and_stores_text(env):
    client, Session = env
    r = client.post("/api/predict", json={"text": "this is great"})
    assert r.status_code == 200
    body = r.json()
    assert body["prediction"] == "positive" and body["decision_source"] == "local:logisticregression"
    with Session() as db:
        row = db.query(PredictionLog).one()
        assert row.text == "this is great"


def test_api_predict_rejects_empty_and_overlong(env, app_module, monkeypatch):
    client, _ = env
    assert client.post("/api/predict", json={"text": "   "}).status_code == 400
    monkeypatch.setattr(app_module.settings, "MAX_TEXT_CHARS", 10)
    assert client.post("/api/predict", json={"text": "x" * 11}).status_code == 413


# ---- /api/upload-csv ------------------------------------------------------------------------

def test_api_upload_csv_returns_distribution_and_errors_and_logs_text(env, app_module, monkeypatch):
    client, Session = env
    monkeypatch.setattr(app_module, "report_agent", None)
    body = io.BytesIO(b"text\nhello there\nbad cell here\n")
    r = client.post("/api/upload-csv", files={"file": ("t.csv", body, "text/csv")})
    assert r.status_code == 200
    data = r.json()
    assert data["total_processed"] == 2
    assert set(data["distribution"]) == {"positive", "neutral", "negative"}
    assert data["distribution"]["positive"] == 2  # FakePipeline always returns positive
    assert data["errors"] == []
    with Session() as db:
        rows = db.query(PredictionLog).all()
        assert {row.text for row in rows} == {"hello there", "bad cell here"}


def test_api_upload_csv_validates_text_column(env):
    client, _ = env
    r = client.post("/api/upload-csv", files={"file": ("t.csv", io.BytesIO(b"a,b\n1,2\n"), "text/csv")})
    assert r.status_code == 400 and "text" in r.json()["detail"]


# ---- /api/prediction-history ----------------------------------------------------------------

def test_prediction_history_returns_text_and_pagination(env):
    client, Session = env
    for t in ("alpha", "beta", "gamma"):
        client.post("/api/predict", json={"text": t})
    body = client.get("/api/prediction-history", params={"limit": 2, "page": 1}).json()
    assert body["total"] == 3 and body["page"] == 1 and body["limit"] == 2 and body["total_pages"] == 2
    assert len(body["logs"]) == 2 and body["logs"][0]["text"] in ("alpha", "beta", "gamma")
    assert all(log["prediction"] == "positive" for log in body["logs"])  # normalised from FakePipeline


def test_prediction_history_filters_by_sentiment_and_search(env):
    client, Session = env
    with Session() as db:
        db.add_all([
            PredictionLog(latency_ms=5, confidence=0.9, model_used="local:logisticregression",
                          prediction="positive", text_length=4, text="happy day"),
            PredictionLog(latency_ms=5, confidence=0.9, model_used="local:logisticregression",
                          prediction="negative", text_length=4, text="sad news"),
            PredictionLog(latency_ms=5, confidence=0.9, model_used="local:logisticregression",
                          prediction="neutral", text_length=4, text="meh whatever"),
        ])
        db.commit()

    neg = client.get("/api/prediction-history", params={"sentiment": "negative"}).json()
    assert neg["total"] == 1 and neg["logs"][0]["text"] == "sad news"

    search = client.get("/api/prediction-history", params={"q": "day"}).json()
    assert search["total"] == 1 and search["logs"][0]["text"] == "happy day"


def test_prediction_history_date_filter_is_inclusive(env):
    client, Session = env
    with Session() as db:
        from datetime import datetime
        db.add_all([
            PredictionLog(latency_ms=5, confidence=0.9, model_used="local:x",
                          prediction="positive", text_length=1, text="a", timestamp=datetime(2026, 10, 29, 9)),
            PredictionLog(latency_ms=5, confidence=0.9, model_used="local:x",
                          prediction="positive", text_length=1, text="b", timestamp=datetime(2026, 10, 30, 9)),
        ])
        db.commit()
    only_29 = client.get("/api/prediction-history", params={"start": "2026-10-29", "end": "2026-10-29"}).json()
    assert only_29["total"] == 1 and only_29["logs"][0]["text"] == "a"


def test_prediction_history_rejects_bad_sentiment_and_date(env):
    client, _ = env
    assert client.get("/api/prediction-history", params={"sentiment": "angry"}).status_code == 400
    assert client.get("/api/prediction-history", params={"start": "not-a-date"}).status_code == 400


def test_prediction_history_old_null_text_rows_render(env):
    client, Session = env
    # rows written before the text column existed have NULL text
    with Session() as db:
        db.add(PredictionLog(latency_ms=5, confidence=0.9, model_used="local:x",
                             prediction="positive", text_length=4))
        db.commit()
    body = client.get("/api/prediction-history").json()
    assert body["total"] == 1 and body["logs"][0]["text"] is None


# ---- /api/model-info -----------------------------------------------------------------------

def test_model_info_reports_served_model(env, tmp_path, monkeypatch):
    client, _ = env
    monkeypatch.chdir(tmp_path)
    (tmp_path / "artifacts" / "best_model").mkdir(parents=True)
    (tmp_path / "artifacts" / "best_model" / "best_model_metadata.json").write_text(
        json.dumps({"best_model_name": "LinearSVC", "tier_1_fast_fallback": "LinearSVC",
                    "metrics": {"macro_f1": 0.93}, "run_id": "r1"}))
    info = client.get("/api/model-info").json()
    assert info["pipeline_ready"] is True
    assert info["best_model_name"] == "LinearSVC"
    assert info["inference_mode"] == "local"  # FakePipeline.inference_mode
    assert info["transformer_enabled"] is False  # FakePipeline has no db_model
    assert info["label_mapping"]["1"] == "positive"
    assert info["best_metrics"]["macro_f1"] == 0.93


def test_model_info_when_pipeline_not_initialised(env, app_module, monkeypatch, tmp_path):
    client, _ = env
    monkeypatch.setattr(app_module, "pipeline", None)
    monkeypatch.chdir(tmp_path)
    info = client.get("/api/model-info").json()
    assert info["pipeline_ready"] is False and info["inference_mode"] is None


# ---- /api/model-lab ------------------------------------------------------------------------

def test_model_lab_reads_real_artifacts_with_per_class_and_confusion(env, tmp_path, monkeypatch):
    client, _ = env
    monkeypatch.chdir(tmp_path)
    d = tmp_path / "artifacts" / "evaluation"
    d.mkdir(parents=True)
    run = {"run_id": "r1", "split": {"n_test": 10, "label_source": "silver", "test_texts_also_in_train_pct": 0.0},
           "results": {
               "LinearSVC": {"macro_f1": 0.93, "macro_f1_ci": [0.92, 0.94], "accuracy": 0.94, "weighted_f1": 0.94,
                             "kappa": 0.9, "ece": 0.05,
                             "per_class": {"negative": {"precision": 0.9, "recall": 0.9, "f1": 0.9, "support": 3}},
                             "confusion_matrix": [[3, 0, 0], [0, 4, 0], [0, 0, 3]],
                             "latency": {"n": 200, "p50_ms": 5.0, "p95_ms": 7.0, "mean_ms": 5.5}},
               "Majority": {"macro_f1": 0.2, "macro_f1_ci": [0.19, 0.21], "accuracy": 0.4, "weighted_f1": 0.27,
                            "kappa": 0.0}},
           "comparisons": [{"model_a": "LinearSVC", "model_b": "Majority", "macro_f1_diff": 0.73,
                             "ci_low": 0.7, "ci_high": 0.76, "p_bootstrap": 0.001}]}
    (d / "results.json").write_text(json.dumps(run))
    (tmp_path / "artifacts" / "best_model").mkdir()
    (tmp_path / "artifacts" / "best_model" / "best_model_metadata.json").write_text(
        json.dumps({"best_model_name": "LinearSVC"}))
    body = client.get("/api/model-lab").json()
    assert body["gold"] is None
    assert body["best_model_name"] == "LinearSVC"
    assert body["label_order"] == ["negative", "neutral", "positive"]
    assert "Silver" in body["limitation"]
    svc = {m["model"]: m for m in body["silver"]["models"]}["LinearSVC"]
    assert svc["macro_f1"] == 0.93 and svc["ci_low"] == 0.92 and svc["ece"] == 0.05
    assert svc["baseline"] is False
    assert svc["per_class"]["negative"]["f1"] == 0.9
    assert svc["confusion_matrix"] == [[3, 0, 0], [0, 4, 0], [0, 0, 3]]
    assert svc["latency"]["p50_ms"] == 5.0
    maj = {m["model"]: m for m in body["silver"]["models"]}["Majority"]
    assert maj["baseline"] is True and maj["ece"] is None
    assert body["silver"]["comparisons"][0]["macro_f1_diff"] == 0.73


def test_model_lab_empty_when_no_artifacts(env, tmp_path, monkeypatch):
    client, _ = env
    monkeypatch.chdir(tmp_path)
    body = client.get("/api/model-lab").json()
    assert body["silver"] is None and body["gold"] is None
