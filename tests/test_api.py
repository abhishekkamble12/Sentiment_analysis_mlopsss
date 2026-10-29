"""API behaviour with a fake pipeline: logging, monitoring, analytics and experiments report real data."""
import csv
import io
import json
from datetime import datetime, timedelta
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from tweet_sentiment_classifier.config.database import Base, get_db
from tweet_sentiment_classifier.entity.models import PredictionLog, ReportLog

REPO = Path(__file__).resolve().parents[1]

LOCAL = {"prediction": "positive", "confidence": 0.9, "decision_source": "local:logisticregression",
         "model_used": "local:logisticregression", "explanation": None}
LLM = {"prediction": "negative", "confidence": None, "decision_source": "llm:groq/llama-3.1-8b-instant",
       "model_used": "llm:groq/llama-3.1-8b-instant", "explanation": "angry"}


class FakePipeline:
    def __init__(self, by_text=None, default=LOCAL):
        self.by_text, self.default = by_text or {}, default

    async def predict(self, text):
        return dict(self.by_text.get(text, self.default))


@pytest.fixture(scope="module")
def app_module():
    import app as app_mod
    return app_mod


@pytest.fixture
def env(app_module, monkeypatch):
    engine = create_engine("sqlite://", poolclass=StaticPool, connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)

    def _get_db():
        db = Session()
        try:
            yield db
        finally:
            db.close()

    app_module.app.dependency_overrides[get_db] = _get_db
    app_module.predict_cache.clear()
    app_module.search_analysis_cache.clear()
    monkeypatch.setattr(app_module, "pipeline", FakePipeline())
    yield TestClient(app_module.app), Session
    app_module.app.dependency_overrides.clear()


def add_logs(Session, rows):
    with Session() as db:
        for r in rows:
            db.add(PredictionLog(latency_ms=r.get("latency_ms", 5.0), confidence=r.get("confidence", 0.9),
                                 model_used=r.get("model_used", "local:logisticregression"),
                                 prediction=r["prediction"], text_length=10, timestamp=r.get("timestamp")))
        db.commit()


# ---- /predict ------------------------------------------------------------------------------

def test_predict_logs_the_real_decision_source(env):
    client, Session = env
    r = client.post("/predict", params={"text": "good day"})
    assert r.status_code == 200 and r.json()["decision_source"] == "local:logisticregression"
    with Session() as db:
        row = db.query(PredictionLog).one()
        assert (row.model_used, row.confidence, row.prediction) == ("local:logisticregression", 0.9, "positive")


def test_predict_stores_null_confidence_for_llm_decisions(env, app_module, monkeypatch):
    client, Session = env
    monkeypatch.setattr(app_module, "pipeline", FakePipeline(default=LLM))
    assert client.post("/predict", params={"text": "awful"}).status_code == 200
    with Session() as db:
        row = db.query(PredictionLog).one()
        assert row.confidence is None and row.model_used.startswith("llm:")


def test_predict_rejects_blank_text(env):
    client, _ = env
    assert client.post("/predict", params={"text": "   "}).status_code == 400


# ---- /monitoring ---------------------------------------------------------------------------

def test_monitoring_is_computed_from_logged_predictions(env, app_module, monkeypatch):
    client, _ = env
    monkeypatch.setattr(app_module, "pipeline", FakePipeline(
        {"a": LOCAL, "b": {**LOCAL, "confidence": 0.7}, "c": LLM}))
    for t in "abc":
        client.post("/predict", params={"text": t})
    m = client.get("/monitoring").json()
    assert m["total_predictions"] == 3
    assert m["model_usage"] == {"local:logisticregression": 2, "llm:groq/llama-3.1-8b-instant": 1}
    assert m["llm_share"] == pytest.approx(1 / 3, abs=1e-4)
    assert m["avg_confidence"] == pytest.approx(0.8)  # the LLM row (NULL) is not counted as 0
    assert m["confidence_n"] == 2
    assert m["prediction_distribution"] == {"positive": 2, "negative": 1}
    assert "fallback_rate" not in m  # it used to be the RSS-search rate, mislabelled as the LLM rate


def test_monitoring_with_no_data_does_not_invent_numbers(env):
    client, _ = env
    m = client.get("/monitoring").json()
    assert m["total_predictions"] == 0 and m["avg_confidence"] is None
    assert m["model_usage"] == {} and m["llm_share"] == 0.0


def test_recent_logs_are_real_newest_first_and_limited(env):
    client, Session = env
    add_logs(Session, [{"prediction": "positive", "model_used": f"m{i}"} for i in range(5)])
    add_logs(Session, [{"prediction": "negative", "model_used": "llm:x", "confidence": None}])
    logs = client.get("/api/recent-logs", params={"limit": 3}).json()["logs"]
    assert [x["model_used"] for x in logs] == ["llm:x", "m4", "m3"]
    assert logs[0]["confidence"] is None and logs[0]["llm"] is True and logs[1]["llm"] is False
    assert len({x["id"] for x in logs}) == 3
    assert len(client.get("/api/recent-logs").json()["logs"]) == 6  # default limit (10) returns all rows


# ---- /api/analytics ------------------------------------------------------------------------

def test_analytics_normalises_legacy_numeric_labels_and_buckets_hours(env):
    client, Session = env
    now = datetime(2026, 10, 29, 13, 30)
    add_logs(Session, [
        {"prediction": "1", "timestamp": now},                                      # legacy numeric
        {"prediction": "positive", "timestamp": now + timedelta(minutes=5)},
        {"prediction": "-1", "confidence": 0.4, "timestamp": now},
        {"prediction": "neutral", "confidence": None, "timestamp": now + timedelta(hours=1)},
    ])
    a = client.get("/api/analytics").json()  # used to raise KeyError on '1' and fall back to zeros
    assert a["sentiment_distribution"] == {"positive": 2, "negative": 1, "neutral": 1}
    assert a["total_insights"] == 4
    assert a["sentiment_index"] == pytest.approx((2 - 1) / 4 * 50 + 50)
    assert a["low_confidence_count"] == 1  # NULL confidence is not "low"
    assert a["trends"]["categories"] == ["10-29 13:00", "10-29 14:00"]
    assert a["trends"]["positive"] == [2, 0] and a["trends"]["negative"] == [1, 0] and a["trends"]["neutral"] == [0, 1]
    assert a["top_keywords"] == []  # no placeholder keywords when there are no reports


def test_analytics_with_no_data_reports_absence_not_neutrality(env):
    client, _ = env
    a = client.get("/api/analytics").json()
    assert a["total_insights"] == 0 and a["sentiment_index"] is None
    assert a["trends"]["categories"] == [] and a["top_keywords"] == []


def test_analytics_keywords_come_from_stored_reports(env):
    client, Session = env
    with Session() as db:
        db.add_all([
            ReportLog(query="q1", top_keywords=["economy", "jobs"], sentiment_distribution={}, fallback_used=False),
            ReportLog(query="q2", top_keywords=json.dumps(["economy", "x"]), sentiment_distribution={}, fallback_used=False),
            ReportLog(query="q3", top_keywords=["economy", "jobs", "tax"], sentiment_distribution={}, fallback_used=True),
        ])
        db.commit()
    a = client.get("/api/analytics").json()
    assert a["top_keywords"][:2] == ["economy", "jobs"] and "x" not in a["top_keywords"]
    assert a["reports_considered"] == 3


def test_analytics_errors_are_not_masked_as_zeros(env, app_module):
    client, _ = env

    def broken():
        raise RuntimeError("db down")
    app_module.app.dependency_overrides[get_db] = broken
    safe_client = TestClient(app_module.app, raise_server_exceptions=False)
    assert safe_client.get("/api/analytics").status_code == 500


# ---- /api/experiments ----------------------------------------------------------------------

def write_summary(root):
    d = root / "artifacts" / "model_evaluation"
    d.mkdir(parents=True)
    rows = [["Model Name", "Accuracy", "F1 Score (Weighted)", "F1 Score (Macro)", "Latency"],
            ["LogisticRegression", 0.91, 0.91, 0.90, 1.8],
            ["LinearSVC", 0.94, 0.94, 0.93, 1.6],
            ["MultinomialNB", 0.77, 0.76, 0.74, 1.5]]
    with open(d / "all_models_summary.csv", "w", newline="") as f:
        csv.writer(f).writerows(rows)


def test_experiment_status_follows_best_model_metadata(env, tmp_path, monkeypatch):
    client, _ = env
    monkeypatch.chdir(tmp_path)
    write_summary(tmp_path)
    (tmp_path / "artifacts" / "best_model").mkdir()
    (tmp_path / "artifacts" / "best_model" / "best_model_metadata.json").write_text(
        json.dumps({"best_model_name": "LinearSVC", "tier_1_fast_fallback": "LinearSVC"}))
    status = {e["model_name"]: e["status"] for e in client.get("/api/experiments").json()["experiments"]}
    assert status == {"LogisticRegression": "Evaluated", "MultinomialNB": "Evaluated",
                      "LinearSVC": "Serving (fast tier) + Best by macro-F1"}


def test_experiment_status_without_metadata_matches_pipeline_default(env, tmp_path, monkeypatch):
    client, _ = env
    monkeypatch.chdir(tmp_path)
    write_summary(tmp_path)
    exps = {e["model_name"]: e for e in client.get("/api/experiments").json()["experiments"]}
    assert exps["LogisticRegression"]["status"] == "Serving (fast tier)"  # PredictionPipeline's default
    assert exps["LinearSVC"]["status"] == "Evaluated"
    assert "eval_seconds" in exps["LinearSVC"] and "latency_seconds" not in exps["LinearSVC"]


# ---- optional components -------------------------------------------------------------------

def test_search_analysis_says_so_when_agents_are_unavailable(env, app_module, monkeypatch):
    client, _ = env
    monkeypatch.setattr(app_module, "search_agent", None)
    r = client.post("/search-analysis", json={"query": "economy"})
    assert r.status_code == 503


def test_csv_upload_works_without_a_report_agent(env, app_module, monkeypatch):
    client, Session = env
    monkeypatch.setattr(app_module, "report_agent", None)
    body = io.BytesIO(b"text\nhello there\nsecond one\n")
    r = client.post("/upload_csv", files={"file": ("t.csv", body, "text/csv")})
    assert r.status_code == 200
    data = r.json()
    assert data["total_processed"] == 2 and data["report"] is None
    with Session() as db:
        assert db.query(PredictionLog).count() == 2


# ---- no simulated data ---------------------------------------------------------------------

def test_no_simulated_data_in_pages_or_queries():
    for page in (REPO / "templates" / "pages").glob("*.html"):
        text = page.read_text(encoding="utf-8")
        assert "Math.random" not in text, page.name
        assert "updateSimulatedLogs" not in text, page.name
    assert "func.strftime" not in (REPO / "app.py").read_text(encoding="utf-8")  # SQLite-only SQL
