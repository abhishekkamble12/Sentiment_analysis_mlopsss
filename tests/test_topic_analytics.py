"""Trends, drift and alerts for monitored topics."""
import itertools
from datetime import datetime, timedelta
from types import SimpleNamespace

import pytest

from tweet_sentiment_classifier.app.services import topic_analytics as ta
from tweet_sentiment_classifier.entity.models import Alert, Post, Topic

NOW = datetime(2026, 10, 30, 12, 0)
_ids = itertools.count()


@pytest.fixture
def seeded(env):
    """(client, Session, topic_id, add) where add(when, n, prediction, confidence) stores posts."""
    client, Session = env
    with Session() as db:
        topic = Topic(name="t", query="q", sources=["fake"])
        db.add(topic)
        db.commit()
        topic_id = topic.id

    def add(when, n, prediction, confidence=0.9, source="fake"):
        with Session() as db:
            for _ in range(n):
                i = next(_ids)
                db.add(Post(topic_id=topic_id, source=source, external_id=f"e{i}", content_hash=f"h{i}",
                            text="some text", prediction=prediction, confidence=confidence,
                            published_at=when, fetched_at=when))
            db.commit()

    return client, Session, topic_id, add


def run_alerts(Session, topic_id, settings, now=NOW):
    with Session() as db:
        return [a.kind for a in ta.evaluate_alerts(db, db.get(Topic, topic_id), None, settings, now)]


@pytest.fixture
def settings(app_module):
    return app_module.settings


# ---- statistics -----------------------------------------------------------------------------------------

def test_wilson_interval_matches_known_values():
    assert ta.wilson(5, 10) == pytest.approx((0.2366, 0.7634), abs=1e-3)
    assert ta.wilson(0, 10) == pytest.approx((0.0, 0.2775), abs=1e-3)
    assert ta.wilson(0, 0) == (0.0, 0.0)


def test_two_proportion_p_value():
    assert ta.two_proportion_p(50, 100, 30, 100) == pytest.approx(0.00389, abs=1e-4)  # z = 2.887
    assert ta.two_proportion_p(30, 100, 30, 100) == pytest.approx(1.0)
    assert ta.two_proportion_p(0, 50, 0, 50) == 1.0  # no variance: nothing to detect


def test_psi_is_zero_for_equal_mixes_and_large_for_a_shift():
    mix = {"positive": 50, "neutral": 30, "negative": 20}
    assert ta.psi(mix, dict(mix)) == pytest.approx(0.0, abs=1e-9)
    assert ta.psi({"positive": 10, "negative": 90}, {"positive": 90, "negative": 10}) > 0.25


def test_oov_rate_counts_words_outside_the_vocabulary():
    vec = SimpleNamespace(vocabulary_={"great": 0, "day": 1})
    assert ta.oov_rate(vec, ["Great day, wow!"]) == pytest.approx(1 / 3, abs=1e-3)
    assert ta.oov_rate(None, ["Great day"]) is None
    assert ta.oov_rate(vec, ["!!!"]) is None


# ---- trends ---------------------------------------------------------------------------------------------

def test_trends_bucket_by_day_with_counts_shares_and_sources(seeded):
    _, Session, tid, add = seeded
    add(NOW - timedelta(days=1), 2, "positive", 0.9)
    add(NOW - timedelta(days=1), 1, "negative", 0.6, source="rss")
    add(NOW, 1, "neutral", 0.5)
    add(NOW - timedelta(days=40), 5, "negative")  # outside the 14-day range
    with Session() as db:
        points = ta.trends(db, tid, "day", 14, now=NOW)
    assert [p["bucket"] for p in points] == ["2026-10-29", "2026-10-30"]
    first = points[0]
    assert first["n"] == 3 and first["counts"] == {"positive": 2, "neutral": 0, "negative": 1, "uncertain": 0}
    assert first["mean_confidence"] == pytest.approx((0.9 + 0.9 + 0.6) / 3, abs=1e-4)
    assert first["by_source"] == {"fake": 2, "rss": 1}
    assert first["shares"]["positive"]["share"] == pytest.approx(0.6667, abs=1e-4)
    assert first["shares"]["positive"]["low"] < 0.6667 < first["shares"]["positive"]["high"]


# ---- alerts ---------------------------------------------------------------------------------------------

def test_negative_spike_fires_once_and_respects_the_cooldown(seeded, settings):
    _, Session, tid, add = seeded
    add(NOW - timedelta(days=3), 32, "positive")
    add(NOW - timedelta(days=3), 8, "negative")  # baseline: 20% negative
    add(NOW - timedelta(hours=1), 16, "positive")
    add(NOW - timedelta(hours=1), 24, "negative")  # last day: 60% negative
    kinds = run_alerts(Session, tid, settings)
    assert set(kinds) == {"negative_spike", "class_drift"}
    with Session() as db:
        spike = db.query(Alert).filter_by(kind="negative_spike").one()
        assert (spike.baseline, spike.observed) == (pytest.approx(0.2), pytest.approx(0.6))
        assert spike.p_value < 0.01 and "20%" in spike.message and "60%" in spike.message
    assert run_alerts(Session, tid, settings) == []  # same alerts are not repeated inside the cooldown


def test_no_alert_without_a_change_or_without_enough_posts_or_when_negativity_falls(seeded, settings):
    _, Session, tid, add = seeded
    add(NOW - timedelta(days=3), 32, "positive")
    add(NOW - timedelta(days=3), 8, "negative")
    add(NOW - timedelta(hours=1), 32, "positive")
    add(NOW - timedelta(hours=1), 8, "negative")  # identical mix
    assert run_alerts(Session, tid, settings) == []

    few = SimpleNamespace(**{**vars(settings), "ALERT_MIN_POSTS": 100})
    assert run_alerts(Session, tid, few) == []  # 40 posts per window is below the minimum


def test_a_fall_in_negativity_is_not_a_negative_spike(seeded, settings):
    _, Session, tid, add = seeded
    add(NOW - timedelta(days=3), 20, "negative")
    add(NOW - timedelta(days=3), 20, "positive")
    add(NOW - timedelta(hours=1), 38, "positive")
    add(NOW - timedelta(hours=1), 2, "negative")
    assert "negative_spike" not in run_alerts(Session, tid, settings)


def test_volume_spike(seeded, settings):
    _, Session, tid, add = seeded
    add(NOW - timedelta(hours=24) - timedelta(days=6, hours=23), 21, "neutral")  # about 3 posts a day
    add(NOW - timedelta(hours=1), 100, "neutral")
    assert run_alerts(Session, tid, settings) == ["volume_spike"]


# ---- API ------------------------------------------------------------------------------------------------

def test_trend_drift_and_alert_endpoints(seeded, settings):
    client, Session, tid, add = seeded
    now = datetime.utcnow()
    add(now - timedelta(days=3), 32, "positive")
    add(now - timedelta(days=3), 8, "negative")
    add(now - timedelta(hours=1), 16, "positive")
    add(now - timedelta(hours=1), 24, "negative")

    trend = client.get(f"/api/v1/topics/{tid}/trends?bucket=day").json()
    assert sum(p["n"] for p in trend["points"]) == 80
    assert client.get(f"/api/v1/topics/{tid}/trends?bucket=week").status_code == 400
    assert client.get("/api/v1/topics/999/trends").status_code == 404

    drift = client.get(f"/api/v1/topics/{tid}/drift").json()
    assert drift["window"]["n"] == 40 and drift["baseline"]["n"] == 40 and drift["psi"] > 0.25
    assert drift["window"]["oov_rate"] is None  # the test pipeline has no vocabulary

    with Session() as db:
        ta.evaluate_alerts(db, db.get(Topic, tid), None, settings)
    alerts = client.get(f"/api/v1/topics/{tid}/alerts").json()["alerts"]
    assert {a["kind"] for a in alerts} == {"negative_spike", "class_drift"} and not alerts[0]["acknowledged"]
    acked = client.post(f"/api/v1/alerts/{alerts[0]['id']}/ack").json()
    assert acked["acknowledged"] is True
    assert client.post("/api/v1/alerts/9999/ack").status_code == 404


def test_a_failure_in_alerting_does_not_fail_the_ingestion(env, app_module, monkeypatch):
    from tests.test_topics_ingestion import FakeSource, post
    client, _ = env
    monkeypatch.setattr(app_module.settings, "INGEST_TOKEN", "secret")
    monkeypatch.setattr(app_module, "sources", {"fake": FakeSource([post(1)])})
    monkeypatch.setattr(ta, "evaluate_alerts", lambda *a, **k: 1 / 0)
    tid = client.post("/api/v1/topics", json={"name": "T", "query": "q", "sources": ["fake"]}).json()["id"]
    runs = client.post(f"/api/v1/topics/{tid}/ingest", headers={"X-Ingest-Token": "secret"}).json()["runs"]
    assert runs[0]["new"] == 1 and runs[0]["error"] is None
