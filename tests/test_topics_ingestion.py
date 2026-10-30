"""Topics, the RSS sources, ingestion (dedupe, failure isolation, caps) and the endpoints around them."""
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from tests.conftest import FakePipeline
from tweet_sentiment_classifier.app.services import ingestion_service
from tweet_sentiment_classifier.app.sources import rss
from tweet_sentiment_classifier.app.sources.rss import RawPost
from tweet_sentiment_classifier.entity.models import IngestionRun, Post, Topic

FIXTURE = Path(__file__).parent / "fixtures" / "sources" / "google_news.xml"


def post(i, text=None, source="fake"):
    return RawPost(source, f"id-{i}", f"https://x.example/{i}", text or f"story number {i} about fuel", None)


class FakeSource:
    def __init__(self, posts):
        self.posts = posts

    def fetch(self, query):
        return list(self.posts)


class BoomSource:
    def fetch(self, query):
        raise RuntimeError("network down")


@pytest.fixture
def topics_env(env, app_module, monkeypatch):
    monkeypatch.setattr(app_module.settings, "INGEST_TOKEN", "secret")
    monkeypatch.setattr(app_module, "sources", {"fake": FakeSource([post(1), post(2)]), "boom": BoomSource()})
    return env


H = {"X-Ingest-Token": "secret"}


def new_topic(client, name="Fuel", sources=("fake",)):
    r = client.post("/api/v1/topics", json={"name": name, "query": "fuel prices", "sources": list(sources)})
    assert r.status_code == 201, r.text
    return r.json()["id"]


# ---- sources --------------------------------------------------------------------------------------------

def test_parse_rss_fixture_strips_html_dedupes_title_and_normalises_dates():
    posts = rss.parse_rss(FIXTURE.read_bytes(), "google_news")
    assert [p.external_id for p in posts][:2] == ["CBMi-a1", "CBMi-a2"]
    assert posts[0].text == "Fuel prices rise again - The Daily"  # description only repeated the title
    assert posts[1].text == "Commuters angry over petrol hike. Commuters say the hike hurts & nobody listens."
    assert posts[0].published_at == datetime(2026, 10, 28, 8, 30)
    assert posts[1].published_at == datetime(2026, 10, 28, 4, 30)  # +05:30 converted to naive UTC
    assert posts[2].external_id == "https://news.example/a3" and posts[2].published_at is None
    assert len(posts) == 3  # the item with no text is dropped


def test_hostile_feed_and_non_https_urls_are_rejected():
    bomb = (b'<?xml version="1.0"?><!DOCTYPE r [<!ENTITY a "aaaa">]>'
            b'<rss><channel><item><title>&a;</title></item></channel></rss>')
    with pytest.raises(ValueError):
        rss.parse_rss(bomb, "x")
    with pytest.raises(ValueError):
        rss.fetch_url("http://news.example/feed")


def test_rss_feeds_keep_only_relevant_entries_and_survive_a_dead_feed(monkeypatch):
    def fake_fetch(url, timeout=10.0):
        if "dead" in url:
            raise OSError("down")
        return FIXTURE.read_bytes()
    monkeypatch.setattr(rss, "fetch_url", fake_fetch)
    feeds = rss.RssFeeds(["https://dead.example/f", "https://ok.example/f"])
    assert [p.external_id for p in feeds.fetch("petrol")] == ["CBMi-a2"]
    with pytest.raises(RuntimeError):
        rss.RssFeeds(["https://dead.example/f"]).fetch("petrol")


def test_build_sources_registers_rss_only_when_configured():
    assert set(rss.build_sources("")) == {"google_news"}
    assert set(rss.build_sources("https://a.example/f, https://b.example/f")) == {"google_news", "rss"}


# ---- ingestion ------------------------------------------------------------------------------------------

def test_ingest_stores_classified_posts_and_skips_exact_and_near_duplicates(topics_env, app_module):
    client, Session = topics_env
    tid = new_topic(client)
    first = client.post(f"/api/v1/topics/{tid}/ingest", headers=H).json()["runs"][0]
    assert (first["fetched"], first["new"], first["error"]) == (2, 2, None)

    app_module.sources["fake"].posts = [post(1), post(2), post(3, text="STORY number 1 about fuel!!!"), post(4)]
    second = client.post(f"/api/v1/topics/{tid}/ingest", headers=H).json()["runs"][0]
    assert second["fetched"] == 4 and second["new"] == 1  # id-3 is a near-duplicate of id-1 after cleaning

    listing = client.get(f"/api/v1/topics/{tid}/posts").json()
    assert listing["total"] == 3
    assert {p["prediction"] for p in listing["posts"]} == {"positive"}  # FakePipeline answers positive
    assert listing["posts"][0]["decision_source"] == "local:logisticregression"
    assert client.get(f"/api/v1/topics/{tid}/posts?sentiment=negative").json()["total"] == 0
    assert client.get(f"/api/v1/topics/{tid}/posts?sentiment=happy").status_code == 400


def test_a_failing_source_is_recorded_and_does_not_stop_the_others(topics_env):
    client, Session = topics_env
    tid = new_topic(client, sources=("boom", "fake"))
    runs = client.post(f"/api/v1/topics/{tid}/ingest", headers=H).json()["runs"]
    assert runs[0]["source"] == "boom" and "network down" in runs[0]["error"]
    assert runs[1]["source"] == "fake" and runs[1]["new"] == 2 and runs[1]["error"] is None
    with Session() as db:
        stored = {(r.source, r.error is not None, r.finished_at is not None) for r in db.query(IngestionRun)}
        assert stored == {("boom", True, True), ("fake", False, True)}


def test_the_per_run_cap_limits_how_many_posts_are_classified(topics_env, app_module, monkeypatch):
    client, _ = topics_env
    monkeypatch.setattr(app_module.settings, "MAX_POSTS_PER_RUN", 3)
    app_module.sources["fake"].posts = [post(i) for i in range(10)]
    tid = new_topic(client)
    assert client.post(f"/api/v1/topics/{tid}/ingest", headers=H).json()["runs"][0]["new"] == 3


def test_a_post_the_pipeline_cannot_classify_is_skipped_not_fatal(topics_env, app_module, monkeypatch):
    client, _ = topics_env

    class Flaky(FakePipeline):
        async def predict(self, text):
            if "number 1" in text:
                raise RuntimeError("model error")
            return await super().predict(text)

    monkeypatch.setattr(app_module, "pipeline", Flaky())
    tid = new_topic(client)
    run = client.post(f"/api/v1/topics/{tid}/ingest", headers=H).json()["runs"][0]
    assert (run["fetched"], run["new"], run["error"]) == (2, 1, None)


def test_without_a_pipeline_the_run_records_the_error(topics_env, app_module, monkeypatch):
    client, _ = topics_env
    monkeypatch.setattr(app_module, "pipeline", None)
    tid = new_topic(client)
    run = client.post(f"/api/v1/topics/{tid}/ingest", headers=H).json()["runs"][0]
    assert run["new"] == 0 and "pipeline" in run["error"]


def test_ingest_endpoints_need_the_token(topics_env, app_module, monkeypatch):
    client, _ = topics_env
    tid = new_topic(client)
    assert client.post(f"/api/v1/topics/{tid}/ingest").status_code == 403
    assert client.post("/api/v1/ingest/run", headers={"X-Ingest-Token": "wrong"}).status_code == 403
    assert client.post("/api/v1/ingest/run", headers=H).json() == {
        "Fuel": [{"source": "fake", "fetched": 2, "new": 2, "error": None}]}
    monkeypatch.setattr(app_module.settings, "INGEST_TOKEN", "")
    assert client.post("/api/v1/ingest/run", headers=H).status_code == 503  # disabled, not open


def test_purge_old_posts_keeps_recent_ones(topics_env):
    _, Session = topics_env
    with Session() as db:
        topic = Topic(name="t", query="q", sources=["fake"])
        db.add(topic)
        db.commit()
        old, fresh = datetime.utcnow() - timedelta(days=100), datetime.utcnow()
        for i, when in enumerate((old, fresh)):
            db.add(Post(topic_id=topic.id, source="fake", external_id=str(i), content_hash=str(i), text="t",
                        prediction="neutral", fetched_at=when))
        db.commit()
        assert ingestion_service.purge_old_posts(db, 90) == 1
        assert [p.external_id for p in db.query(Post)] == ["1"]


# ---- topics API -----------------------------------------------------------------------------------------

def test_topic_validation_listing_and_delete(topics_env, app_module, monkeypatch):
    client, Session = topics_env
    tid = new_topic(client)
    bad = [
        ({"name": " ", "query": "q"}, 400),
        ({"name": "A", "query": "q", "sources": ["nope"]}, 400),
        ({"name": "A" * 101, "query": "q"}, 413),
        ({"name": "Fuel", "query": "other", "sources": ["fake"]}, 409),  # duplicate name
    ]
    assert [client.post("/api/v1/topics", json=b).status_code for b, _ in bad] == [c for _, c in bad]
    monkeypatch.setattr(app_module.settings, "MAX_TOPICS", 1)
    assert client.post("/api/v1/topics", json={"name": "B", "query": "q", "sources": ["fake"]}).status_code == 409

    client.post(f"/api/v1/topics/{tid}/ingest", headers=H)
    listed = client.get("/api/v1/topics").json()["topics"]
    assert listed[0]["posts"] == 2 and listed[0]["last_ingested"].endswith("Z") and listed[0]["sources"] == ["fake"]

    assert client.delete(f"/api/v1/topics/{tid}").status_code == 200
    assert client.get(f"/api/v1/topics/{tid}/posts").status_code == 404
    with Session() as db:
        assert db.query(Post).count() == 0 and db.query(IngestionRun).count() == 0


def test_topics_page_renders(env):
    client, _ = env
    page = client.get("/topics")
    assert page.status_code == 200 and "Topic Monitor" in page.text
