"""API keys, write protection, rate limiting and Prometheus metrics."""
import pytest

from tests.test_agent import Scripted, final, run_once, use_model, world  # noqa: F401  (world is a fixture)
from tests.test_topics_ingestion import H, new_topic, topics_env  # noqa: F401  (topics_env is a fixture)
from tweet_sentiment_classifier.app.core import auth, metrics
from tweet_sentiment_classifier.app.core.ratelimit import RateLimiter
from tweet_sentiment_classifier.entity.models import ApiKey

# ---- API keys -------------------------------------------------------------------------------------------


def test_a_key_is_stored_only_as_a_hash_and_authorizes_until_revoked(env):
    _, Session = env
    with Session() as db:
        assert auth.authorize(db, ["anything"], "") == "disabled"  # no token, no key: protected endpoints stay shut
        key = auth.create_key(db, "ci cron")
        row = db.query(ApiKey).one()
        assert key.startswith("sk_") and len(key) > 40
        assert row.key_hash == auth.hash_key(key) and key not in (row.key_hash, row.name)
        assert auth.authorize(db, [key], "") == "ok"
        assert auth.authorize(db, ["", "wrong"], "") == "denied" and auth.authorize(db, [""], "") == "denied"
        assert auth.authorize(db, ["static"], "static") == "ok" and auth.authorize(db, [key], "static") == "ok"
        assert auth.revoke_key(db, row.id) is True and auth.revoke_key(db, 999) is False
        assert auth.authorize(db, [key], "") == "disabled"  # the only key is revoked
        assert auth.authorize(db, [key], "static") == "denied"


def test_protected_endpoints_accept_a_key_in_either_header(topics_env, app_module, monkeypatch):  # noqa: F811
    client, Session = topics_env
    tid = new_topic(client)
    with Session() as db:
        key = auth.create_key(db, "k")
    monkeypatch.setattr(app_module.settings, "INGEST_TOKEN", "")
    assert client.post(f"/api/v1/topics/{tid}/ingest").status_code == 403  # a key exists, none presented
    assert client.post(f"/api/v1/topics/{tid}/ingest", headers={"X-API-Key": key}).status_code == 200
    assert client.post(f"/api/v1/topics/{tid}/ingest", headers={"X-Ingest-Token": key}).status_code == 200
    assert client.post(f"/api/v1/topics/{tid}/ingest", headers={"X-API-Key": "sk_forged"}).status_code == 403
    with Session() as db:
        auth.revoke_key(db, 1)
    assert client.post(f"/api/v1/topics/{tid}/ingest", headers={"X-API-Key": key}).status_code == 503


def test_api_auth_protects_topic_writes_and_the_annotation_queue_but_not_public_feedback(env, app_module, monkeypatch):
    client, _ = env
    monkeypatch.setattr(app_module.settings, "API_AUTH", True)
    body = {"name": "T", "query": "q", "sources": ["google_news"]}
    assert client.post("/api/v1/topics", json=body).status_code == 503  # nothing configured to accept

    monkeypatch.setattr(app_module.settings, "INGEST_TOKEN", "secret")
    assert client.post("/api/v1/topics", json=body).status_code == 403
    created = client.post("/api/v1/topics", json=body, headers={"X-API-Key": "secret"})
    assert created.status_code == 201
    tid = created.json()["id"]
    assert client.post(f"/api/v1/annotation/refresh?topic_id={tid}").status_code == 403
    assert client.post(f"/api/v1/annotation/refresh?topic_id={tid}", headers={"X-API-Key": "secret"}).status_code == 200
    assert client.delete(f"/api/v1/topics/{tid}").status_code == 403
    assert client.delete(f"/api/v1/topics/{tid}", headers={"X-API-Key": "secret"}).status_code == 200
    # reading, predicting and reporting a wrong label stay open
    assert client.get("/api/v1/topics").status_code == 200
    assert client.post("/api/v1/predict", json={"text": "fine day"}).status_code == 200
    assert client.post("/api/v1/corrections", json={"text": "x", "predicted": "positive",
                                                    "corrected": "negative"}).status_code == 201


def test_auth_cli_creates_lists_and_revokes_keys(capsys):
    assert auth.main(["create", "cron", "job"]) == 0
    key = capsys.readouterr().out.strip().splitlines()[-1]
    assert key.startswith("sk_")
    assert auth.main(["list"]) == 0
    listing = capsys.readouterr().out
    assert "cron job" in listing and "active" in listing
    key_id = int(listing.strip().splitlines()[-1].split("\t")[0])
    assert auth.main(["revoke", str(key_id)]) == 0 and auth.main(["revoke", "99999"]) == 1
    assert auth.main(["bogus"]) == 2


# ---- rate limiting --------------------------------------------------------------------------------------

def test_sliding_window_limits_per_key_and_recovers():
    rl = RateLimiter()
    assert [rl.allow("a", 3, now=t)[0] for t in (0, 1, 2)] == [True, True, True]
    allowed, retry = rl.allow("a", 3, now=3)
    assert allowed is False and retry == pytest.approx(57.0)
    assert rl.allow("b", 3, now=3)[0] is True  # another client is unaffected
    assert rl.allow("a", 3, now=60.5)[0] is True  # the first hit left the window


def test_the_middleware_returns_429_with_retry_after_and_separates_clients(env, app_module, monkeypatch):
    client, _ = env
    monkeypatch.setattr(app_module.settings, "TRUST_PROXY", True)
    monkeypatch.setattr(app_module.settings, "RATE_LIMIT_WRITE_PER_MINUTE", 2)
    post = lambda ip, text: client.post("/api/v1/predict", json={"text": text},  # noqa: E731
                                        headers={"X-Forwarded-For": f"{ip}, 10.9.9.9"})
    assert [post("203.0.113.1", f"rl one {i}").status_code for i in range(3)] == [200, 200, 429]
    refused = post("203.0.113.1", "rl one again")
    assert refused.status_code == 429 and int(refused.headers["Retry-After"]) >= 1 and "Too many" in refused.json()["detail"]
    assert post("203.0.113.2", "rl two").status_code == 200  # a different client has its own budget
    assert client.get("/api/v1/health", headers={"X-Forwarded-For": "203.0.113.1"}).status_code == 200  # reads: no limit
    monkeypatch.setattr(app_module.settings, "RATE_LIMIT_WRITE_PER_MINUTE", 0)
    assert post("203.0.113.1", "rl off").status_code == 200  # 0 turns it off


def test_x_forwarded_for_is_ignored_unless_the_proxy_is_trusted(env, app_module, monkeypatch):
    client, _ = env
    monkeypatch.setattr(app_module.settings, "RATE_LIMIT_WRITE_PER_MINUTE", 1)
    monkeypatch.setattr(app_module.settings, "TRUST_PROXY", False)
    codes = [client.post("/api/v1/predict", json={"text": f"spoof {i}"},
                         headers={"X-Forwarded-For": f"198.51.100.{i}"}).status_code for i in range(3)]
    assert codes == [200, 429, 429]  # a spoofed header does not buy a fresh budget


# ---- metrics --------------------------------------------------------------------------------------------

def test_metrics_endpoint_uses_route_templates_and_counts_predictions(env, app_module):
    client, _ = env
    before = metrics.value("predictions_total", {"decision_source": "local:logisticregression"})
    client.post("/api/v1/predict", json={"text": "metrics unique text one"})
    client.get("/api/v1/topics/12345/posts")  # 404, but the route matched
    body = client.get("/metrics").text
    assert metrics.value("predictions_total", {"decision_source": "local:logisticregression"}) == before + 1
    assert 'http_requests_total{method="GET",route="/api/v1/topics/{topic_id}/posts",status="404"}' in body
    assert "/api/v1/topics/12345" not in body  # raw ids never become label values
    assert 'http_request_seconds_bucket{le="0.05",method="POST",route="/api/v1/predict"}' in body


def test_metric_labels_separate_v1_from_legacy_and_do_not_grow_with_static_paths(env):
    client, _ = env
    for path in ("/health", "/api/v1/health", "/static/css/style.css", "/static/does-not-exist-1", "/static/does-not-exist-2",
                 "/no/such/page"):
        client.get(path)
    body = client.get("/metrics").text
    assert 'route="/health"' in body and 'route="/api/v1/health"' in body
    assert 'route="/static"' in body and "does-not-exist" not in body and "/no/such/page" not in body
    assert 'route="unmatched"' in body


def test_metrics_can_be_protected_with_a_bearer_token(env, app_module, monkeypatch):
    client, _ = env
    monkeypatch.setattr(app_module.settings, "METRICS_TOKEN", "m3trics")
    assert client.get("/metrics").status_code == 401
    assert client.get("/metrics", headers={"Authorization": "Bearer wrong"}).status_code == 401
    assert client.get("/metrics", headers={"Authorization": "Bearer m3trics"}).status_code == 200


def test_rate_limited_requests_are_counted(env, app_module, monkeypatch):
    client, _ = env
    before = metrics.value("rate_limited_requests_total", {"kind": "write"})
    monkeypatch.setattr(app_module.settings, "TRUST_PROXY", True)
    monkeypatch.setattr(app_module.settings, "RATE_LIMIT_WRITE_PER_MINUTE", 1)
    for i in range(2):
        client.post("/api/v1/predict", json={"text": f"count rl {i}"}, headers={"X-Forwarded-For": "192.0.2.77"})
    assert metrics.value("rate_limited_requests_total", {"kind": "write"}) == before + 1


def test_llm_tokens_ingestion_and_agent_runs_are_counted(world, topics_env, app_module, monkeypatch):  # noqa: F811
    client, Session, tid, _ = world
    tokens = lambda d: metrics.value("llm_tokens_total", {"provider": "fake", "direction": d})  # noqa: E731
    t_in, t_out = tokens("in"), tokens("out")
    agent = lambda s: metrics.value("agent_runs_total", {"status": s})  # noqa: E731
    done = agent("done")
    use_model(app_module, monkeypatch, Scripted([final("Nothing to report.")]))
    run_once(Session, app_module, "q", tid)
    assert tokens("in") == t_in + 10 and tokens("out") == t_out + 5 and agent("done") == done + 1

    posts, errors = (lambda: metrics.value("ingested_posts_total", {"source": "fake"})), (
        lambda: metrics.value("ingestion_errors_total", {"source": "boom"}))
    p0, e0 = posts(), errors()
    topic = new_topic(client, "Counted", sources=("boom", "fake"))
    client.post(f"/api/v1/topics/{topic}/ingest", headers=H)
    assert posts() == p0 + 2 and errors() == e0 + 1
