"""The /api/v1 routes, their legacy aliases, the request-id header and the TTL cache."""
import io

from tweet_sentiment_classifier.app.core.cache import TTLCache


def test_v1_predict_matches_legacy_alias(env):
    client, _ = env
    v1 = client.post("/api/v1/predict", json={"text": "great work"})
    old = client.post("/api/predict", json={"text": "great work"})
    assert v1.status_code == old.status_code == 200
    assert v1.json() == old.json()
    assert v1.json()["prediction"] == "positive"


def test_v1_and_legacy_read_endpoints_return_identical_json(env):
    client, _ = env
    client.post("/api/v1/predict", json={"text": "one"})
    pairs = [
        ("/api/v1/analytics/monitoring", "/monitoring"),
        ("/api/v1/analytics/recent", "/api/recent-logs"),
        ("/api/v1/analytics/history", "/api/prediction-history"),
        ("/api/v1/analytics/summary", "/api/analytics"),
        ("/api/v1/models/info", "/api/model-info"),
        ("/api/v1/models/experiments", "/api/experiments"),
        ("/api/v1/models/results", "/api/results"),
        ("/api/v1/models/lab", "/api/model-lab"),
        ("/api/v1/health", "/health"),
    ]
    for new, old in pairs:
        a, b = client.get(new), client.get(old)
        assert a.status_code == b.status_code == 200, (new, old)
        assert a.json() == b.json(), (new, old)


def test_v1_batch_matches_legacy_batch(env):
    client, _ = env
    csv = b"text\nlovely\nawful\n"
    a = client.post("/api/v1/predict/batch", files={"file": ("t.csv", io.BytesIO(csv), "text/csv")})
    b = client.post("/api/upload-csv", files={"file": ("t.csv", io.BytesIO(csv), "text/csv")})
    assert a.status_code == b.status_code == 200
    assert a.json() == b.json()
    assert a.json()["total_processed"] == 2


def test_validation_errors_are_the_same_on_v1(env):
    client, _ = env
    assert client.post("/api/v1/predict", json={"text": "   "}).status_code == 400


def test_responses_carry_a_request_id_and_echo_a_supplied_one(env):
    client, _ = env
    assert client.get("/api/v1/health").headers["X-Request-ID"]
    assert client.get("/api/v1/health", headers={"X-Request-ID": "abc123"}).headers["X-Request-ID"] == "abc123"


def test_legacy_routes_are_marked_deprecated_and_v1_routes_are_not(env, app_module):
    schema = app_module.app.openapi()
    assert schema["paths"]["/api/predict"]["post"].get("deprecated") is True
    assert not schema["paths"]["/api/v1/predict"]["post"].get("deprecated")


def test_ttl_cache_expires_evicts_and_clears(monkeypatch):
    import tweet_sentiment_classifier.app.core.cache as cache_mod

    now = [100.0]
    monkeypatch.setattr(cache_mod.time, "monotonic", lambda: now[0])
    cache = TTLCache(ttl_seconds=10, max_size=2)
    cache["a"] = 1
    cache["b"] = 2
    assert "a" in cache and cache["b"] == 2 and len(cache) == 2

    cache["c"] = 3  # full: the oldest entry ("a") is evicted
    assert "a" not in cache and "c" in cache

    now[0] += 11  # everything has expired
    assert "b" not in cache and "c" not in cache and len(cache) == 0

    cache["d"] = 4
    cache.clear()
    assert len(cache) == 0
