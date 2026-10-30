"""Prometheus metrics, in a registry of their own (no default process metrics, safe to import from tests).

Labels are kept low-cardinality on purpose: HTTP series use the route template ("/api/v1/topics/{topic_id}/posts"),
never the raw path, and nothing here is labelled with user text or ids.
"""
from prometheus_client import CollectorRegistry, Counter, Histogram, generate_latest

REGISTRY = CollectorRegistry()

HTTP_REQUESTS = Counter("http_requests_total", "HTTP requests", ["method", "route", "status"], registry=REGISTRY)
HTTP_SECONDS = Histogram("http_request_seconds", "HTTP request latency in seconds", ["method", "route"],
                         registry=REGISTRY)
PREDICTIONS = Counter("predictions_total", "Predictions by the model that decided", ["decision_source"],
                      registry=REGISTRY)
LLM_TOKENS = Counter("llm_tokens_total", "LLM tokens spent", ["provider", "direction"], registry=REGISTRY)
INGESTED_POSTS = Counter("ingested_posts_total", "New posts stored", ["source"], registry=REGISTRY)
INGEST_ERRORS = Counter("ingestion_errors_total", "Failed source fetches", ["source"], registry=REGISTRY)
AGENT_RUNS = Counter("agent_runs_total", "Finished agent sessions", ["status"], registry=REGISTRY)
RATE_LIMITED = Counter("rate_limited_requests_total", "Requests refused by the rate limiter", ["kind"],
                       registry=REGISTRY)


def render() -> bytes:
    return generate_latest(REGISTRY)


def value(name: str, labels: dict) -> float:
    """Current value of one series (0 if it has never been touched). Used by tests."""
    return REGISTRY.get_sample_value(name, labels) or 0.0
