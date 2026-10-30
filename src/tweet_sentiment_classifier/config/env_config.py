import os
from pydantic_settings import BaseSettings

from functools import lru_cache

class Settings(BaseSettings):
    DATABASE_URL: str = "sqlite:///./monitoring.db"
    GROQ_API_KEY: str = ""
    HF_API_TOKEN: str = ""
    HF_TOKEN: str = ""
    LOG_LEVEL: str = "INFO"
    # local: trained models only | llm: LLM decides | hybrid: LLM only when local confidence is low
    INFERENCE_MODE: str = "local"
    HYBRID_CONFIDENCE_THRESHOLD: float = 0.75
    # Local decisions below this confidence are returned as "uncertain" (off when unset). Choose it on the
    # validation split: python -m tweet_sentiment_classifier.evaluation.abstain
    ABSTAIN_THRESHOLD: float | None = None
    # Request limits (a CSV row can cost one LLM call in llm/hybrid mode)
    MAX_UPLOAD_BYTES: int = 5_000_000
    MAX_CSV_ROWS: int = 1000
    MAX_CSV_ROWS_LLM: int = 100
    MAX_TEXT_CHARS: int = 2000
    MAX_QUERY_CHARS: int = 200
    # Topic monitoring. /ingest endpoints are disabled until INGEST_TOKEN is set. In llm/hybrid mode every
    # classified post can cost an LLM call, so MAX_POSTS_PER_RUN is also the per-source LLM cap.
    INGEST_TOKEN: str = ""
    INGEST_INTERVAL_MIN: int = 30
    MAX_POSTS_PER_RUN: int = 100
    MAX_TOPICS: int = 20
    POST_RETENTION_DAYS: int = 90
    # Alerts on a topic: the last ALERT_WINDOW_HOURS against the 7 days before it
    ALERT_WINDOW_HOURS: int = 24
    ALERT_MIN_POSTS: int = 20  # both windows need at least this many posts before any alert can fire
    ALERT_P_VALUE: float = 0.01
    ALERT_COOLDOWN_HOURS: int = 24
    DRIFT_PSI_ALERT: float = 0.25  # the usual "significant shift" level for the population stability index
    # LLM insights (themes, daily digest). Empty LLM_PROVIDER = groq when GROQ_API_KEY is set, else disabled;
    # the themes still work without an LLM (labelled by their top words).
    LLM_PROVIDER: str = ""  # groq | openai | anthropic
    LLM_MODEL: str = ""  # groq/anthropic have defaults; openai needs it set
    ANTHROPIC_API_KEY: str = ""
    OPENAI_API_KEY: str = ""
    LLM_DAILY_TOKEN_BUDGET: int = 200_000
    LLM_TIMEOUT_SECONDS: int = 30
    INSIGHT_INTERVAL_HOURS: int = 6
    INSIGHT_MIN_POSTS: int = 20
    THEME_WINDOW_HOURS: int = 72
    # Agent runs (every run is capped on steps, tokens and time; the endpoints need INGEST_TOKEN)
    AGENT_MAX_STEPS: int = 8
    AGENT_MAX_TOKENS: int = 20_000
    AGENT_TIMEOUT_SECONDS: int = 90
    # Hardening. The defaults keep local use frictionless; .env.docker turns them on for deployments.
    API_AUTH: bool = False  # true: creating/deleting topics and queueing annotations also need a key
    RATE_LIMIT_PER_MINUTE: int = 0  # per client, read requests (0 = off)
    RATE_LIMIT_WRITE_PER_MINUTE: int = 0  # per client, POST/PUT/DELETE requests (0 = off)
    TRUST_PROXY: bool = False  # true only behind a proxy you control: the client is the first X-Forwarded-For entry
    CORS_ORIGINS: str = ""  # comma-separated extra origins, e.g. the Vercel frontend https://my-app.vercel.app
    METRICS_TOKEN: str = ""  # when set, /metrics needs "Authorization: Bearer <token>"
    RSS_FEEDS: str = ""  # comma-separated https feed URLs, filtered by each topic's query

    class Config:
        env_file = ".env"
        extra = "ignore"

@lru_cache()
def get_settings() -> Settings:
    return Settings()

settings = get_settings()
