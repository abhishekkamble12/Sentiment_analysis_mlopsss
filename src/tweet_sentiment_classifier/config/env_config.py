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
    RSS_FEEDS: str = ""  # comma-separated https feed URLs, filtered by each topic's query

    class Config:
        env_file = ".env"
        extra = "ignore"

@lru_cache()
def get_settings() -> Settings:
    return Settings()

settings = get_settings()
