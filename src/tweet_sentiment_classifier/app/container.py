"""Everything the request handlers share: settings, the prediction pipeline, the agents and the caches.

One object, created at startup and kept on `app.state.container`; handlers receive it through
`deps.get_container`, so tests can swap any member (for example `container.pipeline`).
"""
import logging

from tweet_sentiment_classifier.app.core.cache import TTLCache
from tweet_sentiment_classifier.app.sources.rss import build_sources
from tweet_sentiment_classifier.config.env_config import get_settings

logger = logging.getLogger("api")


def _build_agent(name: str):
    """Build a LangChain agent, or None if its dependencies/keys are missing (the endpoints say so)."""
    try:
        from tweet_sentiment_classifier.components import agents
        return getattr(agents, name)()
    except Exception as e:
        logger.warning(f"{name} unavailable: {e}")
        return None


class Container:
    def __init__(self, pipeline=None, search_agent=None, report_agent=None):
        self.settings = get_settings()
        self.pipeline = pipeline
        self.search_agent = search_agent
        self.report_agent = report_agent
        self.predict_cache = TTLCache(ttl_seconds=3600, max_size=1000)
        self.search_analysis_cache = TTLCache(ttl_seconds=3600, max_size=500)
        self.sources = build_sources(self.settings.RSS_FEEDS)  # tests replace these with fakes
        self.app = None  # set by create_app

    @classmethod
    def build(cls) -> "Container":
        try:
            logger.info("Initializing PredictionPipeline...")
            from tweet_sentiment_classifier.pipeline.prediction_pipeline import PredictionPipeline
            # INFERENCE_MODE (local | llm | hybrid) is read from the environment; default is local-only.
            pipeline = PredictionPipeline()
        except Exception as e:
            logger.error(f"Failed to initialize full pipeline: {e}")
            pipeline = None
        logger.info("Initializing LangChain Agents...")
        return cls(pipeline, _build_agent("SearchAgent"), _build_agent("ReportAgent"))
