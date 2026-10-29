import os
import tempfile

# Must run before any application module reads settings: tests never touch the repo's monitoring.db
# and never see real API keys, whatever a developer has in .env.
os.environ["DATABASE_URL"] = "sqlite:///" + os.path.join(tempfile.mkdtemp(prefix="sentiment_tests_"), "test.db").replace("\\", "/")
for _key in ("GROQ_API_KEY", "HF_API_TOKEN", "HF_TOKEN"):
    os.environ[_key] = ""
os.environ["INFERENCE_MODE"] = "local"

import numpy as np
import pytest

from tweet_sentiment_classifier.pipeline.prediction_pipeline import PredictionPipeline


class StubVectorizer:
    def transform(self, texts):
        return texts


class StubProbaModel:
    """Behaves like a fitted sklearn classifier exposing predict_proba."""
    def __init__(self, probs, classes=(-1, 0, 1)):
        self.classes_ = np.array(classes)
        self._probs = np.array([probs])

    def predict_proba(self, X):
        return self._probs


class StubMarginModel:
    """LinearSVC-like: decision_function returns one score per class, no predict_proba."""
    classes_ = np.array([-1, 0, 1])

    def decision_function(self, X):
        return np.array([[-1.0, 0.2, 1.5]])

    def predict(self, X):
        return np.array([1])


@pytest.fixture
def make_pipeline():
    """Build a PredictionPipeline without loading config, models, or any network client."""
    def _make(mode="local", fast_model=None, threshold=0.75, distilbert=None, llm=None):
        p = object.__new__(PredictionPipeline)
        p.inference_mode = mode
        p.confidence_threshold = threshold
        p.fast_model_name = "LogisticRegression"
        p.fast_model = fast_model
        p.vectorizer = StubVectorizer()
        p.db_model = None
        p.db_tokenizer = None
        p.data_transformer = type("T", (), {"clean_text": staticmethod(lambda t: t.lower())})()
        p.llm_calls = 0

        if distilbert is not None:  # (label, confidence) or an Exception to raise
            p.db_model = object()  # marks the tier as loaded
            def _db(_text):
                if isinstance(distilbert, Exception):
                    raise distilbert
                return distilbert
            p._predict_distilbert = _db

        async def _llm(_text):
            p.llm_calls += 1
            return llm
        p._predict_llm_async = _llm
        return p
    return _make


# ---- shared API test fixtures ---------------------------------------------------------------

LOCAL = {"prediction": "positive", "confidence": 0.9, "decision_source": "local:logisticregression",
         "model_used": "local:logisticregression", "explanation": None}
LLM = {"prediction": "negative", "confidence": None, "decision_source": "llm:groq/llama-3.1-8b-instant",
       "model_used": "llm:groq/llama-3.1-8b-instant", "explanation": "angry"}


class FakePipeline:
    inference_mode = "local"

    def __init__(self, by_text=None, default=LOCAL):
        self.by_text, self.default = by_text or {}, default

    async def predict(self, text):
        return dict(self.by_text.get(text, self.default))


@pytest.fixture(scope="module")
def app_module():
    import app as app_mod
    return app_mod


@pytest.fixture
def env(app_module, monkeypatch, tmp_path_factory):
    from fastapi.testclient import TestClient
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from tweet_sentiment_classifier.config.database import Base, get_db

    # A file-backed DB (not one shared in-memory connection): commits run in worker threads, and like a real
    # deployment each session needs its own connection.
    db_file = tmp_path_factory.mktemp("db") / "api.db"
    engine = create_engine(f"sqlite:///{db_file.as_posix()}", connect_args={"check_same_thread": False, "timeout": 30})
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
