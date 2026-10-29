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
