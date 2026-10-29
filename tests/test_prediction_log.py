"""PredictionLog must accept the null confidence an LLM decision produces."""
from sqlalchemy import create_engine, func
from sqlalchemy.orm import sessionmaker

from tweet_sentiment_classifier.config.database import Base
from tweet_sentiment_classifier.entity.models import PredictionLog


def test_null_confidence_is_stored_and_ignored_by_average():
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    with sessionmaker(bind=engine)() as db:
        db.add_all([
            PredictionLog(latency_ms=5, confidence=0.8, model_used="local:logisticregression", prediction="positive", text_length=10),
            PredictionLog(latency_ms=900, confidence=None, model_used="llm:groq/llama-3.1-8b-instant", prediction="negative", text_length=10),
        ])
        db.commit()
        assert db.query(PredictionLog).count() == 2
        assert db.query(func.avg(PredictionLog.confidence)).scalar() == 0.8  # NULL is not counted as 0
