from sqlalchemy import Column, Integer, Float, String, DateTime, Boolean, JSON
from datetime import datetime
from tweet_sentiment_classifier.config.database import Base

class PredictionLog(Base):
    __tablename__ = "prediction_logs"

    id = Column(Integer, primary_key=True, index=True)
    latency_ms = Column(Float, nullable=False)
    confidence = Column(Float, nullable=True)  # NULL when the decider (an LLM) reports no confidence
    model_used = Column(String, nullable=False)
    prediction = Column(String, nullable=False)
    text_length = Column(Integer, nullable=False)
    # The analysed text, so the Prediction History page can show it. Nullable so rows written by
    # older versions (and the anonymous monitoring flow) keep working; bounded to MAX_TEXT_CHARS.
    text = Column(String, nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow)

class ReportLog(Base):
    __tablename__ = "reports"
    
    id = Column(Integer, primary_key=True, index=True)
    query = Column(String, nullable=False) # Only metadata
    top_keywords = Column(JSON, nullable=True)
    sentiment_distribution = Column(JSON, nullable=True)
    fallback_used = Column(Boolean, default=False)
    timestamp = Column(DateTime, default=datetime.utcnow)

class MonitoringMetric(Base):
    __tablename__ = "monitoring_metrics"
    
    id = Column(Integer, primary_key=True, index=True)
    avg_latency = Column(Float, nullable=False)
    avg_confidence = Column(Float, nullable=False)
    fallback_rate = Column(Float, nullable=False)
    prediction_distribution = Column(JSON, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)


class LabelCorrection(Base):
    """A user saying a prediction is wrong. A candidate for annotation, never a training label on its own."""
    __tablename__ = "label_corrections"

    id = Column(Integer, primary_key=True, index=True)
    text = Column(String, nullable=False)
    predicted = Column(String, nullable=False)
    corrected = Column(String, nullable=False)
    decision_source = Column(String, nullable=True)
    reviewed = Column(Boolean, default=False, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
