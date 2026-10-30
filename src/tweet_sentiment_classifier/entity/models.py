from sqlalchemy import Column, Integer, Float, String, DateTime, Boolean, JSON, ForeignKey, UniqueConstraint
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


class Topic(Base):
    """Something to monitor (a policy, brand, person or event) and where to look for posts about it."""
    __tablename__ = "topics"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False, unique=True)
    query = Column(String, nullable=False)
    sources = Column(JSON, nullable=False)  # list of source names, e.g. ["google_news"]
    active = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)


class Post(Base):
    """One collected public post, already classified. Unique per topic by source id and by normalised content."""
    __tablename__ = "posts"
    __table_args__ = (UniqueConstraint("topic_id", "external_id"), UniqueConstraint("topic_id", "content_hash"))

    id = Column(Integer, primary_key=True, index=True)
    topic_id = Column(Integer, ForeignKey("topics.id"), nullable=False, index=True)
    source = Column(String, nullable=False)
    external_id = Column(String, nullable=False)
    content_hash = Column(String, nullable=False)
    url = Column(String, nullable=True)
    text = Column(String, nullable=False)
    published_at = Column(DateTime, nullable=True)
    fetched_at = Column(DateTime, default=datetime.utcnow, index=True)
    prediction = Column(String, nullable=False)
    confidence = Column(Float, nullable=True)
    decision_source = Column(String, nullable=True)


class IngestionRun(Base):
    __tablename__ = "ingestion_runs"

    id = Column(Integer, primary_key=True, index=True)
    topic_id = Column(Integer, ForeignKey("topics.id"), nullable=False, index=True)
    source = Column(String, nullable=False)
    started_at = Column(DateTime, nullable=False)
    finished_at = Column(DateTime, nullable=True)
    fetched = Column(Integer, default=0, nullable=False)
    new = Column(Integer, default=0, nullable=False)
    error = Column(String, nullable=True)


class Alert(Base):
    """A statistically unusual change on a topic (see services/topic_analytics.py)."""
    __tablename__ = "alerts"

    id = Column(Integer, primary_key=True, index=True)
    topic_id = Column(Integer, ForeignKey("topics.id"), nullable=False, index=True)
    kind = Column(String, nullable=False)  # negative_spike | volume_spike | class_drift
    window_hours = Column(Integer, nullable=False)
    baseline = Column(Float, nullable=False)
    observed = Column(Float, nullable=False)
    p_value = Column(Float, nullable=True)
    message = Column(String, nullable=False)
    acknowledged = Column(Boolean, default=False, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)


class LlmCall(Base):
    """One LLM request: the cache (same key, ok=True) and the token ledger behind the daily budget."""
    __tablename__ = "llm_calls"

    id = Column(Integer, primary_key=True, index=True)
    key = Column(String, nullable=False, index=True)  # sha256 of provider, model, prompt version and input
    provider = Column(String, nullable=False)
    model = Column(String, nullable=False)
    prompt_version = Column(String, nullable=False)
    tokens_in = Column(Integer, default=0, nullable=False)
    tokens_out = Column(Integer, default=0, nullable=False)
    ok = Column(Boolean, default=False, nullable=False)
    response = Column(JSON, nullable=True)  # the validated answer when ok
    created_at = Column(DateTime, default=datetime.utcnow, index=True)


class Insight(Base):
    """Generated understanding of a topic: its themes, or the daily "what changed" digest."""
    __tablename__ = "insights"

    id = Column(Integer, primary_key=True, index=True)
    topic_id = Column(Integer, ForeignKey("topics.id"), nullable=False, index=True)
    kind = Column(String, nullable=False)  # themes | digest
    day = Column(String, nullable=False, index=True)  # YYYY-MM-DD (UTC)
    payload = Column(JSON, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)


class AgentRun(Base):
    """One agent session: the question, every step it took (the trace), its verified answer and what it cost."""
    __tablename__ = "agent_runs"

    id = Column(Integer, primary_key=True, index=True)
    kind = Column(String, nullable=False)  # report | ask
    question = Column(String, nullable=False)
    topic_id = Column(Integer, nullable=True, index=True)
    status = Column(String, nullable=False)  # running | done | failed | incomplete | limit_reached
    steps = Column(JSON, nullable=False)
    answer = Column(JSON, nullable=True)
    error = Column(String, nullable=True)
    tokens_in = Column(Integer, default=0, nullable=False)
    tokens_out = Column(Integer, default=0, nullable=False)
    latency_ms = Column(Float, nullable=True)
    provider = Column(String, nullable=True)
    model = Column(String, nullable=True)
    prompt_version = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    finished_at = Column(DateTime, nullable=True)


class AnnotationItem(Base):
    """A post worth a human label (the model was unsure). Candidates for annotation, never labels."""
    __tablename__ = "annotation_queue"

    id = Column(Integer, primary_key=True, index=True)
    post_id = Column(Integer, ForeignKey("posts.id"), nullable=False, unique=True)
    topic_id = Column(Integer, nullable=False, index=True)
    reason = Column(String, nullable=False)  # uncertain | low_confidence
    priority = Column(Float, nullable=False)  # higher = less sure
    status = Column(String, default="pending", nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)


class ApiKey(Base):
    """Only the SHA-256 of a key is stored: the key itself is shown once, when it is created."""
    __tablename__ = "api_keys"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    key_hash = Column(String, nullable=False, unique=True)
    revoked = Column(Boolean, default=False, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

