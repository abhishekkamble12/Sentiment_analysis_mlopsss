"""Topics to monitor, the posts collected for them, and the endpoints that trigger collection."""
import hmac
from typing import List

from fastapi import APIRouter, Depends, Header, HTTPException
from pydantic import BaseModel
from sqlalchemy import func
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from tweet_sentiment_classifier.app.deps import get_container, get_db
from tweet_sentiment_classifier.app.services import ingestion_service
from tweet_sentiment_classifier.entity.models import Alert, IngestionRun, Insight, Post, Topic

router = APIRouter(tags=["topics"])

SENTIMENTS = {"positive", "neutral", "negative", "uncertain"}


class TopicRequest(BaseModel):
    name: str
    query: str
    sources: List[str] = ["google_news"]


def require_ingest_token(x_ingest_token: str = Header(default=""), c=Depends(get_container)) -> None:
    expected = c.settings.INGEST_TOKEN
    if not expected:
        raise HTTPException(status_code=503, detail="Ingestion endpoints are disabled: set INGEST_TOKEN.")
    if not hmac.compare_digest(x_ingest_token.encode(), expected.encode()):
        raise HTTPException(status_code=403, detail="Invalid ingest token.")


def _topic_or_404(db: Session, topic_id: int) -> Topic:
    topic = db.get(Topic, topic_id)
    if topic is None:
        raise HTTPException(status_code=404, detail="Topic not found.")
    return topic


@router.post("/topics", status_code=201)
def create_topic(req: TopicRequest, db: Session = Depends(get_db), c=Depends(get_container)):
    name, query = req.name.strip(), req.query.strip()
    if not name or not query:
        raise HTTPException(status_code=400, detail="Name and query cannot be empty.")
    if len(name) > 100 or len(query) > c.settings.MAX_QUERY_CHARS:
        raise HTTPException(status_code=413, detail="Name or query is too long.")
    sources = list(dict.fromkeys(req.sources)) or ["google_news"]
    unknown = [s for s in sources if s not in c.sources]
    if unknown:
        raise HTTPException(status_code=400, detail=f"Unknown or unconfigured source(s): {', '.join(unknown)}.")
    if db.query(func.count(Topic.id)).scalar() >= c.settings.MAX_TOPICS:
        raise HTTPException(status_code=409, detail=f"At most {c.settings.MAX_TOPICS} topics are allowed.")
    topic = Topic(name=name, query=query, sources=sources)
    db.add(topic)
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        raise HTTPException(status_code=409, detail="A topic with this name already exists.")
    return {"id": topic.id, "name": topic.name, "query": topic.query, "sources": topic.sources}


@router.get("/topics")
def list_topics(db: Session = Depends(get_db)):
    counts = dict(db.query(Post.topic_id, func.count(Post.id)).group_by(Post.topic_id).all())
    last = dict(db.query(IngestionRun.topic_id, func.max(IngestionRun.finished_at)).group_by(IngestionRun.topic_id).all())
    return {"topics": [
        {"id": t.id, "name": t.name, "query": t.query, "sources": t.sources, "active": t.active,
         "posts": counts.get(t.id, 0), "last_ingested": last[t.id].isoformat() + "Z" if last.get(t.id) else None}
        for t in db.query(Topic).order_by(Topic.id).all()
    ]}


@router.delete("/topics/{topic_id}")
def delete_topic(topic_id: int, db: Session = Depends(get_db)):
    topic = _topic_or_404(db, topic_id)
    db.query(Post).filter(Post.topic_id == topic_id).delete()
    db.query(IngestionRun).filter(IngestionRun.topic_id == topic_id).delete()
    db.query(Alert).filter(Alert.topic_id == topic_id).delete()
    db.query(Insight).filter(Insight.topic_id == topic_id).delete()
    db.delete(topic)
    db.commit()
    return {"status": "deleted"}


@router.get("/topics/{topic_id}/posts")
def list_posts(topic_id: int, source: str | None = None, sentiment: str | None = None,
               page: int = 1, limit: int = 20, db: Session = Depends(get_db)):
    _topic_or_404(db, topic_id)
    if sentiment and sentiment not in SENTIMENTS:
        raise HTTPException(status_code=400, detail="sentiment must be positive, neutral, negative or uncertain.")
    page, limit = max(1, page), max(1, min(limit, 100))
    query = db.query(Post).filter(Post.topic_id == topic_id)
    if source:
        query = query.filter(Post.source == source)
    if sentiment:
        query = query.filter(Post.prediction == sentiment)
    total = query.count()
    rows = query.order_by(Post.published_at.desc().nullslast(), Post.id.desc()).offset((page - 1) * limit).limit(limit).all()
    return {"total": total, "page": page, "limit": limit, "total_pages": (total + limit - 1) // limit, "posts": [
        {"id": p.id, "source": p.source, "url": p.url, "text": p.text, "prediction": p.prediction,
         "confidence": p.confidence, "decision_source": p.decision_source,
         "published_at": p.published_at.isoformat() + "Z" if p.published_at else None}
        for p in rows
    ]}


@router.post("/topics/{topic_id}/ingest", dependencies=[Depends(require_ingest_token)])
async def ingest_one(topic_id: int, db: Session = Depends(get_db), c=Depends(get_container)):
    return {"runs": await ingestion_service.ingest_topic(c, db, _topic_or_404(db, topic_id))}


@router.post("/ingest/run", dependencies=[Depends(require_ingest_token)])
async def ingest_everything(db: Session = Depends(get_db), c=Depends(get_container)):
    """For hosts without a background worker: call this from a cron (for example a GitHub Actions schedule)."""
    return await ingestion_service.ingest_all(c, db)
