"""Collect posts for a topic from its sources, drop duplicates, classify the new ones and store them."""
import asyncio
import hashlib
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List

from sqlalchemy.orm import Session

from tweet_sentiment_classifier.app.core import metrics
from tweet_sentiment_classifier.app.services import topic_analytics
from tweet_sentiment_classifier.components.data_transformation import clean_text
from tweet_sentiment_classifier.entity.models import IngestionRun, Post, Topic

logger = logging.getLogger("api")
MAX_FETCHED = 300  # shortcut: the duplicate lookups use one IN() list, so cap what a single fetch can bring


def content_hash(text: str) -> str:
    return hashlib.sha1(clean_text(text).encode()).hexdigest()


def _unseen(db: Session, topic_id: int, raw: list) -> list:
    """Posts not yet stored for this topic, by source id and by normalised content (also within the batch)."""
    ids = {p.external_id for p in raw}
    hashes = {content_hash(p.text) for p in raw}
    known_ids = {r[0] for r in db.query(Post.external_id).filter(Post.topic_id == topic_id, Post.external_id.in_(ids))}
    known_hashes = {r[0] for r in db.query(Post.content_hash).filter(
        Post.topic_id == topic_id, Post.content_hash.in_(hashes))}
    fresh = []
    for p in raw:
        h = content_hash(p.text)
        if p.external_id in known_ids or h in known_hashes:
            continue
        known_ids.add(p.external_id)
        known_hashes.add(h)
        fresh.append((p, h))
    return fresh


async def _ingest_source(c, db: Session, topic: Topic, name: str) -> Dict[str, Any]:
    run = IngestionRun(topic_id=topic.id, source=name, started_at=datetime.utcnow(), fetched=0, new=0)
    try:
        source = c.sources.get(name)
        if source is None:
            raise ValueError(f"unknown or unconfigured source '{name}'")
        if c.pipeline is None:
            raise RuntimeError("prediction pipeline is not available")
        raw = (await asyncio.to_thread(source.fetch, topic.query))[:MAX_FETCHED]
        run.fetched = len(raw)
        for post, digest in _unseen(db, topic.id, raw)[:c.settings.MAX_POSTS_PER_RUN]:
            try:
                res = await c.pipeline.predict(post.text[:c.settings.MAX_TEXT_CHARS])
            except Exception as e:  # one unclassifiable post is skipped, not fatal
                logger.warning(f"classification failed for a post of topic {topic.id}: {e}")
                continue
            db.add(Post(topic_id=topic.id, source=name, external_id=post.external_id, content_hash=digest,
                        url=post.url, text=post.text, published_at=post.published_at,
                        prediction=res.get("prediction", "unknown"), confidence=res.get("confidence"),
                        decision_source=res.get("decision_source")))
            run.new += 1
        metrics.INGESTED_POSTS.labels(name).inc(run.new)
    except Exception as e:
        db.rollback()  # drops this source's half-added posts; the run row below is still recorded
        metrics.INGEST_ERRORS.labels(name).inc()
        logger.warning(f"ingestion of '{name}' for topic {topic.id} failed: {e}")
        run = IngestionRun(topic_id=topic.id, source=name, started_at=run.started_at, fetched=0, new=0,
                           error=str(e)[:300])
    run.finished_at = datetime.utcnow()
    db.add(run)
    db.commit()
    return {"source": name, "fetched": run.fetched, "new": run.new, "error": run.error}


async def ingest_topic(c, db: Session, topic: Topic) -> List[Dict[str, Any]]:
    """Every source of the topic, one after the other; a failing source does not stop the rest."""
    runs = [await _ingest_source(c, db, topic, name) for name in topic.sources]
    try:
        topic_analytics.evaluate_alerts(db, topic, getattr(c.pipeline, "vectorizer", None), c.settings)
    except Exception as e:  # alerting must never turn a successful ingestion into a failure
        db.rollback()
        logger.warning(f"alert evaluation failed for topic {topic.id}: {e}")
    return runs


async def ingest_all(c, db: Session) -> Dict[str, Any]:
    topics = db.query(Topic).filter(Topic.active.is_(True)).all()
    return {t.name: await ingest_topic(c, db, t) for t in topics}


def purge_old_posts(db: Session, days: int) -> int:
    deleted = db.query(Post).filter(Post.fetched_at < datetime.utcnow() - timedelta(days=days)).delete()
    db.commit()
    return deleted
