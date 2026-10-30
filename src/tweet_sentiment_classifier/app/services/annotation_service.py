"""Posts worth a human label: the ones the model was unsure about.

These are candidates for new annotation rounds. They must never be added to the frozen gold test set, and the
export deliberately omits the model's label so annotators are not anchored by it (see docs/annotation_guidelines.md).
"""
from typing import Any, Dict, List, Optional

from sqlalchemy import or_
from sqlalchemy.orm import Session

from tweet_sentiment_classifier.entity.models import AnnotationItem, Post

LOW_CONFIDENCE = 0.6


def queue_uncertain(db: Session, topic_id: int, limit: int = 50, threshold: float = LOW_CONFIDENCE) -> int:
    """Queue the least sure posts of a topic that are not queued yet; returns how many were added."""
    queued = db.query(AnnotationItem.post_id).filter(AnnotationItem.topic_id == topic_id)
    candidates = (db.query(Post.id, Post.prediction, Post.confidence)
                  .filter(Post.topic_id == topic_id, Post.id.notin_(queued),
                          or_(Post.prediction == "uncertain", Post.confidence < threshold)).limit(2000).all())
    ranked = sorted(((1.0 if pred == "uncertain" or conf is None else 1.0 - conf, pid, pred)
                     for pid, pred, conf in candidates), reverse=True)[:limit]
    for priority, post_id, prediction in ranked:
        db.add(AnnotationItem(post_id=post_id, topic_id=topic_id, priority=round(priority, 4),
                              reason="uncertain" if prediction == "uncertain" else "low_confidence"))
    db.commit()
    return len(ranked)


def pending(db: Session, topic_id: Optional[int] = None, limit: int = 100) -> List[Dict[str, Any]]:
    query = (db.query(AnnotationItem, Post).join(Post, Post.id == AnnotationItem.post_id)
             .filter(AnnotationItem.status == "pending"))
    if topic_id is not None:
        query = query.filter(AnnotationItem.topic_id == topic_id)
    return [{"id": item.id, "post_id": post.id, "topic_id": item.topic_id, "text": post.text, "url": post.url,
             "reason": item.reason, "priority": item.priority}
            for item, post in query.order_by(AnnotationItem.priority.desc(), AnnotationItem.id).limit(limit).all()]


def label_studio_tasks(db: Session, topic_id: Optional[int] = None, limit: int = 100) -> List[Dict[str, Any]]:
    """Import file for Label Studio (docs/label_studio_config.xml reads $text). No model label is included."""
    return [{"data": {"text": row["text"], "id": f"post-{row['post_id']}"}} for row in pending(db, topic_id, limit)]
