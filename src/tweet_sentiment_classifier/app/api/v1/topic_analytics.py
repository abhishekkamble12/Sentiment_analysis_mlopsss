"""Trends, drift and alerts for a topic."""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from tweet_sentiment_classifier.app.api.v1.topics import _topic_or_404
from tweet_sentiment_classifier.app.deps import get_container, get_db
from tweet_sentiment_classifier.app.services import topic_analytics
from tweet_sentiment_classifier.entity.models import Alert

router = APIRouter(tags=["topic analytics"])


def _alert_json(a: Alert) -> dict:
    return {"id": a.id, "topic_id": a.topic_id, "kind": a.kind, "window_hours": a.window_hours,
            "baseline": a.baseline, "observed": a.observed, "p_value": a.p_value, "message": a.message,
            "acknowledged": a.acknowledged, "created_at": a.created_at.isoformat() + "Z"}


@router.get("/topics/{topic_id}/trends")
def get_trends(topic_id: int, bucket: str = "day", days: int = 14, db: Session = Depends(get_db)):
    _topic_or_404(db, topic_id)
    if bucket not in ("hour", "day"):
        raise HTTPException(status_code=400, detail="bucket must be hour or day.")
    return {"bucket": bucket, "points": topic_analytics.trends(db, topic_id, bucket, max(1, min(days, 90)))}


@router.get("/topics/{topic_id}/drift")
def get_drift(topic_id: int, window_hours: int | None = None, db: Session = Depends(get_db), c=Depends(get_container)):
    _topic_or_404(db, topic_id)
    hours = max(1, min(window_hours or c.settings.ALERT_WINDOW_HOURS, 24 * 7))
    return topic_analytics.drift_report(db, topic_id, getattr(c.pipeline, "vectorizer", None), hours)


@router.get("/topics/{topic_id}/alerts")
def list_alerts(topic_id: int, db: Session = Depends(get_db)):
    _topic_or_404(db, topic_id)
    rows = db.query(Alert).filter(Alert.topic_id == topic_id).order_by(Alert.id.desc()).limit(50).all()
    return {"alerts": [_alert_json(a) for a in rows]}


@router.post("/alerts/{alert_id}/ack")
def acknowledge_alert(alert_id: int, db: Session = Depends(get_db)):
    alert = db.get(Alert, alert_id)
    if alert is None:
        raise HTTPException(status_code=404, detail="Alert not found.")
    alert.acknowledged = True
    db.commit()
    return _alert_json(alert)
