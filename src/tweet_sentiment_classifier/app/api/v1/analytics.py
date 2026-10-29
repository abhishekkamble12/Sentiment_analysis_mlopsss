"""Read-only analytics over the prediction log."""
import json
from datetime import datetime, timedelta
from typing import Dict

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import func
from sqlalchemy.orm import Session

from tweet_sentiment_classifier.config.database import get_db
from tweet_sentiment_classifier.entity.models import PredictionLog, ReportLog

router = APIRouter(tags=["analytics"])

LABEL_ALIASES = {"-1": "negative", "-1.0": "negative", "0": "neutral", "0.0": "neutral", "1": "positive", "1.0": "positive"}


def _normalize_label(label) -> str:
    """Older log rows stored numeric labels ('-1','0','1'); newer ones store names."""
    return LABEL_ALIASES.get(str(label), str(label))


def _is_llm(model_used: str) -> bool:
    m = (model_used or "").lower()
    return m.startswith("llm:") or "gemma" in m or "groq" in m  # the latter two match legacy rows


def _label_counts(db: Session) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    rows = db.query(PredictionLog.prediction, func.count(PredictionLog.id)).group_by(PredictionLog.prediction).all()
    for label, n in rows:
        key = _normalize_label(label)
        counts[key] = counts.get(key, 0) + n
    return counts


@router.get("/analytics/monitoring")
def get_monitoring(db: Session = Depends(get_db)):
    total = db.query(func.count(PredictionLog.id)).scalar() or 0
    avg_lat = db.query(func.avg(PredictionLog.latency_ms)).scalar() or 0.0
    avg_conf = db.query(func.avg(PredictionLog.confidence)).scalar()  # NULL (LLM) rows are ignored
    confidence_n = db.query(func.count(PredictionLog.confidence)).scalar() or 0

    usage = dict(db.query(PredictionLog.model_used, func.count(PredictionLog.id)).group_by(PredictionLog.model_used).all())
    llm_total = sum(n for model, n in usage.items() if _is_llm(model))

    total_reports = db.query(ReportLog).count()
    search_fallbacks = db.query(ReportLog).filter(ReportLog.fallback_used == True).count()  # noqa: E712

    return {
        "total_predictions": total,
        "avg_latency_ms": round(avg_lat, 2),
        "avg_confidence": None if avg_conf is None else round(avg_conf, 4),
        "confidence_n": confidence_n,
        "prediction_distribution": _label_counts(db),
        "model_usage": usage,
        "llm_share": round(llm_total / total, 4) if total else 0.0,
        # share of search analyses that had to use the RSS fallback (unrelated to the LLM)
        "search_fallback_rate": round(search_fallbacks / total_reports, 4) if total_reports else 0.0,
    }


@router.get("/analytics/recent")
def get_recent_logs(limit: int = 10, db: Session = Depends(get_db)):
    limit = max(1, min(limit, 100))
    rows = db.query(PredictionLog).order_by(PredictionLog.id.desc()).limit(limit).all()
    return {"logs": [
        {
            "id": r.id,
            "timestamp": r.timestamp.isoformat() + "Z" if r.timestamp else None,  # stored as naive UTC
            "model_used": r.model_used,
            "prediction": _normalize_label(r.prediction),
            "latency_ms": round(r.latency_ms, 1),
            "confidence": r.confidence,
            "llm": _is_llm(r.model_used),
        }
        for r in rows
    ]}


def _aliases_for(label: str) -> list[str]:
    """Stored values that normalise to one sentiment (newer rows use names, older rows use numbers)."""
    aliases = [label]
    for stored, normalized in LABEL_ALIASES.items():
        if normalized == label:
            aliases.append(stored)
    return aliases


@router.get("/analytics/history")
def get_prediction_history(
        q: str | None = None,
        sentiment: str | None = None,
        start: str | None = None,
        end: str | None = None,
        page: int = 1,
        limit: int = 20,
        db: Session = Depends(get_db)):
    """Persisted predictions, searchable and filterable. Older rows have NULL text."""
    sentiment = (sentiment or "").strip().lower() or None
    if sentiment and sentiment not in ("positive", "neutral", "negative"):
        raise HTTPException(status_code=400, detail="sentiment must be positive, neutral or negative.")
    page = max(1, page)
    limit = max(1, min(limit, 100))

    query = db.query(PredictionLog)
    if q:
        query = query.filter(PredictionLog.text.ilike(f"%{q.strip()}%"))
    if sentiment:
        query = query.filter(PredictionLog.prediction.in_(_aliases_for(sentiment)))
    if start:
        try:
            query = query.filter(PredictionLog.timestamp >= datetime.fromisoformat(start))
        except ValueError:
            raise HTTPException(status_code=400, detail="start must be an ISO date (e.g. 2026-10-29).")
    if end:
        try:
            end_dt = datetime.fromisoformat(end)
            # A date-only end is treated as inclusive of that whole day
            if end_dt.hour == 0 and end_dt.minute == 0 and end_dt.second == 0:
                end_dt = end_dt + timedelta(days=1)
            query = query.filter(PredictionLog.timestamp < end_dt)
        except ValueError:
            raise HTTPException(status_code=400, detail="end must be an ISO date (e.g. 2026-10-29).")

    total = query.count()
    rows = query.order_by(PredictionLog.id.desc()).offset((page - 1) * limit).limit(limit).all()
    logs = [{
        "id": r.id,
        "timestamp": r.timestamp.isoformat() + "Z" if r.timestamp else None,
        "text": r.text,
        "prediction": _normalize_label(r.prediction),
        "model_used": r.model_used,
        "confidence": r.confidence,
        "latency_ms": round(r.latency_ms, 1),
        "text_length": r.text_length,
        "llm": _is_llm(r.model_used),
    } for r in rows]

    return {"logs": logs, "total": total, "page": page, "limit": limit,
            "total_pages": (total + limit - 1) // limit}


@router.get("/analytics/summary")
def get_analytics(db: Session = Depends(get_db)):
    total = db.query(func.count(PredictionLog.id)).scalar() or 0
    avg_lat = db.query(func.avg(PredictionLog.latency_ms)).scalar() or 0.0
    avg_conf = db.query(func.avg(PredictionLog.confidence)).scalar()

    dist = _label_counts(db)
    for key in ("positive", "neutral", "negative"):
        dist.setdefault(key, 0)

    # (positive - negative) / total mapped to 0..100; undefined when there is no data
    sentiment_index = None
    if total:
        sentiment_index = round((dist["positive"] - dist["negative"]) / total * 50 + 50, 1)

    low_confidence = db.query(PredictionLog).filter(PredictionLog.confidence < 0.6).count()

    # Hourly trend bucketed in Python, so it works on every database (strftime is SQLite-only)
    rows = db.query(PredictionLog.timestamp, PredictionLog.prediction) \
        .order_by(PredictionLog.timestamp.desc()).limit(20000).all()
    buckets: Dict[str, Dict[str, int]] = {}
    for ts, pred in rows:
        if ts is None:
            continue
        hour = ts.strftime("%m-%d %H:00")
        bucket = buckets.setdefault(hour, {"positive": 0, "neutral": 0, "negative": 0})
        key = _normalize_label(pred)
        if key in bucket:
            bucket[key] += 1
    hours = sorted(buckets)[-10:]
    trends = {
        "categories": hours,
        "positive": [buckets[h]["positive"] for h in hours],
        "neutral": [buckets[h]["neutral"] for h in hours],
        "negative": [buckets[h]["negative"] for h in hours],
    }

    recent_reports = db.query(ReportLog).order_by(ReportLog.timestamp.desc()).limit(10).all()
    kw_freq: Dict[str, int] = {}
    for r in recent_reports:
        kws = r.top_keywords
        if isinstance(kws, str):
            try:
                kws = json.loads(kws)
            except ValueError:
                kws = []
        for kw in kws or []:
            if isinstance(kw, str) and len(kw.strip()) > 1:
                kw_freq[kw] = kw_freq.get(kw, 0) + 1
    top_keywords = [k for k, _ in sorted(kw_freq.items(), key=lambda x: x[1], reverse=True)[:6]]

    return {
        "total_insights": total,
        "avg_latency_ms": round(avg_lat, 2),
        "avg_confidence": None if avg_conf is None else round(avg_conf, 4),
        "sentiment_index": sentiment_index,
        "low_confidence_count": low_confidence,
        "sentiment_distribution": dist,
        "trends": trends,
        "top_keywords": top_keywords,  # [] when no search reports exist
        "reports_considered": len(recent_reports),
    }

