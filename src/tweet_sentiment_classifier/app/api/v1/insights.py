"""LLM-generated understanding of a topic (themes, daily digest) and the LLM token budget."""
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from tweet_sentiment_classifier.app.api.v1.topics import _topic_or_404, require_ingest_token
from tweet_sentiment_classifier.app.deps import get_container, get_db
from tweet_sentiment_classifier.app.services import insight_service

router = APIRouter(tags=["insights"])


def _llm_info(c) -> dict:
    p = c.llm.provider
    return {"configured": p is not None, "provider": p.name if p else None, "model": p.model if p else None}


@router.get("/topics/{topic_id}/insights")
def get_insights(topic_id: int, db: Session = Depends(get_db), c=Depends(get_container)):
    _topic_or_404(db, topic_id)
    return {"themes": insight_service.latest_json(db, topic_id, "themes"),
            "digest": insight_service.latest_json(db, topic_id, "digest"), "llm": _llm_info(c)}


@router.post("/topics/{topic_id}/insights/run", dependencies=[Depends(require_ingest_token)])
def run_insights(topic_id: int, db: Session = Depends(get_db), c=Depends(get_container)):
    """Rebuild now (this can spend LLM tokens, so it needs the ingest token). Sync on purpose: LLM calls block."""
    topic = _topic_or_404(db, topic_id)
    return insight_service.refresh_topic(db, topic, c.llm, c.settings, force=True)


@router.get("/llm/usage")
def llm_usage(db: Session = Depends(get_db), c=Depends(get_container)):
    return {**_llm_info(c), "tokens_today": c.llm.tokens_today(db), "daily_budget": c.llm.daily_token_budget}
