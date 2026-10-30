"""Agent sessions (research report, ask your data) and the human-annotation queue."""
from typing import Literal, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session, sessionmaker

from tweet_sentiment_classifier.app import prompts
from tweet_sentiment_classifier.app.agent import runner
from tweet_sentiment_classifier.app.api.v1.topics import _topic_or_404, require_ingest_token
from tweet_sentiment_classifier.app.deps import get_container, get_db
from tweet_sentiment_classifier.app.services import annotation_service
from tweet_sentiment_classifier.entity.models import AgentRun

router = APIRouter(tags=["agent"])
MAX_RUNNING = 3


class RunRequest(BaseModel):
    kind: Literal["report", "ask"]
    question: Optional[str] = None
    topic_id: Optional[int] = None


def _run_json(r: AgentRun) -> dict:
    return {"id": r.id, "kind": r.kind, "question": r.question, "topic_id": r.topic_id, "status": r.status,
            "steps": r.steps, "answer": r.answer, "error": r.error, "tokens_in": r.tokens_in,
            "tokens_out": r.tokens_out, "latency_ms": r.latency_ms, "provider": r.provider, "model": r.model,
            "created_at": r.created_at.isoformat() + "Z"}


@router.post("/agent/runs", status_code=202, dependencies=[Depends(require_ingest_token)])
def start_run(req: RunRequest, background: BackgroundTasks, db: Session = Depends(get_db), c=Depends(get_container)):
    """Start a session in the background; poll GET /agent/runs/{id}. It spends LLM tokens, so it needs the token."""
    if not c.llm.configured:
        raise HTTPException(status_code=503, detail="The agent needs an LLM: set GROQ_API_KEY or LLM_PROVIDER.")
    question = (req.question or "").strip()
    if req.kind == "ask" and not question:
        raise HTTPException(status_code=400, detail="An ask session needs a question.")
    if len(question) > 500:
        raise HTTPException(status_code=413, detail="The question is too long (max 500 characters).")
    if req.kind == "report":
        if req.topic_id is None:
            raise HTTPException(status_code=400, detail="A report needs a topic_id.")
        question = question or f"Write a research report on the topic '{_topic_or_404(db, req.topic_id).name}'."
    elif req.topic_id is not None:
        _topic_or_404(db, req.topic_id)
    if db.query(AgentRun).filter(AgentRun.status == "running").count() >= MAX_RUNNING:
        raise HTTPException(status_code=429, detail="Too many agent sessions are running; try again shortly.")

    p = c.llm.provider
    run = AgentRun(kind=req.kind, question=question, topic_id=req.topic_id, status="running", steps=[],
                   provider=p.name, model=p.model, prompt_version=prompts.AGENT_VERSION)
    db.add(run)
    db.commit()
    background.add_task(runner.execute, sessionmaker(bind=db.get_bind()), c, run.id)
    return {"id": run.id, "status": "running"}


@router.get("/agent/runs/{run_id}")
def get_run(run_id: int, db: Session = Depends(get_db)):
    run = db.get(AgentRun, run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found.")
    return _run_json(run)


@router.get("/agent/runs")
def list_runs(limit: int = 20, db: Session = Depends(get_db)):
    rows = db.query(AgentRun).order_by(AgentRun.id.desc()).limit(max(1, min(limit, 100))).all()
    return {"runs": [{k: v for k, v in _run_json(r).items() if k not in ("steps", "answer")} for r in rows]}


@router.post("/annotation/refresh")
def refresh_queue(topic_id: int, limit: int = 50, db: Session = Depends(get_db)):
    _topic_or_404(db, topic_id)
    return {"queued": annotation_service.queue_uncertain(db, topic_id, max(1, min(limit, 500)))}


@router.get("/annotation/queue")
def get_queue(topic_id: Optional[int] = None, limit: int = 100, db: Session = Depends(get_db)):
    return {"items": annotation_service.pending(db, topic_id, max(1, min(limit, 500)))}


@router.get("/annotation/export")
def export_queue(topic_id: Optional[int] = None, limit: int = 100, db: Session = Depends(get_db)):
    """Label Studio import file. It carries the text only: no model label, to avoid anchoring the annotators."""
    return annotation_service.label_studio_tasks(db, topic_id, max(1, min(limit, 500)))
