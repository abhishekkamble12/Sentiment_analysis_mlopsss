"""The pre-v1 URLs, kept so the Jinja pages, the React app and external callers keep working.

Each path points at the same handler as its /api/v1 equivalent (or a thin wrapper for the two whose
contract differs) and is marked deprecated in the OpenAPI schema.
"""
import asyncio

from fastapi import APIRouter, Depends, File, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from tweet_sentiment_classifier.app.api.v1 import analytics, models, predict
from tweet_sentiment_classifier.app.deps import get_container, get_db
from tweet_sentiment_classifier.app.services import prediction_service

router = APIRouter(deprecated=True)


@router.post("/predict")
async def predict_query_param(text: str, db: Session = Depends(get_db), c=Depends(get_container)):
    """Original contract: the text arrives as a query parameter."""
    return await prediction_service.predict_and_log(c, text, db)


@router.post("/upload_csv")
async def upload_csv(file: UploadFile = File(...), db: Session = Depends(get_db), c=Depends(get_container)):
    """Original contract: results plus an LLM report of the first 50 rows."""
    processed = await prediction_service.process_csv(c, file, db)
    await run_in_threadpool(db.commit)
    # Limit the report to the first 50 rows to bound prompt size; the blocking LLM call runs in a thread
    report = (await asyncio.to_thread(c.report_agent.generate_report, "Bulk CSV Upload", processed["results"][:50])
              if c.report_agent else None)
    return JSONResponse(content={"results": processed["results"],
                                 "total_processed": processed["total_processed"], "report": report})


_ALIASES = [
    ("/health", predict.health_check, "GET"),
    ("/api/predict", predict.predict_json, "POST"),
    ("/api/upload-csv", predict.predict_batch, "POST"),
    ("/api/form-submission", predict.feedback_submission, "POST"),
    ("/search-analysis", predict.search_report, "POST"),
    ("/monitoring", analytics.get_monitoring, "GET"),
    ("/api/recent-logs", analytics.get_recent_logs, "GET"),
    ("/api/prediction-history", analytics.get_prediction_history, "GET"),
    ("/api/analytics", analytics.get_analytics, "GET"),
    ("/api/model-info", models.get_model_info, "GET"),
    ("/api/experiments", models.get_experiments, "GET"),
    ("/api/results", models.get_results, "GET"),
    ("/api/model-lab", models.get_model_lab, "GET"),
]
for _path, _endpoint, _method in _ALIASES:
    router.add_api_route(_path, _endpoint, methods=[_method])
