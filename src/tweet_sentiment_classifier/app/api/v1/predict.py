"""Prediction endpoints: single text, CSV batch, feedback form, web-search report."""
from typing import Any, Dict

from fastapi import APIRouter, Depends, File, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from tweet_sentiment_classifier.app.deps import get_container, get_db
from tweet_sentiment_classifier.app.schemas import FormSubmissionRequest, PredictRequest, SearchRequest
from tweet_sentiment_classifier.app.services import prediction_service, report_service

router = APIRouter(tags=["predict"])


@router.get("/health")
async def health_check():
    return {"status": "ok", "message": "Backend is running!"}


@router.post("/predict")
async def predict_json(req: PredictRequest, db: Session = Depends(get_db), c=Depends(get_container)) -> Dict[str, Any]:
    """JSON prediction: validate, predict, persist, cache."""
    return await prediction_service.predict_and_log(c, req.text, db)


@router.post("/predict/batch")
async def predict_batch(file: UploadFile = File(...), db: Session = Depends(get_db), c=Depends(get_container)):
    """Batch CSV analysis: per-row results, a sentiment distribution and row errors."""
    processed = await prediction_service.process_csv(c, file, db)
    await run_in_threadpool(db.commit)
    return JSONResponse(content=processed)


@router.post("/feedback/submission")
async def feedback_submission(form_req: FormSubmissionRequest, db: Session = Depends(get_db),
                              c=Depends(get_container)):
    return await prediction_service.submit_form(c, form_req, db)


@router.post("/reports/search")
async def search_report(request: SearchRequest, db: Session = Depends(get_db), c=Depends(get_container)):
    return await report_service.search_analysis(c, request, db)
