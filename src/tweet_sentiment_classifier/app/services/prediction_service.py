"""Prediction use-cases: single text, CSV batch and the feedback form. Persists every prediction."""
import logging
import time
from typing import Any, Dict

import io

import pandas as pd
from fastapi import HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from sqlalchemy.orm import Session

from tweet_sentiment_classifier.config.env_config import get_settings
from tweet_sentiment_classifier.entity.models import PredictionLog

logger = logging.getLogger("api")
settings = get_settings()


def log_prediction(db: Session, result: Dict[str, Any], latency_ms: float, text_length: int,
                    text: str | None = None) -> None:
    """Stage a PredictionLog row. confidence stays None when the decider reports none (LLM)."""
    db.add(PredictionLog(
        latency_ms=latency_ms,
        confidence=result.get("confidence"),
        model_used=result.get("decision_source") or result.get("model_used", "unknown"),
        prediction=result.get("prediction", "unknown"),
        text_length=text_length,
        text=(text[:settings.MAX_TEXT_CHARS] if text else None),
    ))

# --- Endpoints ---


async def predict_and_log(c, text: str, db: Session) -> Dict[str, Any]:
    """Shared prediction path for the query-param and JSON endpoints: validate, predict, persist, cache."""
    if not c.pipeline:
        raise HTTPException(status_code=500, detail="Pipeline not initialized (models missing).")
    text = (text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text cannot be empty.")
    if len(text) > settings.MAX_TEXT_CHARS:
        raise HTTPException(status_code=413, detail=f"Text is too long (max {settings.MAX_TEXT_CHARS} characters).")

    if text in c.predict_cache:
        logger.info("Cache hit for /predict")
        return c.predict_cache[text]

    start_time = time.time()
    result = await c.pipeline.predict(text)
    latency_ms = (time.time() - start_time) * 1000

    try:
        log_prediction(db, result, latency_ms, len(text), text=text)
        await run_in_threadpool(db.commit)
    except Exception as e:
        logger.warning(f"Failed to save prediction log: {e}")
        await run_in_threadpool(db.rollback)

    # Keep cache size bounded
    if len(c.predict_cache) > 1000:
        c.predict_cache.clear()
    c.predict_cache[text] = result
    return result



async def process_csv(c, file: UploadFile, db: Session) -> Dict[str, Any]:
    """Validate a CSV upload, predict per row and stage PredictionLog rows (caller commits).

    Returns {results, total_processed, distribution, errors}. distribution/errors are real counts
    derived from the predictions; nothing is invented.
    """
    if not c.pipeline:
        raise HTTPException(status_code=500, detail="Pipeline not initialized.")
    if not (file.filename or "").lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are allowed.")

    # Read one byte past the limit so an oversize upload is detected without buffering all of it
    contents = await file.read(settings.MAX_UPLOAD_BYTES + 1)
    if len(contents) > settings.MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail=f"File is larger than {settings.MAX_UPLOAD_BYTES // 1_000_000} MB.")

    try:
        df = pd.read_csv(io.BytesIO(contents))
    except Exception:
        raise HTTPException(status_code=400, detail="Could not parse the file as CSV.")
    if 'text' not in df.columns:
        raise HTTPException(status_code=400, detail="CSV must contain a 'text' column.")

    mode = getattr(c.pipeline, "inference_mode", "local")
    max_rows = settings.MAX_CSV_ROWS if mode == "local" else settings.MAX_CSV_ROWS_LLM
    if len(df) > max_rows:
        raise HTTPException(status_code=413, detail=f"CSV has {len(df)} rows; the limit is {max_rows} in '{mode}' mode.")

    results = []
    errors: list[Dict[str, str]] = []
    distribution = {"positive": 0, "neutral": 0, "negative": 0}
    for text in df['text'].dropna().astype(str):  # blank cells are skipped, not classified as "nan"
        text = text.strip()[:settings.MAX_TEXT_CHARS]
        if not text:
            continue

        try:
            start_time = time.time()
            res = await c.pipeline.predict(text)
            latency_ms = (time.time() - start_time) * 1000
        except Exception as e:
            logger.warning(f"CSV row failed: {e}")
            errors.append({"text": text, "error": "prediction failed"})
            continue

        log_prediction(db, res, latency_ms, len(text), text=text)
        pred = res["prediction"]
        distribution[pred] = distribution.get(pred, 0) + 1
        results.append({
            "text": text,
            "prediction": pred,
            "confidence": res.get("confidence"),
            "model_used": res.get("decision_source", "unknown")
        })

    return {"results": results, "total_processed": len(results), "distribution": distribution, "errors": errors}



async def submit_form(c, form_req, db: Session) -> Dict[str, Any]:
    if not c.pipeline:
        raise HTTPException(status_code=500, detail="Pipeline not initialized.")
    text = (form_req.feedback_text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="Feedback text cannot be empty.")
    if len(text) > settings.MAX_TEXT_CHARS:
        raise HTTPException(status_code=413, detail=f"Text is too long (max {settings.MAX_TEXT_CHARS} characters).")

    start_time = time.time()
    res = await c.pipeline.predict(text)
    latency_ms = (time.time() - start_time) * 1000

    try:
        log_prediction(db, res, latency_ms, len(text), text=text)
        await run_in_threadpool(db.commit)
    except Exception as e:
        logger.warning(f"Failed to save prediction log from form submission: {e}")
        await run_in_threadpool(db.rollback)

    return {
        "status": "success",
        "feedback": text,
        "prediction": res.get("prediction", "neutral"),
        "sentiment": res.get("prediction", "neutral"),
        "confidence": res.get("confidence"),
        "decision_source": res.get("decision_source", "unknown"),
        "model_used": res.get("decision_source", "unknown"),
        "respondent": form_req.respondent or "Anonymous",
    }
