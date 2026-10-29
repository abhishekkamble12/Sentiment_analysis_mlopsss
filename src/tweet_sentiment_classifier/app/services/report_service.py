"""Web-search sentiment report: search agent, per-result prediction, LLM report."""
import asyncio
import logging
import time

from fastapi import HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from tweet_sentiment_classifier.app.services.prediction_service import log_prediction
from tweet_sentiment_classifier.config.env_config import get_settings
from tweet_sentiment_classifier.entity.models import ReportLog

logger = logging.getLogger("api")
settings = get_settings()

MIN_SEARCH_RESULTS = 3


async def search_analysis(c, request, db: Session):
    if not c.pipeline:
        raise HTTPException(status_code=500, detail="Pipeline not initialized.")
    if not c.search_agent or not c.report_agent:
        raise HTTPException(status_code=503, detail="Search analysis needs the LangChain agents and GROQ_API_KEY.")

    query = request.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    if len(query) > settings.MAX_QUERY_CHARS:
        raise HTTPException(status_code=413, detail=f"Query is too long (max {settings.MAX_QUERY_CHARS} characters).")

    if query in c.search_analysis_cache:
        logger.info("Cache hit for /search-analysis")
        return JSONResponse(content=c.search_analysis_cache[query])

    # Search and report generation are blocking network calls: keep them off the event loop
    search_res = await asyncio.to_thread(c.search_agent.search, query)
    search_results = [t[:settings.MAX_TEXT_CHARS] for t in search_res.get("results", [])]
    fallback_used = search_res.get("fallback_used", False)

    if not search_results:
        return JSONResponse(content={"message": "No results found."})
    if len(search_results) < MIN_SEARCH_RESULTS:
        return JSONResponse(content={
            "message": f"Only {len(search_results)} relevant result(s) found; at least {MIN_SEARCH_RESULTS} are needed for an analysis."
        })

    predictions = []
    sentiment_counts = {}

    for text in search_results:
        start_time = time.time()
        res = await c.pipeline.predict(text)
        latency_ms = (time.time() - start_time) * 1000

        log_prediction(db, res, latency_ms, len(text), text=text)

        predictions.append({
            "text": text,
            "prediction": res["prediction"],
            "confidence": res.get("confidence")
        })

        pred = res["prediction"]
        sentiment_counts[pred] = sentiment_counts.get(pred, 0) + 1

    await run_in_threadpool(db.commit)
    report = await asyncio.to_thread(c.report_agent.generate_report, query, predictions)

    db.add(ReportLog(
        query=query,
        top_keywords=report.get("keyword_extraction", []),
        sentiment_distribution=sentiment_counts,
        fallback_used=fallback_used
    ))
    await run_in_threadpool(db.commit)

    response_data = {
        "distribution": sentiment_counts,
        "sources": search_res.get("sources", {}),
        "top_keywords": report.get("keyword_extraction", []),
        "confidence_summary": report.get("confidence_summary", ""),
        "report": {
            "prediction_explanation": report.get("prediction_explanation", ""),
            "sentiment_report": report.get("sentiment_report", ""),
            "insights": report.get("insights", "")
        }
    }

    if len(c.search_analysis_cache) > 500:
        c.search_analysis_cache.clear()
    c.search_analysis_cache[query] = response_data


    return JSONResponse(content=response_data)
