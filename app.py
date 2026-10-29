from fastapi.concurrency import run_in_threadpool
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Request
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import asyncio
import io
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
import uvicorn
import logging
from typing import Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy import func
from pydantic import BaseModel

from tweet_sentiment_classifier.pipeline.prediction_pipeline import PredictionPipeline
from tweet_sentiment_classifier.config.database import engine, Base, get_db
from tweet_sentiment_classifier.entity.models import PredictionLog, ReportLog
from tweet_sentiment_classifier.config.env_config import get_settings

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("api")

# --- App Initialization ---
settings = get_settings()
Base.metadata.create_all(bind=engine)

app = FastAPI(title="Tweet Sentiment Classification API", version="2.0")

# --- CORS (local development: the React dev server runs on a different port) ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173", "http://127.0.0.1:5173",
        "http://localhost:5174", "http://127.0.0.1:5174",
        "http://localhost:8000", "http://127.0.0.1:8000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- UI Setup ---
templates = Jinja2Templates(directory="templates")

# --- UI Routes ---
@app.get("/", response_class=HTMLResponse)
async def landing_page(request: Request):
    return templates.TemplateResponse(request, "pages/landing.html")

@app.get("/predict-ui", response_class=HTMLResponse)
async def predict_page(request: Request):
    return templates.TemplateResponse(request, "pages/predict.html")

@app.get("/csv-analysis", response_class=HTMLResponse)
async def csv_analysis_page(request: Request):
    return templates.TemplateResponse(request, "pages/csv_analysis.html")

@app.get("/search-analysis-ui", response_class=HTMLResponse)
async def search_analysis_page(request: Request):
    return templates.TemplateResponse(request, "pages/search_analysis.html")

@app.get("/monitoring-dashboard", response_class=HTMLResponse)
async def monitoring_page(request: Request):
    return templates.TemplateResponse(request, "pages/monitoring.html")

@app.get("/experiments", response_class=HTMLResponse)
async def experiments_page(request: Request):
    return templates.TemplateResponse(request, "pages/experiments.html")

@app.get("/analytics-dashboard", response_class=HTMLResponse)
async def analytics_dashboard_page(request: Request):
    return templates.TemplateResponse(request, "pages/analytics_dashboard.html")


@app.get("/feedback", response_class=HTMLResponse)
async def feedback_page(request: Request):
    return templates.TemplateResponse(request, "pages/feedback.html")

# --- Exception Handlers ---
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception during {request.method} {request.url}: {str(exc)}", exc_info=True)
    return JSONResponse(status_code=500, content={"detail": "Internal server error. Please try again later."})

# --- ML Pipeline & Agents ---
try:
    logger.info("Initializing PredictionPipeline...")
    # INFERENCE_MODE (local | llm | hybrid) is read from the environment; default is local-only.
    pipeline = PredictionPipeline()
except Exception as e:
    logger.error(f"Failed to initialize full pipeline: {e}")
    pipeline = None

def _build_agent(name: str):
    """Build a LangChain agent, or None if its dependencies/keys are missing (the endpoints say so)."""
    try:
        from tweet_sentiment_classifier.components import agents
        return getattr(agents, name)()
    except Exception as e:
        logger.warning(f"{name} unavailable: {e}")
        return None


logger.info("Initializing LangChain Agents...")
search_agent = _build_agent("SearchAgent")
report_agent = _build_agent("ReportAgent")

# --- In-Memory Cache (Optimization) ---
predict_cache: Dict[str, Dict[str, Any]] = {}
search_analysis_cache: Dict[str, Dict[str, Any]] = {}

class FormSubmissionRequest(BaseModel):
    feedback_text: str
    respondent: str | None = "Anonymous"


class SearchRequest(BaseModel):
    query: str


class PredictRequest(BaseModel):
    text: str

def _log_prediction(db: Session, result: Dict[str, Any], latency_ms: float, text_length: int,
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

@app.get("/health")
async def health_check():
    return {"status": "ok", "message": "Backend is running!"}

MIN_SEARCH_RESULTS = 3


async def _predict_and_log(text: str, db: Session) -> Dict[str, Any]:
    """Shared prediction path for the query-param and JSON endpoints: validate, predict, persist, cache."""
    if not pipeline:
        raise HTTPException(status_code=500, detail="Pipeline not initialized (models missing).")
    text = (text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text cannot be empty.")
    if len(text) > settings.MAX_TEXT_CHARS:
        raise HTTPException(status_code=413, detail=f"Text is too long (max {settings.MAX_TEXT_CHARS} characters).")

    if text in predict_cache:
        logger.info("Cache hit for /predict")
        return predict_cache[text]

    start_time = time.time()
    result = await pipeline.predict(text)
    latency_ms = (time.time() - start_time) * 1000

    try:
        _log_prediction(db, result, latency_ms, len(text), text=text)
        await run_in_threadpool(db.commit)
    except Exception as e:
        logger.warning(f"Failed to save prediction log: {e}")
        await run_in_threadpool(db.rollback)

    # Keep cache size bounded
    if len(predict_cache) > 1000:
        predict_cache.clear()
    predict_cache[text] = result
    return result


@app.post("/predict")
async def predict_sentiment(text: str, db: Session = Depends(get_db)) -> Dict[str, Any]:
    return await _predict_and_log(text, db)


@app.post("/api/predict")
async def api_predict(req: PredictRequest, db: Session = Depends(get_db)) -> Dict[str, Any]:
    """JSON prediction for the React app: same contract as /predict but with a request body."""
    return await _predict_and_log(req.text, db)


async def _process_csv(file: UploadFile, db: Session) -> Dict[str, Any]:
    """Validate a CSV upload, predict per row and stage PredictionLog rows (caller commits).

    Returns {results, total_processed, distribution, errors}. distribution/errors are real counts
    derived from the predictions; nothing is invented.
    """
    if not pipeline:
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

    mode = getattr(pipeline, "inference_mode", "local")
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
            res = await pipeline.predict(text)
            latency_ms = (time.time() - start_time) * 1000
        except Exception as e:
            logger.warning(f"CSV row failed: {e}")
            errors.append({"text": text, "error": "prediction failed"})
            continue

        _log_prediction(db, res, latency_ms, len(text), text=text)
        pred = res["prediction"]
        distribution[pred] = distribution.get(pred, 0) + 1
        results.append({
            "text": text,
            "prediction": pred,
            "confidence": res.get("confidence"),
            "model_used": res.get("decision_source", "unknown")
        })

    return {"results": results, "total_processed": len(results), "distribution": distribution, "errors": errors}


@app.post("/upload_csv")
async def upload_csv(file: UploadFile = File(...), db: Session = Depends(get_db)):
    processed = await _process_csv(file, db)
    await run_in_threadpool(db.commit)

    # Limit the report to the first 50 rows to bound prompt size; the blocking LLM call runs in a thread
    report = await asyncio.to_thread(report_agent.generate_report, "Bulk CSV Upload", processed["results"][:50]) if report_agent else None

    return JSONResponse(content={"results": processed["results"], "total_processed": processed["total_processed"], "report": report})


@app.post("/api/upload-csv")
async def api_upload_csv(file: UploadFile = File(...), db: Session = Depends(get_db)):
    """Batch CSV analysis for the React app: returns per-row results, a sentiment distribution and row errors."""
    processed = await _process_csv(file, db)
    await run_in_threadpool(db.commit)
    return JSONResponse(content=processed)


@app.post("/api/form-submission")
async def handle_form_submission(form_req: FormSubmissionRequest, db: Session = Depends(get_db)):
    if not pipeline:
        raise HTTPException(status_code=500, detail="Pipeline not initialized.")
    text = (form_req.feedback_text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="Feedback text cannot be empty.")
    if len(text) > settings.MAX_TEXT_CHARS:
        raise HTTPException(status_code=413, detail=f"Text is too long (max {settings.MAX_TEXT_CHARS} characters).")

    start_time = time.time()
    res = await pipeline.predict(text)
    latency_ms = (time.time() - start_time) * 1000

    try:
        _log_prediction(db, res, latency_ms, len(text), text=text)
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


@app.post("/search-analysis")
async def search_analysis(request: SearchRequest, db: Session = Depends(get_db)):
    if not pipeline:
        raise HTTPException(status_code=500, detail="Pipeline not initialized.")
    if not search_agent or not report_agent:
        raise HTTPException(status_code=503, detail="Search analysis needs the LangChain agents and GROQ_API_KEY.")

    query = request.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    if len(query) > settings.MAX_QUERY_CHARS:
        raise HTTPException(status_code=413, detail=f"Query is too long (max {settings.MAX_QUERY_CHARS} characters).")

    if query in search_analysis_cache:
        logger.info("Cache hit for /search-analysis")
        return JSONResponse(content=search_analysis_cache[query])

    # Search and report generation are blocking network calls: keep them off the event loop
    search_res = await asyncio.to_thread(search_agent.search, query)
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
        res = await pipeline.predict(text)
        latency_ms = (time.time() - start_time) * 1000

        _log_prediction(db, res, latency_ms, len(text), text=text)

        predictions.append({
            "text": text,
            "prediction": res["prediction"],
            "confidence": res.get("confidence")
        })

        pred = res["prediction"]
        sentiment_counts[pred] = sentiment_counts.get(pred, 0) + 1

    await run_in_threadpool(db.commit)
    report = await asyncio.to_thread(report_agent.generate_report, query, predictions)

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

    if len(search_analysis_cache) > 500:
        search_analysis_cache.clear()
    search_analysis_cache[query] = response_data

    return JSONResponse(content=response_data)

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


@app.get("/monitoring")
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


@app.get("/api/recent-logs")
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


@app.get("/api/prediction-history")
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


@app.get("/api/analytics")
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


def _load_best_model_meta() -> Dict[str, Any]:
    try:
        return json.loads(Path("artifacts/best_model/best_model_metadata.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


BASELINE_MODELS = {"Majority", "TextBlob", "VADER"}


def _optional_float(value):
    return None if value is None or pd.isna(value) else round(float(value), 4)


@app.get("/api/experiments")
def get_experiments():
    """Results of the latest single evaluation run (artifacts/evaluation/results.csv)."""
    summary_path = Path("artifacts/evaluation/results.csv")
    if not summary_path.exists():
        return {"experiments": [], "run_id": None}

    try:
        df = pd.read_csv(summary_path)
        run_id = None
        results_json = summary_path.with_suffix(".json")
        if results_json.exists():
            run_id = json.loads(results_json.read_text(encoding="utf-8")).get("run_id")

        # Mirror what PredictionPipeline actually serves (see prediction_pipeline.py)
        meta = _load_best_model_meta()
        best = meta.get("best_model_name")
        fast = meta.get("tier_1_fast_fallback", "LogisticRegression")

        experiments = []
        for _, row in df.iterrows():
            name = str(row["model"])
            labels = []
            if name in BASELINE_MODELS:
                labels.append("Baseline (label source)" if name == "TextBlob" else "Baseline")
            if name == fast:
                labels.append("Serving (fast tier)")
            if name == best:
                labels.append("Best by macro-F1")
            experiments.append({
                "run_id": run_id,
                "model_name": name,
                "n_test": int(row["n"]),
                "accuracy": round(float(row["accuracy"]), 4),
                "f1_macro": round(float(row["macro_f1"]), 4),
                "f1_weighted": round(float(row["weighted_f1"]), 4),
                "ci_low": _optional_float(row.get("macro_f1_ci_low")),
                "ci_high": _optional_float(row.get("macro_f1_ci_high")),
                "ece": _optional_float(row.get("ece")),
                "latency_p50_ms": _optional_float(row.get("latency_p50_ms")),
                "status": " + ".join(labels) or "Evaluated",
            })
        return {"experiments": experiments, "run_id": run_id}
    except Exception as e:
        logger.error(f"Error reading experiments: {e}")
        return {"experiments": [], "error": str(e)}


@app.get("/api/model-info")
def get_model_info():
    """What the API is actually serving: model name, inference mode, transformer status, label map."""
    meta = _load_best_model_meta()
    ready = pipeline is not None
    return {
        "pipeline_ready": ready,
        "best_model_name": meta.get("best_model_name"),
        "fast_model_name": meta.get("tier_1_fast_fallback", "LogisticRegression"),
        "inference_mode": getattr(pipeline, "inference_mode", None) if ready else None,
        "transformer_enabled": bool(getattr(pipeline, "db_model", None)) if ready else False,
        "label_mapping": {"-1": "negative", "0": "neutral", "1": "positive"},
        "best_metrics": meta.get("metrics"),
        "run_id": meta.get("run_id"),
    }


RESULTS_DIR = Path("artifacts/evaluation")
LABEL_WORDS = {-1: "negative", 0: "neutral", 1: "positive"}


def _read_json(path: Path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _model_table(run: Dict[str, Any]) -> list:
    rows = []
    for name, r in run.get("results", {}).items():
        ci = r.get("macro_f1_ci") or (None, None)
        rows.append({"model": name, "macro_f1": r["macro_f1"], "ci_low": ci[0], "ci_high": ci[1],
                     "accuracy": r["accuracy"], "kappa": r["kappa"], "ece": r.get("ece"),
                     "baseline": name in BASELINE_MODELS})
    return rows


def _run_summary(run):
    if not run:
        return None
    split = run.get("split", {})
    return {"run_id": run.get("run_id"), "n_test": split.get("n_test"), "label_source": split.get("label_source"),
            "overlap_pct": split.get("test_texts_also_in_train_pct"), "weights_column": split.get("weights_column"),
            "models": _model_table(run), "comparisons": run.get("comparisons", [])}


def _lab_run(run):
    """Full per-model detail for the Model Lab page: per-class F1, confusion matrix and latency.

    Read straight from the evaluation artifacts; nothing is invented or rounded away.
    """
    if not run:
        return None
    split = run.get("split", {})
    models = []
    for name, r in run.get("results", {}).items():
        ci = r.get("macro_f1_ci") or (None, None)
        models.append({
            "model": name,
            "accuracy": r.get("accuracy"),
            "macro_f1": r.get("macro_f1"),
            "weighted_f1": r.get("weighted_f1"),
            "kappa": r.get("kappa"),
            "ece": r.get("ece"),
            "ci_low": ci[0], "ci_high": ci[1],
            "baseline": name in BASELINE_MODELS,
            "per_class": r.get("per_class"),
            "confusion_matrix": r.get("confusion_matrix"),
            "latency": r.get("latency"),
        })
    return {"run_id": run.get("run_id"), "split": split, "models": models,
            "comparisons": run.get("comparisons", [])}


@app.get("/results", response_class=HTMLResponse)
async def results_page(request: Request):
    return templates.TemplateResponse(request, "pages/results.html")


@app.get("/api/results")
def get_results():
    """Everything on the results page, read from artifact files (nothing is hardcoded)."""
    seeds = _read_json(RESULTS_DIR / "seed_study.json")
    tuning = _read_json(RESULTS_DIR / "tuning.json")
    budget = _read_json(Path("artifacts/experiments/gold_budget/results.json"))

    disagreements = []
    gold_predictions = RESULTS_DIR / "gold" / "predictions.csv"
    if gold_predictions.exists():
        pred = pd.read_csv(gold_predictions)
        if "pred_TextBlob" in pred:
            diff = pred[pred["pred_TextBlob"] != pred["label"]].head(25)
            disagreements = [{"text": r["text"], "human": LABEL_WORDS.get(int(r["label"])),
                              "textblob": LABEL_WORDS.get(int(r["pred_TextBlob"]))} for _, r in diff.iterrows()]

    return {
        "silver": _run_summary(_read_json(RESULTS_DIR / "results.json")),
        "gold": _run_summary(_read_json(RESULTS_DIR / "gold" / "results.json")),
        "seed_study": None if not seeds else {
            "seeds": seeds["seeds"],
            "models": [{"model": m, "mean": s["macro_f1"]["mean"], "sd": s["macro_f1"]["sd"],
                        "min": s["macro_f1"]["min"], "max": s["macro_f1"]["max"]} for m, s in seeds["summary"].items()]},
        "tuning_best": None if not tuning else tuning.get("best"),
        "gold_budget": None if not budget else budget.get("summary"),
        "disagreements": disagreements,
    }


@app.get("/api/model-lab")
def get_model_lab():
    """Real evaluation detail for the Model Lab page: per-class F1, confusion matrices, latency,
    pairwise comparisons and the silver/gold split, read straight from the evaluation artifacts."""
    return {
        "silver": _lab_run(_read_json(RESULTS_DIR / "results.json")),
        "gold": _lab_run(_read_json(RESULTS_DIR / "gold" / "results.json")),
        "label_order": ["negative", "neutral", "positive"],
        "best_model_name": _load_best_model_meta().get("best_model_name"),
        "limitation": ("Silver metrics use TextBlob-generated labels, not human annotations: they measure how "
                       "well a model imitates that lexicon, not human-perceived sentiment. Gold (human-labelled) "
                       "results appear only once that set exists."),
    }


# Mount static files at the end to prevent route conflicts
app.mount("/static", StaticFiles(directory="static"), name="static")

# Serve the built React app at /app when it has been compiled (frontend/dist). The legacy Jinja
# UI below / stays available; this only adds /app and its assets, so no existing route is shadowed.
_FRONTEND_DIST = Path("frontend/dist")
if _FRONTEND_DIST.exists():
    app.mount("/app", StaticFiles(directory=str(_FRONTEND_DIST), html=True), name="frontend")

if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8000, workers=1, log_level="info")
