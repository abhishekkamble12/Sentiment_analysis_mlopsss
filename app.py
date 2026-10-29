from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Request
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import pandas as pd
import io
import json
import time
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

# --- UI Setup ---
templates = Jinja2Templates(directory="templates")

# --- UI Routes ---
@app.get("/", response_class=HTMLResponse)
async def landing_page(request: Request):
    return templates.TemplateResponse("pages/landing.html", {"request": request})

@app.get("/predict-ui", response_class=HTMLResponse)
async def predict_page(request: Request):
    return templates.TemplateResponse("pages/predict.html", {"request": request})

@app.get("/csv-analysis", response_class=HTMLResponse)
async def csv_analysis_page(request: Request):
    return templates.TemplateResponse("pages/csv_analysis.html", {"request": request})

@app.get("/search-analysis-ui", response_class=HTMLResponse)
async def search_analysis_page(request: Request):
    return templates.TemplateResponse("pages/search_analysis.html", {"request": request})

@app.get("/monitoring-dashboard", response_class=HTMLResponse)
async def monitoring_page(request: Request):
    return templates.TemplateResponse("pages/monitoring.html", {"request": request})

@app.get("/experiments", response_class=HTMLResponse)
async def experiments_page(request: Request):
    return templates.TemplateResponse("pages/experiments.html", {"request": request})

@app.get("/analytics-dashboard", response_class=HTMLResponse)
async def analytics_dashboard_page(request: Request):
    return templates.TemplateResponse("pages/analytics_dashboard.html", {"request": request})

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

class SearchRequest(BaseModel):
    query: str

def _log_prediction(db: Session, result: Dict[str, Any], latency_ms: float, text_length: int) -> None:
    """Stage a PredictionLog row. confidence stays None when the decider reports none (LLM)."""
    db.add(PredictionLog(
        latency_ms=latency_ms,
        confidence=result.get("confidence"),
        model_used=result.get("decision_source") or result.get("model_used", "unknown"),
        prediction=result.get("prediction", "unknown"),
        text_length=text_length,
    ))

# --- Endpoints ---

@app.get("/health")
async def health_check():
    return {"status": "ok", "message": "Backend is running!"}

@app.post("/predict")
async def predict_sentiment(text: str, db: Session = Depends(get_db)) -> Dict[str, Any]:
    if not pipeline:
        raise HTTPException(status_code=500, detail="Pipeline not initialized (models missing).")
    if not text or not text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")
        
    text = text.strip()
    if text in predict_cache:
        logger.info("Cache hit for /predict")
        return predict_cache[text]
    
    start_time = time.time()
    result = await pipeline.predict(text)
    latency_ms = (time.time() - start_time) * 1000
    
    try:
        _log_prediction(db, result, latency_ms, len(text))
        db.commit()
    except Exception as e:
        logger.warning(f"Failed to save prediction log: {e}")
        db.rollback()
        
    # Keep cache size bounded
    if len(predict_cache) > 1000:
        predict_cache.clear()
    predict_cache[text] = result
    return result

@app.post("/upload_csv")
async def upload_csv(file: UploadFile = File(...), db: Session = Depends(get_db)):
    if not pipeline:
        raise HTTPException(status_code=500, detail="Pipeline not initialized.")
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are allowed.")
    
    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        
        if 'text' not in df.columns:
            raise HTTPException(status_code=400, detail="CSV must contain a 'text' column.")
            
        results = []
        for text in df['text'].astype(str):
            text = text.strip()
            if not text:
                continue
                
            start_time = time.time()
            res = await pipeline.predict(text)
            latency_ms = (time.time() - start_time) * 1000
            
            _log_prediction(db, res, latency_ms, len(text))

            results.append({
                "text": text,
                "prediction": res["prediction"],
                "confidence": res["confidence"],
                "model_used": res.get("decision_source", "unknown")
            })
            
        db.commit()
        
        # Generate an AI Report for the bulk upload
        # Limit to first 50 to avoid massive prompt token usage, but provide enough context
        report = report_agent.generate_report("Bulk CSV Upload", results[:50]) if report_agent else None
        
        return JSONResponse(content={"results": results, "total_processed": len(results), "report": report})
    finally:
        del contents
        if 'df' in locals():
            del df

@app.post("/search-analysis")
async def search_analysis(request: SearchRequest, db: Session = Depends(get_db)):
    if not pipeline:
        raise HTTPException(status_code=500, detail="Pipeline not initialized.")
        
    if not search_agent or not report_agent:
        raise HTTPException(status_code=503, detail="Search analysis needs the LangChain agents and GROQ_API_KEY.")
    query = request.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
        
    if query in search_analysis_cache:
        logger.info("Cache hit for /search-analysis")
        return JSONResponse(content=search_analysis_cache[query])
        
    try:
        search_res = search_agent.search(query)
        if isinstance(search_res, dict):
            search_results = search_res.get("results", [])
            fallback_used = search_res.get("fallback_used", False)
        else:
            search_results = search_res
            fallback_used = False
        
        if not search_results:
            return JSONResponse(content={"message": "No results found."})
            
        df = pd.DataFrame({"text": search_results})
        predictions = []
        sentiment_counts = {}
        
        for text in df["text"].astype(str):
            start_time = time.time()
            res = await pipeline.predict(text)
            latency_ms = (time.time() - start_time) * 1000
            
            _log_prediction(db, res, latency_ms, len(text))

            predictions.append({
                "text": text,
                "prediction": res["prediction"],
                "confidence": res.get("confidence")
            })
            
            pred = res["prediction"]
            sentiment_counts[pred] = sentiment_counts.get(pred, 0) + 1
            
        db.commit()
        report = report_agent.generate_report(query, predictions)
        
        report_log = ReportLog(
            query=query,
            top_keywords=report.get("keyword_extraction", []),
            sentiment_distribution=sentiment_counts,
            fallback_used=fallback_used
        )
        db.add(report_log)
        db.commit()
        
        response_data = {
            "distribution": sentiment_counts,
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
        
    finally:
        if 'df' in locals():
            del df

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


@app.get("/api/experiments")
def get_experiments():
    summary_path = Path("artifacts/model_evaluation/all_models_summary.csv")
    if not summary_path.exists():
        return {"experiments": []}

    try:
        df = pd.read_csv(summary_path)
        df = df.fillna({"Accuracy": 0.0, "F1 Score (Weighted)": 0.0, "F1 Score (Macro)": 0.0, "Latency": 0.0, "Timestamp": ""})

        # Mirror what PredictionPipeline actually serves (see prediction_pipeline.py)
        meta = _load_best_model_meta()
        best = meta.get("best_model_name")
        fast = meta.get("tier_1_fast_fallback", "LogisticRegression")

        experiments = []
        for _, row in df.iterrows():
            name = row.get("Model Name", "")
            labels = []
            if name == fast:
                labels.append("Serving (fast tier)")
            if name == best:
                labels.append("Best by macro-F1")
            experiments.append({
                "timestamp": row.get("Timestamp", "") or row.get("timestamp", ""),
                "model_name": name,
                "accuracy": round(float(row.get("Accuracy", 0.0)), 4),
                "f1_weighted": round(float(row.get("F1 Score (Weighted)", 0.0)), 4),
                "f1_macro": round(float(row.get("F1 Score (Macro)", 0.0)), 4),
                # wall-clock time to evaluate the whole test set, not per-prediction latency
                "eval_seconds": round(float(row.get("Latency", 0.0)), 4),
                "status": " + ".join(labels) or "Evaluated",
            })
        return {"experiments": experiments}
    except Exception as e:
        logger.error(f"Error reading experiments: {e}")
        return {"experiments": [], "error": str(e)}


# Mount static files at the end to prevent route conflicts
app.mount("/static", StaticFiles(directory="static"), name="static")

if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8000, workers=1, log_level="info")
