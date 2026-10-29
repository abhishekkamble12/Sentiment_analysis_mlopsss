from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Request
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import pandas as pd
import io
import time
import uvicorn
import logging
from typing import Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy import func
from pydantic import BaseModel

from tweet_sentiment_classifier.pipeline.prediction_pipeline import PredictionPipeline
from tweet_sentiment_classifier.components.agents import SearchAgent, ReportAgent
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

logger.info("Initializing LangChain Agents...")
search_agent = SearchAgent()
report_agent = ReportAgent()

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
        report = report_agent.generate_report("Bulk CSV Upload", results[:50])
        
        return JSONResponse(content={"results": results, "total_processed": len(results), "report": report})
    finally:
        del contents
        if 'df' in locals():
            del df

@app.post("/search-analysis")
async def search_analysis(request: SearchRequest, db: Session = Depends(get_db)):
    if not pipeline:
        raise HTTPException(status_code=500, detail="Pipeline not initialized.")
        
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

@app.get("/monitoring")
def get_monitoring(db: Session = Depends(get_db)):
    avg_lat = db.query(func.avg(PredictionLog.latency_ms)).scalar() or 0.0
    avg_conf = db.query(func.avg(PredictionLog.confidence)).scalar() or 0.0
    
    total_reports = db.query(ReportLog).count()
    fallback_count = db.query(ReportLog).filter(ReportLog.fallback_used == True).count()
    fallback_rate = (fallback_count / total_reports) if total_reports > 0 else 0.0
    
    preds = db.query(PredictionLog.prediction, func.count(PredictionLog.id)).group_by(PredictionLog.prediction).all()
    dist = {str(p[0]): p[1] for p in preds}
    
    return {
        "avg_latency_ms": round(avg_lat, 2),
        "avg_confidence": round(avg_conf, 4),
        "fallback_rate": round(fallback_rate, 4),
        "prediction_distribution": dist
    }

@app.get("/api/analytics")
def get_analytics(db: Session = Depends(get_db)):
    try:
        import json
        total_predictions = db.query(PredictionLog).count()
        
        avg_lat = db.query(func.avg(PredictionLog.latency_ms)).scalar() or 0.0
        avg_conf = db.query(func.avg(PredictionLog.confidence)).scalar() or 0.0
        
        preds = db.query(PredictionLog.prediction, func.count(PredictionLog.id)).group_by(PredictionLog.prediction).all()
        sentiment_dist = {str(p[0]): p[1] for p in preds}
        
        for key in ["positive", "neutral", "negative"]:
            if key not in sentiment_dist:
                sentiment_dist[key] = 0
                
        pos_count = sentiment_dist.get("positive", 0)
        neg_count = sentiment_dist.get("negative", 0)
        if total_predictions > 0:
            sentiment_index = round(((pos_count - neg_count) / total_predictions) * 50 + 50, 1)
        else:
            sentiment_index = 50.0
            
        critical_alerts_count = db.query(PredictionLog).filter(PredictionLog.confidence < 0.6).count()
        
        trend_data = db.query(
            func.strftime('%Y-%m-%d %H:00', PredictionLog.timestamp).label('hour'),
            PredictionLog.prediction,
            func.count(PredictionLog.id)
        ).group_by('hour', PredictionLog.prediction).order_by('hour').all()
        
        trends = {}
        for hour, pred, count in trend_data:
            if not hour:
                continue
            if hour not in trends:
                trends[hour] = {"positive": 0, "neutral": 0, "negative": 0}
            trends[hour][pred] = count
            
        sorted_hours = sorted(trends.keys())[-10:]
        trends_formatted = {
            "categories": [h.split(" ")[1] if " " in h else h for h in sorted_hours],
            "positive": [trends[h]["positive"] for h in sorted_hours],
            "neutral": [trends[h]["neutral"] for h in sorted_hours],
            "negative": [trends[h]["negative"] for h in sorted_hours]
        }
        
        recent_reports = db.query(ReportLog).order_by(ReportLog.timestamp.desc()).limit(10).all()
        all_keywords = []
        for r in recent_reports:
            if r.top_keywords:
                if isinstance(r.top_keywords, list):
                    all_keywords.extend(r.top_keywords)
                elif isinstance(r.top_keywords, str):
                    try:
                        kws = json.loads(r.top_keywords)
                        if isinstance(kws, list):
                            all_keywords.extend(kws)
                    except:
                        pass
        
        kw_freq = {}
        for kw in all_keywords:
            if kw and len(kw.strip()) > 1:
                kw_freq[kw] = kw_freq.get(kw, 0) + 1
        sorted_kws = sorted(kw_freq.items(), key=lambda x: x[1], reverse=True)
        top_kws = [kw[0] for kw in sorted_kws[:6]]
        
        if not top_kws:
            top_kws = ["sentiment", "feedback", "latency", "confidence", "ensemble", "gemma"]
            
        return {
            "total_insights": total_predictions,
            "avg_latency_ms": round(avg_lat, 2),
            "avg_confidence": round(avg_conf, 4),
            "sentiment_index": sentiment_index,
            "critical_alerts": critical_alerts_count,
            "sentiment_distribution": sentiment_dist,
            "trends": trends_formatted,
            "top_keywords": top_kws
        }
    except Exception as e:
        logger.error(f"Error serving /api/analytics: {e}")
        return {
            "total_insights": 0,
            "avg_latency_ms": 0.0,
            "avg_confidence": 0.0,
            "sentiment_index": 50.0,
            "critical_alerts": 0,
            "sentiment_distribution": {"positive": 0, "neutral": 0, "negative": 0},
            "trends": {"categories": [], "positive": [], "neutral": [], "negative": []},
            "top_keywords": ["sentiment", "feedback", "latency", "confidence", "ensemble", "gemma"]
        }

@app.get("/api/experiments")
def get_experiments():
    from pathlib import Path
    summary_path = Path("artifacts/model_evaluation/all_models_summary.csv")
    if not summary_path.exists():
        return {"experiments": []}
    
    try:
        df = pd.read_csv(summary_path)
        df = df.fillna({
            "Accuracy": 0.0,
            "F1 Score (Weighted)": 0.0,
            "F1 Score (Macro)": 0.0,
            "Precision (Weighted)": 0.0,
            "Recall (Weighted)": 0.0,
            "Latency": 0.0,
            "Timestamp": ""
        })
        
        experiments = []
        for _, row in df.iterrows():
            model_name = row.get("Model Name", "")
            if model_name == "LinearSVC":
                status = "Deployed (Tier 1)"
            elif model_name == "DistilBERT":
                status = "Deployed (Tier 2)"
            elif model_name in ["LogisticRegression", "RidgeClassifier"]:
                status = "Standby (Ensemble)"
            else:
                status = "Trained"
                
            experiments.append({
                "timestamp": row.get("Timestamp", "") or row.get("timestamp", ""),
                "model_name": model_name,
                "accuracy": round(float(row.get("Accuracy", 0.0)), 4),
                "f1_weighted": round(float(row.get("F1 Score (Weighted)", 0.0)), 4),
                "f1_macro": round(float(row.get("F1 Score (Macro)", 0.0)), 4),
                "latency_seconds": round(float(row.get("Latency", 0.0)), 4),
                "status": status
            })
        return {"experiments": experiments}
    except Exception as e:
        logger.error(f"Error reading experiments: {e}")
        return {"experiments": [], "error": str(e)}


# Mount static files at the end to prevent route conflicts
app.mount("/static", StaticFiles(directory="static"), name="static")

if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8000, workers=1, log_level="info")
