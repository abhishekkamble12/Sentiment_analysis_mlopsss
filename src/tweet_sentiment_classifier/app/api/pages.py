"""Server-rendered (Jinja) pages."""
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

templates = Jinja2Templates(directory="templates")
router = APIRouter(include_in_schema=False)

@router.get("/", response_class=HTMLResponse)
async def landing_page(request: Request):
    return templates.TemplateResponse(request, "pages/landing.html")

@router.get("/predict-ui", response_class=HTMLResponse)
async def predict_page(request: Request):
    return templates.TemplateResponse(request, "pages/predict.html")

@router.get("/csv-analysis", response_class=HTMLResponse)
async def csv_analysis_page(request: Request):
    return templates.TemplateResponse(request, "pages/csv_analysis.html")

@router.get("/search-analysis-ui", response_class=HTMLResponse)
async def search_analysis_page(request: Request):
    return templates.TemplateResponse(request, "pages/search_analysis.html")

@router.get("/monitoring-dashboard", response_class=HTMLResponse)
async def monitoring_page(request: Request):
    return templates.TemplateResponse(request, "pages/monitoring.html")

@router.get("/experiments", response_class=HTMLResponse)
async def experiments_page(request: Request):
    return templates.TemplateResponse(request, "pages/experiments.html")

@router.get("/analytics-dashboard", response_class=HTMLResponse)
async def analytics_dashboard_page(request: Request):
    return templates.TemplateResponse(request, "pages/analytics_dashboard.html")


@router.get("/feedback", response_class=HTMLResponse)
async def feedback_page(request: Request):
    return templates.TemplateResponse(request, "pages/feedback.html")



@router.get("/results", response_class=HTMLResponse)
async def results_page(request: Request):
    return templates.TemplateResponse(request, "pages/results.html")


@router.get("/topics", response_class=HTMLResponse)
async def topics_page(request: Request):
    return templates.TemplateResponse(request, "pages/topics.html")
