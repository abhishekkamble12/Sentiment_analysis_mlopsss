"""Application factory: middleware, routers, static files and the shared container."""
import logging
import uuid
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from tweet_sentiment_classifier.app.api import legacy, pages, v1
from tweet_sentiment_classifier.app.container import Container
from tweet_sentiment_classifier.config.database import Base, engine

logger = logging.getLogger("api")


def create_app(container: Container | None = None) -> FastAPI:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler()],
    )
    Base.metadata.create_all(bind=engine)

    app = FastAPI(title="Tweet Sentiment Classification API", version="2.0")
    container = container or Container.build()
    container.app = app
    app.state.container = container

    # CORS: the React dev server (Vite) runs on a different port during development
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

    @app.middleware("http")
    async def request_id(request: Request, call_next):
        rid = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:12]
        request.state.request_id = rid
        response = await call_next(request)
        response.headers["X-Request-ID"] = rid
        return response

    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        rid = getattr(request.state, "request_id", "-")
        logger.error(f"[{rid}] Unhandled exception during {request.method} {request.url}: {exc}", exc_info=True)
        return JSONResponse(status_code=500, content={"detail": "Internal server error. Please try again later."},
                            headers={"X-Request-ID": rid})

    app.include_router(pages.router)
    app.include_router(v1.router, prefix="/api/v1")
    app.include_router(legacy.router)

    # Mount static files last to prevent route conflicts
    app.mount("/static", StaticFiles(directory="static"), name="static")
    # Serve the built React app at /app when it has been compiled (frontend/dist). The Jinja UI at / stays.
    frontend_dist = Path("frontend/dist")
    if frontend_dist.exists():
        app.mount("/app", StaticFiles(directory=str(frontend_dist), html=True), name="frontend")
    return app
