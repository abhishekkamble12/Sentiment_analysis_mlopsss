"""Application factory: middleware, routers, static files and the shared container."""
import hmac
import logging
import math
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from tweet_sentiment_classifier.app.api import legacy, pages, v1
from prometheus_client import CONTENT_TYPE_LATEST
from starlette.routing import Mount

from tweet_sentiment_classifier.app.container import Container
from tweet_sentiment_classifier.app.core import metrics
from tweet_sentiment_classifier.app.core.ratelimit import RateLimiter
from tweet_sentiment_classifier.config.database import Base, engine

logger = logging.getLogger("api")


def route_label(request: Request) -> str:
    """A low-cardinality name for the endpoint: path parameters become {names}; mounts are named by mount point."""
    path = request.url.path
    for mount in ("/static", "/app"):  # files under a mount share one series, however many paths are requested
        if path == mount or path.startswith(mount + "/"):
            return mount
    route = request.scope.get("route")
    if route is None or isinstance(route, Mount):
        return "unmatched"
    params = {str(v): k for k, v in request.path_params.items()}
    return "/".join("{%s}" % params[seg] if seg in params else seg for seg in path.split("/"))


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

    limiter = RateLimiter()

    def client_id(request: Request) -> str:
        if container.settings.TRUST_PROXY:
            forwarded = request.headers.get("X-Forwarded-For", "")
            if forwarded:
                return forwarded.split(",")[0].strip()
        return request.client.host if request.client else "unknown"

    @app.middleware("http")
    async def rate_limit(request: Request, call_next):
        s = container.settings
        write = request.method not in ("GET", "HEAD", "OPTIONS")
        limit = s.RATE_LIMIT_WRITE_PER_MINUTE if write else s.RATE_LIMIT_PER_MINUTE
        if limit > 0 and request.url.path != "/metrics":
            allowed, retry = limiter.allow((client_id(request), write), limit)
            if not allowed:
                metrics.RATE_LIMITED.labels("write" if write else "read").inc()
                return JSONResponse(status_code=429, content={"detail": "Too many requests. Slow down and retry."},
                                    headers={"Retry-After": str(math.ceil(retry))})
        return await call_next(request)

    @app.middleware("http")
    async def record_metrics(request: Request, call_next):  # defined last = outermost: it also sees 429s
        started = time.perf_counter()
        response = await call_next(request)
        route = route_label(request)
        metrics.HTTP_REQUESTS.labels(request.method, route, str(response.status_code)).inc()
        metrics.HTTP_SECONDS.labels(request.method, route).observe(time.perf_counter() - started)
        return response

    @app.get("/metrics", include_in_schema=False)
    async def prometheus_metrics(request: Request):
        token = container.settings.METRICS_TOKEN
        if token and not hmac.compare_digest(request.headers.get("Authorization", "").encode(),
                                             f"Bearer {token}".encode()):
            return JSONResponse(status_code=401, content={"detail": "metrics token required"})
        return Response(metrics.render(), media_type=CONTENT_TYPE_LATEST)

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
