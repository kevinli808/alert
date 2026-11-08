"""FastAPI application entry point.

Run locally:  uvicorn app.main:app --reload
Swagger UI:   http://localhost:8000/docs
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from app.api import detection, history, settings as settings_api, users
from app.core.config import Settings, get_settings
from app.core.errors import register_exception_handlers
from app.core.security import install_http_middleware
from app.services.container import ServiceContainer, build_services

logger = logging.getLogger("app")

MULTIPART_OVERHEAD_BYTES = 64 * 1024


def create_app(settings: Settings | None = None, services: ServiceContainer | None = None) -> FastAPI:
    settings = settings or get_settings()
    settings.validate_for_startup()
    logging.basicConfig(level=settings.log_level.upper(), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if settings.anonymous_detection_enabled:
        logger.warning("Auth0 not configured: /api/detect-sound accepts requests without a token (dev only)")

    @asynccontextmanager
    async def lifespan(app: FastAPI):  # type: ignore[no-untyped-def]
        container = services or build_services(settings)
        app.state.services = container
        if settings.preload_model:
            # Load the model in a worker thread so the API is up immediately; /health shows progress.
            app.state.model_loader = asyncio.create_task(asyncio.to_thread(container.detection.warm_up))
        try:
            yield
        finally:
            await container.aclose()

    app = FastAPI(
        title=settings.app_name,
        version="0.1.0",
        description="Backend for visual alerts of important environmental sounds. "
        "Authenticate with an Auth0 access token (Authorize button → Bearer token).",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_credentials=False,  # bearer tokens, not cookies
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type"],
        expose_headers=["X-Process-Time-Ms", "Retry-After"],
        max_age=600,
    )
    install_http_middleware(app, settings.max_upload_bytes + MULTIPART_OVERHEAD_BYTES)
    register_exception_handlers(app)

    app.include_router(users.router)
    app.include_router(settings_api.router)
    app.include_router(detection.router)
    app.include_router(history.router)

    @app.get("/health", tags=["health"], summary="Liveness and readiness of dependencies")
    def health(request: Request) -> dict:
        c: ServiceContainer = request.app.state.services
        model_status = c.detection.classifier.status
        return {
            "status": "ok" if model_status == "ready" else "degraded",
            "model": {"provider": c.detection.classifier.name, "status": model_status},
            "auth_configured": c.token_verifier is not None,
            # True when the frontend may call /api/detect-sound without logging in (local dev only).
            "anonymous_detection": c.settings.anonymous_detection_enabled,
            "database_configured": c.database is not None,
        }

    return app


app = create_app()
