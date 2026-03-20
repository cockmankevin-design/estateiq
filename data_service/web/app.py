"""
FastAPI application factory.

Creates and configures the EstateIQ data service web application.
"""

import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api import router
from ..config import config

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown lifecycle handler."""
    logger.info("EstateIQ Data Service starting up (env=%s)", config.environment)
    yield
    logger.info("EstateIQ Data Service shutting down")


def create_app(**kwargs: Any) -> FastAPI:
    """
    Application factory.

    Parameters
    ----------
    **kwargs : forwarded to FastAPI constructor.
    """
    app = FastAPI(
        title="EstateIQ Data Service",
        description=(
            "Unified API for market data, strategy backtesting, "
            "ML predictions, factor analysis, and real-estate analytics."
        ),
        version="1.0.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        **kwargs,
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.web.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Routes
    app.include_router(router)

    # Root redirect
    @app.get("/", include_in_schema=False)
    async def root():
        return {"message": "EstateIQ Data Service", "docs": "/docs"}

    return app


# Convenience entry point for `uvicorn data_service.web.app:application`
application = create_app()
