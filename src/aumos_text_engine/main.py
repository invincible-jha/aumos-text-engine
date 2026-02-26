"""FastAPI application entry point for aumos-text-engine.

Uses aumos-common create_app factory for standard middleware, health checks,
CORS, request ID propagation, and OpenTelemetry tracing.
"""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import structlog
from aumos_common.app import create_app
from aumos_common.health import create_health_router
from aumos_common.logging import get_logger
from fastapi import FastAPI

from aumos_text_engine.adapters.llm_client import LLMServingClient
from aumos_text_engine.adapters.pii_detector import PresidioPIIDetector
from aumos_text_engine.adapters.privacy_client import PrivacyEngineClient
from aumos_text_engine.adapters.quality_validator import SemanticQualityValidator
from aumos_text_engine.api.router import create_text_router
from aumos_text_engine.settings import get_settings

logger: structlog.BoundLogger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manage application startup and shutdown lifecycle.

    Initializes adapters (PII detector, LLM client, quality validator)
    and stores them in app.state for dependency injection.

    Args:
        app: The FastAPI application instance.

    Yields:
        None: Control is yielded to the application while running.
    """
    settings = get_settings()
    log = get_logger(__name__)

    log.info(
        "aumos-text-engine starting",
        llm_serving_url=settings.text.llm_serving_url,
        spacy_model=settings.text.spacy_model,
    )

    # Initialize PII detector (loads spaCy model — may take a moment)
    pii_detector = PresidioPIIDetector(
        spacy_model=settings.text.spacy_model,
        score_threshold=settings.text.presidio_score_threshold,
    )
    await pii_detector.initialize()
    app.state.pii_detector = pii_detector

    # Initialize LLM serving client
    llm_client = LLMServingClient(
        base_url=settings.text.llm_serving_url,
        timeout_seconds=settings.text.llm_serving_timeout_seconds,
        default_model=settings.text.default_model,
    )
    app.state.llm_client = llm_client

    # Initialize privacy engine client
    privacy_client = PrivacyEngineClient(
        base_url=settings.text.privacy_engine_url,
        timeout_seconds=settings.text.privacy_engine_timeout_seconds,
    )
    app.state.privacy_client = privacy_client

    # Initialize quality validator (loads sentence-transformer)
    quality_validator = SemanticQualityValidator(
        model_name=settings.text.embedding_model,
        similarity_threshold=settings.text.semantic_similarity_threshold,
    )
    await quality_validator.initialize()
    app.state.quality_validator = quality_validator

    log.info("aumos-text-engine startup complete")

    yield

    # Shutdown
    log.info("aumos-text-engine shutting down")
    await llm_client.close()
    await privacy_client.close()
    log.info("aumos-text-engine shutdown complete")


def create_text_engine_app() -> FastAPI:
    """Create and configure the aumos-text-engine FastAPI application.

    Returns:
        FastAPI: Configured application with all routes and middleware.
    """
    settings = get_settings()

    app = create_app(
        title="AumOS Text Engine",
        description="LLM-powered synthetic text generation with PII-aware entity replacement",
        version="0.1.0",
        service_name=settings.service_name,
        lifespan=lifespan,
    )

    # Health checks
    health_router = create_health_router(service_name=settings.service_name)
    app.include_router(health_router)

    # Text engine API routes
    text_router = create_text_router()
    app.include_router(text_router, prefix="/api/v1")

    return app


app = create_text_engine_app()
