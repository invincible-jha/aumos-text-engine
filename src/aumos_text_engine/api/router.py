"""FastAPI route handlers for aumos-text-engine.

All endpoints are under the /api/v1/text/ prefix. Job-based endpoints
return immediately with a job_id; clients poll GET /text/jobs/{id}.
"""

from __future__ import annotations

import uuid
from typing import Annotated

import structlog
from aumos_common.auth import get_current_tenant, get_current_user
from aumos_common.database import get_db_session
from aumos_common.errors import NotFoundError
from aumos_common.logging import get_logger
from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from aumos_text_engine.adapters.entity_replacer import ContextAwareEntityReplacer
from aumos_text_engine.adapters.kafka import TextEventPublisher
from aumos_text_engine.adapters.storage import MinIOStorageAdapter
from aumos_text_engine.api.schemas import (
    BatchRequest,
    DomainResponse,
    FineTuneJobResponse,
    FineTuneRequest,
    JobResponse,
    PIIReplaceJobResponse,
    PIIReplaceRequest,
    SynthesisRequest,
)
from aumos_text_engine.core.models import TextSynthesisJob
from aumos_text_engine.core.services import (
    BatchService,
    DomainService,
    FineTuneService,
    PIIService,
    SynthesisService,
)

logger: structlog.BoundLogger = get_logger(__name__)


def create_text_router() -> APIRouter:
    """Create and configure the text engine API router.

    Returns:
        APIRouter with all text engine endpoints registered.
    """
    router = APIRouter(prefix="/text", tags=["Text Engine"])

    @router.post(
        "/synthesize",
        response_model=JobResponse,
        status_code=status.HTTP_202_ACCEPTED,
        summary="Generate synthetic text",
        description=(
            "Generates a synthetic document from a domain template or example. "
            "Returns a job immediately; poll GET /text/jobs/{id} for completion."
        ),
    )
    async def synthesize_text(
        request: Request,
        body: SynthesisRequest,
        tenant_id: Annotated[str, Depends(get_current_tenant)],
        session: Annotated[AsyncSession, Depends(get_db_session)],
    ) -> JobResponse:
        """Generate synthetic text via LLM with PII-aware style preservation.

        Args:
            request: FastAPI request object (for accessing app.state).
            body: Synthesis request parameters.
            tenant_id: Authenticated tenant from JWT.
            session: Async database session.

        Returns:
            JobResponse with job_id for polling.
        """
        log = get_logger(__name__).bind(tenant_id=tenant_id, domain=body.domain)
        log.info("synthesis request received")

        pii_service = _build_pii_service(request, session)
        synthesis_service = _build_synthesis_service(request, session, pii_service)

        return await synthesis_service.synthesize(
            request=body,
            tenant_id=tenant_id,
            session=session,
        )

    @router.post(
        "/pii-replace",
        response_model=PIIReplaceJobResponse,
        status_code=status.HTTP_200_OK,
        summary="Detect and replace PII",
        description=(
            "Detects all PII in the input text and replaces entities with "
            "context-aware fake values. Returns the anonymized text inline."
        ),
    )
    async def replace_pii(
        request: Request,
        body: PIIReplaceRequest,
        tenant_id: Annotated[str, Depends(get_current_tenant)],
        session: Annotated[AsyncSession, Depends(get_db_session)],
    ) -> PIIReplaceJobResponse:
        """Detect and replace PII in text with plausible fake values.

        Args:
            request: FastAPI request object.
            body: PII replacement request.
            tenant_id: Authenticated tenant from JWT.
            session: Async database session.

        Returns:
            PIIReplaceJobResponse with anonymized text and entity list.

        Raises:
            HTTPException: 422 if PII validation fails (residual PII in output).
        """
        log = get_logger(__name__).bind(tenant_id=tenant_id)
        log.info("PII replace request received", text_length=len(body.text))

        try:
            pii_service = _build_pii_service(request, session)
            return await pii_service.anonymize(
                request=body,
                tenant_id=tenant_id,
                session=session,
            )
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=str(exc),
            ) from exc

    @router.post(
        "/batch",
        response_model=JobResponse,
        status_code=status.HTTP_202_ACCEPTED,
        summary="Batch document processing",
        description="Process multiple documents in parallel. Returns a batch job for polling.",
    )
    async def batch_process(
        request: Request,
        body: BatchRequest,
        tenant_id: Annotated[str, Depends(get_current_tenant)],
        session: Annotated[AsyncSession, Depends(get_db_session)],
    ) -> JobResponse:
        """Process a batch of documents with controlled concurrency.

        Args:
            request: FastAPI request object.
            body: Batch request with documents and operation type.
            tenant_id: Authenticated tenant from JWT.
            session: Async database session.

        Returns:
            JobResponse with job_id for polling.
        """
        log = get_logger(__name__).bind(tenant_id=tenant_id, doc_count=len(body.documents))
        log.info("batch request received", operation=body.operation)

        try:
            pii_service = _build_pii_service(request, session)
            synthesis_service = _build_synthesis_service(request, session, pii_service)
            storage = _build_storage(request)

            batch_service = BatchService(
                pii_service=pii_service,
                synthesis_service=synthesis_service,
                storage=storage,
            )
            return await batch_service.process_batch(
                request=body,
                tenant_id=tenant_id,
                session=session,
            )
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=str(exc),
            ) from exc

    @router.get(
        "/jobs/{job_id}",
        response_model=JobResponse,
        summary="Get job status",
        description="Poll the status of a synthesis or batch job by ID.",
    )
    async def get_job_status(
        job_id: uuid.UUID,
        tenant_id: Annotated[str, Depends(get_current_tenant)],
        session: Annotated[AsyncSession, Depends(get_db_session)],
    ) -> JobResponse:
        """Fetch the current status of a text synthesis or batch job.

        Args:
            job_id: UUID of the job to query.
            tenant_id: Authenticated tenant from JWT.
            session: Async database session.

        Returns:
            JobResponse with current status, metrics, and output URI.

        Raises:
            HTTPException: 404 if job not found for this tenant.
        """
        stmt = (
            select(TextSynthesisJob)
            .where(TextSynthesisJob.id == job_id)
            .where(TextSynthesisJob.tenant_id == tenant_id)
        )
        result = await session.execute(stmt)
        job = result.scalar_one_or_none()

        if job is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Job {job_id} not found",
            )

        return JobResponse(
            job_id=job.id,
            status=job.status.value,
            job_type=job.job_type.value,
            domain=job.domain,
            pii_entities_found=job.pii_entities_found,
            pii_entities_replaced=job.pii_entities_replaced,
            semantic_similarity=job.semantic_similarity,
            output_uri=job.output_uri,
            document_count=job.document_count,
            completed_documents=job.completed_documents,
            error_message=job.error_message,
            created_at=job.created_at,
            updated_at=job.updated_at,
        )

    @router.get(
        "/domains",
        response_model=list[DomainResponse],
        summary="List domain generators",
        description="List all available domain-specific text generators for the current tenant.",
    )
    async def list_domains(
        tenant_id: Annotated[str, Depends(get_current_tenant)],
        session: Annotated[AsyncSession, Depends(get_db_session)],
        include_defaults: bool = True,
    ) -> list[DomainResponse]:
        """List domain-specific text generators available to this tenant.

        Args:
            tenant_id: Authenticated tenant from JWT.
            session: Async database session.
            include_defaults: Whether to include platform-wide templates.

        Returns:
            List of available domain generators.
        """
        domain_service = DomainService(session=session)
        return await domain_service.list_domains(
            tenant_id=tenant_id,
            include_defaults=include_defaults,
        )

    @router.post(
        "/fine-tune",
        response_model=FineTuneJobResponse,
        status_code=status.HTTP_202_ACCEPTED,
        summary="Fine-tune generation model",
        description=(
            "Fine-tune a base LLM on an enterprise text corpus using LoRA. "
            "Returns a job for polling. The resulting model artifact can be "
            "referenced in future synthesis requests."
        ),
    )
    async def create_fine_tune(
        request: Request,
        body: FineTuneRequest,
        tenant_id: Annotated[str, Depends(get_current_tenant)],
        session: Annotated[AsyncSession, Depends(get_db_session)],
    ) -> FineTuneJobResponse:
        """Create a LoRA fine-tuning job on an enterprise text corpus.

        Args:
            request: FastAPI request object.
            body: Fine-tuning request with corpus URI and LoRA config.
            tenant_id: Authenticated tenant from JWT.
            session: Async database session.

        Returns:
            FineTuneJobResponse with job_id for polling.

        Raises:
            HTTPException: 422 if LoRA config is invalid.
        """
        log = get_logger(__name__).bind(tenant_id=tenant_id, base_model=body.base_model)
        log.info("fine-tune request received")

        try:
            storage = _build_storage(request)
            finetune_service = FineTuneService(storage=storage, session=session)
            return await finetune_service.create_finetune_job(
                request=body,
                tenant_id=tenant_id,
            )
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=str(exc),
            ) from exc

    @router.get(
        "/fine-tune/{job_id}",
        response_model=FineTuneJobResponse,
        summary="Get fine-tune job status",
        description="Poll the status of a LoRA fine-tuning job.",
    )
    async def get_fine_tune_status(
        job_id: uuid.UUID,
        tenant_id: Annotated[str, Depends(get_current_tenant)],
        session: Annotated[AsyncSession, Depends(get_db_session)],
    ) -> FineTuneJobResponse:
        """Get the current status of a fine-tuning job.

        Args:
            job_id: UUID of the fine-tuning job.
            tenant_id: Authenticated tenant from JWT.
            session: Async database session.

        Returns:
            FineTuneJobResponse with current training status and metrics.

        Raises:
            HTTPException: 404 if job not found.
        """
        try:
            # Use a temporary storage reference; FineTuneService only needs session here
            from aumos_text_engine.adapters.storage import MinIOStorageAdapter
            from aumos_text_engine.settings import get_settings

            settings = get_settings()
            storage = MinIOStorageAdapter(
                endpoint=settings.minio.endpoint,
                access_key=settings.minio.access_key,
                secret_key=settings.minio.secret_key,
                bucket=settings.minio.bucket,
                secure=settings.minio.secure,
            )
            finetune_service = FineTuneService(storage=storage, session=session)
            return await finetune_service.get_finetune_status(
                job_id=job_id,
                tenant_id=tenant_id,
            )
        except NotFoundError as exc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=str(exc),
            ) from exc

    return router


# ---------------------------------------------------------------------------
# Dependency factories (build adapters from app.state)
# ---------------------------------------------------------------------------


def _build_pii_service(request: Request, session: AsyncSession) -> PIIService:
    """Build a PIIService from app.state adapters.

    Args:
        request: FastAPI request for app.state access.
        session: Database session for job tracking.

    Returns:
        Configured PIIService instance.
    """
    pii_detector = request.app.state.pii_detector
    privacy_client = request.app.state.privacy_client
    entity_replacer = ContextAwareEntityReplacer()

    return PIIService(
        pii_detector=pii_detector,
        entity_replacer=entity_replacer,
        privacy_client=privacy_client,
    )


def _build_synthesis_service(
    request: Request,
    session: AsyncSession,
    pii_service: PIIService,
) -> SynthesisService:
    """Build a SynthesisService from app.state adapters.

    Args:
        request: FastAPI request for app.state access.
        session: Database session.
        pii_service: Pre-built PIIService for the pipeline.

    Returns:
        Configured SynthesisService instance.
    """
    from aumos_text_engine.adapters.style_preserver import FewShotStylePreserver

    llm_client = request.app.state.llm_client
    quality_validator = request.app.state.quality_validator
    privacy_client = request.app.state.privacy_client
    storage = _build_storage(request)
    style_preserver = FewShotStylePreserver(llm_client=llm_client)

    return SynthesisService(
        pii_service=pii_service,
        text_generator=llm_client,
        style_preserver=style_preserver,
        quality_validator=quality_validator,
        storage=storage,
        privacy_client=privacy_client,
    )


def _build_storage(request: Request) -> MinIOStorageAdapter:
    """Build a MinIOStorageAdapter from settings.

    Args:
        request: FastAPI request (unused but kept for consistency).

    Returns:
        Configured MinIOStorageAdapter instance.
    """
    from aumos_text_engine.settings import get_settings

    settings = get_settings()
    return MinIOStorageAdapter(
        endpoint=settings.minio.endpoint,
        access_key=settings.minio.access_key,
        secret_key=settings.minio.secret_key,
        bucket=settings.minio.bucket,
        secure=settings.minio.secure,
    )
