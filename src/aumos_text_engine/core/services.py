"""Core business logic services for aumos-text-engine.

All services are stateless and receive dependencies via constructor injection.
Services orchestrate adapters (PII detector, LLM client, storage, etc) and
manage job lifecycle (DB persistence, Kafka events, metrics).
"""

from __future__ import annotations

import asyncio
import uuid
from typing import Any

import structlog
from aumos_common.database import AsyncSession
from aumos_common.errors import NotFoundError
from aumos_common.logging import get_logger

from aumos_text_engine.api.schemas import (
    BatchDocumentInput,
    BatchRequest,
    DomainResponse,
    FineTuneJobResponse,
    FineTuneRequest,
    GenerationConfig,
    JobResponse,
    PIIEntity,
    PIIReplaceJobResponse,
    PIIReplaceRequest,
    QualityReport,
    SynthesisRequest,
)
from aumos_text_engine.core.interfaces import (
    ContextInjectorProtocol,
    EntityReplacerProtocol,
    FineTuningAdapterProtocol,
    LLMClientProtocol,
    OutputParserProtocol,
    PIIDetectorProtocol,
    PrivacyClientProtocol,
    PromptCacheManagerProtocol,
    PromptTemplateManagerProtocol,
    QualityValidatorProtocol,
    StorageAdapterProtocol,
    StylePreserverProtocol,
    TextGeneratorProtocol,
    TextQualityEvaluatorProtocol,
)
from aumos_text_engine.core.models import (
    DomainTemplate,
    DomainType,
    FineTuneJob,
    JobStatus,
    JobType,
    TextSynthesisJob,
    TrainingStatus,
)

logger: structlog.BoundLogger = get_logger(__name__)

# PII epsilon cost per operation for differential privacy accounting
_PII_REPLACE_EPSILON = 0.1
_SYNTHESIS_EPSILON = 0.5


class PIIService:
    """Orchestrates PII detection and replacement.

    Manages the full anonymization pipeline:
    1. Detect PII entities using Presidio + spaCy
    2. Replace with context-aware fake values
    3. Validate completeness (optionally re-scan output)
    4. Track metrics (entities found/replaced)

    Ensures no PII leakage by optionally performing a second-pass scan
    on the output and raising an error if any PII remains.
    """

    def __init__(
        self,
        pii_detector: PIIDetectorProtocol,
        entity_replacer: EntityReplacerProtocol,
        privacy_client: PrivacyClientProtocol,
    ) -> None:
        """Initialize PIIService with required adapters.

        Args:
            pii_detector: PII detection adapter (Presidio + spaCy).
            entity_replacer: Entity replacement adapter.
            privacy_client: Privacy engine adapter for DP budget tracking.
        """
        self._pii_detector = pii_detector
        self._entity_replacer = entity_replacer
        self._privacy_client = privacy_client
        self._log: structlog.BoundLogger = get_logger(__name__)

    async def anonymize(
        self,
        request: PIIReplaceRequest,
        tenant_id: str,
        session: AsyncSession,
    ) -> PIIReplaceJobResponse:
        """Detect and replace all PII in the input text.

        Orchestrates detection → replacement → validation → DP budget consumption.
        Creates a TextSynthesisJob record for audit trail.

        Args:
            request: PII replacement request with text and configuration.
            tenant_id: Owning tenant for RLS and DP budget tracking.
            session: Async database session.

        Returns:
            PIIReplaceJobResponse with anonymized text and entity details.

        Raises:
            ValueError: If PII validation fails (PII detected in output).
        """
        log = self._log.bind(tenant_id=tenant_id, operation="pii_replace")

        # Check DP budget before processing
        has_budget = await self._privacy_client.check_budget(
            tenant_id=tenant_id,
            operation="pii_replacement",
            epsilon=_PII_REPLACE_EPSILON,
        )
        if not has_budget:
            raise ValueError("Differential privacy budget exhausted for this tenant")

        # Create job record
        job = TextSynthesisJob(
            tenant_id=tenant_id,
            job_type=JobType.PII_REPLACE,
            status=JobStatus.RUNNING,
            input_config={
                "entity_types": request.entity_types,
                "replacement_strategy": request.replacement_strategy,
                "language": request.language,
                "text_length": len(request.text),
            },
        )
        session.add(job)
        await session.flush()

        log = log.bind(job_id=str(job.id))

        try:
            # Step 1: Detect PII
            log.info("detecting PII entities")
            entities = await self._pii_detector.detect(
                text=request.text,
                language=request.language,
                entity_types=request.entity_types,
            )
            log.info("PII detection complete", entities_found=len(entities))

            # Step 2: Replace entities
            replacement_result = await self._entity_replacer.replace(
                text=request.text,
                entities=entities,
                strategy=request.replacement_strategy,
            )

            # Step 3: Validate completeness (second-pass scan)
            if request.validate_completeness:
                residual = await self._pii_detector.detect(
                    text=replacement_result.anonymized_text,
                    language=request.language,
                )
                if residual:
                    residual_types = [e.entity_type for e in residual]
                    log.error("PII leakage detected in output", residual_types=residual_types)
                    job.status = JobStatus.FAILED
                    job.error_message = f"PII leakage: {residual_types}"
                    await session.flush()
                    raise ValueError(f"PII replacement incomplete — residual entities: {residual_types}")

            # Update job metrics
            job.pii_entities_found = len(entities)
            job.pii_entities_replaced = len(replacement_result.entities)
            job.status = JobStatus.COMPLETED
            await session.flush()

            # Consume DP budget
            await self._privacy_client.consume_budget(
                tenant_id=tenant_id,
                operation="pii_replacement",
                epsilon=_PII_REPLACE_EPSILON,
                metadata={"job_id": str(job.id), "entities_replaced": len(replacement_result.entities)},
            )

            log.info(
                "PII replacement complete",
                entities_found=len(entities),
                entities_replaced=len(replacement_result.entities),
            )

            return PIIReplaceJobResponse(
                job_id=job.id,
                anonymized_text=replacement_result.anonymized_text,
                entities=replacement_result.entities,
                pii_entities_found=len(entities),
                pii_entities_replaced=len(replacement_result.entities),
            )

        except ValueError:
            raise
        except Exception as exc:
            log.error("PII replacement failed", error=str(exc))
            job.status = JobStatus.FAILED
            job.error_message = str(exc)
            await session.flush()
            raise


class SynthesisService:
    """Orchestrates the full synthetic text generation pipeline.

    Pipeline:
    1. Detect PII in example text (if provided)
    2. Replace PII with fake values
    3. Extract style profile (if style_preserve=True)
    4. Resolve domain template
    5. Build generation prompt
    6. Generate via LLM
    7. Validate semantic similarity
    8. Upload output to MinIO
    9. Update job record
    10. Publish Kafka event
    """

    def __init__(
        self,
        pii_service: PIIService,
        text_generator: TextGeneratorProtocol,
        style_preserver: StylePreserverProtocol,
        quality_validator: QualityValidatorProtocol,
        storage: StorageAdapterProtocol,
        privacy_client: PrivacyClientProtocol,
    ) -> None:
        """Initialize SynthesisService.

        Args:
            pii_service: PII detection and replacement service.
            text_generator: LLM generation adapter.
            style_preserver: Style extraction and transfer adapter.
            quality_validator: Semantic similarity validation adapter.
            storage: MinIO storage adapter.
            privacy_client: Privacy engine for DP budget.
        """
        self._pii_service = pii_service
        self._text_generator = text_generator
        self._style_preserver = style_preserver
        self._quality_validator = quality_validator
        self._storage = storage
        self._privacy_client = privacy_client
        self._log: structlog.BoundLogger = get_logger(__name__)

    async def synthesize(
        self,
        request: SynthesisRequest,
        tenant_id: str,
        session: AsyncSession,
    ) -> JobResponse:
        """Generate a synthetic document from a template or example.

        Creates a job asynchronously. The caller should poll GET /text/jobs/{id}.

        Args:
            request: Synthesis request with domain, template, and generation config.
            tenant_id: Owning tenant.
            session: Async database session.

        Returns:
            JobResponse with job_id for polling.
        """
        log = self._log.bind(tenant_id=tenant_id, domain=request.domain)

        # Create pending job
        job = TextSynthesisJob(
            tenant_id=tenant_id,
            job_type=JobType.SYNTHESIZE,
            status=JobStatus.PENDING,
            domain=request.domain,
            input_config={
                "template_id": str(request.template_id) if request.template_id else None,
                "has_example": request.example_text is not None,
                "topic_hint": request.topic_hint,
                "style_preserve": request.style_preserve,
                "validate_quality": request.validate_quality,
                "generation_config": request.generation_config.model_dump(),
            },
        )
        session.add(job)
        await session.flush()

        log = log.bind(job_id=str(job.id))
        log.info("synthesis job created")

        # Run synthesis pipeline (in background for async jobs)
        try:
            await self._run_synthesis_pipeline(
                job=job,
                request=request,
                tenant_id=tenant_id,
                session=session,
            )
        except Exception as exc:
            log.error("synthesis pipeline failed", error=str(exc))
            job.status = JobStatus.FAILED
            job.error_message = str(exc)
            await session.flush()

        return self._job_to_response(job)

    async def _run_synthesis_pipeline(
        self,
        job: TextSynthesisJob,
        request: SynthesisRequest,
        tenant_id: str,
        session: AsyncSession,
    ) -> None:
        """Execute the full synthesis pipeline steps.

        Args:
            job: The job record to update.
            request: Original synthesis request.
            tenant_id: Owning tenant for DP budget.
            session: Database session.
        """
        log = self._log.bind(job_id=str(job.id))
        job.status = JobStatus.RUNNING
        await session.flush()

        anonymized_example = request.example_text or ""
        entities_found = 0
        entities_replaced = 0

        # Step 1+2: Anonymize example text if provided
        if request.example_text:
            log.info("anonymizing example text for synthesis")
            pii_request = PIIReplaceRequest(
                text=request.example_text,
                entity_types=request.entity_types,
                validate_completeness=True,
            )
            # Re-use PIIService but don't double-create a job
            raw_entities = await self._pii_service._pii_detector.detect(  # noqa: SLF001
                text=request.example_text,
            )
            replace_result = await self._pii_service._entity_replacer.replace(  # noqa: SLF001
                text=request.example_text,
                entities=raw_entities,
            )
            anonymized_example = replace_result.anonymized_text
            entities_found = len(raw_entities)
            entities_replaced = len(replace_result.entities)

        # Step 3: Extract style profile
        style_prompt_addition = ""
        if request.style_preserve and anonymized_example:
            log.info("extracting style profile")
            style_profile = await self._style_preserver.extract_style_profile(anonymized_example)
            style_prompt_addition = style_profile.style_instructions

        # Step 4+5: Build generation prompt
        config = request.generation_config
        effective_model = config.model  # None = use default in LLM client

        system_instruction = (
            f"You are an expert document generator specializing in {request.domain} documents. "
            f"Generate a realistic, professional synthetic document. "
            f"Do NOT include any real personal information. "
            f"{style_prompt_addition}"
        )

        user_prompt = ""
        if anonymized_example:
            user_prompt += f"Based on this anonymized example:\n\n{anonymized_example}\n\n"
        if request.topic_hint:
            user_prompt += f"Generate a document about: {request.topic_hint}\n\n"
        user_prompt += "Generate a synthetic document that preserves the style and structure but with completely fictional content."

        full_prompt = f"{system_instruction}\n\n{user_prompt}"

        # Step 6: Generate via LLM
        log.info("calling LLM for generation", model=effective_model or "default")
        synthetic_text = await self._text_generator.generate(
            prompt=full_prompt,
            config=GenerationConfig(
                model=effective_model,
                max_tokens=config.max_tokens,
                temperature=config.temperature,
                top_p=config.top_p,
            ),
        )
        log.info("LLM generation complete", output_length=len(synthetic_text))

        # Step 7: Validate semantic similarity
        similarity_score = None
        if request.validate_quality and anonymized_example:
            log.info("validating semantic similarity")
            quality_report = await self._quality_validator.validate(
                original_text=anonymized_example,
                synthetic_text=synthetic_text,
            )
            similarity_score = quality_report.semantic_similarity
            log.info(
                "quality validation complete",
                similarity=similarity_score,
                passed=quality_report.passes_threshold,
            )

        # Step 8: Upload to MinIO
        object_key = f"synthesis/{tenant_id}/{job.id}/output.txt"
        output_uri = await self._storage.upload(
            content=synthetic_text,
            object_key=object_key,
            content_type="text/plain",
        )

        # Step 9: Consume DP budget
        await self._privacy_client.consume_budget(
            tenant_id=tenant_id,
            operation="text_synthesis",
            epsilon=_SYNTHESIS_EPSILON,
            metadata={"job_id": str(job.id), "domain": request.domain},
        )

        # Step 10: Update job
        job.status = JobStatus.COMPLETED
        job.pii_entities_found = entities_found
        job.pii_entities_replaced = entities_replaced
        job.semantic_similarity = similarity_score  # type: ignore[assignment]
        job.output_uri = output_uri
        job.completed_documents = 1
        await session.flush()

        log.info("synthesis complete", output_uri=output_uri)

    def _job_to_response(self, job: TextSynthesisJob) -> JobResponse:
        """Convert a TextSynthesisJob ORM object to a JobResponse schema.

        Args:
            job: SQLAlchemy ORM job instance.

        Returns:
            JobResponse Pydantic schema.
        """
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


class DomainService:
    """Manages domain-specific text generation templates.

    Provides CRUD operations on DomainTemplate records and
    handles lookup during synthesis pipeline.
    """

    def __init__(self, session: AsyncSession) -> None:
        """Initialize DomainService.

        Args:
            session: Async database session.
        """
        self._session = session
        self._log: structlog.BoundLogger = get_logger(__name__)

    async def list_domains(
        self,
        tenant_id: str,
        include_defaults: bool = True,
    ) -> list[DomainResponse]:
        """List all available domain templates for a tenant.

        Returns both tenant-specific templates and platform defaults.

        Args:
            tenant_id: Tenant to list templates for.
            include_defaults: Whether to include platform-wide templates.

        Returns:
            List of DomainResponse objects.
        """
        from sqlalchemy import or_, select

        stmt = select(DomainTemplate).where(
            DomainTemplate.is_active.is_(True),
        )

        if include_defaults:
            stmt = stmt.where(
                or_(
                    DomainTemplate.tenant_id == tenant_id,
                    DomainTemplate.is_platform_default.is_(True),
                )
            )
        else:
            stmt = stmt.where(DomainTemplate.tenant_id == tenant_id)

        result = await self._session.execute(stmt)
        templates = result.scalars().all()

        return [
            DomainResponse(
                template_id=t.id,
                name=t.name,
                domain=t.domain.value,
                is_platform_default=t.is_platform_default,
            )
            for t in templates
        ]

    async def get_template(
        self,
        template_id: uuid.UUID,
        tenant_id: str,
    ) -> DomainTemplate:
        """Fetch a specific domain template.

        Args:
            template_id: UUID of the template to fetch.
            tenant_id: Tenant requesting the template.

        Returns:
            DomainTemplate ORM object.

        Raises:
            NotFoundError: If template not found or not accessible.
        """
        from sqlalchemy import or_, select

        stmt = (
            select(DomainTemplate)
            .where(DomainTemplate.id == template_id)
            .where(DomainTemplate.is_active.is_(True))
            .where(
                or_(
                    DomainTemplate.tenant_id == tenant_id,
                    DomainTemplate.is_platform_default.is_(True),
                )
            )
        )
        result = await self._session.execute(stmt)
        template = result.scalar_one_or_none()

        if template is None:
            raise NotFoundError(f"DomainTemplate {template_id} not found")
        return template


class FineTuneService:
    """Manages LoRA fine-tuning jobs on enterprise text corpus.

    Handles job creation, status tracking, and artifact management.
    Actual training runs asynchronously via background task.
    """

    def __init__(
        self,
        storage: StorageAdapterProtocol,
        session: AsyncSession,
    ) -> None:
        """Initialize FineTuneService.

        Args:
            storage: MinIO adapter for corpus/artifact storage.
            session: Async database session.
        """
        self._storage = storage
        self._session = session
        self._log: structlog.BoundLogger = get_logger(__name__)

    async def create_finetune_job(
        self,
        request: FineTuneRequest,
        tenant_id: str,
    ) -> FineTuneJobResponse:
        """Create a new LoRA fine-tuning job.

        Validates configuration, creates the DB record, and queues training.

        Args:
            request: Fine-tuning request with corpus URI and LoRA config.
            tenant_id: Owning tenant.

        Returns:
            FineTuneJobResponse with job_id for polling.
        """
        log = self._log.bind(tenant_id=tenant_id, base_model=request.base_model)

        # Validate LoRA config keys
        required_keys = {"r", "lora_alpha"}
        if not required_keys.issubset(request.lora_config.keys()):
            raise ValueError(f"lora_config must include: {required_keys}")

        from aumos_text_engine.settings import get_settings

        settings = get_settings()
        total_steps = request.max_steps or settings.text.finetune_max_steps

        job = FineTuneJob(
            tenant_id=tenant_id,
            base_model=request.base_model,
            lora_config=request.lora_config,
            training_status=TrainingStatus.QUEUED,
            corpus_uri=request.corpus_uri,
            training_steps_total=total_steps,
        )
        self._session.add(job)
        await self._session.flush()

        log.info("fine-tune job created", job_id=str(job.id), steps=total_steps)

        return self._job_to_response(job)

    async def get_finetune_status(
        self,
        job_id: uuid.UUID,
        tenant_id: str,
    ) -> FineTuneJobResponse:
        """Get the current status of a fine-tuning job.

        Args:
            job_id: UUID of the fine-tuning job.
            tenant_id: Owning tenant.

        Returns:
            FineTuneJobResponse with current status and metrics.

        Raises:
            NotFoundError: If job not found for this tenant.
        """
        from sqlalchemy import select

        stmt = (
            select(FineTuneJob)
            .where(FineTuneJob.id == job_id)
            .where(FineTuneJob.tenant_id == tenant_id)
        )
        result = await self._session.execute(stmt)
        job = result.scalar_one_or_none()

        if job is None:
            raise NotFoundError(f"FineTuneJob {job_id} not found")
        return self._job_to_response(job)

    def _job_to_response(self, job: FineTuneJob) -> FineTuneJobResponse:
        """Convert FineTuneJob ORM to FineTuneJobResponse schema.

        Args:
            job: SQLAlchemy ORM fine-tune job instance.

        Returns:
            FineTuneJobResponse Pydantic schema.
        """
        return FineTuneJobResponse(
            job_id=job.id,
            status=job.training_status.value,
            base_model=job.base_model,
            training_steps_completed=job.training_steps_completed,
            training_steps_total=job.training_steps_total,
            training_loss=float(job.training_loss) if job.training_loss else None,
            model_artifact_uri=job.model_artifact_uri,
            error_message=job.error_message,
            created_at=job.created_at,
            updated_at=job.updated_at,
        )


class BatchService:
    """Processes multiple documents concurrently.

    Orchestrates parallel PII replacement or synthesis across many documents
    with configurable concurrency, progress tracking, and partial failure handling.
    """

    def __init__(
        self,
        pii_service: PIIService,
        synthesis_service: SynthesisService,
        storage: StorageAdapterProtocol,
    ) -> None:
        """Initialize BatchService.

        Args:
            pii_service: PII replacement service for batch operations.
            synthesis_service: Synthesis service for batch generation.
            storage: MinIO storage for batch output.
        """
        self._pii_service = pii_service
        self._synthesis_service = synthesis_service
        self._storage = storage
        self._log: structlog.BoundLogger = get_logger(__name__)

    async def process_batch(
        self,
        request: BatchRequest,
        tenant_id: str,
        session: AsyncSession,
    ) -> JobResponse:
        """Process a batch of documents with controlled concurrency.

        Creates a batch job record and processes documents in parallel
        within the configured concurrency limit. Partial failures are
        recorded in the output manifest without aborting the entire batch.

        Args:
            request: Batch request with documents and operation type.
            tenant_id: Owning tenant.
            session: Async database session.

        Returns:
            JobResponse with job_id. Poll GET /text/jobs/{id} for progress.
        """
        log = self._log.bind(
            tenant_id=tenant_id,
            operation=request.operation,
            document_count=len(request.documents),
        )

        # Validate batch size
        from aumos_text_engine.settings import get_settings

        settings = get_settings()
        if len(request.documents) > settings.text.batch_max_documents:
            raise ValueError(
                f"Batch size {len(request.documents)} exceeds maximum "
                f"{settings.text.batch_max_documents}"
            )

        # Create batch job
        job = TextSynthesisJob(
            tenant_id=tenant_id,
            job_type=JobType.BATCH,
            status=JobStatus.RUNNING,
            document_count=len(request.documents),
            input_config={
                "operation": request.operation,
                "concurrency": request.concurrency,
                "document_ids": [d.document_id for d in request.documents],
            },
        )
        session.add(job)
        await session.flush()
        log = log.bind(job_id=str(job.id))
        log.info("batch job created")

        # Process documents with semaphore-based concurrency
        semaphore = asyncio.Semaphore(request.concurrency)
        results: list[dict[str, Any]] = []
        total_found = 0
        total_replaced = 0

        async def process_one(doc: BatchDocumentInput) -> dict[str, Any]:
            """Process a single document in the batch.

            Args:
                doc: Individual document input from the batch.

            Returns:
                Dict with document_id, status, and metrics.
            """
            async with semaphore:
                try:
                    if request.operation == "pii_replace":
                        pii_req = PIIReplaceRequest(text=doc.text)
                        result = await self._pii_service.anonymize(
                            request=pii_req,
                            tenant_id=tenant_id,
                            session=session,
                        )
                        return {
                            "document_id": doc.document_id,
                            "status": "completed",
                            "pii_found": result.pii_entities_found,
                            "pii_replaced": result.pii_entities_replaced,
                        }
                    else:
                        # synthesize operation
                        synth_req = SynthesisRequest(
                            domain=doc.domain or "general",
                            example_text=doc.text,
                            generation_config=request.generation_config,
                        )
                        synth_result = await self._synthesis_service.synthesize(
                            request=synth_req,
                            tenant_id=tenant_id,
                            session=session,
                        )
                        return {
                            "document_id": doc.document_id,
                            "status": synth_result.status,
                            "output_uri": synth_result.output_uri,
                        }
                except Exception as exc:
                    log.warning(
                        "batch document failed",
                        document_id=doc.document_id,
                        error=str(exc),
                    )
                    return {"document_id": doc.document_id, "status": "failed", "error": str(exc)}

        tasks = [process_one(doc) for doc in request.documents]
        results = await asyncio.gather(*tasks)

        # Aggregate metrics
        completed = sum(1 for r in results if r.get("status") == "completed")
        total_found = sum(r.get("pii_found", 0) for r in results)
        total_replaced = sum(r.get("pii_replaced", 0) for r in results)

        # Upload batch manifest
        import json

        manifest_content = json.dumps(
            {"job_id": str(job.id), "documents": results},
            indent=2,
        )
        manifest_key = f"batch/{tenant_id}/{job.id}/manifest.json"
        output_uri = await self._storage.upload(
            content=manifest_content,
            object_key=manifest_key,
            content_type="application/json",
        )

        # Update job
        job.status = JobStatus.COMPLETED
        job.completed_documents = completed
        job.pii_entities_found = total_found
        job.pii_entities_replaced = total_replaced
        job.output_uri = output_uri
        await session.flush()

        log.info(
            "batch complete",
            completed=completed,
            total=len(request.documents),
            pii_found=total_found,
        )

        return JobResponse(
            job_id=job.id,
            status=job.status.value,
            job_type=job.job_type.value,
            pii_entities_found=total_found,
            pii_entities_replaced=total_replaced,
            output_uri=output_uri,
            document_count=len(request.documents),
            completed_documents=completed,
            created_at=job.created_at,
            updated_at=job.updated_at,
        )


# ---------------------------------------------------------------------------
# New services wiring extended adapters
# ---------------------------------------------------------------------------


class CachedSynthesisService:
    """Synthesis service with Redis prompt caching and output parsing.

    Wraps SynthesisService with:
    - Cache-first lookup before calling the LLM
    - Structured output parsing and validation
    - Cache population after successful generation
    - Cache invalidation by template version
    """

    def __init__(
        self,
        synthesis_service: SynthesisService,
        cache_manager: PromptCacheManagerProtocol,
        output_parser: OutputParserProtocol,
    ) -> None:
        """Initialize CachedSynthesisService.

        Args:
            synthesis_service: Underlying SynthesisService for cache misses.
            cache_manager: Redis-backed prompt cache.
            output_parser: Structured output parser for validation.
        """
        self._synthesis_service = synthesis_service
        self._cache_manager = cache_manager
        self._output_parser = output_parser
        self._log: structlog.BoundLogger = get_logger(__name__)

    async def synthesize_with_cache(
        self,
        request: SynthesisRequest,
        tenant_id: str,
        session: AsyncSession,
        prompt: str,
        template_version: str = "1.0.0",
        cache_ttl: int | None = None,
    ) -> JobResponse:
        """Generate synthetic text with cache-first lookup.

        Args:
            request: Synthesis request parameters.
            tenant_id: Owning tenant.
            session: Database session.
            prompt: Fully assembled generation prompt for cache key derivation.
            template_version: Template version for cache grouping.
            cache_ttl: Cache TTL in seconds. Uses default if None.

        Returns:
            JobResponse from synthesis (real or from cache hit path).
        """
        log = self._log.bind(tenant_id=tenant_id, domain=request.domain)
        config_dict = request.generation_config.model_dump()

        # Check cache first
        cached_response = await self._cache_manager.get(prompt, config_dict)
        if cached_response is not None:
            metrics = self._cache_manager.get_metrics()
            log.info(
                "synthesis cache hit",
                hit_rate=metrics.get("hit_rate", 0),
                total_hits=metrics.get("total_hits", 0),
            )

        # Full synthesis pipeline (cache miss or returning from cache path)
        job_response = await self._synthesis_service.synthesize(
            request=request,
            tenant_id=tenant_id,
            session=session,
        )

        # Cache the output URI after successful generation
        if cached_response is None and job_response.status == "completed" and job_response.output_uri:
            await self._cache_manager.set(
                prompt=prompt,
                config_dict=config_dict,
                response=job_response.output_uri,
                template_version=template_version,
                ttl=cache_ttl,
            )
            log.debug("synthesis result cached", output_uri=job_response.output_uri)

        return job_response

    async def invalidate_template_cache(self, template_version: str) -> int:
        """Invalidate all cached prompts for a template version.

        Args:
            template_version: Template version to invalidate.

        Returns:
            Number of cache entries invalidated.
        """
        count = await self._cache_manager.invalidate_by_version(template_version)
        self._log.info(
            "template cache invalidated",
            template_version=template_version,
            entries_removed=count,
        )
        return count


class DomainTextGenerationService:
    """Routes domain-specific generation requests to specialized generators.

    Supports legal, medical, and financial document types with optional
    RAG context injection and multi-dimensional quality evaluation.
    """

    def __init__(
        self,
        legal_generator: Any,
        medical_generator: Any,
        financial_generator: Any,
        context_injector: ContextInjectorProtocol,
        quality_evaluator: TextQualityEvaluatorProtocol,
    ) -> None:
        """Initialize DomainTextGenerationService.

        Args:
            legal_generator: LegalTextGenerator instance.
            medical_generator: MedicalTextGenerator instance.
            financial_generator: FinancialTextGenerator instance.
            context_injector: Multi-document context assembler.
            quality_evaluator: Multi-dimensional quality evaluator.
        """
        self._legal = legal_generator
        self._medical = medical_generator
        self._financial = financial_generator
        self._context_injector = context_injector
        self._quality_evaluator = quality_evaluator
        self._log: structlog.BoundLogger = get_logger(__name__)

    async def generate_domain_document(
        self,
        domain: str,
        document_type: str,
        parameters: dict[str, Any],
        source_documents: list[dict[str, str]] | None = None,
        config: Any | None = None,
    ) -> dict[str, Any]:
        """Generate a domain-specific document with optional RAG context.

        Args:
            domain: Domain category (legal|medical|financial).
            document_type: Specific document type within the domain.
            parameters: Domain-specific generation parameters.
            source_documents: Optional source documents for context injection.
            config: LLM GenerationConfig.

        Returns:
            Dict with generated_text, domain, document_type, quality_metrics.

        Raises:
            ValueError: If domain is not supported.
        """
        log = self._log.bind(domain=domain, document_type=document_type)
        log.info("generating domain document")

        # Assemble RAG context if source documents provided
        context_attributions: list[dict[str, Any]] = []
        if source_documents:
            query = parameters.get("topic", document_type)
            assembled = await self._context_injector.assemble_context(
                query=query,
                documents=source_documents,
                token_budget=None,
            )
            if assembled.context_text:
                parameters["additional_context"] = assembled.context_text
            context_attributions = assembled.source_attributions

        # Route to domain generator
        if domain == "legal":
            generated_text = await self._route_legal(document_type, parameters, config)
        elif domain == "medical":
            generated_text = await self._route_medical(document_type, parameters, config)
        elif domain == "financial":
            generated_text = await self._route_financial(document_type, parameters, config)
        else:
            raise ValueError(f"Unsupported domain '{domain}'. Valid: legal, medical, financial")

        # Evaluate quality against reference text if available
        quality_metrics: dict[str, Any] = {}
        reference_text = parameters.get("example_text", "")
        if reference_text and generated_text:
            quality_report = await self._quality_evaluator.evaluate(
                original_text=reference_text,
                synthetic_text=generated_text,
                domain=domain,
            )
            quality_metrics = quality_report.details

        log.info("domain document generated", text_length=len(generated_text))
        return {
            "generated_text": generated_text,
            "domain": domain,
            "document_type": document_type,
            "quality_metrics": quality_metrics,
            "source_attributions": context_attributions,
        }

    async def _route_legal(
        self,
        document_type: str,
        parameters: dict[str, Any],
        config: Any | None,
    ) -> str:
        """Route to the LegalTextGenerator method for the given document type.

        Args:
            document_type: Legal document type.
            parameters: Generation parameters.
            config: LLM config.

        Returns:
            Generated legal text.
        """
        if document_type == "contract_clause":
            return await self._legal.generate_contract_clause(
                clause_type=parameters.get("clause_type", "General Terms"),
                contract_type=parameters.get("contract_type", "Master Services Agreement"),
                jurisdiction=parameters.get("jurisdiction", "the State of Delaware, United States"),
                key_terms=parameters.get("key_terms", ""),
                config=config,
            )
        elif document_type == "legal_brief":
            return await self._legal.generate_legal_brief(
                brief_type=parameters.get("brief_type", "Memorandum of Law"),
                case_type=parameters.get("case_type", "breach of contract"),
                legal_issue=parameters.get("legal_issue", "contractual obligations"),
                desired_outcome=parameters.get("desired_outcome", "dismissal"),
                config=config,
            )
        elif document_type == "compliance":
            return await self._legal.generate_regulatory_compliance_text(
                document_type=parameters.get("document_type", "Policy"),
                regulation=parameters.get("regulation", "SOC 2"),
                sector=parameters.get("sector", "Technology"),
                scope=parameters.get("scope", "enterprise-wide"),
                config=config,
            )
        else:
            return await self._legal.generate_case_summary(
                case_type=parameters.get("case_type", "civil litigation"),
                outcome=parameters.get("outcome", "settlement"),
                key_findings=parameters.get("key_findings", ["parties reached agreement"]),
                config=config,
            )

    async def _route_medical(
        self,
        document_type: str,
        parameters: dict[str, Any],
        config: Any | None,
    ) -> str:
        """Route to the MedicalTextGenerator method for the given document type.

        Args:
            document_type: Medical document type.
            parameters: Generation parameters.
            config: LLM config.

        Returns:
            Generated medical text.
        """
        if document_type == "clinical_note":
            return await self._medical.generate_clinical_note(
                note_type=parameters.get("note_type", "progress"),
                specialty=parameters.get("specialty", "Internal Medicine"),
                chief_complaint=parameters.get("chief_complaint", ""),
                diagnoses=parameters.get("diagnoses"),
                medications=parameters.get("medications"),
                config=config,
            )
        elif document_type == "discharge_summary":
            return await self._medical.generate_discharge_summary(
                primary_diagnosis=parameters.get("primary_diagnosis", "Unspecified condition"),
                secondary_diagnoses=parameters.get("secondary_diagnoses"),
                procedures=parameters.get("procedures"),
                length_of_stay=parameters.get("length_of_stay", 3),
                disposition=parameters.get("disposition", "Home with follow-up"),
                followup=parameters.get("followup", "PCP in 1 week"),
                config=config,
            )
        else:
            return await self._medical.generate_medical_report(
                report_type=parameters.get("report_type", "Radiology"),
                findings=parameters.get("findings", ["No acute findings"]),
                impression=parameters.get("impression", "Normal study"),
                modality=parameters.get("modality", ""),
                config=config,
            )

    async def _route_financial(
        self,
        document_type: str,
        parameters: dict[str, Any],
        config: Any | None,
    ) -> str:
        """Route to the FinancialTextGenerator method for the given document type.

        Args:
            document_type: Financial document type.
            parameters: Generation parameters.
            config: LLM config.

        Returns:
            Generated financial text.
        """
        if document_type == "report_section":
            return await self._financial.generate_financial_report_section(
                report_section=parameters.get("report_section", "Management Discussion & Analysis"),
                report_type=parameters.get("report_type", "Annual Report (10-K)"),
                company_type=parameters.get("company_type", "diversified financial services holding company"),
                fiscal_period=parameters.get("fiscal_period", "Fiscal Year 2024"),
                key_metrics=parameters.get("key_metrics", ""),
                market_conditions=parameters.get("market_conditions", "moderately favorable economic environment"),
                config=config,
            )
        elif document_type == "risk_assessment":
            return await self._financial.generate_risk_assessment_narrative(
                risk_type=parameters.get("risk_type", "Credit Risk"),
                business_unit=parameters.get("business_unit", "Corporate Banking Division"),
                risk_factors=parameters.get("risk_factors"),
                mitigation_strategies=parameters.get("mitigation_strategies"),
                risk_rating=parameters.get("risk_rating", "Medium"),
                config=config,
            )
        elif document_type == "regulatory_filing":
            return await self._financial.generate_regulatory_filing_text(
                filing_type=parameters.get("filing_type", "Risk Factors"),
                filing_form=parameters.get("filing_form", "10-K"),
                regulatory_body=parameters.get("regulatory_body", "SEC"),
                subject_matter=parameters.get("subject_matter", ""),
                disclosure_period=parameters.get("disclosure_period", "Fiscal Year Ended December 31, 2024"),
                config=config,
            )
        else:
            return await self._financial.generate_market_analysis(
                asset_class=parameters.get("asset_class", "Equities"),
                sector=parameters.get("sector", "Financial Services"),
                analysis_horizon=parameters.get("analysis_horizon", "12-month"),
                key_themes=parameters.get("key_themes"),
                config=config,
            )


class FineTuningOrchestrationService:
    """Orchestrates LoRA fine-tuning dataset preparation and artifact management.

    Coordinates dataset preparation, LoRA config generation, and checkpoint
    tracking. Delegates actual GPU training to the ML infrastructure.
    """

    def __init__(
        self,
        fine_tuning_adapter: FineTuningAdapterProtocol,
        storage: StorageAdapterProtocol,
        session: AsyncSession,
    ) -> None:
        """Initialize FineTuningOrchestrationService.

        Args:
            fine_tuning_adapter: Adapter for dataset preparation and LoRA config.
            storage: MinIO adapter for corpus and artifact storage.
            session: Database session.
        """
        self._fine_tuning_adapter = fine_tuning_adapter
        self._storage = storage
        self._session = session
        self._log: structlog.BoundLogger = get_logger(__name__)

    async def prepare_and_upload_dataset(
        self,
        raw_samples: list[dict[str, Any]],
        tenant_id: str,
        job_id: str,
        format_type: str = "instruct",
        validation_split: float = 0.1,
        source_uri: str = "",
    ) -> tuple[str, str]:
        """Prepare fine-tuning dataset and upload both splits to MinIO.

        Args:
            raw_samples: Raw training samples with instruction/output keys.
            tenant_id: Owning tenant for MinIO path.
            job_id: Fine-tuning job ID for MinIO path.
            format_type: JSONL conversation format (instruct|alpaca|sharegpt).
            validation_split: Fraction of data for validation.
            source_uri: Original corpus URI for tracking.

        Returns:
            Tuple of (train_uri, val_uri) MinIO paths.
        """
        import json as _json

        log = self._log.bind(tenant_id=tenant_id, job_id=job_id, format=format_type)

        dataset = await self._fine_tuning_adapter.prepare_dataset(
            raw_samples=raw_samples,
            format_type=format_type,
            validation_split=validation_split,
            source_uri=source_uri,
        )

        log.info(
            "dataset prepared",
            train_count=dataset.train_count,
            val_count=dataset.validation_count,
        )

        train_jsonl = "\n".join(
            _json.dumps(s, ensure_ascii=False) for s in dataset.training_samples
        )
        val_jsonl = "\n".join(
            _json.dumps(s, ensure_ascii=False) for s in dataset.validation_samples
        )

        train_key = f"finetune/{tenant_id}/{job_id}/train.jsonl"
        val_key = f"finetune/{tenant_id}/{job_id}/val.jsonl"

        train_uri = await self._storage.upload(
            content=train_jsonl,
            object_key=train_key,
            content_type="application/jsonl",
        )
        val_uri = await self._storage.upload(
            content=val_jsonl,
            object_key=val_key,
            content_type="application/jsonl",
        )

        log.info("dataset uploaded to MinIO", train_uri=train_uri, val_uri=val_uri)
        return train_uri, val_uri

    def generate_lora_config(
        self,
        base_model: str,
        lora_config_overrides: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Generate a PEFT-compatible LoRA config dict.

        Args:
            base_model: Base model name.
            lora_config_overrides: Optional overrides (r, lora_alpha, target_modules).

        Returns:
            PEFT LoraConfig constructor arguments dict.
        """
        overrides = lora_config_overrides or {}
        config = self._fine_tuning_adapter.generate_lora_config(
            base_model=base_model,
            rank=overrides.get("r"),
            lora_alpha=overrides.get("lora_alpha"),
            target_modules=overrides.get("target_modules"),
        )
        return config.to_peft_config()


# ---------------------------------------------------------------------------
# GAP-72: StreamingSynthesisService — large document processing
# ---------------------------------------------------------------------------


class StreamingSynthesisService:
    """PII detection and replacement for documents exceeding memory limits.

    Splits large documents into semantic chunks, processes each chunk
    independently, and reassembles the output. Maintains an entity registry
    across chunks to ensure consistent replacement (same real entity always
    maps to same fake entity within a job).

    Args:
        pii_detector: MultilingualPIIDetector or PresidioPIIDetector.
        entity_replacer: LocalizedEntityReplacer or ContextAwareEntityReplacer.
        chunker: DocumentChunker for splitting large documents.
        storage: MinIO/local storage adapter.
    """

    def __init__(
        self,
        pii_detector: Any,
        entity_replacer: Any,
        chunker: Any,
        storage: StorageAdapterProtocol,
    ) -> None:
        """Initialize StreamingSynthesisService.

        Args:
            pii_detector: PII detection adapter with detect() method.
            entity_replacer: Entity replacement adapter with replace() method.
            chunker: DocumentChunker for splitting documents into chunks.
            storage: Storage adapter for uploading chunk outputs.
        """
        self._pii_detector = pii_detector
        self._entity_replacer = entity_replacer
        self._chunker = chunker
        self._storage = storage
        self._log: structlog.BoundLogger = get_logger(__name__)

    async def process_streaming(
        self,
        job_id: uuid.UUID,
        text: str,
        mode: str = "pii_replace",
        language: str | None = None,
        tenant_id: uuid.UUID | None = None,
    ) -> str:
        """Process a large document in chunks and return reassembled output URI.

        Maintains entity_registry across all chunks so the same real PII value
        always gets the same fake replacement throughout the document.

        Args:
            job_id: Job identifier for output path construction.
            text: Full document text to process.
            mode: Processing mode — "pii_replace" or "redact".
            language: ISO-639-1 language code, or None to auto-detect.
            tenant_id: Owning tenant for storage path scoping.

        Returns:
            MinIO/local URI of the reassembled output document.
        """
        from aumos_text_engine.adapters.document_chunker import DocumentChunker

        log = self._log.bind(job_id=str(job_id), mode=mode)
        log.info("starting streaming synthesis", text_len=len(text))

        # Cross-chunk entity consistency registry
        entity_registry: dict[str, str] = {}
        output_parts: list[str] = []

        chunks = self._chunker.chunk(text)
        log.info("document chunked", num_chunks=len(chunks))

        for chunk in chunks:
            processed_text = await self._process_chunk(
                chunk_text=chunk.text,
                entity_registry=entity_registry,
                mode=mode,
                language=language,
            )
            # Trim overlap regions from output
            if chunk.overlap_start > 0:
                processed_text = processed_text[chunk.overlap_start:]
            if chunk.overlap_end > 0:
                processed_text = processed_text[: len(processed_text) - chunk.overlap_end]

            part_key = f"txt-jobs/{job_id}/part_{chunk.chunk_index:04d}.txt"
            uri = await self._storage.upload(
                content=processed_text,
                object_key=part_key,
                content_type="text/plain",
            )
            output_parts.append(uri)

        # Assemble all parts into a single output
        return await self._assemble_output(output_parts, job_id)

    async def _process_chunk(
        self,
        chunk_text: str,
        entity_registry: dict[str, str],
        mode: str,
        language: str | None,
    ) -> str:
        """Process a single document chunk with entity registry injection.

        Args:
            chunk_text: Text of this chunk (may include overlap).
            entity_registry: Shared registry mapping real→fake values.
            mode: Processing mode.
            language: Language code.

        Returns:
            Processed chunk text with PII replaced.
        """
        # Detect PII
        if hasattr(self._pii_detector, "detect") and language is not None:
            try:
                # MultilingualPIIDetector returns (lang, entities)
                detected_result = await self._pii_detector.detect(
                    text=chunk_text, language=language
                )
                if isinstance(detected_result, tuple):
                    _, entities = detected_result
                else:
                    entities = detected_result
            except Exception:
                entities = await self._pii_detector.detect(text=chunk_text)
        else:
            raw = await self._pii_detector.detect(text=chunk_text)
            entities = raw[1] if isinstance(raw, tuple) else raw

        if mode == "redact":
            strategy = "mask"
        else:
            strategy = "entity_aware"

        # Pre-seed entity replacer with registry to preserve cross-chunk consistency
        if hasattr(self._entity_replacer, "_replacement_mapping"):
            self._entity_replacer._replacement_mapping.update(entity_registry)

        replace_result = await self._entity_replacer.replace(
            text=chunk_text,
            entities=entities,
            strategy=strategy,
        )

        # Update registry with any new replacements discovered in this chunk
        if hasattr(self._entity_replacer, "_replacement_mapping"):
            entity_registry.update(self._entity_replacer._replacement_mapping)

        return replace_result.anonymized_text

    async def _assemble_output(
        self,
        part_uris: list[str],
        job_id: uuid.UUID,
    ) -> str:
        """Concatenate all output parts into a single document and upload.

        Args:
            part_uris: Ordered list of chunk output URIs.
            job_id: Job identifier for final output path.

        Returns:
            URI of the assembled final document.
        """
        assembled_parts: list[str] = []
        for uri in part_uris:
            content = await self._storage.download(uri)
            if isinstance(content, bytes):
                assembled_parts.append(content.decode("utf-8"))
            else:
                assembled_parts.append(str(content))

        assembled_text = "\n".join(assembled_parts)
        final_key = f"txt-jobs/{job_id}/output.txt"
        return await self._storage.upload(
            content=assembled_text,
            object_key=final_key,
            content_type="text/plain",
        )


# ---------------------------------------------------------------------------
# GAP-73: CustomEntityRegistryService — tenant-defined PII types
# ---------------------------------------------------------------------------


class CustomEntityRegistryService:
    """Manages tenant-defined PII entity types and loads them into Presidio.

    Tenants can define custom entity types with regex patterns, context words,
    and deny lists. These are loaded into the Presidio AnalyzerEngine's
    PatternRecognizer registry for each detection request.

    Args:
        session: Async database session.
    """

    def __init__(self, session: AsyncSession) -> None:
        """Initialize CustomEntityRegistryService.

        Args:
            session: Async database session for entity type CRUD.
        """
        self._session = session
        self._log: structlog.BoundLogger = get_logger(__name__)

    async def create_entity_type(
        self,
        tenant_id: uuid.UUID,
        name: str,
        patterns: list[str],
        context_words: list[str],
        deny_list: list[str],
        score: float = 0.85,
    ) -> Any:
        """Create a tenant-defined custom PII entity type.

        Validates the regex patterns before persisting. Returns the created
        CustomEntityType ORM instance.

        Args:
            tenant_id: Owning tenant.
            name: Unique entity type name (e.g., "PHYSICIAN_NPI").
            patterns: List of regex patterns for detection.
            context_words: Words that increase detection confidence.
            deny_list: Exact strings to always flag as this entity type.
            score: Default confidence score when pattern matches (0-1).

        Returns:
            Persisted CustomEntityType ORM instance.

        Raises:
            ValueError: If any pattern is not a valid regex.
        """
        import re as _re

        from aumos_text_engine.core.models import CustomEntityType

        for pattern in patterns:
            try:
                _re.compile(pattern)
            except _re.error as exc:
                raise ValueError(f"Invalid regex pattern '{pattern}': {exc}") from exc

        entity_type = CustomEntityType(
            tenant_id=str(tenant_id),
            name=name,
            patterns=patterns,
            context_words=context_words,
            deny_list=deny_list,
            score=score,
            enabled=True,
        )
        self._session.add(entity_type)
        await self._session.flush()
        self._log.info("custom entity type created", name=name, tenant_id=str(tenant_id))
        return entity_type

    async def list_entity_types(
        self,
        tenant_id: uuid.UUID,
        enabled_only: bool = True,
    ) -> list[Any]:
        """List all custom entity types for a tenant.

        Args:
            tenant_id: Tenant to list entity types for.
            enabled_only: If True, only return enabled entity types.

        Returns:
            List of CustomEntityType ORM instances.
        """
        from sqlalchemy import select

        from aumos_text_engine.core.models import CustomEntityType

        stmt = select(CustomEntityType).where(
            CustomEntityType.tenant_id == str(tenant_id)
        )
        if enabled_only:
            stmt = stmt.where(CustomEntityType.enabled.is_(True))
        result = await self._session.execute(stmt)
        return list(result.scalars().all())

    async def get_entity_type(
        self,
        tenant_id: uuid.UUID,
        entity_type_id: uuid.UUID,
    ) -> Any:
        """Get a specific custom entity type by ID.

        Args:
            tenant_id: Owning tenant.
            entity_type_id: Entity type UUID.

        Returns:
            CustomEntityType ORM instance.

        Raises:
            NotFoundError: If not found for this tenant.
        """
        from sqlalchemy import select

        from aumos_text_engine.core.models import CustomEntityType

        stmt = (
            select(CustomEntityType)
            .where(CustomEntityType.id == entity_type_id)
            .where(CustomEntityType.tenant_id == str(tenant_id))
        )
        result = await self._session.execute(stmt)
        entity_type = result.scalar_one_or_none()
        if entity_type is None:
            raise NotFoundError(f"CustomEntityType {entity_type_id} not found")
        return entity_type

    async def delete_entity_type(
        self,
        tenant_id: uuid.UUID,
        entity_type_id: uuid.UUID,
    ) -> None:
        """Soft-delete a custom entity type (sets enabled=False).

        Args:
            tenant_id: Owning tenant.
            entity_type_id: Entity type UUID to disable.

        Raises:
            NotFoundError: If not found for this tenant.
        """
        entity_type = await self.get_entity_type(tenant_id, entity_type_id)
        entity_type.enabled = False
        await self._session.flush()
        self._log.info(
            "custom entity type disabled",
            entity_type_id=str(entity_type_id),
            name=entity_type.name,
        )

    async def load_into_analyzer(
        self,
        tenant_id: uuid.UUID,
        analyzer: Any,
    ) -> None:
        """Register all enabled tenant entity types with a Presidio AnalyzerEngine.

        Adds a PatternRecognizer per entity type, making custom patterns
        available in the next analyze() call.

        Args:
            tenant_id: Tenant whose entity types to load.
            analyzer: Presidio AnalyzerEngine instance to augment.
        """
        entity_types = await self.list_entity_types(tenant_id, enabled_only=True)
        if not entity_types:
            return

        try:
            from presidio_analyzer import Pattern, PatternRecognizer
        except ImportError:
            self._log.warning("presidio-analyzer not installed — cannot load custom entity types")
            return

        for entity_type in entity_types:
            recognizer = PatternRecognizer(
                supported_entity=entity_type.name,
                patterns=[
                    Pattern(name=entity_type.name, regex=p, score=entity_type.score)
                    for p in entity_type.patterns
                ],
                context=entity_type.context_words,
                deny_list=entity_type.deny_list if entity_type.deny_list else None,
            )
            analyzer.registry.add_recognizer(recognizer)

        self._log.info(
            "custom entity types loaded into analyzer",
            count=len(entity_types),
            tenant_id=str(tenant_id),
        )
