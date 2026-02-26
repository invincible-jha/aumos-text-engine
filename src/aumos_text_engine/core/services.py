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
    EntityReplacerProtocol,
    PIIDetectorProtocol,
    PrivacyClientProtocol,
    QualityValidatorProtocol,
    StorageAdapterProtocol,
    StylePreserverProtocol,
    TextGeneratorProtocol,
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
