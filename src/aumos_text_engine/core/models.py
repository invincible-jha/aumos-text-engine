"""SQLAlchemy ORM models for aumos-text-engine.

All tables use the txt_ prefix. All tenant-scoped tables extend AumOSModel
for RLS tenant isolation. TimestampMixin provides created_at/updated_at.
"""

import enum
from decimal import Decimal
from typing import Any

from aumos_common.database import AumOSModel, Base, TimestampMixin
from sqlalchemy import (
    DECIMAL,
    JSON,
    VARCHAR,
    BigInteger,
    Boolean,
    Enum,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import Mapped, mapped_column


class JobType(str, enum.Enum):
    """Type of text synthesis operation."""

    SYNTHESIZE = "synthesize"
    PII_REPLACE = "pii_replace"
    BATCH = "batch"


class JobStatus(str, enum.Enum):
    """Processing status of a text synthesis job."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class DomainType(str, enum.Enum):
    """Supported domain-specific generators."""

    LEGAL = "legal"
    MEDICAL = "medical"
    FINANCIAL = "financial"
    HR = "hr"
    CUSTOM = "custom"


class TrainingStatus(str, enum.Enum):
    """Status of a LoRA fine-tuning job."""

    QUEUED = "queued"
    PREPARING = "preparing"
    TRAINING = "training"
    COMPLETED = "completed"
    FAILED = "failed"


class TextSynthesisJob(AumOSModel, TimestampMixin, Base):  # type: ignore[misc]
    """Tracks a single text synthesis, PII replacement, or batch operation.

    Stores configuration, progress metrics, and output location.
    Uses txt_ prefix per AumOS table naming convention.

    Attributes:
        id: Auto-generated primary key.
        tenant_id: Owning tenant (for RLS isolation).
        job_type: synthesize | pii_replace | batch.
        status: Current processing state.
        input_config: JSON config for this job (template, params, etc).
        pii_entities_found: Count of detected PII entities across all documents.
        pii_entities_replaced: Count of successfully replaced PII entities.
        semantic_similarity: Quality score comparing input/output semantics.
        output_uri: MinIO URI of the synthetic output.
        domain: Domain context used for generation.
        error_message: Failure reason if status=failed.
        document_count: Number of documents in batch jobs.
    """

    __tablename__ = "txt_synthesis_jobs"

    job_type: Mapped[JobType] = mapped_column(
        Enum(JobType, name="txt_job_type"),
        nullable=False,
    )
    status: Mapped[JobStatus] = mapped_column(
        Enum(JobStatus, name="txt_job_status"),
        nullable=False,
        default=JobStatus.PENDING,
        index=True,
    )
    input_config: Mapped[dict[str, Any]] = mapped_column(
        JSON,
        nullable=False,
        default=dict,
    )
    pii_entities_found: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
    )
    pii_entities_replaced: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
    )
    semantic_similarity: Mapped[Decimal | None] = mapped_column(
        DECIMAL(precision=5, scale=4),
        nullable=True,
    )
    output_uri: Mapped[str | None] = mapped_column(
        VARCHAR(2048),
        nullable=True,
    )
    domain: Mapped[str | None] = mapped_column(
        VARCHAR(64),
        nullable=True,
        index=True,
    )
    error_message: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )
    document_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=1,
    )
    completed_documents: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
    )


class DomainTemplate(AumOSModel, TimestampMixin, Base):  # type: ignore[misc]
    """Domain-specific prompt template for text generation.

    Stores prompt templates and example documents for domain-specific
    generation (legal, medical, financial, HR, custom).

    Attributes:
        id: Auto-generated primary key.
        tenant_id: Owning tenant (NULL = platform-wide template).
        name: Human-readable template name.
        domain: Domain category.
        prompt_template: Jinja2-style prompt template with {input_text} placeholder.
        example_documents: List of example documents for few-shot prompting.
        custom_entity_types: Domain-specific PII entity type definitions.
        is_active: Whether this template is available for use.
        is_platform_default: If true, available to all tenants.
    """

    __tablename__ = "txt_domain_templates"

    name: Mapped[str] = mapped_column(VARCHAR(256), nullable=False)
    domain: Mapped[DomainType] = mapped_column(
        Enum(DomainType, name="txt_domain_type"),
        nullable=False,
        index=True,
    )
    prompt_template: Mapped[str] = mapped_column(Text, nullable=False)
    example_documents: Mapped[list[dict[str, Any]]] = mapped_column(
        JSON,
        nullable=False,
        default=list,
    )
    custom_entity_types: Mapped[list[dict[str, Any]]] = mapped_column(
        JSON,
        nullable=False,
        default=list,
    )
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    is_platform_default: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)


class FineTuneJob(AumOSModel, TimestampMixin, Base):  # type: ignore[misc]
    """Tracks a LoRA fine-tuning job on an enterprise corpus.

    Stores training configuration, progress, and artifact location.
    Fine-tuned models can then be referenced in generation requests.

    Attributes:
        id: Auto-generated primary key.
        tenant_id: Owning tenant.
        base_model: Base LLM model identifier (e.g., llama3-8b-instruct).
        lora_config: LoRA hyperparameters (r, alpha, target_modules, etc).
        training_status: Current training state.
        corpus_uri: MinIO URI of the training corpus.
        model_artifact_uri: MinIO URI of the fine-tuned LoRA adapter.
        training_steps_completed: Current training step.
        training_steps_total: Target training steps.
        training_loss: Latest training loss value.
        error_message: Failure reason if status=failed.
    """

    __tablename__ = "txt_finetune_jobs"

    base_model: Mapped[str] = mapped_column(VARCHAR(256), nullable=False)
    lora_config: Mapped[dict[str, Any]] = mapped_column(
        JSON,
        nullable=False,
        default=dict,
    )
    training_status: Mapped[TrainingStatus] = mapped_column(
        Enum(TrainingStatus, name="txt_training_status"),
        nullable=False,
        default=TrainingStatus.QUEUED,
        index=True,
    )
    corpus_uri: Mapped[str] = mapped_column(VARCHAR(2048), nullable=False)
    model_artifact_uri: Mapped[str | None] = mapped_column(VARCHAR(2048), nullable=True)
    training_steps_completed: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
    )
    training_steps_total: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
    )
    training_loss: Mapped[float | None] = mapped_column(
        DECIMAL(precision=10, scale=6),
        nullable=True,
    )
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)


class CustomEntityType(AumOSModel, TimestampMixin, Base):  # type: ignore[misc]
    """Tenant-defined PII entity type with detection patterns.

    Allows tenants to register custom sensitive data patterns (e.g., NPI numbers,
    project code names) that are not in Presidio's built-in recognizer set.

    Attributes:
        name: Unique entity type name within the tenant (e.g., "PHYSICIAN_NPI").
        patterns: List of regex pattern strings for detection.
        context_words: Words near the entity that increase detection confidence.
        deny_list: Exact strings to always flag as this entity type.
        score: Default confidence score when a pattern matches (0-1).
        enabled: Soft-delete flag; False means the type is disabled.
    """

    __tablename__ = "txt_custom_entity_types"

    name: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    patterns: Mapped[list[Any]] = mapped_column(JSON, nullable=False, default=list)
    context_words: Mapped[list[Any]] = mapped_column(JSON, nullable=False, default=list)
    deny_list: Mapped[list[Any]] = mapped_column(JSON, nullable=False, default=list)
    score: Mapped[float] = mapped_column(Float, nullable=False, default=0.85)
    enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True, index=True)
