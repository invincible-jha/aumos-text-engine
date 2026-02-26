"""Pydantic v2 schemas for all aumos-text-engine API inputs and outputs.

All models use strict validation. Enums match SQLAlchemy model enums.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from decimal import Decimal
from typing import Any

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Shared primitives
# ---------------------------------------------------------------------------


class PIIEntity(BaseModel):
    """A detected PII entity in text.

    Attributes:
        entity_type: PII category (PERSON, EMAIL_ADDRESS, etc).
        value: The original PII value found in the text.
        replacement: The fake value substituted in its place.
        start: Character offset where entity starts.
        end: Character offset where entity ends.
        confidence: Presidio confidence score (0.0–1.0).
    """

    entity_type: str = Field(description="PII entity type (e.g. PERSON, EMAIL_ADDRESS)")
    value: str = Field(description="Original PII value detected in text")
    replacement: str = Field(default="", description="Fake value substituted for this entity")
    start: int = Field(description="Character start offset in original text")
    end: int = Field(description="Character end offset in original text")
    confidence: float = Field(ge=0.0, le=1.0, description="Detection confidence score")


class PIIReplaceResult(BaseModel):
    """Result of PII detection and replacement.

    Attributes:
        anonymized_text: Text with all PII replaced by fake values.
        entities: All detected PII entities with their replacements.
        replacement_mapping: Original→replacement mapping for consistency.
    """

    anonymized_text: str = Field(description="Text with all PII replaced")
    entities: list[PIIEntity] = Field(description="All detected PII entities")
    replacement_mapping: dict[str, str] = Field(
        default_factory=dict,
        description="Mapping of original values to their replacements",
    )


class StyleProfile(BaseModel):
    """Style features extracted from a reference document.

    Attributes:
        avg_sentence_length: Average sentence length in words.
        formality_score: Formality level (0.0=informal, 1.0=very formal).
        domain_markers: Key domain vocabulary terms.
        tone: Detected tone (neutral, assertive, persuasive, empathetic).
        style_instructions: Human-readable style instructions for the LLM prompt.
    """

    avg_sentence_length: float = Field(default=0.0)
    formality_score: float = Field(default=0.5, ge=0.0, le=1.0)
    domain_markers: list[str] = Field(default_factory=list)
    tone: str = Field(default="neutral")
    style_instructions: str = Field(default="")


class QualityReport(BaseModel):
    """Quality assessment for a synthetic document.

    Attributes:
        semantic_similarity: Cosine similarity between original and synthetic embeddings.
        passes_threshold: Whether similarity meets the configured minimum.
        details: Additional quality metrics (perplexity, diversity score, etc).
    """

    semantic_similarity: float = Field(ge=0.0, le=1.0)
    passes_threshold: bool
    details: dict[str, Any] = Field(default_factory=dict)


class GenerationConfig(BaseModel):
    """LLM generation parameters.

    Attributes:
        model: LLM model identifier. Uses default_model from settings if omitted.
        max_tokens: Maximum tokens to generate.
        temperature: Sampling temperature.
        top_p: Nucleus sampling probability.
    """

    model: str | None = Field(default=None, description="LLM model identifier")
    max_tokens: int = Field(default=2048, gt=0, le=32768)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.95, ge=0.0, le=1.0)


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------


class SynthesisRequest(BaseModel):
    """Request body for POST /text/synthesize.

    Generates a synthetic document from a template or example.

    Attributes:
        domain: Domain context for generation.
        template_id: UUID of a DomainTemplate to use (optional).
        example_text: Reference text whose style to preserve (optional).
        topic_hint: Short description of the desired document topic.
        generation_config: LLM generation parameters.
        style_preserve: Whether to extract and preserve style from example_text.
        validate_quality: Whether to run semantic similarity check on output.
        entity_types: PII entity types to detect and replace (None = all).
    """

    domain: str = Field(
        default="general",
        description="Domain for generation (legal|medical|financial|hr|custom|general)",
    )
    template_id: uuid.UUID | None = Field(
        default=None,
        description="Optional domain template UUID to use",
    )
    example_text: str | None = Field(
        default=None,
        description="Example document for style preservation and context",
        max_length=100_000,
    )
    topic_hint: str = Field(
        default="",
        description="Short description of the document to generate",
        max_length=1000,
    )
    generation_config: GenerationConfig = Field(default_factory=GenerationConfig)
    style_preserve: bool = Field(
        default=True,
        description="Whether to extract and preserve style from example_text",
    )
    validate_quality: bool = Field(
        default=True,
        description="Whether to validate semantic similarity of output",
    )
    entity_types: list[str] | None = Field(
        default=None,
        description="PII entity types to detect. None means all types.",
    )


class PIIReplaceRequest(BaseModel):
    """Request body for POST /text/pii-replace.

    Detects and replaces PII in an existing document.

    Attributes:
        text: The text to anonymize.
        entity_types: Optional filter for specific entity types.
        replacement_strategy: Strategy for replacing entities.
        language: Language of the input text.
        validate_completeness: Whether to re-scan output to verify all PII removed.
    """

    text: str = Field(
        description="Text to scan and anonymize",
        min_length=1,
        max_length=500_000,
    )
    entity_types: list[str] | None = Field(
        default=None,
        description="Specific PII types to detect. None = all types.",
    )
    replacement_strategy: str = Field(
        default="entity_aware",
        description="Replacement strategy: entity_aware | random | mask",
    )
    language: str = Field(default="en", description="Language code for NER model selection")
    validate_completeness: bool = Field(
        default=True,
        description="Re-scan output to verify no PII remains",
    )


class BatchDocumentInput(BaseModel):
    """Single document in a batch request.

    Attributes:
        document_id: Client-supplied identifier for tracking.
        text: Document content.
        domain: Optional per-document domain override.
        metadata: Optional metadata to pass through to output.
    """

    document_id: str = Field(description="Client-supplied document identifier")
    text: str = Field(min_length=1, max_length=500_000)
    domain: str | None = Field(default=None)
    metadata: dict[str, Any] = Field(default_factory=dict)


class BatchRequest(BaseModel):
    """Request body for POST /text/batch.

    Attributes:
        documents: List of documents to process.
        operation: Operation to perform (pii_replace | synthesize).
        generation_config: Generation parameters (used when operation=synthesize).
        concurrency: How many documents to process in parallel.
    """

    documents: list[BatchDocumentInput] = Field(
        min_length=1,
        description="Documents to process",
    )
    operation: str = Field(
        default="pii_replace",
        description="Operation: pii_replace | synthesize",
    )
    generation_config: GenerationConfig = Field(default_factory=GenerationConfig)
    concurrency: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Parallel processing concurrency limit",
    )


class FineTuneRequest(BaseModel):
    """Request body for POST /text/fine-tune.

    Attributes:
        base_model: LLM model identifier to fine-tune.
        corpus_uri: MinIO URI of the training corpus.
        lora_config: LoRA hyperparameters.
        validation_split: Fraction of corpus to use for validation.
    """

    base_model: str = Field(
        description="Base LLM model to fine-tune (e.g. llama3-8b-instruct)",
    )
    corpus_uri: str = Field(
        description="MinIO URI of the training corpus (e.g. s3://bucket/path/)",
    )
    lora_config: dict[str, Any] = Field(
        default_factory=lambda: {
            "r": 16,
            "lora_alpha": 32,
            "target_modules": ["q_proj", "v_proj"],
            "lora_dropout": 0.05,
            "bias": "none",
        },
        description="LoRA configuration (r, alpha, target_modules, etc)",
    )
    validation_split: float = Field(
        default=0.1,
        ge=0.0,
        le=0.5,
        description="Fraction of corpus for validation",
    )
    max_steps: int | None = Field(
        default=None,
        description="Override default training steps",
    )


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------


class JobResponse(BaseModel):
    """Response for any job-based operation.

    Returned immediately after job creation. Use GET /text/jobs/{id} to poll.

    Attributes:
        job_id: UUID to poll for status.
        status: Current processing status.
        job_type: Operation type.
        domain: Domain context for generation jobs.
        pii_entities_found: PII entity count (populated after completion).
        pii_entities_replaced: Replacement count (populated after completion).
        semantic_similarity: Quality score (populated after completion).
        output_uri: MinIO URI of the output (populated after completion).
        document_count: Total documents for batch jobs.
        completed_documents: Documents completed so far (batch jobs).
        error_message: Failure reason if status=failed.
        created_at: Job creation timestamp.
        updated_at: Last status update timestamp.
    """

    job_id: uuid.UUID
    status: str
    job_type: str
    domain: str | None = None
    pii_entities_found: int = 0
    pii_entities_replaced: int = 0
    semantic_similarity: Decimal | None = None
    output_uri: str | None = None
    document_count: int = 1
    completed_documents: int = 0
    error_message: str | None = None
    created_at: datetime
    updated_at: datetime


class PIIReplaceJobResponse(BaseModel):
    """Inline (synchronous) response for simple PII replacement.

    Returned directly for single-document PII replacement requests.

    Attributes:
        job_id: UUID of the tracking record.
        anonymized_text: Text with all PII replaced.
        entities: All detected PII entities.
        pii_entities_found: Total PII entities detected.
        pii_entities_replaced: Total PII entities replaced.
        quality_report: Optional semantic similarity report.
    """

    job_id: uuid.UUID
    anonymized_text: str
    entities: list[PIIEntity]
    pii_entities_found: int
    pii_entities_replaced: int
    quality_report: QualityReport | None = None


class DomainResponse(BaseModel):
    """A domain-specific generator entry.

    Attributes:
        template_id: UUID of the domain template.
        name: Human-readable name.
        domain: Domain category.
        description: What this template generates.
        is_platform_default: Whether this is a platform-provided template.
    """

    template_id: uuid.UUID
    name: str
    domain: str
    description: str = ""
    is_platform_default: bool = False


class FineTuneJobResponse(BaseModel):
    """Response for a fine-tuning job.

    Attributes:
        job_id: UUID of the fine-tuning job.
        status: Current training status.
        base_model: Base model being fine-tuned.
        training_steps_completed: Current step.
        training_steps_total: Target steps.
        training_loss: Latest training loss.
        model_artifact_uri: MinIO URI when complete.
        error_message: Failure reason if failed.
        created_at: Job creation timestamp.
        updated_at: Last update timestamp.
    """

    job_id: uuid.UUID
    status: str
    base_model: str
    training_steps_completed: int = 0
    training_steps_total: int = 0
    training_loss: float | None = None
    model_artifact_uri: str | None = None
    error_message: str | None = None
    created_at: datetime
    updated_at: datetime
