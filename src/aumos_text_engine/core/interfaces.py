"""Protocol interfaces for text engine adapters.

Defines structural subtypes for all external adapters, enabling
dependency injection and testability via mock implementations.
"""

from typing import Any, Protocol, runtime_checkable

from aumos_text_engine.api.schemas import (
    BatchDocumentInput,
    GenerationConfig,
    PIIEntity,
    PIIReplaceResult,
    QualityReport,
    StyleProfile,
)


# ---------------------------------------------------------------------------
# Core adapter protocols (original)
# ---------------------------------------------------------------------------


@runtime_checkable
class PIIDetectorProtocol(Protocol):
    """Detects PII entities in text using NER and pattern matching."""

    async def detect(
        self,
        text: str,
        language: str = "en",
        entity_types: list[str] | None = None,
    ) -> list[PIIEntity]:
        """Detect all PII entities in the given text.

        Args:
            text: Input text to analyze for PII.
            language: Language code for NER model selection.
            entity_types: Optional filter to detect only specific entity types.

        Returns:
            List of detected PII entities with type, value, position, and confidence.
        """
        ...

    async def initialize(self) -> None:
        """Initialize the detector (load models, warm up).

        Args:
            None

        Returns:
            None
        """
        ...


@runtime_checkable
class EntityReplacerProtocol(Protocol):
    """Replaces detected PII entities with context-aware fake values."""

    async def replace(
        self,
        text: str,
        entities: list[PIIEntity],
        strategy: str = "entity_aware",
    ) -> PIIReplaceResult:
        """Replace PII entities in text with plausible fake values.

        Args:
            text: Original text containing PII.
            entities: List of detected PII entities (from PIIDetectorProtocol).
            strategy: Replacement strategy (entity_aware | random | mask).

        Returns:
            PIIReplaceResult with anonymized text and replacement mapping.
        """
        ...

    def get_replacement_mapping(self) -> dict[str, str]:
        """Return the mapping of original -> replacement values used so far.

        Args:
            None

        Returns:
            Dict mapping original PII values to their replacements (for consistency).
        """
        ...


@runtime_checkable
class TextGeneratorProtocol(Protocol):
    """Generates synthetic text via LLM inference."""

    async def generate(
        self,
        prompt: str,
        config: GenerationConfig,
    ) -> str:
        """Generate text from a prompt using the configured LLM.

        Args:
            prompt: Full prompt including system instruction and input context.
            config: Generation parameters (model, temperature, max_tokens, etc).

        Returns:
            Generated text string.
        """
        ...

    async def close(self) -> None:
        """Close the underlying HTTP client.

        Args:
            None

        Returns:
            None
        """
        ...


@runtime_checkable
class StylePreserverProtocol(Protocol):
    """Preserves writing style from example documents in generated output."""

    async def extract_style_profile(self, example_text: str) -> StyleProfile:
        """Extract style features from an example document.

        Analyzes vocabulary, sentence length, formality, domain markers.

        Args:
            example_text: Reference document to extract style from.

        Returns:
            StyleProfile with style features for use in generation prompts.
        """
        ...

    async def build_style_prompt(
        self,
        style_profile: StyleProfile,
        base_prompt: str,
    ) -> str:
        """Augment a generation prompt with style-preserving instructions.

        Args:
            style_profile: Style features extracted from a reference document.
            base_prompt: Original generation prompt to augment.

        Returns:
            Augmented prompt incorporating style preservation instructions.
        """
        ...


@runtime_checkable
class QualityValidatorProtocol(Protocol):
    """Validates quality of generated text using semantic similarity."""

    async def validate(
        self,
        original_text: str,
        synthetic_text: str,
    ) -> QualityReport:
        """Compute quality metrics between original and synthetic text.

        Args:
            original_text: Source document (after PII replacement).
            synthetic_text: Generated synthetic document.

        Returns:
            QualityReport with semantic similarity score and pass/fail status.
        """
        ...

    async def initialize(self) -> None:
        """Initialize the validator (load embedding model).

        Args:
            None

        Returns:
            None
        """
        ...


@runtime_checkable
class StorageAdapterProtocol(Protocol):
    """Stores and retrieves documents from object storage (MinIO)."""

    async def upload(
        self,
        content: bytes | str,
        object_key: str,
        content_type: str = "text/plain",
    ) -> str:
        """Upload content to object storage and return the URI.

        Args:
            content: Document content (text or binary).
            object_key: Storage path within the bucket.
            content_type: MIME type of the content.

        Returns:
            Full URI to the stored object (e.g., s3://bucket/path).
        """
        ...

    async def download(self, object_key: str) -> bytes:
        """Download content from object storage.

        Args:
            object_key: Storage path within the bucket.

        Returns:
            Raw content bytes.
        """
        ...


@runtime_checkable
class PrivacyClientProtocol(Protocol):
    """Checks differential privacy budget with aumos-privacy-engine."""

    async def check_budget(
        self,
        tenant_id: str,
        operation: str,
        epsilon: float,
    ) -> bool:
        """Check whether the tenant has sufficient DP budget for an operation.

        Args:
            tenant_id: Tenant requesting the operation.
            operation: Operation type (e.g., text_synthesis, pii_replacement).
            epsilon: Required epsilon budget for this operation.

        Returns:
            True if budget is available, False if budget exhausted.
        """
        ...

    async def consume_budget(
        self,
        tenant_id: str,
        operation: str,
        epsilon: float,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Consume DP budget for a completed operation.

        Args:
            tenant_id: Tenant performing the operation.
            operation: Operation type.
            epsilon: Epsilon budget consumed.
            metadata: Optional metadata about the operation.

        Returns:
            None
        """
        ...

    async def close(self) -> None:
        """Close the underlying HTTP client.

        Args:
            None

        Returns:
            None
        """
        ...


# ---------------------------------------------------------------------------
# Extended adapter protocols — new capabilities
# ---------------------------------------------------------------------------


@runtime_checkable
class LLMClientProtocol(Protocol):
    """Unified async LLM client supporting vLLM, Ollama, and LiteLLM backends."""

    async def generate(self, prompt: str, config: GenerationConfig) -> str:
        """Generate text from a prompt.

        Args:
            prompt: Full generation prompt.
            config: Generation parameters.

        Returns:
            Generated text string.
        """
        ...

    async def generate_structured(
        self,
        prompt: str,
        config: GenerationConfig,
        json_schema: dict[str, Any],
    ) -> dict[str, Any]:
        """Generate structured JSON output validated against a schema.

        Args:
            prompt: Generation prompt.
            config: Generation parameters.
            json_schema: JSON schema describing expected output.

        Returns:
            Parsed and validated JSON dict.
        """
        ...

    def estimate_token_count(self, text: str) -> int:
        """Estimate token count for a given text string.

        Args:
            text: Input text to estimate.

        Returns:
            Estimated token count.
        """
        ...

    async def close(self) -> None:
        """Close the underlying HTTP client.

        Args:
            None

        Returns:
            None
        """
        ...


@runtime_checkable
class PromptTemplateManagerProtocol(Protocol):
    """Domain-specific prompt template registry with variable substitution."""

    def get_template(self, template_id: str) -> Any:
        """Retrieve a template by ID.

        Args:
            template_id: Template identifier.

        Returns:
            PromptTemplate object.
        """
        ...

    def render(
        self,
        template_id: str,
        variables: dict[str, Any],
        include_few_shot: bool = True,
    ) -> str:
        """Render a template to a single prompt string.

        Args:
            template_id: Template to render.
            variables: Variable substitution dict.
            include_few_shot: Whether to inject few-shot examples.

        Returns:
            Rendered prompt string.
        """
        ...

    def list_templates(self, domain: str | None = None) -> list[dict[str, str]]:
        """List all registered templates.

        Args:
            domain: Optional domain filter (legal|medical|financial|general).

        Returns:
            List of template metadata dicts.
        """
        ...


@runtime_checkable
class TextQualityEvaluatorProtocol(Protocol):
    """Multi-dimensional text quality evaluator with semantic and linguistic metrics."""

    async def initialize(self) -> None:
        """Load embedding models.

        Args:
            None

        Returns:
            None
        """
        ...

    async def validate(self, original_text: str, synthetic_text: str) -> QualityReport:
        """Validate quality between original and synthetic text.

        Args:
            original_text: Reference document (after PII replacement).
            synthetic_text: Generated document to evaluate.

        Returns:
            QualityReport with semantic similarity and pass/fail status.
        """
        ...

    async def evaluate(
        self,
        original_text: str,
        synthetic_text: str,
        domain: str,
    ) -> QualityReport:
        """Full multi-dimensional quality evaluation.

        Args:
            original_text: Reference document.
            synthetic_text: Generated document.
            domain: Domain for domain-specific quality scoring.

        Returns:
            QualityReport with all quality dimensions and aggregate score.
        """
        ...


@runtime_checkable
class ContextInjectorProtocol(Protocol):
    """Multi-document context assembler for retrieval-augmented generation."""

    async def initialize(self) -> None:
        """Load ranking models.

        Args:
            None

        Returns:
            None
        """
        ...

    async def assemble_context(
        self,
        query: str,
        documents: list[dict[str, str]],
        token_budget: int | None,
    ) -> Any:
        """Assemble relevant context from multiple documents for a generation query.

        Args:
            query: Generation query or topic hint.
            documents: Source documents (dicts with id, name, text keys).
            token_budget: Maximum total context tokens.

        Returns:
            AssembledContext with formatted context string and source attributions.
        """
        ...

    async def chunk_document(
        self,
        text: str,
        source_id: str,
        source_name: str,
    ) -> list[Any]:
        """Split a document into overlapping chunks.

        Args:
            text: Document text to chunk.
            source_id: Document identifier.
            source_name: Human-readable document name.

        Returns:
            List of DocumentChunk objects.
        """
        ...


@runtime_checkable
class PromptCacheManagerProtocol(Protocol):
    """Redis-backed prompt/response cache with semantic approximate matching."""

    async def initialize(self) -> None:
        """Connect to Redis and load embedding models.

        Args:
            None

        Returns:
            None
        """
        ...

    async def get(self, prompt: str, config_dict: dict[str, Any]) -> str | None:
        """Look up a cached response for the given prompt and config.

        Args:
            prompt: Generation prompt to look up.
            config_dict: Config dict for cache key derivation.

        Returns:
            Cached response string, or None if not found.
        """
        ...

    async def set(
        self,
        prompt: str,
        config_dict: dict[str, Any],
        response: str,
        template_version: str,
        ttl: int | None,
    ) -> None:
        """Store a prompt/response pair in the cache.

        Args:
            prompt: Generation prompt.
            config_dict: Config dict.
            response: LLM response to cache.
            template_version: Template version tag for batch invalidation.
            ttl: Cache TTL in seconds. Uses default if None.

        Returns:
            None
        """
        ...

    async def invalidate_by_version(self, template_version: str) -> int:
        """Invalidate all cache entries for a template version.

        Args:
            template_version: Template version to invalidate.

        Returns:
            Number of entries invalidated.
        """
        ...

    def get_metrics(self) -> dict[str, Any]:
        """Return current cache hit/miss/memory metrics.

        Args:
            None

        Returns:
            Metrics dict with total_hits, total_misses, hit_rate, etc.
        """
        ...

    async def close(self) -> None:
        """Close the Redis connection.

        Args:
            None

        Returns:
            None
        """
        ...


@runtime_checkable
class OutputParserProtocol(Protocol):
    """Structured LLM output parser with JSON schema validation."""

    def sanitize(self, raw_text: str) -> str:
        """Remove markdown artifacts and normalize LLM output.

        Args:
            raw_text: Raw LLM output string.

        Returns:
            Sanitized text with fences and preamble removed.
        """
        ...

    def parse_json(self, raw_text: str) -> dict[str, Any]:
        """Parse JSON from LLM output using multiple extraction strategies.

        Args:
            raw_text: Raw LLM output containing JSON.

        Returns:
            Parsed JSON dict.
        """
        ...

    def parse_and_validate(
        self,
        raw_text: str,
        schema: dict[str, Any],
        format_type: str,
    ) -> tuple[dict[str, Any], list[str]]:
        """Parse output and validate against a JSON schema.

        Args:
            raw_text: Raw LLM output.
            schema: JSON schema to validate against.
            format_type: Expected format ("json" or "yaml").

        Returns:
            Tuple of (parsed_data, validation_errors).
        """
        ...

    def build_retry_prompt(
        self,
        original_prompt: str,
        failed_output: str,
        validation_errors: list[str],
        schema: dict[str, Any],
    ) -> str:
        """Build a retry prompt after schema validation failure.

        Args:
            original_prompt: Original generation prompt.
            failed_output: Invalid output from previous attempt.
            validation_errors: List of validation error messages.
            schema: Expected output schema.

        Returns:
            Augmented retry prompt string.
        """
        ...


@runtime_checkable
class FineTuningAdapterProtocol(Protocol):
    """LoRA fine-tuning dataset preparation and checkpoint tracking."""

    async def prepare_dataset(
        self,
        raw_samples: list[dict[str, Any]],
        format_type: str,
        validation_split: float,
        source_uri: str,
    ) -> Any:
        """Prepare a training dataset from raw text samples.

        Args:
            raw_samples: Raw sample dicts with instruction/output keys.
            format_type: Target format (instruct|alpaca|sharegpt).
            validation_split: Fraction of data to reserve for validation.
            source_uri: Original corpus URI for tracking.

        Returns:
            TrainingDataset with formatted training and validation splits.
        """
        ...

    def generate_lora_config(
        self,
        base_model: str,
        rank: int | None,
        lora_alpha: int | None,
        target_modules: list[str] | None,
    ) -> Any:
        """Generate a LoRA configuration for the given base model.

        Args:
            base_model: Base model name (e.g. llama3-8b-instruct).
            rank: LoRA rank (r). Auto-selected if None.
            lora_alpha: Scaling factor. Defaults to 2x rank.
            target_modules: Module names to apply LoRA. Auto-detected if None.

        Returns:
            LoRAConfig with all parameters set.
        """
        ...

    def get_best_checkpoint(self, job_id: str) -> Any | None:
        """Return the best training checkpoint for a fine-tuning job.

        Args:
            job_id: Fine-tuning job identifier.

        Returns:
            CheckpointInfo for the best checkpoint, or None if none recorded.
        """
        ...
