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
        """Return the mapping of original → replacement values used so far.

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
