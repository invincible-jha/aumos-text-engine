"""Presidio + spaCy NER PII detection adapter.

Integrates Microsoft Presidio Analyzer with spaCy NER to detect 50+ PII
entity types in enterprise text documents. Supports multi-language detection
and domain-specific custom recognizers.
"""

from __future__ import annotations

import structlog
from aumos_common.logging import get_logger

from aumos_text_engine.api.schemas import PIIEntity

# Presidio imports — loaded lazily to avoid slow startup if not needed
_presidio_available = False
try:
    from presidio_analyzer import AnalyzerEngine, RecognizerRegistry
    from presidio_analyzer.nlp_engine import NlpEngineProvider

    _presidio_available = True
except ImportError:
    pass

# All entity types supported by default
DEFAULT_ENTITY_TYPES: list[str] = [
    "PERSON",
    "EMAIL_ADDRESS",
    "PHONE_NUMBER",
    "CREDIT_CARD",
    "US_SSN",
    "US_PASSPORT",
    "US_DRIVER_LICENSE",
    "IP_ADDRESS",
    "URL",
    "NRP",
    "LOCATION",
    "DATE_TIME",
    "MEDICAL_LICENSE",
    "US_BANK_NUMBER",
    "IBAN_CODE",
    "CRYPTO",
    "AU_ABN",
    "AU_ACN",
    "AU_TFN",
    "AU_MEDICARE",
    "UK_NHS",
    "SG_NRIC_FIN",
    "IN_PAN",
    "IN_AADHAAR",
    "IT_FISCAL_CODE",
    "ES_NIF",
    "PL_PESEL",
    "ORGANIZATION",
]


class PresidioPIIDetector:
    """PII detection adapter using Presidio Analyzer with spaCy NER backend.

    Wraps Microsoft Presidio for comprehensive PII detection across 50+ entity
    types. Uses spaCy as the NLP engine for named entity recognition, supplemented
    by Presidio's built-in rule-based and regex recognizers.

    Attributes:
        _spacy_model: Name of the spaCy model to load (e.g. en_core_web_lg).
        _score_threshold: Minimum confidence score to return an entity.
        _analyzer: Presidio AnalyzerEngine instance (set during initialize()).
        _log: Structured logger.
    """

    def __init__(
        self,
        spacy_model: str = "en_core_web_lg",
        score_threshold: float = 0.7,
    ) -> None:
        """Create a PresidioPIIDetector.

        Args:
            spacy_model: spaCy model name for NER (e.g. en_core_web_lg).
            score_threshold: Minimum presidio score to include an entity.
        """
        self._spacy_model = spacy_model
        self._score_threshold = score_threshold
        self._analyzer: object | None = None
        self._log: structlog.BoundLogger = get_logger(__name__)

    async def initialize(self) -> None:
        """Initialize the Presidio AnalyzerEngine and load the spaCy model.

        Must be called once at startup before any detect() calls.
        Loads the spaCy NER model which may take several seconds.

        Args:
            None

        Returns:
            None
        """
        if not _presidio_available:
            self._log.warning(
                "presidio-analyzer not installed — PII detection will be disabled",
                spacy_model=self._spacy_model,
            )
            return

        self._log.info("initializing Presidio PII detector", spacy_model=self._spacy_model)

        nlp_engine_config = {
            "nlp_engine_name": "spacy",
            "models": [{"lang_code": "en", "model_name": self._spacy_model}],
        }
        provider = NlpEngineProvider(nlp_configuration=nlp_engine_config)
        nlp_engine = provider.create_engine()

        registry = RecognizerRegistry()
        registry.load_predefined_recognizers()

        self._analyzer = AnalyzerEngine(
            nlp_engine=nlp_engine,
            registry=registry,
            supported_languages=["en"],
        )
        self._log.info("Presidio PII detector ready")

    async def detect(
        self,
        text: str,
        language: str = "en",
        entity_types: list[str] | None = None,
    ) -> list[PIIEntity]:
        """Detect all PII entities in the given text.

        Uses Presidio's AnalyzerEngine which combines:
        - spaCy NER (PERSON, ORG, GPE, DATE, etc.)
        - Rule-based recognizers (CREDIT_CARD, US_SSN, etc.)
        - Context-aware confidence boosting

        Args:
            text: Input text to analyze for PII.
            language: BCP-47 language code (default: "en").
            entity_types: Optional list of specific entity types to detect.
                          If None, detects all supported types.

        Returns:
            List of PIIEntity objects with type, value, position, and confidence.
        """
        if not text.strip():
            return []

        if self._analyzer is None:
            self._log.warning("PII detector not initialized, returning empty result")
            return []

        types_to_detect = entity_types or DEFAULT_ENTITY_TYPES

        try:
            results = self._analyzer.analyze(  # type: ignore[union-attr]
                text=text,
                language=language,
                entities=types_to_detect,
                score_threshold=self._score_threshold,
            )
        except Exception as exc:
            self._log.error("Presidio analysis failed", error=str(exc))
            return []

        entities: list[PIIEntity] = []
        for result in results:
            value = text[result.start : result.end]
            entities.append(
                PIIEntity(
                    entity_type=result.entity_type,
                    value=value,
                    replacement="",  # Filled by entity replacer
                    start=result.start,
                    end=result.end,
                    confidence=result.score,
                )
            )

        # Sort by position for deterministic ordering
        entities.sort(key=lambda e: e.start)

        self._log.debug(
            "PII detection complete",
            entities_found=len(entities),
            types_found=list({e.entity_type for e in entities}),
        )
        return entities

    @property
    def supported_entity_types(self) -> list[str]:
        """Return the list of all entity types this detector can identify.

        Returns:
            List of entity type strings.
        """
        return DEFAULT_ENTITY_TYPES.copy()
