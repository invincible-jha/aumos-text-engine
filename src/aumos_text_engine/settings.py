"""Settings for AumOS Text Engine.

Extends AumOSSettings with text-engine-specific configuration.
All env vars use the AUMOS_TEXT__ prefix.
"""

from aumos_common.config import AumOSSettings
from pydantic import Field
from pydantic_settings import BaseSettings


class TextEngineSettings(BaseSettings):
    """Text-engine-specific configuration.

    All variables use the AUMOS_TEXT__ env prefix.

    Example:
        AUMOS_TEXT__LLM_SERVING_URL=http://llm-serving:8001
        AUMOS_TEXT__DEFAULT_MODEL=llama3-8b-instruct
    """

    # LLM serving
    llm_serving_url: str = Field(
        default="http://localhost:8001",
        description="aumos-llm-serving base URL",
    )
    llm_serving_timeout_seconds: int = Field(
        default=120,
        description="LLM HTTP request timeout in seconds",
    )
    default_model: str = Field(
        default="llama3-8b-instruct",
        description="Default LLM model identifier for generation",
    )
    max_tokens: int = Field(default=4096, description="Max tokens for LLM generation")
    temperature: float = Field(default=0.7, description="Default generation temperature")

    # Privacy engine
    privacy_engine_url: str = Field(
        default="http://localhost:8002",
        description="aumos-privacy-engine base URL",
    )
    privacy_engine_timeout_seconds: int = Field(
        default=30,
        description="Privacy engine HTTP request timeout in seconds",
    )

    # PII detection
    spacy_model: str = Field(
        default="en_core_web_lg",
        description="spaCy NER model name for PII detection",
    )
    presidio_score_threshold: float = Field(
        default=0.7,
        description="Minimum presidio confidence score to consider an entity as PII",
        ge=0.0,
        le=1.0,
    )
    pii_replacement_strategy: str = Field(
        default="entity_aware",
        description="PII replacement strategy: entity_aware | random | mask",
    )

    # Quality validation
    semantic_similarity_threshold: float = Field(
        default=0.7,
        description="Minimum semantic similarity score between input and output",
        ge=0.0,
        le=1.0,
    )
    embedding_model: str = Field(
        default="all-MiniLM-L6-v2",
        description="Sentence transformer model for semantic similarity",
    )

    # Batch processing
    batch_max_documents: int = Field(
        default=100,
        description="Maximum number of documents per batch request",
    )
    batch_concurrency: int = Field(
        default=5,
        description="Number of documents processed concurrently in batch",
    )

    # Fine-tuning (LoRA)
    finetune_output_dir: str = Field(
        default="/tmp/aumos-text-engine/models",
        description="Local directory for LoRA fine-tuned model artifacts",
    )
    finetune_max_steps: int = Field(default=1000, description="Max training steps for LoRA")
    finetune_learning_rate: float = Field(
        default=3e-4,
        description="Learning rate for LoRA fine-tuning",
    )

    model_config = {"env_prefix": "AUMOS_TEXT__", "env_nested_delimiter": "__"}


class Settings(AumOSSettings):
    """Combined settings for aumos-text-engine.

    Composes core AumOS settings with text-engine-specific config.
    """

    text: TextEngineSettings = Field(default_factory=TextEngineSettings)


_settings: Settings | None = None


def get_settings() -> Settings:
    """Return the cached settings singleton.

    Returns:
        Settings: Validated settings instance loaded from environment.
    """
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
