# Changelog

All notable changes to `aumos-text-engine` will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-02-26

### Added
- Initial scaffolding for aumos-text-engine
- Presidio + spaCy PII detection (50+ entity types) with context-aware replacement
- Domain-specific text generation (legal, medical, financial, HR, custom)
- LoRA fine-tuning on enterprise corpus via PEFT
- Semantic consistency validation using sentence-transformers
- Batch document processing with progress tracking
- Integration with aumos-llm-serving (vLLM/Ollama) for generation
- Integration with aumos-privacy-engine for differential privacy budget
- MinIO object storage for document I/O
- Kafka event publishing for synthesis lifecycle events
- Full hexagonal architecture (api/ + core/ + adapters/)
- FastAPI REST API with 7 endpoints
- SQLAlchemy async ORM with txt_ table prefix
- Pydantic v2 schemas for all request/response models
- Structured logging via structlog
- OpenTelemetry tracing
- Multi-stage Docker build (non-root)
- GitHub Actions CI (lint, typecheck, test, docker, license-check)
- docker-compose.dev.yml for local development
