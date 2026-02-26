# CLAUDE.md — AumOS Text Engine

## Project Overview

AumOS Enterprise is a composable enterprise AI platform with 9 products + 2 services
across 62 repositories. This repo (`aumos-text-engine`) is part of **Phase 1A Data Factory**:
the core synthetic data generation engine for unstructured text.

**Release Tier:** B (Core Platform)
**Product Mapping:** Data Factory — Text Engine
**Phase:** 1A (Months 1-4)

## Repo Purpose

The Text Engine provides LLM-powered synthetic text generation with PII-aware entity
replacement, domain-specific fine-tuning, and semantic consistency validation. It is
the primary service for generating synthetic documents that preserve statistical and
linguistic properties of real enterprise text while eliminating all personally identifiable
information.

## Architecture Position

```
aumos-common          ──► aumos-text-engine ──► MinIO (document storage)
aumos-proto           ──►                   ──► aumos-event-bus (synthesis events)
aumos-llm-serving     ──►                   ──► aumos-fidelity-validator (quality check)
aumos-privacy-engine  ──►

External:
  Presidio + spaCy  ◄──► aumos-text-engine (local PII detection)
  Transformers/PEFT ◄──► aumos-text-engine (LoRA fine-tuning)
```

**Upstream dependencies (this repo IMPORTS from):**
- `aumos-common` — auth, database, events, errors, config, health, pagination
- `aumos-proto` — Protobuf message definitions for Kafka events
- `aumos-llm-serving` — LLM inference (vLLM/Ollama) via HTTP
- `aumos-privacy-engine` — Differential privacy budget tracking via HTTP

**Downstream dependents (other repos depend on this):**
- `aumos-fidelity-validator` — validates quality of generated text
- `aumos-data-pipeline` — orchestrates text generation as part of synthetic datasets

## Tech Stack (DO NOT DEVIATE)

| Component | Version | Purpose |
|-----------|---------|---------|
| Python | 3.11+ | Runtime |
| FastAPI | 0.110+ | REST API framework |
| SQLAlchemy | 2.0+ (async) | Database ORM |
| asyncpg | 0.29+ | PostgreSQL async driver |
| Pydantic | 2.6+ | Data validation, settings, API schemas |
| Presidio | 2.2+ | PII detection and anonymization |
| spaCy | 3.7+ | NER for PII entity detection |
| Transformers | 4.38+ | LLM base and fine-tuning |
| PEFT | 0.9+ | LoRA fine-tuning |
| sentence-transformers | 2.7+ | Semantic similarity validation |
| confluent-kafka | 2.3+ | Kafka producer |
| aiobotocore | 2.12+ | Async MinIO/S3 client |
| structlog | 24.1+ | Structured JSON logging |
| OpenTelemetry | 1.23+ | Distributed tracing |
| httpx | 0.27+ | Async HTTP client for LLM/privacy engine |
| pytest | 8.0+ | Testing framework |
| ruff | 0.3+ | Linting and formatting |
| mypy | 1.8+ | Type checking |

## Coding Standards

### ABSOLUTE RULES (violations will break integration with other repos)

1. **Import aumos-common, never reimplement.** If aumos-common provides it, use it.
   ```python
   # CORRECT
   from aumos_common.auth import get_current_tenant, get_current_user
   from aumos_common.database import get_db_session, Base, AumOSModel, BaseRepository
   from aumos_common.events import EventPublisher, Topics
   from aumos_common.errors import NotFoundError, ErrorCode
   from aumos_common.config import AumOSSettings
   from aumos_common.health import create_health_router
   from aumos_common.pagination import PageRequest, PageResponse, paginate
   from aumos_common.app import create_app
   from aumos_common.logging import get_logger
   ```

2. **Type hints on EVERY function.** No exceptions.

3. **Pydantic models for ALL API inputs/outputs.** Never return raw dicts.

4. **RLS tenant isolation via aumos-common.** Never write raw SQL that bypasses RLS.

5. **Structured logging via structlog.** Never use print() or logging.getLogger().

6. **Publish domain events to Kafka after state changes.**

7. **Async by default.** All I/O operations must be async.

8. **Google-style docstrings** on all public classes and functions.

9. **PII leakage is a critical bug.** Always verify no PII remains before returning output.

### Style Rules

- Max line length: **120 characters**
- Import order: stdlib → third-party → aumos-common → local
- Linter: `ruff` (select E, W, F, I, N, UP, ANN, B, A, COM, C4, PT, RUF)
- Type checker: `mypy` strict mode
- Formatter: `ruff format`

### File Structure Convention

```
src/aumos_text_engine/
├── __init__.py
├── main.py                    # FastAPI app entry point using create_app()
├── settings.py                # Extends AumOSSettings (env prefix AUMOS_TEXT_)
├── api/
│   ├── __init__.py
│   ├── router.py              # /api/v1/text/* endpoints
│   └── schemas.py             # All Pydantic request/response models
├── core/
│   ├── __init__.py
│   ├── models.py              # SQLAlchemy ORM models (txt_ prefix)
│   ├── interfaces.py          # Protocol interfaces for all adapters
│   └── services.py            # PIIService, SynthesisService, DomainService, etc.
└── adapters/
    ├── __init__.py
    ├── pii_detector.py        # Presidio + spaCy NER integration
    ├── entity_replacer.py     # Context-aware PII entity replacement
    ├── llm_client.py          # HTTP client for aumos-llm-serving
    ├── style_preserver.py     # Style transfer via few-shot prompting
    ├── quality_validator.py   # Semantic similarity scoring
    ├── repositories.py        # SQLAlchemy repos for all models
    ├── kafka.py               # TextEventPublisher
    ├── storage.py             # MinIO adapter for document I/O
    └── privacy_client.py      # HTTP client for privacy engine DP budget
```

## API Conventions

- All endpoints under: `/api/v1/text/`
- Auth: Bearer JWT token (validated by aumos-common)
- Tenant: `X-Tenant-ID` header (set by auth middleware)
- Request ID: `X-Request-ID` header (auto-generated if missing)
- Job pattern: POST creates job, GET /jobs/{id} polls status, output URI in response

## Database Conventions

- Table prefix: `txt_` (text-engine)
- `txt_synthesis_jobs` — all text synthesis/replacement/batch jobs
- `txt_domain_templates` — domain-specific prompt templates
- `txt_finetune_jobs` — LoRA fine-tuning job tracking
- ALL tenant-scoped tables: extend `AumOSModel`

## Kafka Events Published

| Event Type | Topic | Trigger |
|------------|-------|---------|
| text.synthesis.started | TEXT_SYNTHESIS_EVENTS | Job created |
| text.synthesis.completed | TEXT_SYNTHESIS_EVENTS | Generation complete |
| text.synthesis.failed | TEXT_SYNTHESIS_EVENTS | Job failed |
| text.pii.detected | TEXT_PII_EVENTS | PII entities found |
| text.pii.replaced | TEXT_PII_EVENTS | PII replacement complete |
| text.finetune.started | TEXT_FINETUNE_EVENTS | LoRA training started |
| text.finetune.completed | TEXT_FINETUNE_EVENTS | Model artifact ready |

## PII Entity Types (50+)

Presidio detects all standard entity types including:
PERSON, EMAIL_ADDRESS, PHONE_NUMBER, CREDIT_CARD, US_SSN, US_PASSPORT, US_DRIVER_LICENSE,
IP_ADDRESS, URL, NRP (Nationality/Religious/Political group), LOCATION, DATE_TIME,
MEDICAL_LICENSE, US_BANK_NUMBER, IBAN_CODE, CRYPTO, AU_ABN, AU_ACN, AU_TFN, AU_MEDICARE,
UK_NHS, SG_NRIC_FIN, IN_PAN, IN_AADHAAR, IT_FISCAL_CODE, ES_NIF, PL_PESEL, plus
custom enterprise entity types defined in DomainTemplate.

## Entity Replacement Strategy

Context-aware replacement preserves format and plausibility:
- PERSON → realistic fake name (matching gender/cultural context)
- EMAIL_ADDRESS → fake email at generated domain
- PHONE_NUMBER → format-preserving fake phone
- DATE_TIME → date-shifted by random offset (preserving relative ordering)
- LOCATION → nearby city of similar size
- CREDIT_CARD → Luhn-valid fake card number
- ORGANIZATION → fake company name in same industry

## What Claude Code Should NOT Do

1. **Do NOT reimplement anything in aumos-common.** JWT, auth, DB, Kafka, health — import it.
2. **Do NOT return text with PII intact.** Always verify replacement completeness.
3. **Do NOT log document content.** Never log text that might contain PII.
4. **Do NOT load model weights from untrusted sources.** Validate artifact URIs.
5. **Do NOT skip tenant isolation.** Use RLS on every DB query.
6. **Do NOT hardcode LLM model names.** Always use settings.default_model.
7. **Do NOT return raw dicts.** Every API response must be a typed Pydantic model.
8. **Do NOT store document content in the database.** Use MinIO; DB holds only URIs.
