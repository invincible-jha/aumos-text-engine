# aumos-text-engine

[![CI](https://github.com/aumos-enterprise/aumos-text-engine/actions/workflows/ci.yml/badge.svg)](https://github.com/aumos-enterprise/aumos-text-engine/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Tier B](https://img.shields.io/badge/tier-B%20Core-green.svg)]()

LLM-powered synthetic text generation with PII-aware entity replacement, domain-specific fine-tuning, and semantic consistency validation. Part of the **AumOS Enterprise Data Factory**.

## Overview

`aumos-text-engine` generates high-quality synthetic text documents that:

- **Preserve linguistic properties** — style, tone, domain vocabulary, formatting
- **Eliminate all PII** — 50+ entity types detected by Presidio + spaCy, replaced with plausible fakes
- **Domain-specific quality** — dedicated generators for legal, medical, financial, HR, and custom domains
- **Measurable fidelity** — semantic similarity scoring validates output quality
- **Enterprise-scale** — batch processing for thousands of documents with async I/O throughout

## Architecture

```
                    ┌─────────────────────────────────────┐
                    │         aumos-text-engine            │
                    │                                     │
  Input doc ──────► │  ┌─────────────────────────────┐   │ ──► MinIO (synthetic doc)
                    │  │  PII Detection (Presidio)    │   │
                    │  │  Entity Replacement           │   │
                    │  │  LLM Generation               │   │ ──► Kafka (lifecycle events)
                    │  │  Style Preservation           │   │
                    │  │  Quality Validation           │   │ ──► Privacy Engine (DP budget)
                    │  └─────────────────────────────┘   │
                    └─────────────────────────────────────┘
                              │               ▲
                              ▼               │
                    aumos-llm-serving    aumos-privacy-engine
                    (vLLM / Ollama)      (DP budget tracking)
```

### Hexagonal Architecture

```
src/aumos_text_engine/
├── api/              # HTTP layer (FastAPI routes, Pydantic schemas)
├── core/             # Business logic (services, interfaces, ORM models)
└── adapters/         # External integrations (Presidio, LLM, MinIO, Kafka)
```

## Quick Start

### Prerequisites

- Python 3.11+
- Docker + Docker Compose
- aumos-common and aumos-proto installed (AumOS internal PyPI)

### Local Development

```bash
# Clone
git clone https://github.com/aumos-enterprise/aumos-text-engine.git
cd aumos-text-engine

# Install
make install

# Configure
cp .env.example .env
# Edit .env with your local settings

# Start infrastructure (postgres, redis, kafka, minio)
make docker-run

# Run tests
make test

# Start service directly
uvicorn aumos_text_engine.main:app --reload
```

### Docker

```bash
# Build image
make docker-build

# Run with compose (full stack)
docker compose -f docker-compose.dev.yml up -d
```

## API Reference

Base URL: `http://localhost:8000/api/v1`

### Text Synthesis

**POST /text/synthesize** — Generate synthetic text from a template or example document.

```json
{
  "domain": "legal",
  "template_id": "uuid-optional",
  "example_text": "This agreement is between John Smith (johnn@email.com) and...",
  "generation_config": {
    "max_tokens": 2048,
    "temperature": 0.7,
    "model": "llama3-8b-instruct"
  },
  "style_preserve": true
}
```

Response: `JobResponse` with `job_id` for polling.

**POST /text/pii-replace** — Detect and replace PII in existing text.

```json
{
  "text": "Patient Jane Doe, SSN 123-45-6789, was admitted on...",
  "entity_types": ["PERSON", "US_SSN", "DATE_TIME"],
  "replacement_strategy": "entity_aware"
}
```

**POST /text/batch** — Process multiple documents in parallel.

```json
{
  "documents": [
    {"id": "doc-1", "text": "...", "domain": "medical"},
    {"id": "doc-2", "text": "...", "domain": "financial"}
  ],
  "operation": "pii_replace",
  "concurrency": 5
}
```

**GET /text/jobs/{job_id}** — Poll job status.

**GET /text/domains** — List available domain-specific generators.

**POST /text/fine-tune** — Fine-tune generation model on enterprise corpus.

```json
{
  "base_model": "llama3-8b-instruct",
  "corpus_uri": "s3://aumos-text-engine/corpus/legal-contracts/",
  "lora_config": {
    "r": 16,
    "lora_alpha": 32,
    "target_modules": ["q_proj", "v_proj"]
  }
}
```

**GET /text/fine-tune/{job_id}** — Fine-tune job status and artifact URI.

### Health Endpoints

- `GET /live` — Liveness probe
- `GET /ready` — Readiness probe (checks DB, LLM serving, privacy engine)
- `GET /health` — Detailed health with all dependency statuses

## Configuration

All settings use the `AUMOS_TEXT__` prefix. See `.env.example` for full reference.

| Variable | Default | Description |
|----------|---------|-------------|
| `AUMOS_TEXT__LLM_SERVING_URL` | `http://localhost:8001` | aumos-llm-serving endpoint |
| `AUMOS_TEXT__PRIVACY_ENGINE_URL` | `http://localhost:8002` | aumos-privacy-engine endpoint |
| `AUMOS_TEXT__DEFAULT_MODEL` | `llama3-8b-instruct` | Default LLM for generation |
| `AUMOS_TEXT__SPACY_MODEL` | `en_core_web_lg` | spaCy NER model |
| `AUMOS_TEXT__PRESIDIO_SCORE_THRESHOLD` | `0.7` | Minimum PII confidence score |
| `AUMOS_TEXT__SEMANTIC_SIMILARITY_THRESHOLD` | `0.7` | Minimum output quality score |
| `AUMOS_TEXT__BATCH_MAX_DOCUMENTS` | `100` | Max documents per batch request |
| `AUMOS_TEXT__BATCH_CONCURRENCY` | `5` | Parallel document processing |

## Development Guide

### Running Tests

```bash
make test           # Full suite with coverage
make test-quick     # Fast run, stop on first failure
pytest tests/test_services.py -v  # Single file
```

### Code Quality

```bash
make lint           # Check style + formatting
make format         # Auto-fix style + formatting
make typecheck      # mypy strict mode
make all            # lint + typecheck + test
```

### Adding a Domain Template

1. Add template SQL migration in `migrations/`
2. Seed via `POST /api/v1/text/domains` (admin endpoint)
3. The template includes: `prompt_template`, `example_documents`, `custom_entity_types`

### Adding a New PII Entity Type

1. Define a `PatternRecognizer` subclass in `adapters/pii_detector.py`
2. Register it with the `AnalyzerEngine` in `PresidioPIIDetector.__init__`
3. Add replacement logic to `adapters/entity_replacer.py`
4. Add test case in `tests/test_pii_detector.py`

## Related Repos

| Repo | Relationship |
|------|-------------|
| [aumos-common](../aumos-common) | Core utilities — auth, DB, Kafka, health |
| [aumos-proto](../aumos-proto) | Protobuf event schema definitions |
| [aumos-llm-serving](../aumos-llm-serving) | LLM inference backend (vLLM/Ollama) |
| [aumos-privacy-engine](../aumos-privacy-engine) | Differential privacy budget management |
| [aumos-tabular-engine](../aumos-tabular-engine) | Synthetic tabular data generation |
| [aumos-image-engine](../aumos-image-engine) | Synthetic image generation |
| [aumos-fidelity-validator](../aumos-fidelity-validator) | Cross-modal quality validation |
| [aumos-data-pipeline](../aumos-data-pipeline) | Orchestrates all synthesis engines |

## License

Apache 2.0 — see [LICENSE](LICENSE) for details.

**IMPORTANT:** All dependencies must use Apache 2.0, MIT, BSD, ISC, PSF, or MPL 2.0 licenses.
GPL and AGPL dependencies are strictly prohibited.
