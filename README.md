# LLM Gateway

Multi-provider LLM routing gateway with rate limiting, budget management, and resilience.

## Overview

This gateway provides a unified API for multiple LLM providers with:
- Multi-provider routing and fallback
- Rate limiting (RPM/TPM)
- Budget management and cost tracking
- Circuit breakers and resilience
- Comprehensive observability

## Requirements

- Python 3.11 or higher
- Redis 7+
- PostgreSQL 15+
- [uv](https://github.com/astral-sh/uv) for dependency management

## Setup

1. Install `uv`:
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. Clone the repository and navigate to the project directory

3. Copy `.env.example` to `.env` and configure with your values:
   ```bash
   cp .env.example .env
   ```

4. Install dependencies:
   ```bash
   uv sync --all-extras
   ```

## Development

Run tests:
```bash
uv run pytest
```

Run linter:
```bash
uv run ruff check .
```

Run formatter:
```bash
uv run ruff format .
```

Run type checker:
```bash
uv run mypy src/
```

## Project Structure

```
src/gateway/           # Main application package
├── api/              # FastAPI routes
├── domain/           # Domain models
├── auth/             # Authentication
├── providers/        # LLM provider integrations
├── routing/          # Request routing and fallback
├── rate_limiting/    # RPM/TPM rate limiting
├── budget/           # Financial budget management
├── resilience/       # Circuit breakers and health checks
├── observability/    # Logging, metrics, tracing
├── admin/            # Administrative API
└── config/           # Configuration loading

database/migrations/  # Database schema migrations
tests/               # Test suite
├── unit/            # Unit tests
├── integration/     # Integration tests
└── load/            # Load tests

config/              # Configuration files
├── pricing.yaml     # Provider pricing
└── routing.yaml     # Routing policies
```

## Architecture

See `docs/prd/LLM_Gateway_PRD_V2.2.pdf` for the complete Product Requirements Document.

## License

Copyright © 2026. All rights reserved.
