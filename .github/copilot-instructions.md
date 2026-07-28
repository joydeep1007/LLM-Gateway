# Copilot Instructions — LLM Gateway

## Source of Truth

The architectural and behavioural source of truth is:

`docs/prd/LLM_Gateway_PRD_V2.2.pdf`

- Use the current numbered implementation prompt as the primary specification for the current task.
- Consult the PRD only when additional architectural context or clarification is needed.
- Do not load/read the entire PRD for every task when the current prompt and repository provide sufficient context.
- Implement only the current numbered PRD prompt.
- Do not implement future phases or stretch goals unless explicitly requested.
- Preserve interfaces, schemas, migrations, and behaviour created by earlier phases.
- Prefer the smallest coherent change that satisfies the current task.

When requirements differ, use this precedence unless the user explicitly overrides it:

1. Current explicit user instruction
2. PRD V2.2 architectural invariants
3. Current numbered PRD implementation prompt
4. Existing implementation

Do not silently resolve meaningful conflicts. Report them before making architectural changes.

## Engineering Principles

Prioritise:

1. correctness;
2. security;
3. testability;
4. readability;
5. finish-ability;
6. measured performance where required.

Prefer explicit, understandable implementations over clever abstractions.

Do not introduce frameworks, architectural layers, dependencies, or design patterns without a current requirement.

Keep FastAPI routes thin. Business logic belongs in the appropriate module.

Use async I/O for network, Redis, PostgreSQL, and provider operations.

Use full type hints and clear module boundaries.

## Tooling

Use:

- Python 3.11+
- FastAPI / Pydantic v2
- Redis 7+
- PostgreSQL 15+ / asyncpg
- httpx
- structlog
- OpenTelemetry
- Prometheus / Grafana / Jaeger
- pytest / pytest-asyncio
- Locust
- Docker Compose
- GitHub Actions

Use `uv` for Python environments and dependencies.

Prefer:

    uv sync
    uv run pytest
    uv run ruff check .
    uv run ruff format --check .
    uv run mypy src/

`pyproject.toml` is the canonical dependency configuration.

Do not introduce `requirements.txt` or a pip-centric workflow unless explicitly requested.

## Architecture Boundaries

Maintain the `src/gateway/` architecture:

    api/
    domain/
    auth/
    providers/
    routing/
    rate_limiting/
    budget/
    resilience/
    observability/
    admin/
    config/

Do not move unrelated responsibilities into `main.py` or API routes.

Provider-specific translation belongs inside provider adapters. Higher layers operate on gateway domain models.

Routing, retry, fallback, budgeting, rate limiting, authentication, and observability remain separate concerns.

## Datastores

Redis is for distributed/ephemeral coordination:

- RPM/TPM token buckets;
- budget reservations/counters;
- circuit-breaker state;
- provider-health state;
- short-lived coordination/idempotency state.

PostgreSQL is for durable data:

- teams;
- API-key metadata;
- audit records;
- other explicitly migrated durable records.

Do not blur these responsibilities.

## Request Identity

One gateway request has one `request_id`.

Every actual provider call has a unique `attempt_id`, including retries and fallbacks.

Never reuse an `attempt_id`.

Budget reconciliation is idempotent per:

    (request_id, attempt_id)

Never use `request_id` alone for attempt reconciliation.

## Rate-Limit Invariants

RPM + TPM admission MUST be one atomic Redis Lua operation.

If either bucket lacks capacity:

- reject admission;
- consume neither bucket.

TPM admission uses:

    estimated_tpm_tokens =
        estimated_input_tokens + max_output_tokens

TPM accounting is deliberately conservative.

DO NOT refund unused TPM capacity after provider execution.

Redis Lua/concurrency correctness must be tested against real Redis, not fakeredis.

## Financial Accounting Invariants

Financial budgeting is separate from TPM accounting.

Use exact integer microdollars for authoritative currency accounting; never authoritative Python floats.

Pricing belongs in:

`config/pricing.yaml`

Never invent or hardcode current provider prices.

Each provider attempt independently performs:

    admission -> reservation -> execution -> reconciliation

Each retry/fallback gets its own reservation and reconciliation.

When authoritative usage exists, reconcile to actual cost.

When failure has clearly zero billable usage, release the reservation.

When billable usage is unknown, apply the PRD's conservative financial policy.

Financial reconciliation MUST NOT refund or modify TPM capacity.

## Authentication & Security

API-key verification uses HMAC-SHA-256 with server-side `API_KEY_PEPPER`.

Do not replace it with bcrypt/PBKDF2.

The pepper must come from the environment and must never be stored in PostgreSQL.

Never log:

- API keys;
- Authorization headers;
- secret peppers;
- provider credentials;
- prompts;
- completions.

Never commit real secrets or `.env`.

`.env.example` contains placeholders only.

Do not introduce insecure fallback defaults for required secrets.

## Provider / Reliability Rules

All providers implement the common `LLMProvider` abstraction.

MockLLMProvider is the default provider for automated testing and failure simulation.

Automated tests and CI must make ZERO paid provider API calls.

Retry and fallback are separate concepts.

Retries stay on the current provider according to `RetryPolicy`.

Fallback occurs after applicable retries are exhausted.

Every provider call receives a new `attempt_id`.

Do not retry explicitly non-retryable errors.

Circuit-breaker state shared across replicas belongs in Redis.

## Streaming Invariant

Fallback is allowed only BEFORE response streaming begins.

Once any response bytes/tokens have been sent to the client:

    NEVER transparently fallback to another provider.

Do not combine output from multiple providers into one client stream.

Do not buffer the full response merely to enable fallback or accounting.

## Configuration

Runtime policy belongs in configuration where specified by the PRD.

Important files include:

    config/pricing.yaml
    config/routing.yaml

Do not scatter provider pricing or routing chains through source code.

Validate required configuration and fail fast for invalid or missing security-critical settings.

Database schema changes must use ordered migrations.

## Observability

Use:

- OpenTelemetry -> OTLP -> Jaeger
- Prometheus -> Grafana
- structlog -> structured JSON

Prometheus labels MUST remain low-cardinality.

Never use unbounded identifiers/content such as `request_id`, `attempt_id`, API keys, or prompt text as metric labels.

Gateway overhead is:

    total_request_duration
    - sum(measured_provider_call_durations)

Never subtract configured/mock latency.

Treat gateway overhead as an application-level approximation.

## Testing

Every numbered PRD prompt must be independently testable before moving forward.

Use:

- unit tests for pure logic;
- integration tests for datastore/infrastructure behaviour;
- real Redis for Lua/concurrency tests;
- MockLLMProvider for provider behaviour.

Test meaningful failure paths as well as success paths.

For bug fixes, add regression tests where practical.

Never delete, weaken, bypass, or rewrite valid tests merely to obtain a passing build.

## Scope Control

Core implementation follows PRD Phases 0–10 sequentially.

Do NOT implement stretch goals during core development, including:

- priority queues;
- request enrichment;
- regex content filtering;
- Slack budget alerts;
- configuration hot reload;
- admin dashboard UI;
- prompt caching.

`InteractiveUser` and `LargeContextUser` are load-test traffic patterns, NOT priority classes.

## Benchmark Integrity

PRD performance numbers are targets, not guaranteed results.

Never fabricate, massage, or pre-write benchmark/CV numbers.

README and CV metrics must come from actual measured runs.

If measured performance misses a target, report the real result.

## Before Editing

For every task:

1. Identify the current PRD prompt and phase.
2. Inspect relevant existing files.
3. Identify dependencies created by earlier phases.
4. Confirm the task does not require unimplemented future functionality.
5. Implement only the requested scope.

If a required earlier dependency is missing, stop and report it instead of inventing architecture.

If the current prompt contains technically ambiguous terminology that could lead to materially different implementations, do not guess.

First inspect the existing architecture and relevant PRD context. If the intended design remains unclear, report the ambiguity before implementation.

## Definition of Done

Before declaring a task complete:

1. Requested behaviour/files exist.
2. Relevant tests exist and pass.
3. Appropriate lint/type checks have been run.
4. PRD invariants remain satisfied.
5. No unrelated or future features were added.
6. No secrets or unsafe logging were introduced.
7. Report files created/modified.
8. Report commands/tests executed and their results.
9. Report unresolved assumptions/issues.
10. Stop and wait for the next prompt.

Do not automatically continue to the next PRD prompt or phase.

## Non-Negotiable Invariants

Never violate these unless the user explicitly changes the architecture:

1. RPM + TPM admission is atomic.
2. TPM has no post-admission refund.
3. Financial accounting is separate from TPM accounting.
4. Authoritative money uses integer microdollars.
5. Every provider call has a unique `attempt_id`.
6. Reconciliation is idempotent per `(request_id, attempt_id)`.
7. Retries/fallbacks create new attempts.
8. API keys use HMAC-SHA-256 + server-side pepper.
9. Secrets, prompts, and completions are never logged.
10. Automated tests make no paid provider calls.
11. Redis Lua/concurrency tests use real Redis.
12. No transparent fallback after streaming begins.
13. Provider pricing and routing policy are configuration-driven.
14. Prometheus labels remain low-cardinality.
15. Gateway overhead uses measured provider-call durations.
16. Benchmark/CV numbers come only from real measurements.
17. Stretch goals stay outside Phases 0–10.
18. Current work must not depend on unimplemented future phases.