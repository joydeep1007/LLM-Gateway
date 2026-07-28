# Infrastructure

Local development infrastructure using Docker Compose.

## Services

- **gateway**: LLM Gateway application (placeholder)
- **redis**: Redis 7 (cache and distributed coordination)
- **postgres**: PostgreSQL 15 (persistent data)
- **prometheus**: Prometheus (metrics collection)
- **grafana**: Grafana (metrics visualization)
- **jaeger**: Jaeger (distributed tracing)

## Prerequisites

- Docker
- Docker Compose
- `.env` file with required variables (copy from `.env.example`)

## Required Environment Variables

The following environment variables MUST be set in `.env`:

- `API_KEY_PEPPER`: Server-side pepper for API key hashing (CRITICAL - never commit real value)
- `POSTGRES_PASSWORD`: PostgreSQL password
- `ADMIN_SECRET_KEY`: Admin API secret key

## Usage

Start all services:
```bash
cd infra
docker-compose up -d
```

View logs:
```bash
docker-compose logs -f gateway
```

Stop all services:
```bash
docker-compose down
```

Stop and remove volumes:
```bash
docker-compose down -v
```

## Service URLs

- Gateway API: http://localhost:8000
- Redis: localhost:6379
- PostgreSQL: localhost:5432
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000 (admin/admin)
- Jaeger UI: http://localhost:16686

## Health Checks

Redis and PostgreSQL include health checks. The gateway service will wait for both to be healthy before starting.

## Volumes

- `redis-data`: Redis persistence
- `postgres-data`: PostgreSQL data
- `grafana-data`: Grafana dashboards and settings
- `prometheus-data`: Prometheus time series data

## TODO

- Configure Prometheus scrape targets for gateway metrics endpoint
- Add Grafana datasource and dashboard provisioning
