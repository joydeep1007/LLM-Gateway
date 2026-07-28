# Placeholder Dockerfile for LLM Gateway
# This will be populated with actual build steps in a later phase

FROM python:3.11-slim

WORKDIR /app

# Install uv
RUN pip install uv

# Copy project files
COPY pyproject.toml .
COPY src/ ./src/

# Install dependencies
RUN uv sync --no-dev

# Placeholder CMD - will be replaced with actual gateway startup
CMD ["echo", "Gateway service not yet implemented"]
