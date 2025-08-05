# Multi-stage build for optimized PostgreSQL FDW Data Connector
FROM python:3.11-slim as builder

# Set build arguments
ARG BUILD_DATE
ARG VERSION
ARG VCS_REF

# Add metadata labels
LABEL maintainer="Data Engineering Team" \
      org.label-schema.build-date=$BUILD_DATE \
      org.label-schema.name="instant-data-connector" \
      org.label-schema.description="PostgreSQL FDW-based data connector" \
      org.label-schema.version=$VERSION \
      org.label-schema.vcs-ref=$VCS_REF \
      org.label-schema.schema-version="1.0" \
      security.non-root=true \
      security.no-new-privileges=true

# Install system dependencies for building
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    libpq-dev \
    libssl-dev \
    libffi-dev \
    pkg-config \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set up Python environment
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements first for better caching
COPY requirements.txt requirements-dev.txt ./

# Install Python dependencies
RUN pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt && \
    pip install -r requirements-dev.txt

# Production stage
FROM python:3.11-slim as production

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    PYTHONPATH="/app/src:$PYTHONPATH"

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    postgresql-client \
    libpq5 \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser -d /app -s /sbin/nologin appuser

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Create application directory
WORKDIR /app

# Copy application code
COPY src/ ./src/
COPY tests/ ./tests/
COPY configs/ ./configs/
COPY scripts/ ./scripts/
COPY pytest.ini ./

# Create necessary directories
RUN mkdir -p /app/logs /app/data /app/tmp && \
    chown -R appuser:appuser /app

# Copy health check script
COPY docker/health_check.sh /usr/local/bin/health_check.sh
RUN chmod +x /usr/local/bin/health_check.sh

# Switch to non-root user
USER appuser

# Expose application port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD /usr/local/bin/health_check.sh

# Default command
CMD ["python", "-m", "uvicorn", "src.instant_connector.api:app", \
     "--host", "0.0.0.0", "--port", "8000", \
     "--log-config", "configs/logging.json"]

# Development stage
FROM production as development

# Switch back to root for development dependencies
USER root

# Install development tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    vim \
    htop \
    net-tools \
    telnet \
    && rm -rf /var/lib/apt/lists/*

# Install additional development Python packages
RUN pip install \
    jupyter \
    ipython \
    pytest-xdist \
    pytest-cov \
    black \
    flake8 \
    mypy

# Switch back to appuser
USER appuser

# Development command with hot reload
CMD ["python", "-m", "uvicorn", "src.instant_connector.api:app", \
     "--host", "0.0.0.0", "--port", "8000", \
     "--reload", "--log-level", "debug"]