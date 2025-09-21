# Multi-stage Dockerfile for AI-Powered Customer Support Ticket Classifier

# --- Base builder image ---
FROM python:3.13-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# System build deps (if needed later for scientific libs)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirement spec first for layer caching
COPY requirements.txt ./
RUN pip install --upgrade pip && pip wheel --no-cache-dir --wheel-dir /wheels -r requirements.txt

# --- Final runtime image ---
FROM python:3.13-slim AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    APP_HOME=/app

WORKDIR ${APP_HOME}

# Create non-root user
RUN useradd -m appuser

# Copy wheels and install
COPY --from=builder /wheels /wheels
COPY requirements.txt ./
RUN pip install --no-cache-dir /wheels/* && rm -rf /wheels

# Copy source code
COPY . .

# Adjust permissions
RUN chown -R appuser:appuser ${APP_HOME}
USER appuser

EXPOSE 8000

# Default environment (override in deployment)
ENV LOG_LEVEL=info STRUCTURED_LOGS=true

# Run with uvicorn
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
