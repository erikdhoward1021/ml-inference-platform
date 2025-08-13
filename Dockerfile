
# Multi-stage build for optimal size and security
# Stage 1: Builder - Download models and install dependencies
FROM python:3.11-slim as builder

# Set working directory
WORKDIR /app

# Install system dependencies needed for building
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --user -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY .env.example .env

# Pre-download the model to include in image
# This prevents download at runtime and improves startup time
RUN python -c "from sentence_transformers import SentenceTransformer; \
    model = SentenceTransformer('all-MiniLM-L6-v2'); \
    model.save('/tmp/model-cache')"

# Stage 2: Runtime - Minimal image for production
FROM python:3.11-slim as runtime

# Security: Create non-root user
RUN useradd -m -u 1000 -s /bin/bash mluser

# Set working directory
WORKDIR /app

# Copy Python packages from builder
COPY --from=builder --chown=mluser:mluser /root/.local /home/mluser/.local

# Copy pre-downloaded model
COPY --from=builder --chown=mluser:mluser /tmp/model-cache /home/mluser/.cache/torch/sentence_transformers/sentence-transformers_all-MiniLM-L6-v2

# Copy application code
COPY --chown=mluser:mluser src/ ./src/
COPY --chown=mluser:mluser .env.example .env

# Set Python path
ENV PYTHONPATH=/app:$PYTHONPATH \
    PATH=/home/mluser/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2 \
    MODEL_CACHE_DIR=/home/mluser/.cache/torch/sentence_transformers

# Switch to non-root user
USER mluser

# Expose ports
EXPOSE 8000 9090

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health/live').raise_for_status()"

# Run the application
CMD ["python", "-m", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]