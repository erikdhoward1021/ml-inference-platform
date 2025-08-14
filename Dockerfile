FROM python:3.11-slim as builder

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the model
RUN python -c "from sentence_transformers import SentenceTransformer; \
    import shutil; \
    model = SentenceTransformer('all-MiniLM-L6-v2'); \
    model.save('/tmp/model-cache/all-MiniLM-L6-v2')"

FROM python:3.11-slim as runtime

WORKDIR /app

# Create non-root user
RUN useradd -m -u 1000 mluser

# Copy application code
COPY --chown=mluser:mluser src/ ./src/

# Copy pre-downloaded model
COPY --from=builder --chown=mluser:mluser /tmp/model-cache /app/models

# Install runtime dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Switch to non-root user
USER mluser

# Set environment variables
ENV MODEL_CACHE_DIR=/app/models
ENV PYTHONPATH=/app

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]