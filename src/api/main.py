"""
FastAPI application for ML model inference.

This is the main entry point for our API. It sets up routes, middleware,
exception handlers, and lifecycle events.
"""

import time
import logging
import sys
from contextlib import asynccontextmanager
from typing import Dict, Any

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from ..models.sentence_transformers import model_manager
from ..utils.config import settings
from .schemas import (
    TextInput,
    BatchTextInput,
    SimilarityInput,
    EmbeddingResponse,
    BatchEmbeddingResponse,
    SimilarityResponse,
    HealthResponse,
    ModelInfoResponse,
    ErrorResponse,
)

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manage application lifecycle.

    This runs before the app starts serving requests (startup)
    and after it stops (shutdown). It's handles lifecycle events in FastAPI.
    """
    # Startup
    logger.info(f"Starting {settings.APP_NAME} v{settings.APP_VERSION}")

    try:
        # Load model during startup
        logger.info("Loading ML model...")
        model_manager.load_model()
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model during startup: {str(e)}")
        # continue anyway - the model can be loaded on first request

    yield  # Application runs

    # Shutdown
    logger.info("Shutting down application...")


# FastAPI application
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Production-ready ML inference service for text embeddings",
    lifespan=lifespan,
)

# Add CORS middleware for browser-based clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict appropriately in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Middleware to add request ID and timing
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """
    Add processing time to response headers.

    Helpful for performance monitoring and debugging.
    """
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(round(process_time * 1000, 2))
    return response


# Exception handler for validation errors
@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    """Handle validation errors with proper error responses."""
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={"error": "Invalid input", "detail": str(exc)},
    )


# Health check endpoints
@app.get(
    "/health/live",
    response_model=HealthResponse,
    tags=["Health"],
    summary="Liveness probe",
    description="Check if the service is running",
)
async def liveness():
    """
    Kubernetes liveness probe endpoint.

    Should return 200 if the service is alive, even if the model
    isn't loaded.
    """
    return HealthResponse(status="alive", model_loaded=model_manager.model_loaded)


@app.get(
    "/health/ready",
    response_model=HealthResponse,
    tags=["Health"],
    summary="Readiness probe",
    description="Check if the service is ready to serve requests",
)
async def readiness():
    """
    Kubernetes readiness probe endpoint.

    Should only return 200 if the service is ready to handle requests.
    """
    if not model_manager.model_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded yet",
        )

    return HealthResponse(status="ready", model_loaded=True)


# Model information endpoint
@app.get(
    "/model/info",
    response_model=ModelInfoResponse,
    tags=["Model"],
    summary="Get model information",
    description="Get details about the loaded model",
)
async def model_info():
    """
    Return information about the loaded model.

    Useful for debugging and monitoring which model version is deployed.
    """
    return ModelInfoResponse(**model_manager.get_model_info())


# Main prediction endpoints
@app.post(
    "/predict",
    response_model=EmbeddingResponse,
    tags=["Inference"],
    summary="Generate text embedding",
    description="Generate a dense vector embedding for the input text",
)
async def predict(input_data: TextInput):
    """
    Generate embedding for a single text input.

    This is the main inference endpoint. It takes text and returns
    a dense vector representation.
    """
    try:
        result = model_manager.predict_single(input_data.text)
        return EmbeddingResponse(**result)
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}",
        )


@app.post(
    "/predict/batch",
    response_model=BatchEmbeddingResponse,
    tags=["Inference"],
    summary="Generate batch embeddings",
    description="Generate embeddings for multiple texts in a single request",
)
async def predict_batch(input_data: BatchTextInput):
    """
    Generate embeddings for multiple texts efficiently.
    """
    try:
        result = model_manager.predict_batch(input_data.texts)
        return BatchEmbeddingResponse(**result)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error(f"Batch prediction failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction failed: {str(e)}",
        )


@app.post(
    "/similarity",
    response_model=SimilarityResponse,
    tags=["Inference"],
    summary="Compute text similarity",
    description="Compute cosine similarity between two texts",
)
async def compute_similarity(input_data: SimilarityInput):
    """
    Compute semantic similarity between two texts.

    Returns a score between -1 and 1, where 1 means identical meaning,
    0 means unrelated, and -1 means opposite meaning.
    """
    try:
        result = model_manager.compute_similarity(input_data.text1, input_data.text2)
        return SimilarityResponse(**result)
    except Exception as e:
        logger.error(f"Similarity computation failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Similarity computation failed: {str(e)}",
        )


# Root endpoint
@app.get(
    "/",
    tags=["General"],
    summary="Root endpoint",
    description="Basic information about the service",
)
async def root():
    """
    Root endpoint providing basic service information.

    Useful for quick checks that the service is running.
    """
    return {
        "service": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "status": "running",
        "documentation": "/docs",
        "model_loaded": model_manager.model_loaded,
    }


# OpenAPI customization
@app.get("/openapi.json", include_in_schema=False)
async def custom_openapi():
    """
    Customize OpenAPI schema.

    Adds extra metadata to improve the auto-generated documentation.
    """
    if app.openapi_schema:
        return app.openapi_schema

    from fastapi.openapi.utils import get_openapi

    openapi_schema = get_openapi(
        title=settings.APP_NAME,
        version=settings.APP_VERSION,
        description="Production-ready ML inference service",
        routes=app.routes,
    )

    # Add server information
    openapi_schema["servers"] = [
        {
            "url": f"http://localhost:{settings.API_PORT}",
            "description": "Local development server",
        }
    ]

    app.openapi_schema = openapi_schema
    return app.openapi_schema


if __name__ == "__main__":
    # This allows running the module directly for development
    import uvicorn

    uvicorn.run(
        "src.api.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=True,  # Enable auto-reload for development
        log_level=settings.LOG_LEVEL.lower(),
    )
