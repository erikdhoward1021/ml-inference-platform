"""
Pydantic models for API request and response validation.

"""

from typing import List, Optional
from pydantic import BaseModel, Field, field_validator, ConfigDict


class TextInput(BaseModel):
    """Single text input for inference."""

    # Pydantic v2 uses model_config instead of nested Config class
    model_config = ConfigDict(
        json_schema_extra={"example": {"text": "This is a sentence to be embedded."}}
    )

    text: str = Field(
        ...,
        min_length=1,
        max_length=512,
        description="Input text to generate embedding for",
    )

    @field_validator("text")
    @classmethod
    def validate_text(cls, v: str) -> str:
        """Ensure text is not just whitespace."""
        if not v.strip():
            raise ValueError("Text cannot be empty or only whitespace")
        return v.strip()


class BatchTextInput(BaseModel):
    """Multiple texts for batch processing."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "texts": [
                    "Sentence 1 to be embedded",
                    "Sentence 2 to be embedded",
                    "Sentence 3 to be embedded",
                ]
            }
        }
    )

    texts: List[str] = Field(
        ...,
        min_length=1,
        max_length=32,
        description="List of texts to process",
    )

    @field_validator("texts")
    @classmethod
    def validate_texts(cls, v: List[str]) -> List[str]:
        """Validate each text in the batch."""
        cleaned = []
        for text in v:
            if not text.strip():
                raise ValueError("Texts cannot be empty or only whitespace")
            if len(text) > 512:
                raise ValueError(f"Text exceeds maximum length of 512 characters")
            cleaned.append(text.strip())
        return cleaned


class SimilarityInput(BaseModel):
    """Input for computing similarity between two texts."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "text1": "Sentence one for batch inference",
                "text2": "Sentence two for batch inference",
            }
        }
    )

    text1: str = Field(
        ..., min_length=1, max_length=512, description="First text for comparison"
    )
    text2: str = Field(
        ..., min_length=1, max_length=512, description="Second text for comparison"
    )

    @field_validator("text1", "text2")
    @classmethod
    def validate_text(cls, v: str) -> str:
        """Ensure texts are not just whitespace."""
        if not v.strip():
            raise ValueError("Text cannot be empty or only whitespace")
        return v.strip()


class EmbeddingResponse(BaseModel):
    """Response containing embedding vector and metadata."""

    embedding: List[float] = Field(
        ..., description="Dense vector representation of the input text"
    )
    dimension: int = Field(..., description="Dimensionality of the embedding vector")
    model_version: str = Field(..., description="Model used to generate the embedding")
    inference_time_ms: float = Field(
        ..., description="Time taken for inference in milliseconds"
    )


class BatchEmbeddingResponse(BaseModel):
    """Response for batch embedding requests."""

    embeddings: List[List[float]] = Field(..., description="List of embedding vectors")
    batch_size: int = Field(..., description="Number of texts processed")
    dimension: int = Field(..., description="Dimensionality of each embedding")
    model_version: str = Field(..., description="Model used to generate embeddings")
    inference_time_ms: float = Field(
        ..., description="Total inference time in milliseconds"
    )
    avg_time_per_item_ms: float = Field(
        ..., description="Average time per text in milliseconds"
    )


class SimilarityResponse(BaseModel):
    """Response containing similarity score."""

    similarity: float = Field(
        ..., ge=-1.0, le=1.0, description="Cosine similarity score between -1 and 1"
    )
    model_version: str = Field(..., description="Model used to compute similarity")
    inference_time_ms: float = Field(
        ..., description="Time taken for inference in milliseconds"
    )


class HealthResponse(BaseModel):
    """Health check response."""

    model_config = ConfigDict(
        json_schema_extra={"example": {"status": "healthy", "model_loaded": True}}
    )

    status: str = Field(..., description="Service status")
    model_loaded: bool = Field(..., description="Whether the ML model is loaded")


class ModelInfoResponse(BaseModel):
    """Detailed model information."""

    loaded: bool = Field(..., description="Whether model is loaded in memory")
    model_name: str = Field(..., description="Name/path of the model")
    embedding_dimension: Optional[int] = Field(
        None, description="Size of embedding vectors"
    )
    load_time_seconds: Optional[float] = Field(
        None, description="Time taken to load model"
    )
    max_sequence_length: Optional[int] = Field(
        None, description="Maximum input sequence length"
    )


class ErrorResponse(BaseModel):
    """Standard error response."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "error": "Model not loaded",
                "detail": "The model failed to load during startup",
            }
        }
    )

    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Additional error details")
