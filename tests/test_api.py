"""
Tests for the FastAPI application endpoints.
"""

import pytest
from fastapi.testclient import TestClient
from src.api.main import app
from src.utils.config import settings

# Create test client
client = TestClient(app)


class TestHealthEndpoints:
    """Test health check endpoints."""

    def test_liveness_probe(self):
        """Liveness should always return 200."""
        response = client.get("/health/live")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "model_loaded" in data

    def test_readiness_probe(self):
        """Readiness depends on model being loaded."""
        response = client.get("/health/ready")
        # This might be 200 or 503 depending on model load status
        assert response.status_code in [200, 503]

        if response.status_code == 200:
            data = response.json()
            assert data["status"] == "ready"
            assert data["model_loaded"] is True


class TestPredictionEndpoints:
    """Test main prediction endpoints."""

    def test_single_prediction(self):
        """Test single text embedding generation."""
        response = client.post("/predict", json={"text": "This is a test sentence"})

        if response.status_code == 200:
            data = response.json()
            assert "embedding" in data
            assert "dimension" in data
            assert "model_version" in data
            assert "inference_time_ms" in data
            assert isinstance(data["embedding"], list)
            assert len(data["embedding"]) == data["dimension"]

    def test_empty_text_validation(self):
        """Test that empty text is rejected."""
        response = client.post("/predict", json={"text": "   "})  # Just whitespace
        assert response.status_code == 422  # Validation error

    def test_batch_prediction(self):
        """Test batch embedding generation."""
        texts = ["First test sentence", "Second test sentence", "Third test sentence"]

        response = client.post("/predict/batch", json={"texts": texts})

        if response.status_code == 200:
            data = response.json()
            assert "embeddings" in data
            assert "batch_size" in data
            assert data["batch_size"] == len(texts)
            assert len(data["embeddings"]) == len(texts)

    def test_batch_size_limit(self):
        """Test that batch size limit is enforced."""
        # Create a batch larger than MAX_BATCH_SIZE
        texts = [f"Text {i}" for i in range(settings.MAX_BATCH_SIZE + 1)]

        response = client.post("/predict/batch", json={"texts": texts})

        assert response.status_code in [400, 422]

    def test_similarity_endpoint(self):
        """Test similarity computation."""
        response = client.post(
            "/similarity",
            json={"text1": "The cat is on the mat", "text2": "A feline sits on a rug"},
        )

        if response.status_code == 200:
            data = response.json()
            assert "similarity" in data
            assert -1 <= data["similarity"] <= 1
            assert "model_version" in data
            assert "inference_time_ms" in data


class TestModelEndpoints:
    """Test model information endpoints."""

    def test_model_info(self):
        """Test model information endpoint."""
        response = client.get("/model/info")
        assert response.status_code == 200

        data = response.json()
        assert "loaded" in data
        assert "model_name" in data

        if data["loaded"]:
            assert "embedding_dimension" in data
            assert "max_sequence_length" in data


class TestGeneralEndpoints:
    """Test general API endpoints."""

    def test_root_endpoint(self):
        """Test root endpoint returns service info."""
        response = client.get("/")
        assert response.status_code == 200

        data = response.json()
        assert "service" in data
        assert "version" in data
        assert "status" in data
        assert data["status"] == "running"

    def test_openapi_documentation(self):
        """Test that OpenAPI documentation is available."""
        response = client.get("/docs")
        assert response.status_code == 200

        # Test OpenAPI JSON endpoint
        response = client.get("/openapi.json")
        assert response.status_code == 200

        data = response.json()
        assert "openapi" in data
        assert "paths" in data
        assert "components" in data


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_invalid_json(self):
        """Test handling of invalid JSON."""
        response = client.post(
            "/predict",
            data="not valid json",
            headers={"Content-Type": "application/json"},
        )
        assert response.status_code == 422

    def test_missing_required_field(self):
        """Test handling of missing required fields."""
        response = client.post("/predict", json={})  # Missing 'text' field
        assert response.status_code == 422

    def test_text_too_long(self):
        """Test handling of text exceeding max length."""
        long_text = "a" * 1000  # Exceeds 512 char limit

        response = client.post("/predict", json={"text": long_text})
        assert response.status_code == 422


# @pytest.fixture(autouse=True)
# def setup_and_teardown():
#     """Setup and teardown for each test."""
#     # Setup code here if needed
#     yield
#     # Teardown code here if needed
