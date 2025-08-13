"""
Tests for the sentence transformer model module.

These tests verify that our model wrapper works correctly,
handles edge cases, and maintains expected performance characteristics.
"""

import pytest
import time
import numpy as np
from src.models.sentence_transformers import ModelManager


class TestModelManager:
    """Test the ModelManager class."""

    @pytest.fixture
    def model_manager(self):
        """Create a fresh model manager instance for testing."""
        return ModelManager()

    def test_model_initialization(self, model_manager):
        """Test that model manager initializes correctly."""
        assert model_manager._model is None
        assert model_manager.model_loaded is False
        assert model_manager.load_time is None

    def test_model_loading(self, model_manager):
        """Test model loading process."""
        # Load model
        model_manager.load_model()

        assert model_manager.model_loaded is True
        assert model_manager._model is not None
        assert model_manager.load_time is not None
        assert model_manager.load_time > 0

        # Test that loading again doesn't reload
        original_load_time = model_manager.load_time
        model_manager.load_model()
        assert model_manager.load_time == original_load_time

    def test_single_prediction(self, model_manager):
        """Test single text prediction."""
        text = "This is a test sentence for embedding generation"

        result = model_manager.predict_single(text)

        assert "embedding" in result
        assert "dimension" in result
        assert "model_version" in result
        assert "inference_time_ms" in result

        # Check embedding properties
        embedding = result["embedding"]
        assert isinstance(embedding, list)
        assert len(embedding) == result["dimension"]
        assert all(isinstance(x, float) for x in embedding)

    def test_similarity_computation(self, model_manager):
        """Test similarity computation between texts."""
        similar_result = model_manager.compute_similarity(
            "The weather is nice today", "Today's weather is pleasant"
        )

        assert "similarity" in similar_result
        assert "model_version" in similar_result
        assert "inference_time_ms" in similar_result
        assert 0.25 < similar_result["similarity"] <= 1.0

    def test_model_info(self, model_manager):
        """Test model info retrieval."""
        # Before loading
        info = model_manager.get_model_info()
        assert info["loaded"] is False

        # After loading
        model_manager.load_model()
        info = model_manager.get_model_info()
        assert info["loaded"] is True
        assert "embedding_dimension" in info
        assert "max_sequence_length" in info
