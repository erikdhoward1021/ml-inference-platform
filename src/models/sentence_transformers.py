"""
Sentence Transformer model wrapper for text embeddings.

This module handles model loading, caching, and inference. The design
pattern here ensures that only one model instance is in memory
to avoid duplicating large model weights.
"""

import time
import logging
from typing import List, Dict, Any
import numpy as np
from sentence_transformers import SentenceTransformer

from ..utils.config import settings

# Set up logging
logger = logging.getLogger(__name__)


class ModelManager:
    """
    Manages the sentence transformer model lifecycle.

    This class implements lazy loading - the model is only loaded after
    the first inference request. This makes testing a bit easier, and it allows
    the API to start quickly even if model loading fails initially.
    """

    def __init__(self):
        self._model = None
        self.model_name = settings.MODEL_NAME
        self.cache_dir = settings.MODEL_CACHE_DIR
        self.model_loaded = False
        self.load_time = None

    def load_model(self) -> None:
        """
        Load the model into memory.

        This is separated to allow for explicit control
        over when model loading happens.
        """
        if self._model is not None:
            logger.info("Model already loaded, skipping...")
            return

        model_path = self.cache_dir / self.model_name

        logger.info(f"Loading model from local path: {model_path}")
        start_time = time.time()

        try:
            # Load model with explicit cache directory
            # This ensures models are cached between container restarts; testing to see if that is the case
            self._model = SentenceTransformer(str(model_path), device="cpu")

            self.load_time = time.time() - start_time
            self.model_loaded = True

            logger.info(f"Model loaded successfully in {self.load_time:.2f} seconds")

            # Log model info for debugging
            embedding_dim = self._model.get_sentence_embedding_dimension()
            logger.info(f"Model embedding dimension: {embedding_dim}")

        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise RuntimeError(f"Model loading failed: {str(e)}")

    def predict_single(self, text: str) -> Dict[str, Any]:
        """
        Generate embedding for a single text input.

        Args:
            text: Input text to encode

        Returns:
            Dictionary containing embedding and metadata
        """
        if not self.model_loaded:
            self.load_model()

        start_time = time.time()

        # Generate embedding
        # Convert to list for JSON serialization
        embedding = self._model.encode(text, convert_to_numpy=True)
        embedding_list = embedding.tolist()

        inference_time = time.time() - start_time

        return {
            "embedding": embedding_list,
            "dimension": len(embedding_list),
            "model_version": self.model_name,
            "inference_time_ms": round(inference_time * 1000, 2),
        }

    def predict_batch(self, texts: List[str]) -> Dict[str, Any]:
        """
        Generate embeddings for multiple texts efficiently.

        Batch processing is more efficient than individual predictions
        due to GPU parallelization and reduced overhead.

        Args:
            texts: List of input texts

        Returns:
            Dictionary containing embeddings and metadata
        """
        if not self.model_loaded:
            self.load_model()

        # Validate batch size
        if len(texts) > settings.MAX_BATCH_SIZE:
            raise ValueError(
                f"Batch size {len(texts)} exceeds maximum {settings.MAX_BATCH_SIZE}"
            )

        start_time = time.time()

        embeddings = self._model.encode(texts, convert_to_numpy=True)
        embeddings_list = embeddings.tolist()

        inference_time = time.time() - start_time

        return {
            "embeddings": embeddings_list,
            "batch_size": len(texts),
            "dimension": embeddings.shape[1],
            "model_version": self.model_name,
            "inference_time_ms": round(inference_time * 1000, 2),
            "avg_time_per_item_ms": round((inference_time * 1000) / len(texts), 2),
        }

    def compute_similarity(self, text1: str, text2: str) -> Dict[str, Any]:
        """
        Compute cosine similarity between two texts.

        Args:
            text1: First text
            text2: Second text

        Returns:
            Dictionary containing similarity score and metadata
        """
        if not self.model_loaded:
            self.load_model()

        start_time = time.time()

        # Get embeddings for both texts
        embeddings = self._model.encode([text1, text2], convert_to_numpy=True)

        # Compute cosine similarity
        similarity = np.dot(embeddings[0], embeddings[1]) / (
            np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
        )

        inference_time = time.time() - start_time

        return {
            "similarity": float(similarity),
            "model_version": self.model_name,
            "inference_time_ms": round(inference_time * 1000, 2),
        }

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.
        """
        if not self.model_loaded:
            return {"loaded": False, "model_name": self.model_name}

        return {
            "loaded": True,
            "model_name": self.model_name,
            "embedding_dimension": self._model.get_sentence_embedding_dimension(),
            "load_time_seconds": round(self.load_time, 2) if self.load_time else None,
            "max_sequence_length": self._model.max_seq_length,
        }


model_manager = ModelManager()
