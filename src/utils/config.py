"""
Configuration management for the ML inference service.

This module makes it easy to adjust settings for different environments (dev, staging, prod).
"""

import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Settings:
    """Application settings loaded from environment variables."""

    # API Configuration
    APP_NAME: str = os.getenv("APP_NAME", "ML Inference Platform")
    APP_VERSION: str = os.getenv("APP_VERSION", "1.0.0")
    API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
    API_PORT: int = int(os.getenv("API_PORT", "8000"))

    # Model Configuration
    MODEL_NAME: str = os.getenv("MODEL_NAME", "all-MiniLM-L6-v2")
    MODEL_CACHE_DIR: str = os.getenv("MODEL_CACHE_DIR", "/tmp/models")

    # Performance Settings
    MAX_BATCH_SIZE: int = int(os.getenv("MAX_BATCH_SIZE", "32"))
    REQUEST_TIMEOUT: int = int(os.getenv("REQUEST_TIMEOUT", "30"))

    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    @classmethod
    def get_model_path(cls) -> Path:
        """Get the full path where models are cached."""
        return Path(cls.MODEL_CACHE_DIR) / cls.MODEL_NAME.replace("/", "_")


# Create a single instance to import throughout the application
settings = Settings()
