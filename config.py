"""
Configuration settings for the Document Knowledge Retrieval Tool.
Loads environment variables and provides configuration constants.

============================================================
Created by: Ishan Chakraborty
License: MIT License
Copyright (c) 2024 Ishan Chakraborty
============================================================
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv(override=True)


# Base paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
UPLOADS_DIR = DATA_DIR / "uploads"

# Ensure directories exist
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

# OpenAI Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL_NAME = os.getenv("OPENAI_MODEL_NAME", "gpt-4o-mini")
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large")

# Milvus / Zilliz Cloud Configuration
MILVUS_URI = os.getenv("MILVUS_URI", "")
MILVUS_TOKEN = os.getenv("MILVUS_TOKEN", "")
MILVUS_COLLECTION_NAME = os.getenv("MILVUS_COLLECTION_NAME", "document_knowledge")

# Document Processing
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))

# Embedding dimensions for text-embedding-3-large
EMBEDDING_DIMENSION = 3072

# Supported file types
SUPPORTED_EXTENSIONS = [".pdf", ".txt", ".csv", ".json"]


def validate_config() -> tuple[bool, str]:
    """Validate configuration settings."""
    errors = []
    
    if not OPENAI_API_KEY:
        errors.append("OpenAI API key not set. Please set OPENAI_API_KEY in .env file.")
    
    if not MILVUS_URI:
        errors.append("Milvus URI not set. Please set MILVUS_URI in .env file.")
    
    if not MILVUS_TOKEN:
        errors.append("Milvus Token not set. Please set MILVUS_TOKEN in .env file.")
    
    if errors:
        return False, "\n".join(errors)
    
    return True, "Configuration valid."
