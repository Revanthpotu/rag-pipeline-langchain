"""Centralised configuration for the RAG pipeline.

Every setting is loaded from an environment variable (or ``.env`` file)
with a sensible default so the application works out of the box.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env file when present ------------------------------------------------
load_dotenv()

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
CHROMA_PERSIST_DIR: str = os.getenv(
    "CHROMA_PERSIST_DIR",
    str(PROJECT_ROOT / "chroma_db"),
)
UPLOAD_DIR: str = os.getenv(
    "UPLOAD_DIR",
    str(PROJECT_ROOT / "uploaded_pdfs"),
)

# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------
CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "200"))

# ---------------------------------------------------------------------------
# Embedding model (HuggingFace sentence-transformers — free, no API key)
# ---------------------------------------------------------------------------
EMBEDDING_MODEL_NAME: str = os.getenv(
    "EMBEDDING_MODEL_NAME",
    "sentence-transformers/all-MiniLM-L6-v2",
)

# ---------------------------------------------------------------------------
# Ollama LLM
# ---------------------------------------------------------------------------
OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "llama3.2")
OLLAMA_TEMPERATURE: float = float(os.getenv("OLLAMA_TEMPERATURE", "0.3"))

# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------
TOP_K: int = int(os.getenv("TOP_K", "5"))

# ---------------------------------------------------------------------------
# ChromaDB collection name
# ---------------------------------------------------------------------------
CHROMA_COLLECTION: str = os.getenv("CHROMA_COLLECTION", "rag_documents")
