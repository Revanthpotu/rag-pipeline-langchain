"""Document ingestion: load PDFs, chunk text, embed, and store in ChromaDB.

This module handles the full ingestion pipeline:
1. Extract text from PDF files using PyMuPDF (fitz).
2. Split text into overlapping chunks via LangChain's
   ``RecursiveCharacterTextSplitter``.
3. Embed chunks using a HuggingFace sentence-transformer model.
4. Persist the embeddings in a ChromaDB collection on disk.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import List

import chromadb
import fitz  # PyMuPDF
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

from rag.config import (
    CHROMA_COLLECTION,
    CHROMA_PERSIST_DIR,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    EMBEDDING_MODEL_NAME,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Singleton caches so we don't reload on every Streamlit re-run
# ---------------------------------------------------------------------------
_embedding_model: SentenceTransformer | None = None
_chroma_client: chromadb.PersistentClient | None = None


def get_embedding_model() -> SentenceTransformer:
    """Return a cached ``SentenceTransformer`` instance.

    The model is downloaded on first use and cached locally by the
    ``sentence-transformers`` library.

    Returns:
        SentenceTransformer: The loaded embedding model.
    """
    global _embedding_model  # noqa: PLW0603
    if _embedding_model is None:
        logger.info("Loading embedding model: %s", EMBEDDING_MODEL_NAME)
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _embedding_model


def get_chroma_client() -> chromadb.PersistentClient:
    """Return a cached ChromaDB persistent client.

    The database directory is created automatically if it does not exist.

    Returns:
        chromadb.PersistentClient: ChromaDB client with persistent storage.
    """
    global _chroma_client  # noqa: PLW0603
    if _chroma_client is None:
        Path(CHROMA_PERSIST_DIR).mkdir(parents=True, exist_ok=True)
        logger.info("Initialising ChromaDB at %s", CHROMA_PERSIST_DIR)
        _chroma_client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    return _chroma_client


def get_collection() -> chromadb.Collection:
    """Get (or create) the default ChromaDB collection.

    Returns:
        chromadb.Collection: The ChromaDB collection used by the pipeline.
    """
    client = get_chroma_client()
    return client.get_or_create_collection(
        name=CHROMA_COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )


# ---------------------------------------------------------------------------
# PDF text extraction
# ---------------------------------------------------------------------------

def extract_text_from_pdf(pdf_path: str | Path) -> str:
    """Extract all text from a PDF file using PyMuPDF.

    Args:
        pdf_path: Filesystem path to the ``.pdf`` file.

    Returns:
        The concatenated text of every page, separated by newlines.

    Raises:
        FileNotFoundError: If the file does not exist.
        RuntimeError: If PyMuPDF cannot open the file.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    try:
        doc = fitz.open(str(pdf_path))
    except Exception as exc:
        raise RuntimeError(f"Failed to open PDF '{pdf_path}': {exc}") from exc

    pages: list[str] = []
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text = page.get_text()
        if text.strip():
            pages.append(text)
    doc.close()

    full_text = "\n".join(pages)
    if not full_text.strip():
        logger.warning("No extractable text found in %s", pdf_path)
    return full_text


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> List[str]:
    """Split a long text into overlapping chunks.

    Uses LangChain's ``RecursiveCharacterTextSplitter`` which tries to
    split along paragraph / sentence boundaries first.

    Args:
        text: The source text to split.
        chunk_size: Maximum characters per chunk.
        chunk_overlap: Number of overlapping characters between chunks.

    Returns:
        A list of text chunks.
    """
    if not text or not text.strip():
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_text(text)


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

def embed_texts(texts: List[str]) -> List[List[float]]:
    """Embed a batch of text strings using the sentence-transformer model.

    Args:
        texts: List of text strings to embed.

    Returns:
        A list of embedding vectors (each a list of floats).
    """
    if not texts:
        return []
    model = get_embedding_model()
    embeddings = model.encode(texts, show_progress_bar=False)
    return embeddings.tolist()


# ---------------------------------------------------------------------------
# Ingest a full PDF
# ---------------------------------------------------------------------------

def _doc_hash(text: str) -> str:
    """Return a short SHA-256 hex digest for deduplication."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def ingest_pdf(pdf_path: str | Path, filename: str | None = None) -> int:
    """Run the full ingestion pipeline for a single PDF.

    Steps:
        1. Extract text from the PDF.
        2. Chunk the text.
        3. Embed each chunk.
        4. Store chunks + embeddings in ChromaDB.

    Args:
        pdf_path: Path to the PDF file on disk.
        filename: Optional human-readable filename for metadata.

    Returns:
        The number of chunks ingested.

    Raises:
        ValueError: If the PDF contains no extractable text.
    """
    pdf_path = Path(pdf_path)
    filename = filename or pdf_path.name

    logger.info("Ingesting PDF: %s", filename)

    # 1. Extract
    text = extract_text_from_pdf(pdf_path)
    if not text.strip():
        raise ValueError(f"No text could be extracted from '{filename}'.")

    # 2. Chunk
    chunks = chunk_text(text)
    if not chunks:
        raise ValueError(f"Chunking produced no results for '{filename}'.")

    logger.info("Created %d chunks from %s", len(chunks), filename)

    # 3. Embed
    embeddings = embed_texts(chunks)

    # 4. Store in ChromaDB
    collection = get_collection()
    doc_prefix = _doc_hash(text)

    ids = [f"{doc_prefix}_{i}" for i in range(len(chunks))]
    metadatas = [
        {"source": filename, "chunk_index": i}
        for i in range(len(chunks))
    ]

    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=chunks,
        metadatas=metadatas,
    )

    logger.info("Stored %d chunks in ChromaDB for %s", len(chunks), filename)
    return len(chunks)


def get_ingested_sources() -> List[str]:
    """Return a sorted list of unique source filenames stored in ChromaDB.

    Returns:
        List of filenames that have been ingested.
    """
    try:
        collection = get_collection()
        results = collection.get(include=["metadatas"])
        sources: set[str] = set()
        if results and results.get("metadatas"):
            for meta in results["metadatas"]:
                if meta and "source" in meta:
                    sources.add(meta["source"])
        return sorted(sources)
    except Exception:
        logger.exception("Failed to list ingested sources")
        return []


def clear_collection() -> None:
    """Delete and recreate the ChromaDB collection.

    Useful for resetting the database during development.
    """
    client = get_chroma_client()
    try:
        client.delete_collection(CHROMA_COLLECTION)
        logger.info("Deleted ChromaDB collection: %s", CHROMA_COLLECTION)
    except Exception:
        logger.debug("Collection %s did not exist — nothing to delete", CHROMA_COLLECTION)
    # Recreate empty collection
    get_collection()
