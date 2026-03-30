"""Unit tests for the ingestion module.

Uses sample text instead of real PDF files where possible.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import fitz  # PyMuPDF
import pytest

from rag.ingestion import (
    chunk_text,
    clear_collection,
    embed_texts,
    extract_text_from_pdf,
    get_collection,
    ingest_pdf,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _create_sample_pdf(text: str, path: str | Path) -> Path:
    """Create a minimal single-page PDF containing *text*."""
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), text, fontsize=12)
    doc.save(str(path))
    doc.close()
    return Path(path)


SAMPLE_TEXT = (
    "Retrieval Augmented Generation (RAG) is a technique that combines "
    "information retrieval with language model generation. It was introduced "
    "by Lewis et al. in 2020. The key idea is to retrieve relevant documents "
    "from a knowledge base and use them as context for generating answers. "
    "This approach allows language models to access up-to-date information "
    "without retraining.\n\n"
    "RAG systems typically consist of two main components: a retriever and "
    "a generator. The retriever searches a document store (often a vector "
    "database) for passages relevant to the input query. The generator then "
    "conditions on both the query and the retrieved passages to produce a "
    "final answer."
)


# ---------------------------------------------------------------------------
# Tests — chunking
# ---------------------------------------------------------------------------

class TestChunking:
    """Tests for the text chunking function."""

    def test_basic_chunking(self) -> None:
        """Chunking should produce a non-empty list of strings."""
        chunks = chunk_text(SAMPLE_TEXT, chunk_size=200, chunk_overlap=50)
        assert len(chunks) > 0
        assert all(isinstance(c, str) for c in chunks)

    def test_chunk_size_respected(self) -> None:
        """No chunk should exceed the specified size."""
        chunks = chunk_text(SAMPLE_TEXT, chunk_size=200, chunk_overlap=50)
        for c in chunks:
            assert len(c) <= 200 + 50  # small tolerance from splitter

    def test_empty_text_returns_empty(self) -> None:
        """Empty or whitespace-only input should return no chunks."""
        assert chunk_text("") == []
        assert chunk_text("   ") == []

    def test_short_text_single_chunk(self) -> None:
        """Text shorter than chunk_size should yield exactly one chunk."""
        chunks = chunk_text("Hello world.", chunk_size=1000, chunk_overlap=100)
        assert len(chunks) == 1


# ---------------------------------------------------------------------------
# Tests — embedding
# ---------------------------------------------------------------------------

class TestEmbedding:
    """Tests for the embedding function."""

    def test_embed_returns_vectors(self) -> None:
        """Embedding should return a list of float vectors."""
        vectors = embed_texts(["Hello world", "Test sentence"])
        assert len(vectors) == 2
        assert all(isinstance(v, list) for v in vectors)
        assert all(isinstance(f, float) for f in vectors[0])

    def test_embedding_dimension(self) -> None:
        """all-MiniLM-L6-v2 produces 384-dimensional embeddings."""
        vectors = embed_texts(["dimension test"])
        assert len(vectors[0]) == 384

    def test_empty_input(self) -> None:
        """Empty list should return empty list."""
        assert embed_texts([]) == []


# ---------------------------------------------------------------------------
# Tests — PDF extraction
# ---------------------------------------------------------------------------

class TestPDFExtraction:
    """Tests for PDF text extraction."""

    def test_extract_from_sample_pdf(self, tmp_path: Path) -> None:
        """Should extract the text we inserted into a test PDF."""
        pdf_path = tmp_path / "sample.pdf"
        _create_sample_pdf("Hello from a test PDF!", pdf_path)
        text = extract_text_from_pdf(pdf_path)
        assert "Hello from a test PDF" in text

    def test_missing_file_raises(self) -> None:
        """Should raise FileNotFoundError for a missing path."""
        with pytest.raises(FileNotFoundError):
            extract_text_from_pdf("/nonexistent/file.pdf")


# ---------------------------------------------------------------------------
# Tests — ChromaDB storage
# ---------------------------------------------------------------------------

class TestChromaStorage:
    """Tests for ChromaDB ingest and retrieval."""

    def setup_method(self) -> None:
        """Clear collection before each test."""
        clear_collection()

    def test_ingest_and_count(self, tmp_path: Path) -> None:
        """Ingesting a PDF should add chunks to the collection."""
        pdf_path = tmp_path / "test.pdf"
        _create_sample_pdf(SAMPLE_TEXT, pdf_path)
        count = ingest_pdf(pdf_path, filename="test.pdf")
        assert count > 0

        collection = get_collection()
        assert collection.count() == count

    def test_ingest_empty_pdf_raises(self, tmp_path: Path) -> None:
        """A PDF with no text should raise ValueError."""
        doc = fitz.open()
        doc.new_page()  # blank page
        pdf_path = tmp_path / "empty.pdf"
        doc.save(str(pdf_path))
        doc.close()

        with pytest.raises(ValueError, match="No text"):
            ingest_pdf(pdf_path, filename="empty.pdf")
