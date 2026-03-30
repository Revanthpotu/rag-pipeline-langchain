"""Unit tests for the retrieval module.

Seeds ChromaDB with sample data, then verifies search and prompt building.
"""

from __future__ import annotations

from pathlib import Path

import fitz
import pytest

from rag.ingestion import clear_collection, ingest_pdf
from rag.retrieval import (
    QueryResult,
    build_context,
    build_prompt,
    retrieve,
    search_similar,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SAMPLE_TEXT = (
    "Machine learning is a subset of artificial intelligence that focuses "
    "on building systems that learn from data. Deep learning is a further "
    "subset that uses neural networks with many layers. Transformers are "
    "a type of deep learning architecture introduced in the 2017 paper "
    "'Attention Is All You Need'. They have become the foundation for "
    "large language models like GPT and BERT.\n\n"
    "Reinforcement learning is another branch of ML where agents learn "
    "to make decisions by interacting with an environment. Q-learning and "
    "policy gradient methods are popular algorithms in this space."
)


def _create_pdf(text: str, path: Path) -> Path:
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), text, fontsize=11)
    doc.save(str(path))
    doc.close()
    return path


@pytest.fixture(autouse=True)
def _seed_collection(tmp_path: Path) -> None:
    """Clear and seed the collection before each test."""
    clear_collection()
    pdf = _create_pdf(SAMPLE_TEXT, tmp_path / "ml.pdf")
    ingest_pdf(pdf, filename="ml.pdf")


# ---------------------------------------------------------------------------
# Tests — similarity search
# ---------------------------------------------------------------------------

class TestSimilaritySearch:
    """Tests for the search_similar function."""

    def test_returns_results(self) -> None:
        """A relevant query should return at least one result."""
        results = search_similar("What is deep learning?", top_k=3)
        assert len(results) > 0

    def test_respects_top_k(self) -> None:
        """Should not return more results than top_k."""
        results = search_similar("transformers", top_k=2)
        assert len(results) <= 2

    def test_result_has_metadata(self) -> None:
        """Each result should contain source and chunk_index."""
        results = search_similar("neural networks", top_k=1)
        assert results[0].source == "ml.pdf"
        assert isinstance(results[0].chunk_index, int)

    def test_empty_query(self) -> None:
        """An empty query should return no results."""
        assert search_similar("") == []

    def test_empty_collection(self, tmp_path: Path) -> None:
        """Search on an empty collection should return an empty list."""
        clear_collection()
        results = search_similar("anything", top_k=5)
        assert results == []


# ---------------------------------------------------------------------------
# Tests — context & prompt building
# ---------------------------------------------------------------------------

class TestContextBuilding:
    """Tests for context and prompt construction."""

    def test_build_context_non_empty(self) -> None:
        """Context built from results should be a non-empty string."""
        results = search_similar("What are transformers?", top_k=2)
        ctx = build_context(results)
        assert isinstance(ctx, str)
        assert len(ctx) > 0

    def test_build_context_empty(self) -> None:
        """Empty results should yield empty context."""
        assert build_context([]) == ""

    def test_build_prompt_contains_question(self) -> None:
        """The prompt should contain the original question."""
        prompt = build_prompt("What is ML?", "Some context here.")
        assert "What is ML?" in prompt
        assert "Some context here." in prompt


# ---------------------------------------------------------------------------
# Tests — full retrieve pipeline
# ---------------------------------------------------------------------------

class TestRetrievePipeline:
    """Tests for the combined retrieve() function."""

    def test_returns_query_result(self) -> None:
        """retrieve() should return a QueryResult dataclass."""
        qr = retrieve("deep learning")
        assert isinstance(qr, QueryResult)
        assert qr.query == "deep learning"
        assert len(qr.results) > 0
        assert len(qr.context) > 0
        assert "QUESTION:" in qr.prompt

    def test_sources_property(self) -> None:
        """The sources property should return deduplicated source info."""
        qr = retrieve("reinforcement learning", top_k=3)
        assert isinstance(qr.sources, list)
        if qr.sources:
            assert "source" in qr.sources[0]
