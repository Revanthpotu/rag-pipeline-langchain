"""Retrieval module: embed queries, search ChromaDB, assemble context.

Responsible for the *retrieval* half of RAG:
1. Embed the user query with the same model used during ingestion.
2. Perform a cosine-similarity search in ChromaDB.
3. Assemble a context string and a structured prompt for the LLM.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List

from rag.config import TOP_K
from rag.ingestion import embed_texts, get_collection

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data class for search results
# ---------------------------------------------------------------------------

@dataclass
class RetrievalResult:
    """Container for a single retrieved chunk and its metadata."""

    text: str
    source: str
    chunk_index: int
    distance: float

    def __repr__(self) -> str:  # noqa: D105
        return (
            f"RetrievalResult(source={self.source!r}, "
            f"chunk={self.chunk_index}, dist={self.distance:.4f})"
        )


@dataclass
class QueryResult:
    """Full result of a retrieval query, including assembled context."""

    query: str
    results: List[RetrievalResult] = field(default_factory=list)
    context: str = ""
    prompt: str = ""

    @property
    def sources(self) -> List[Dict[str, str | int]]:
        """Return deduplicated source metadata for display."""
        seen: set[str] = set()
        sources: list[dict[str, str | int]] = []
        for r in self.results:
            if r.source not in seen:
                seen.add(r.source)
                sources.append({"source": r.source, "chunk_index": r.chunk_index})
        return sources


# ---------------------------------------------------------------------------
# Core retrieval functions
# ---------------------------------------------------------------------------

def search_similar(
    query: str,
    top_k: int = TOP_K,
) -> List[RetrievalResult]:
    """Embed *query* and return the *top_k* most similar chunks.

    Args:
        query: The user's natural-language question.
        top_k: Number of results to return.

    Returns:
        A list of ``RetrievalResult`` objects sorted by relevance.
    """
    if not query or not query.strip():
        return []

    collection = get_collection()

    # Check the collection has documents
    if collection.count() == 0:
        logger.warning("ChromaDB collection is empty — upload documents first.")
        return []

    query_embedding = embed_texts([query])[0]

    try:
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, collection.count()),
            include=["documents", "metadatas", "distances"],
        )
    except Exception:
        logger.exception("ChromaDB query failed")
        return []

    retrieval_results: list[RetrievalResult] = []

    if results and results.get("documents"):
        documents = results["documents"][0]
        metadatas = results["metadatas"][0] if results.get("metadatas") else [{}] * len(documents)
        distances = results["distances"][0] if results.get("distances") else [0.0] * len(documents)

        for doc, meta, dist in zip(documents, metadatas, distances):
            retrieval_results.append(
                RetrievalResult(
                    text=doc,
                    source=meta.get("source", "unknown"),
                    chunk_index=int(meta.get("chunk_index", 0)),
                    distance=float(dist),
                )
            )

    return retrieval_results


# ---------------------------------------------------------------------------
# Context assembly & prompt construction
# ---------------------------------------------------------------------------

_SYSTEM_TEMPLATE = (
    "You are a helpful assistant that answers questions based on the "
    "provided context. If the context does not contain enough information "
    "to answer the question, say so honestly — do not make up facts.\n\n"
    "CONTEXT:\n{context}\n\n"
    "QUESTION:\n{question}\n\n"
    "ANSWER:"
)


def build_context(results: List[RetrievalResult]) -> str:
    """Concatenate retrieved chunks into a single context block.

    Each chunk is prefixed with its source filename so the LLM can
    attribute information.

    Args:
        results: Retrieved chunks from ``search_similar``.

    Returns:
        A formatted context string.
    """
    if not results:
        return ""

    parts: list[str] = []
    for i, r in enumerate(results, 1):
        parts.append(f"[Source: {r.source} | Chunk {r.chunk_index}]\n{r.text}")
    return "\n\n---\n\n".join(parts)


def build_prompt(query: str, context: str) -> str:
    """Build the final prompt to send to the LLM.

    Args:
        query: The user's question.
        context: Assembled context from ``build_context``.

    Returns:
        A fully formatted prompt string.
    """
    return _SYSTEM_TEMPLATE.format(context=context, question=query)


def retrieve(query: str, top_k: int = TOP_K) -> QueryResult:
    """Full retrieval pipeline: search → context → prompt.

    This is the main entry-point used by the Streamlit app.

    Args:
        query: The user's question.
        top_k: Number of chunks to retrieve.

    Returns:
        A ``QueryResult`` containing results, context, and the prompt.
    """
    results = search_similar(query, top_k=top_k)
    context = build_context(results)
    prompt = build_prompt(query, context)

    return QueryResult(
        query=query,
        results=results,
        context=context,
        prompt=prompt,
    )
