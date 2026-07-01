"""Top-k retrieval from the vector store."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from app.config import Config
from app.rag.embeddings import embed_query
from app.rag.vector_store import VectorStore, create_vector_store
from app.utils.logging_utils import get_logger

logger = get_logger(__name__)

_store: Optional[VectorStore] = None


def get_vector_store(force_reload: bool = False) -> VectorStore:
    """Return a cached vector store instance."""
    global _store
    if _store is None or force_reload:
        _store = create_vector_store()
    return _store


def retrieve(
    query: str,
    top_k: Optional[int] = None,
    store: Optional[VectorStore] = None,
) -> List[Dict[str, Any]]:
    """
    Retrieve the most relevant document chunks for a query.

    Args:
        query: User question or search text.
        top_k: Number of chunks to return (defaults to Config.TOP_K).
        store: Optional vector store override.

    Returns:
        List of chunk dicts with text, metadata, and similarity score.
    """
    if not query or not query.strip():
        return []

    top_k = top_k or Config.TOP_K
    store = store or get_vector_store()

    if not store.is_initialized():
        raise RuntimeError(
            "Vector store is not initialized. "
            "Run 'python scripts/build_index.py' to ingest and index content."
        )

    query_embedding = embed_query(query)
    results = store.similarity_search(query_embedding, top_k=top_k)
    logger.debug("Retrieved %d chunks for query: %s", len(results), query[:80])
    return results
