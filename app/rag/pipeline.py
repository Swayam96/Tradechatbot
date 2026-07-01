"""RAG pipeline orchestration: query -> retrieve -> generate."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from app.config import Config
from app.rag.embeddings import embed_documents
from app.rag.ingestion import WebsiteIngestion
from app.rag.llm import generate_answer
from app.rag.preprocess import TextChunk, load_chunks, process_documents
from app.rag.retrieval import get_vector_store, retrieve
from app.rag.vector_store import build_vector_store
from app.utils.logging_utils import get_logger

logger = get_logger(__name__)


def extract_sources(chunks: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """Deduplicate sources from retrieved chunks."""
    seen = set()
    sources: List[Dict[str, str]] = []

    for chunk in chunks:
        url = chunk.get("source_url", "")
        title = chunk.get("title", "Source")
        key = url or title
        if key and key not in seen:
            seen.add(key)
            sources.append({"title": title, "url": url})

    return sources


def answer_query(
    query: str,
    top_k: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Full RAG pipeline: retrieve relevant chunks and generate an answer.

    Returns:
        Dict with 'answer' and 'sources' keys.
    """
    if not query or not query.strip():
        return {
            "answer": "Please enter a question about trading or finance.",
            "sources": [],
        }

    try:
        chunks = retrieve(query, top_k=top_k)
        answer = generate_answer(query, chunks)
        sources = extract_sources(chunks)
        return {"answer": answer, "sources": sources}
    except RuntimeError as exc:
        logger.error("Pipeline error: %s", exc)
        return {"answer": str(exc), "sources": [], "error": True}
    except Exception as exc:
        logger.exception("Unexpected pipeline error")
        return {
            "answer": (
                "Sorry, something went wrong while processing your question. "
                "Please try again later."
            ),
            "sources": [],
            "error": True,
        }


def rebuild_index(
    base_url: Optional[str] = None,
    max_pages: Optional[int] = None,
    max_depth: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Run full ingestion pipeline: crawl -> chunk -> embed -> index.

    Returns summary statistics.
    """
    Config.ensure_directories()
    logger.info("Starting index rebuild...")

    ingestion = WebsiteIngestion(
        base_url=base_url,
        max_pages=max_pages,
        max_depth=max_depth,
    )
    documents = ingestion.crawl()

    if not documents:
        raise RuntimeError(
            "No documents were crawled. Check TARGET_WEBSITE_BASE_URL and network access."
        )

    chunks = process_documents(documents)
    if not chunks:
        raise RuntimeError("No text chunks were produced from crawled documents.")

    texts = [c.text for c in chunks]
    embeddings = embed_documents(texts)
    build_vector_store(chunks, embeddings)

    # Force reload cached store
    get_vector_store(force_reload=True)

    summary = {
        "pages_crawled": len(documents),
        "chunks_indexed": len(chunks),
        "vector_store": Config.VECTOR_STORE_TYPE,
        "base_url": ingestion.base_url,
    }
    logger.info("Index rebuild complete: %s", summary)
    return summary


def build_index_from_existing_chunks() -> Dict[str, Any]:
    """Rebuild vector index from previously saved chunks (skip crawl)."""
    chunks = load_chunks()
    if not chunks:
        raise RuntimeError(
            "No processed chunks found. Run full rebuild_index or crawl first."
        )

    texts = [c.text for c in chunks]
    embeddings = embed_documents(texts)
    build_vector_store(chunks, embeddings)
    get_vector_store(force_reload=True)

    return {
        "chunks_indexed": len(chunks),
        "vector_store": Config.VECTOR_STORE_TYPE,
    }
