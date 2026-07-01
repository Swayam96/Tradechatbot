"""Embedding model — uses OpenAI embeddings API to avoid loading large local models into RAM."""

from __future__ import annotations

from typing import List

import numpy as np

from app.config import Config
from app.utils.logging_utils import get_logger

logger = get_logger(__name__)

# OpenAI embedding model — 1536-dim, fast and cheap
_OPENAI_EMBED_MODEL = "text-embedding-3-small"

_openai_client = None


def _get_openai_client():
    """Lazy-load the OpenAI client (singleton)."""
    global _openai_client
    if _openai_client is None:
        from openai import OpenAI
        _openai_client = OpenAI(api_key=Config.API_KEY)
    return _openai_client


def embed_documents(texts: List[str], batch_size: int = 100) -> np.ndarray:
    """
    Embed a list of document texts using OpenAI API.

    Returns a 2D numpy array of shape (n_docs, embedding_dim).
    """
    if not texts:
        return np.array([])

    client = _get_openai_client()
    all_embeddings = []

    # Process in batches to avoid request size limits
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        # Replace newlines — OpenAI recommends this for cleaner embeddings
        batch = [t.replace("\n", " ") for t in batch]
        response = client.embeddings.create(input=batch, model=_OPENAI_EMBED_MODEL)
        batch_embeddings = [item.embedding for item in response.data]
        all_embeddings.extend(batch_embeddings)
        logger.info("Embedded batch %d-%d / %d", i, i + len(batch), len(texts))

    return np.asarray(all_embeddings, dtype=np.float32)


def embed_query(query_text: str) -> np.ndarray:
    """
    Embed a single query string using OpenAI API.

    Returns a 1D numpy array of shape (embedding_dim,).
    """
    if not query_text or not query_text.strip():
        raise ValueError("Query text cannot be empty.")

    client = _get_openai_client()
    response = client.embeddings.create(
        input=[query_text.strip().replace("\n", " ")],
        model=_OPENAI_EMBED_MODEL,
    )
    return np.asarray(response.data[0].embedding, dtype=np.float32)


def get_embedding_dimension() -> int:
    """Return the embedding dimension for text-embedding-3-small."""
    return 1536
