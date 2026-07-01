"""Embedding model loading and vectorization."""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional

import numpy as np

from app.config import Config
from app.utils.logging_utils import get_logger

logger = get_logger(__name__)

# Keep Hugging Face model cache in project folder (avoids full system drive)
_hf_home = Path(Config.HF_HOME)
os.environ.setdefault("HF_HOME", str(_hf_home))
os.environ.setdefault("HF_HUB_CACHE", str(_hf_home / "hub"))
_hf_home.mkdir(parents=True, exist_ok=True)

_model = None


def get_embedding_model():
    """Load and cache the sentence-transformers model (singleton)."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        model_name = Config.EMBEDDING_MODEL_NAME
        logger.info("Loading embedding model: %s", model_name)
        _model = SentenceTransformer(model_name)
        logger.info("Embedding model loaded (dim=%d)", _model.get_embedding_dimension())
    return _model


def embed_documents(texts: List[str], batch_size: Optional[int] = None) -> np.ndarray:
    """
    Embed a list of document texts.

    Returns a 2D numpy array of shape (n_docs, embedding_dim).
    """
    if not texts:
        return np.array([])

    model = get_embedding_model()
    batch_size = batch_size or Config.EMBEDDING_BATCH_SIZE

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=len(texts) > 50,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return np.asarray(embeddings, dtype=np.float32)


def embed_query(query_text: str) -> np.ndarray:
    """
    Embed a single query string.

    Returns a 1D numpy array of shape (embedding_dim,).
    """
    if not query_text or not query_text.strip():
        raise ValueError("Query text cannot be empty.")

    model = get_embedding_model()
    embedding = model.encode(
        query_text.strip(),
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return np.asarray(embedding, dtype=np.float32)


def get_embedding_dimension() -> int:
    """Return the embedding dimension for the configured model."""
    return get_embedding_model().get_embedding_dimension()
