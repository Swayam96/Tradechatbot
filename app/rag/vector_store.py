"""FAISS and Pinecone vector store backends with a unified interface."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from app.config import Config
from app.rag.preprocess import TextChunk
from app.utils.logging_utils import get_logger

logger = get_logger(__name__)


class VectorStore(ABC):
    """Abstract vector store interface."""

    @abstractmethod
    def add_documents(
        self, documents: List[TextChunk], embeddings: np.ndarray
    ) -> None:
        """Add documents and their embeddings to the store."""

    @abstractmethod
    def similarity_search(
        self, query_embedding: np.ndarray, top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """Return top-k most similar documents with scores."""

    @abstractmethod
    def is_initialized(self) -> bool:
        """Return True if the store contains indexed documents."""

    @abstractmethod
    def save(self) -> None:
        """Persist the store to disk or remote service."""

    def clear(self) -> None:
        """Remove all vectors (override in backends that support it)."""
        pass

    @classmethod
    @abstractmethod
    def load(cls) -> "VectorStore":
        """Load an existing store."""


class FAISSVectorStore(VectorStore):
    """Local FAISS index with JSON metadata sidecar."""

    def __init__(self):
        self.index = None
        self.metadata: List[Dict[str, Any]] = []
        self.index_path = Path(Config.FAISS_INDEX_PATH)
        self.metadata_path = Path(Config.FAISS_METADATA_PATH)

    def add_documents(
        self, documents: List[TextChunk], embeddings: np.ndarray
    ) -> None:
        import faiss

        if len(documents) != len(embeddings):
            raise ValueError("Number of documents must match number of embeddings.")

        if len(documents) == 0:
            return

        dim = embeddings.shape[1]
        if self.index is None:
            self.index = faiss.IndexFlatIP(dim)

        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)
        self.metadata.extend([doc.to_dict() for doc in documents])
        logger.info("Added %d documents to FAISS index (total=%d)", len(documents), self.index.ntotal)

    def similarity_search(
        self, query_embedding: np.ndarray, top_k: int = 5
    ) -> List[Dict[str, Any]]:
        import faiss

        if not self.is_initialized():
            raise RuntimeError(
                "Vector store is not initialized. Run 'python scripts/build_index.py' first."
            )

        query = np.asarray(query_embedding, dtype=np.float32).reshape(1, -1)
        faiss.normalize_L2(query)
        k = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(query, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            meta = dict(self.metadata[idx])
            meta["score"] = float(score)
            results.append(meta)
        return results

    def is_initialized(self) -> bool:
        return self.index is not None and self.index.ntotal > 0

    def clear(self) -> None:
        import faiss

        if self.index is not None:
            dim = self.index.d
            self.index = faiss.IndexFlatIP(dim)
        self.metadata = []

    def save(self) -> None:
        import faiss

        if self.index is None:
            raise RuntimeError("Cannot save an empty FAISS index.")

        Config.ensure_directories()
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(self.index_path))
        self.metadata_path.write_text(
            json.dumps(self.metadata, indent=2), encoding="utf-8"
        )
        logger.info("Saved FAISS index to %s", self.index_path)

    @classmethod
    def load(cls) -> "FAISSVectorStore":
        import faiss

        store = cls()
        if not store.index_path.exists() or not store.metadata_path.exists():
            logger.warning("FAISS index files not found at %s", store.index_path.parent)
            return store

        store.index = faiss.read_index(str(store.index_path))
        store.metadata = json.loads(store.metadata_path.read_text(encoding="utf-8"))
        logger.info("Loaded FAISS index with %d vectors", store.index.ntotal)
        return store


class PineconeVectorStore(VectorStore):
    """Pinecone cloud vector store backend."""

    def __init__(self):
        from pinecone import Pinecone

        self.pc = Pinecone(api_key=Config.PINECONE_API_KEY)
        self.index_name = Config.PINECONE_INDEX_NAME
        self.namespace = Config.PINECONE_NAMESPACE
        self._index = None
        self._doc_count = 0

    def _get_index(self):
        if self._index is None:
            self._index = self.pc.Index(self.index_name)
        return self._index

    def add_documents(
        self, documents: List[TextChunk], embeddings: np.ndarray
    ) -> None:
        index = self._get_index()
        vectors = []
        for doc, emb in zip(documents, embeddings):
            vectors.append(
                {
                    "id": doc.id,
                    "values": emb.tolist(),
                    "metadata": {
                        "text": doc.text[:8000],
                        "source_url": doc.source_url,
                        "title": doc.title,
                        "section": doc.section,
                    },
                }
            )

        batch_size = 100
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i : i + batch_size]
            index.upsert(vectors=batch, namespace=self.namespace)

        self._doc_count += len(documents)
        logger.info("Upserted %d vectors to Pinecone", len(documents))

    def similarity_search(
        self, query_embedding: np.ndarray, top_k: int = 5
    ) -> List[Dict[str, Any]]:
        if not self.is_initialized():
            raise RuntimeError(
                "Pinecone index appears empty or unavailable. Run build_index first."
            )

        index = self._get_index()
        response = index.query(
            vector=query_embedding.tolist(),
            top_k=top_k,
            include_metadata=True,
            namespace=self.namespace,
        )

        results = []
        for match in response.get("matches", []):
            meta = match.get("metadata", {})
            results.append(
                {
                    "id": match.get("id"),
                    "text": meta.get("text", ""),
                    "source_url": meta.get("source_url", ""),
                    "title": meta.get("title", ""),
                    "section": meta.get("section", ""),
                    "score": float(match.get("score", 0)),
                }
            )
        return results

    def is_initialized(self) -> bool:
        try:
            return self._vector_count() > 0
        except Exception as exc:
            logger.warning("Pinecone initialization check failed: %s", exc)
            return False

    def _vector_count(self) -> int:
        index = self._get_index()
        stats = index.describe_index_stats()
        namespaces = stats.get("namespaces", {})
        ns_stats = namespaces.get(self.namespace, {})
        count = ns_stats.get("vector_count", 0)
        if count == 0 and self.namespace == "default":
            count = namespaces.get("", {}).get("vector_count", 0)
        if count == 0:
            count = stats.get("total_vector_count", 0)
        return int(count)

    def clear(self) -> None:
        from pinecone.errors import NotFoundError

        if self._vector_count() == 0:
            logger.info(
                "Pinecone namespace '%s' is empty; skipping delete.", self.namespace
            )
            self._doc_count = 0
            return

        index = self._get_index()
        try:
            index.delete(delete_all=True, namespace=self.namespace)
            logger.info("Cleared Pinecone namespace '%s'", self.namespace)
        except NotFoundError:
            logger.info(
                "Pinecone namespace '%s' not found; nothing to clear.", self.namespace
            )
        self._doc_count = 0

    def save(self) -> None:
        # Pinecone persists automatically on upsert
        logger.info("Pinecone vectors persisted remotely.")

    @classmethod
    def load(cls) -> "PineconeVectorStore":
        store = cls()
        store._get_index()
        return store


def create_vector_store(store_type: Optional[str] = None) -> VectorStore:
    """Factory function to create the configured vector store backend."""
    store_type = (store_type or Config.VECTOR_STORE_TYPE).lower()
    Config.validate_vector_store_config()

    if store_type == "faiss":
        return FAISSVectorStore.load()
    if store_type == "pinecone":
        return PineconeVectorStore.load()
    raise ValueError(f"Unsupported VECTOR_STORE_TYPE: {store_type}")


def build_vector_store(
    documents: List[TextChunk], embeddings: np.ndarray, store_type: Optional[str] = None
) -> VectorStore:
    """Create a new vector store, populate it, and persist."""
    store_type = (store_type or Config.VECTOR_STORE_TYPE).lower()
    Config.validate_vector_store_config()

    if store_type == "faiss":
        store: VectorStore = FAISSVectorStore()
    elif store_type == "pinecone":
        store = PineconeVectorStore()
        store.clear()
    else:
        raise ValueError(f"Unsupported VECTOR_STORE_TYPE: {store_type}")

    store.add_documents(documents, embeddings)
    store.save()
    return store
