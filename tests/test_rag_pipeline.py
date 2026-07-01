"""Tests for the RAG pipeline and retrieval."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from app.rag.llm import build_context_block, generate_answer
from app.rag.pipeline import answer_query, extract_sources
from app.rag.preprocess import TextChunk
from app.rag.vector_store import FAISSVectorStore


SAMPLE_CHUNKS = [
    {
        "id": "1",
        "text": "A stop-loss order automatically sells a security when it reaches a specified price.",
        "source_url": "https://example.com/stop-loss",
        "title": "Stop-Loss Orders Explained",
        "section": "Risk Management",
        "score": 0.92,
    },
    {
        "id": "2",
        "text": "Diversification spreads investment risk across multiple assets.",
        "source_url": "https://example.com/diversification",
        "title": "Portfolio Diversification",
        "section": "Investing Basics",
        "score": 0.85,
    },
]


class TestExtractSources:
    def test_deduplicates_by_url(self):
        chunks = SAMPLE_CHUNKS + [SAMPLE_CHUNKS[0]]
        sources = extract_sources(chunks)
        assert len(sources) == 2
        urls = {s["url"] for s in sources}
        assert "https://example.com/stop-loss" in urls


class TestBuildContextBlock:
    def test_includes_titles_and_urls(self):
        context = build_context_block(SAMPLE_CHUNKS[:1])
        assert "Stop-Loss Orders Explained" in context
        assert "https://example.com/stop-loss" in context


class TestAnswerQuery:
    def test_empty_query(self):
        result = answer_query("")
        assert "enter a question" in result["answer"].lower()
        assert result["sources"] == []

    @patch("app.rag.pipeline.retrieve")
    @patch("app.rag.pipeline.generate_answer")
    def test_successful_query(self, mock_generate, mock_retrieve):
        mock_retrieve.return_value = SAMPLE_CHUNKS
        mock_generate.return_value = "A stop-loss helps limit downside risk."

        result = answer_query("What is a stop-loss?")
        assert "stop-loss" in result["answer"].lower()
        assert len(result["sources"]) == 2
        mock_retrieve.assert_called_once()
        mock_generate.assert_called_once()

    @patch("app.rag.pipeline.retrieve")
    def test_uninitialized_store_error(self, mock_retrieve):
        mock_retrieve.side_effect = RuntimeError("Vector store is not initialized.")
        result = answer_query("What is forex?")
        assert "not initialized" in result["answer"].lower()
        assert result.get("error") is True


class TestFAISSVectorStore:
    def test_add_and_search(self, tmp_path, monkeypatch):
        index_path = tmp_path / "faiss.index"
        meta_path = tmp_path / "metadata.json"
        monkeypatch.setattr("app.rag.vector_store.Config.FAISS_INDEX_PATH", str(index_path))
        monkeypatch.setattr("app.rag.vector_store.Config.FAISS_METADATA_PATH", str(meta_path))

        store = FAISSVectorStore()
        chunks = [
            TextChunk(
                id="c1",
                text="Options give the right but not obligation to buy or sell.",
                source_url="https://example.com/options",
                title="Options Basics",
                section="Derivatives",
            ),
            TextChunk(
                id="c2",
                text="Bonds are fixed-income securities issued by governments or corporations.",
                source_url="https://example.com/bonds",
                title="Bond Investing",
                section="Fixed Income",
            ),
        ]

        # Use random normalized embeddings for unit test
        rng = np.random.default_rng(42)
        embeddings = rng.standard_normal((2, 384)).astype(np.float32)

        store.add_documents(chunks, embeddings)
        assert store.is_initialized()

        query = embeddings[0]
        results = store.similarity_search(query, top_k=1)
        assert len(results) == 1
        assert results[0]["title"] in ("Options Basics", "Bond Investing")

        store.save()
        loaded = FAISSVectorStore.load()
        assert loaded.is_initialized()
        assert loaded.index.ntotal == 2


class TestGenerateAnswer:
    @patch("app.rag.llm.get_llm_client")
    def test_calls_llm_client(self, mock_get_client):
        mock_client = MagicMock()
        mock_client.generate.return_value = "Test answer about stop-loss orders."
        mock_get_client.return_value = mock_client

        answer = generate_answer("What is a stop-loss?", SAMPLE_CHUNKS[:1], llm_client=mock_client)
        assert "stop-loss" in answer.lower()
        mock_client.generate.assert_called_once()
