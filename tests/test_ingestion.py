"""Tests for ingestion and text preprocessing."""

import json
from unittest.mock import MagicMock, patch

import pytest

from app.rag.ingestion import PageDocument, WebsiteIngestion
from app.rag.preprocess import TextChunk, chunk_document, clean_text, process_documents
from app.utils.text_utils import split_text_by_tokens


SAMPLE_HTML = """
<html>
<head><title>What Is a Stock?</title></head>
<body>
<nav>Navigation menu here with many links</nav>
<article>
<h1>What Is a Stock?</h1>
<p>A stock represents ownership in a corporation and constitutes a claim on part of the company's assets and earnings.</p>
<p>Investors buy stocks to participate in company growth and receive dividends when declared by the board.</p>
<h2>Types of Stocks</h2>
<p>Common stocks give voting rights. Preferred stocks typically pay fixed dividends and have priority over common stock in liquidation.</p>
</article>
<footer>Subscribe to our newsletter for daily updates and cookie policy information.</footer>
</body>
</html>
"""


class TestTextUtils:
    def test_split_text_by_tokens(self):
        text = " ".join(["word"] * 200)
        chunks = split_text_by_tokens(text, chunk_size=100, overlap=20)
        assert len(chunks) >= 2
        assert all(len(c) > 0 for c in chunks)


class TestCleanText:
    def test_removes_boilerplate(self):
        raw = (
            "A stock represents ownership in a company.\n"
            "Subscribe to our newsletter today for free updates.\n"
            "Investors may receive dividends from profitable firms."
        )
        cleaned = clean_text(raw)
        assert "Subscribe" not in cleaned
        assert "ownership" in cleaned


class TestChunkDocument:
    def test_chunks_with_metadata(self):
        page = PageDocument(
            url="https://example.com/stocks",
            title="What Is a Stock?",
            text=" ".join(["Financial markets enable capital formation."] * 80),
        )
        chunks = chunk_document(page, chunk_size=100, chunk_overlap=10)
        assert len(chunks) >= 1
        assert chunks[0].source_url == page.url
        assert chunks[0].title == page.title
        assert chunks[0].id


class TestWebsiteIngestion:
    def test_normalize_url_same_domain(self):
        ingestion = WebsiteIngestion(base_url="https://www.investopedia.com")
        assert (
            ingestion._normalize_url("https://www.investopedia.com/terms/s/stock.asp")
            == "https://www.investopedia.com/terms/s/stock.asp"
        )

    def test_normalize_url_rejects_external(self):
        ingestion = WebsiteIngestion(base_url="https://www.investopedia.com")
        assert ingestion._normalize_url("https://google.com/search") is None

    def test_extract_main_text(self):
        from bs4 import BeautifulSoup

        ingestion = WebsiteIngestion(base_url="https://example.com")
        soup = BeautifulSoup(SAMPLE_HTML, "html.parser")
        text = ingestion._extract_main_text(soup)
        assert "ownership" in text.lower()
        assert "navigation menu" not in text.lower()

    @patch("app.rag.ingestion.requests.Session.get")
    def test_fetch_page(self, mock_get):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {"Content-Type": "text/html"}
        mock_response.text = SAMPLE_HTML
        mock_get.return_value = mock_response

        ingestion = WebsiteIngestion(base_url="https://example.com")
        with patch.object(ingestion, "_can_fetch", return_value=True):
            with patch.object(ingestion, "_save_raw", return_value=("a.html", "a.txt")):
                doc = ingestion.fetch_page("https://example.com/stocks")

        assert doc is not None
        assert doc.title == "What Is a Stock?"
        assert "ownership" in doc.text


class TestProcessDocuments:
    def test_process_documents_writes_chunks(self, tmp_path, monkeypatch):
        monkeypatch.setattr("app.rag.preprocess.PROCESSED_DATA_DIR", tmp_path)

        docs = [
            PageDocument(
                url="https://example.com/a",
                title="Article A",
                text=" ".join(["Trading involves risk management and discipline."] * 50),
            )
        ]
        chunks = process_documents(docs, output_filename="chunks.json")
        assert len(chunks) >= 1

        output = json.loads((tmp_path / "chunks.json").read_text(encoding="utf-8"))
        assert output[0]["source_url"] == "https://example.com/a"
