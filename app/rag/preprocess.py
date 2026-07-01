"""Text cleaning, chunking, and metadata handling."""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional

from app.config import Config, PROCESSED_DATA_DIR
from app.rag.ingestion import PageDocument
from app.utils.logging_utils import get_logger
from app.utils.text_utils import normalize_whitespace, split_text_by_tokens

logger = get_logger(__name__)

# Patterns for boilerplate / ad-like lines
BOILERPLATE_PATTERNS = [
    r"^subscribe\b",
    r"^sign up\b",
    r"^cookie\b",
    r"^advertisement\b",
    r"^related articles\b",
    r"^read more\b",
    r"^share this\b",
    r"^follow us\b",
    r"^\d+ min read\b",
]


@dataclass
class TextChunk:
    """A chunk of text with source metadata."""

    id: str
    text: str
    source_url: str
    title: str
    section: str

    def to_dict(self) -> dict:
        return asdict(self)


def clean_text(text: str) -> str:
    """Remove boilerplate lines and normalize whitespace."""
    if not text:
        return ""

    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or len(stripped) < 20:
            continue
        lower = stripped.lower()
        if any(re.search(pat, lower) for pat in BOILERPLATE_PATTERNS):
            continue
        lines.append(stripped)

    cleaned = "\n".join(lines)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return normalize_whitespace(cleaned)


def extract_section_headers(text: str) -> List[str]:
    """Extract likely section headers (short capitalized lines)."""
    headers = []
    for line in text.split("\n"):
        line = line.strip()
        if 3 < len(line) < 80 and line[0].isupper() and line.count(".") == 0:
            headers.append(line)
    return headers[:5]


def chunk_document(
    page: PageDocument,
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
) -> List[TextChunk]:
    """Split a page document into overlapping chunks with metadata."""
    chunk_size = chunk_size or Config.CHUNK_SIZE
    chunk_overlap = chunk_overlap or Config.CHUNK_OVERLAP

    cleaned = clean_text(page.text)
    if not cleaned:
        return []

    sections = extract_section_headers(page.text)
    default_section = sections[0] if sections else page.title

    raw_chunks = split_text_by_tokens(cleaned, chunk_size, chunk_overlap)
    chunks: List[TextChunk] = []

    for i, chunk_text in enumerate(raw_chunks):
        section = default_section
        for header in sections:
            if header.lower() in chunk_text.lower():
                section = header
                break

        chunks.append(
            TextChunk(
                id=str(uuid.uuid4()),
                text=chunk_text,
                source_url=page.url,
                title=page.title,
                section=section,
            )
        )

    return chunks


def process_documents(
    documents: List[PageDocument],
    output_filename: str = "chunks.json",
) -> List[TextChunk]:
    """
    Clean and chunk all documents, saving results to data/processed/.

    Returns the list of all chunks.
    """
    Config.ensure_directories()
    all_chunks: List[TextChunk] = []

    for doc in documents:
        doc_chunks = chunk_document(doc)
        all_chunks.extend(doc_chunks)
        logger.debug("Chunked %s into %d chunks", doc.url, len(doc_chunks))

    output_path = PROCESSED_DATA_DIR / output_filename
    output_path.write_text(
        json.dumps([c.to_dict() for c in all_chunks], indent=2),
        encoding="utf-8",
    )
    logger.info("Saved %d chunks to %s", len(all_chunks), output_path)
    return all_chunks


def load_chunks(filename: str = "chunks.json") -> List[TextChunk]:
    """Load preprocessed chunks from disk."""
    path = PROCESSED_DATA_DIR / filename
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    return [TextChunk(**item) for item in data]
