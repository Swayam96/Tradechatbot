"""Text processing helper functions."""

import re
from typing import List


def normalize_whitespace(text: str) -> str:
    """Collapse repeated whitespace and strip edges."""
    if not text:
        return ""
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def estimate_token_count(text: str) -> int:
    """Rough token estimate (~4 chars per token for English)."""
    if not text:
        return 0
    return max(1, len(text) // 4)


def split_text_by_tokens(
    text: str, chunk_size: int, overlap: int
) -> List[str]:
    """
    Split text into overlapping chunks based on approximate token count.

    Uses word boundaries where possible for cleaner splits.
    """
    if not text:
        return []

    words = text.split()
    if not words:
        return []

    # Convert token targets to approximate word counts
    words_per_chunk = max(1, chunk_size // 4)
    overlap_words = max(0, overlap // 4)

    chunks: List[str] = []
    start = 0
    while start < len(words):
        end = min(start + words_per_chunk, len(words))
        chunk = " ".join(words[start:end])
        if chunk.strip():
            chunks.append(chunk.strip())
        if end >= len(words):
            break
        start = max(start + 1, end - overlap_words)

    return chunks


def truncate_text(text: str, max_chars: int = 500) -> str:
    """Truncate text with ellipsis."""
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."
