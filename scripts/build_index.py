#!/usr/bin/env python3
"""
CLI script to ingest website content and build/update the vector index.

Usage:
    python scripts/build_index.py
    python scripts/build_index.py --from-chunks   # Re-index existing chunks only
    python scripts/build_index.py --max-pages 20 --max-depth 1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.config import Config  # noqa: E402
from app.rag.pipeline import build_index_from_existing_chunks, rebuild_index  # noqa: E402
from app.utils.logging_utils import get_logger  # noqa: E402

logger = get_logger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Ingest website content and build the vector search index."
    )
    parser.add_argument(
        "--from-chunks",
        action="store_true",
        help="Skip crawling; rebuild index from existing data/processed/chunks.json",
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help=f"Override TARGET_WEBSITE_BASE_URL (default: {Config.TARGET_WEBSITE_BASE_URL})",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=None,
        help=f"Max pages to crawl (default: {Config.MAX_PAGES})",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=None,
        help=f"Max crawl depth (default: {Config.MAX_DEPTH})",
    )
    args = parser.parse_args()

    Config.ensure_directories()
    try:
        Config.validate_vector_store_config()
    except ValueError as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        return 1

    logger.info("Vector store type: %s", Config.VECTOR_STORE_TYPE)
    logger.info("Embedding model: %s", Config.EMBEDDING_MODEL_NAME)

    try:
        if args.from_chunks:
            summary = build_index_from_existing_chunks()
        else:
            summary = rebuild_index(
                base_url=args.base_url,
                max_pages=args.max_pages,
                max_depth=args.max_depth,
            )

        print("\n=== Index build complete ===")
        for key, value in summary.items():
            print(f"  {key}: {value}")
        return 0

    except Exception as exc:
        logger.exception("Index build failed")
        print(f"\nERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
