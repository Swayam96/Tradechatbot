#!/usr/bin/env python3
"""
Create the Pinecone serverless index if it does not already exist.

Run once before build_index.py when using VECTOR_STORE_TYPE=pinecone.

Usage:
    python scripts/setup_pinecone.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.config import Config  # noqa: E402
from app.rag.embeddings import get_embedding_dimension  # noqa: E402
from app.utils.logging_utils import get_logger  # noqa: E402

logger = get_logger(__name__)


def main() -> int:
    try:
        Config.validate_vector_store_config()
    except ValueError as exc:
        print(f"\nERROR: {exc}\n", file=sys.stderr)
        print("How to fix:", file=sys.stderr)
        print("  1. Go to https://app.pinecone.io → API keys", file=sys.stderr)
        print("  2. Create / copy a key (starts with pcsk_...)", file=sys.stderr)
        print("  3. Open D:\\trade chat bot\\.env", file=sys.stderr)
        print("  4. Set: PINECONE_API_KEY=pcsk_your_actual_key_here", file=sys.stderr)
        print("  5. Save the file and run this script again.\n", file=sys.stderr)
        return 1

    from pinecone import Pinecone, ServerlessSpec
    from pinecone.errors import UnauthorizedError

    pc = Pinecone(api_key=Config.PINECONE_API_KEY.strip())
    index_name = Config.PINECONE_INDEX_NAME
    dimension = get_embedding_dimension()

    try:
        existing = [idx.name for idx in pc.list_indexes()]
    except UnauthorizedError:
        print("\nERROR: Pinecone rejected your API key (401 Unauthorized).\n", file=sys.stderr)
        print("Check that:", file=sys.stderr)
        print("  • The key in .env is complete (no spaces, no quotes)", file=sys.stderr)
        print("  • You copied the key from the correct Pinecone project", file=sys.stderr)
        print("  • The key was not deleted or rotated in the Pinecone console", file=sys.stderr)
        print("\nGet a new key at: https://app.pinecone.io → API keys\n", file=sys.stderr)
        return 1

    if index_name in existing:
        logger.info("Pinecone index '%s' already exists.", index_name)
        index = pc.Index(index_name)
        logger.info("Index stats: %s", index.describe_index_stats())
        return 0

    logger.info(
        "Creating Pinecone index '%s' (dim=%d, metric=cosine, %s/%s)",
        index_name,
        dimension,
        Config.PINECONE_CLOUD,
        Config.PINECONE_REGION,
    )
    pc.create_index(
        name=index_name,
        dimension=dimension,
        metric="cosine",
        spec=ServerlessSpec(cloud=Config.PINECONE_CLOUD, region=Config.PINECONE_REGION),
    )
    logger.info("Index '%s' created successfully.", index_name)
    print(f"\nPinecone index ready: {index_name} (dimension={dimension})")
    print("Next step: python scripts/build_index.py --from-chunks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
