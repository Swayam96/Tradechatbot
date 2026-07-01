#!/usr/bin/env python3
"""
Recreates the Pinecone index with the correct 1536 dimensions for OpenAI embeddings.
Run this ONCE, then run: python scripts/build_index.py --from-chunks
"""

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.config import Config  # noqa: E402

def main():
    api_key    = Config.PINECONE_API_KEY
    index_name = Config.PINECONE_INDEX_NAME
    dimension  = 1536   # OpenAI text-embedding-3-small

    if not api_key or api_key.startswith("your-"):
        print("ERROR: PINECONE_API_KEY not set in .env")
        return 1

    print(f"Connecting to Pinecone...")
    from pinecone import Pinecone, ServerlessSpec

    pc = Pinecone(api_key=api_key)

    # Delete existing index if it exists
    existing = [i.name for i in pc.list_indexes()]
    if index_name in existing:
        current_dim = pc.describe_index(index_name).dimension
        print(f"Found index '{index_name}' with dimension={current_dim}")
        if current_dim == dimension:
            print(f"Index already has correct dimension ({dimension}). Nothing to do!")
            return 0
        print(f"Deleting index '{index_name}' (wrong dimension: {current_dim})...")
        pc.delete_index(index_name)
        print("Deleted. Waiting 5 seconds...")
        time.sleep(5)
    else:
        print(f"Index '{index_name}' not found. Will create fresh.")

    # Create new index with correct dimensions
    print(f"Creating index '{index_name}' with dimension={dimension}, metric=cosine...")
    pc.create_index(
        name=index_name,
        dimension=dimension,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )

    # Wait until ready
    print("Waiting for index to become ready", end="", flush=True)
    for _ in range(30):
        status = pc.describe_index(index_name).status.ready
        if status:
            break
        print(".", end="", flush=True)
        time.sleep(2)
    print()

    final_dim = pc.describe_index(index_name).dimension
    print(f"\n✅ Index '{index_name}' is READY with dimension={final_dim}")
    print("\nNow run:  python scripts/build_index.py --from-chunks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
