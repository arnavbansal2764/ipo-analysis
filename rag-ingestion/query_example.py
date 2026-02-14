#!/usr/bin/env python3
"""Quick utility to query the IPO RAG from the command line."""

from __future__ import annotations

import argparse
import os

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer


QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "ipo_documents")
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")


def main():
    parser = argparse.ArgumentParser(description="Query the IPO RAG")
    parser.add_argument("query", type=str, help="Your question")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--ipo", type=str, default=None, help="Filter by IPO folder name")
    parser.add_argument("--doc-type", type=str, default=None, choices=["ipo_metadata", "rhp_document"])
    args = parser.parse_args()

    model = SentenceTransformer(EMBED_MODEL)
    client = QdrantClient(url=QDRANT_URL)

    query_vec = model.encode(args.query, normalize_embeddings=True).tolist()

    # Build optional filters
    filter_conditions = []
    if args.ipo:
        from qdrant_client.models import FieldCondition, MatchValue
        filter_conditions.append(
            FieldCondition(key="ipo_folder", match=MatchValue(value=args.ipo))
        )
    if args.doc_type:
        from qdrant_client.models import FieldCondition, MatchValue
        filter_conditions.append(
            FieldCondition(key="doc_type", match=MatchValue(value=args.doc_type))
        )

    query_filter = None
    if filter_conditions:
        from qdrant_client.models import Filter
        query_filter = Filter(must=filter_conditions)

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vec,
        query_filter=query_filter,
        limit=args.top_k,
        with_payload=True,
    )

    if not results.points:
        print("No results found.")
        return

    for i, point in enumerate(results.points, 1):
        p = point.payload
        print(f"\n{'='*70}")
        print(f"[{i}] Score: {point.score:.4f}")
        print(f"    Company:  {p.get('company_name', '?')}")
        print(f"    Type:     {p.get('doc_type', '?')}")
        print(f"    Source:   {p.get('source_file', '?')}")
        print(f"    Chunk:    {p.get('chunk_index', '?')}/{p.get('chunk_total', '?')}")
        print(f"{'='*70}")
        print(p.get("text", "")[:500])


if __name__ == "__main__":
    main()