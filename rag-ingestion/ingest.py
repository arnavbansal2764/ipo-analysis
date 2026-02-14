#!/usr/bin/env python3
"""
Ingest IPO markdown files into a Qdrant vector database for RAG.

Reads .md files from the data-pipeline output (ipo_data_md/),
chunks them by markdown sections, embeds with a sentence-transformer,
and upserts into Qdrant.

Usage:
    # Start Qdrant first
    docker compose up -d

    # Ingest all IPOs
    python ingest.py

    # Ingest from a custom path
    python ingest.py --input ../ipo_data_md

    # Re-ingest (recreate collection)
    python ingest.py --recreate

    # Query test
    python ingest.py --query "What is the subscription status of Shadowfax IPO?"
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import uuid
from pathlib import Path

import yaml
from tqdm import tqdm
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer


# ── Configuration ────────────────────────────────────────────

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "ipo_documents")
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))       # max chars per chunk
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))   # overlap chars
DEFAULT_INPUT = os.path.join(os.path.dirname(__file__), "..", "ipo_data_md")


# ── Markdown Parsing ─────────────────────────────────────────

_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)


def parse_frontmatter(text: str) -> tuple[dict, str]:
    """Extract YAML frontmatter and return (metadata_dict, body)."""
    m = _FRONTMATTER_RE.match(text)
    if m:
        try:
            meta = yaml.safe_load(m.group(1)) or {}
        except yaml.YAMLError:
            meta = {}
        body = text[m.end():]
        return meta, body
    return {}, text


def chunk_markdown_by_sections(text: str, max_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split markdown into chunks at section boundaries (## headers).

    If a section exceeds max_size, it is further split on paragraph boundaries.
    Adjacent chunks overlap by `overlap` characters for context continuity.
    """
    # Split on markdown headers (## or ###)
    section_pattern = re.compile(r"(?=^#{1,3}\s)", re.MULTILINE)
    raw_sections = section_pattern.split(text)
    raw_sections = [s.strip() for s in raw_sections if s.strip()]

    chunks: list[str] = []
    for section in raw_sections:
        if len(section) <= max_size:
            chunks.append(section)
        else:
            # Split large sections on double-newlines (paragraph boundaries)
            paragraphs = re.split(r"\n{2,}", section)
            current = ""
            for para in paragraphs:
                if current and len(current) + len(para) + 2 > max_size:
                    chunks.append(current.strip())
                    # Keep overlap from end of current chunk
                    current = current[-overlap:] + "\n\n" + para if overlap else para
                else:
                    current = current + "\n\n" + para if current else para
            if current.strip():
                chunks.append(current.strip())

    return chunks


# ── File Loading ─────────────────────────────────────────────

def load_ipo_documents(input_dir: str) -> list[dict]:
    """Walk input_dir and load all .md files with their metadata.

    Returns a list of dicts:
        { "text": str, "metadata": dict, "source_file": str }
    """
    input_path = Path(input_dir)
    if not input_path.is_dir():
        print(f"[ERR] Input directory not found: {input_dir}")
        sys.exit(1)

    documents = []

    for ipo_folder in sorted(input_path.iterdir()):
        if not ipo_folder.is_dir():
            continue

        folder_name = ipo_folder.name

        # Load companion .metadata.json if available (for URL metadata)
        url_meta = {}
        for f in ipo_folder.iterdir():
            if f.name.endswith(".metadata.json"):
                try:
                    url_meta = json.loads(f.read_text(encoding="utf-8"))
                except Exception:
                    pass
                break

        # Process metadata markdown (the main .md file, not in rhp_docs)
        for md_file in sorted(ipo_folder.glob("*.md")):
            text = md_file.read_text(encoding="utf-8")
            frontmatter, body = parse_frontmatter(text)

            doc_meta = {
                "ipo_folder": folder_name,
                "source_file": str(md_file.relative_to(input_path)),
                "doc_type": frontmatter.get("doc_type", "ipo_metadata"),
                "ipo_id": frontmatter.get("ipo_id", url_meta.get("id", "")),
                "company_name": frontmatter.get("company_name", url_meta.get("name", "")),
                "open_date": frontmatter.get("open_date", ""),
                "close_date": frontmatter.get("close_date", ""),
                "ipo_period": frontmatter.get("ipo_period", ""),
                "is_open": frontmatter.get("is_open", False),
            }

            # Add URL info
            if url_meta.get("urls"):
                doc_meta["rhp_link"] = url_meta["urls"].get("rhp_link", "")
                doc_meta["gmp_link"] = url_meta["urls"].get("gmp_link", "")

            documents.append({
                "text": body,
                "metadata": doc_meta,
                "source_file": str(md_file),
            })

        # Process RHP markdown files
        rhp_dir = ipo_folder / "rhp_docs"
        if rhp_dir.is_dir():
            for md_file in sorted(rhp_dir.glob("*.md")):
                text = md_file.read_text(encoding="utf-8")
                frontmatter, body = parse_frontmatter(text)

                doc_meta = {
                    "ipo_folder": folder_name,
                    "source_file": str(md_file.relative_to(input_path)),
                    "doc_type": "rhp_document",
                    "ipo_id": url_meta.get("id", ""),
                    "company_name": url_meta.get("name", ""),
                    "rhp_filename": md_file.stem,
                }

                documents.append({
                    "text": body,
                    "metadata": doc_meta,
                    "source_file": str(md_file),
                })

    return documents


# ── Qdrant Operations ────────────────────────────────────────

def init_qdrant(client: QdrantClient, embedding_dim: int, recreate: bool = False):
    """Create (or recreate) the Qdrant collection."""
    exists = client.collection_exists(COLLECTION_NAME)

    if exists and recreate:
        print(f"  Deleting existing collection: {COLLECTION_NAME}")
        client.delete_collection(COLLECTION_NAME)
        exists = False

    if not exists:
        print(f"  Creating collection: {COLLECTION_NAME} (dim={embedding_dim})")
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(
                size=embedding_dim,
                distance=models.Distance.COSINE,
            ),
        )
        # Create payload indexes for common filter fields
        for field in ["ipo_id", "company_name", "doc_type", "ipo_folder", "is_open"]:
            try:
                client.create_payload_index(
                    collection_name=COLLECTION_NAME,
                    field_name=field,
                    field_schema=models.PayloadSchemaType.KEYWORD
                    if field != "is_open"
                    else models.PayloadSchemaType.BOOL,
                )
            except Exception:
                pass
    else:
        print(f"  Collection exists: {COLLECTION_NAME}")


def ingest_documents(
    client: QdrantClient,
    model: SentenceTransformer,
    documents: list[dict],
    batch_size: int = 64,
):
    """Chunk, embed, and upsert documents into Qdrant."""
    all_points: list[models.PointStruct] = []

    print(f"\n  Chunking {len(documents)} documents...")
    for doc in tqdm(documents, desc="  Chunking"):
        chunks = chunk_markdown_by_sections(doc["text"])
        for i, chunk in enumerate(chunks):
            if not chunk.strip():
                continue
            point_meta = dict(doc["metadata"])
            point_meta["chunk_index"] = i
            point_meta["chunk_total"] = len(chunks)
            point_meta["text"] = chunk  # store text in payload for retrieval
            all_points.append(
                models.PointStruct(
                    id=str(uuid.uuid4()),
                    vector=[],  # placeholder, filled below
                    payload=point_meta,
                )
            )

    total_chunks = len(all_points)
    print(f"  Total chunks to embed: {total_chunks}")

    # Embed in batches
    texts = [p.payload["text"] for p in all_points]
    print(f"  Embedding with {EMBED_MODEL}...")

    embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="  Embedding"):
        batch = texts[i : i + batch_size]
        batch_emb = model.encode(batch, normalize_embeddings=True, show_progress_bar=False)
        embeddings.extend(batch_emb.tolist())

    for point, emb in zip(all_points, embeddings):
        point.vector = emb

    # Upsert in batches
    print(f"  Upserting {total_chunks} points into Qdrant...")
    for i in tqdm(range(0, len(all_points), batch_size), desc="  Upserting"):
        batch = all_points[i : i + batch_size]
        client.upsert(collection_name=COLLECTION_NAME, points=batch)

    print(f"  Done — {total_chunks} chunks ingested into '{COLLECTION_NAME}'")


# ── Query (test) ─────────────────────────────────────────────

def query_rag(client: QdrantClient, model: SentenceTransformer, query: str, top_k: int = 5):
    """Run a similarity search and print results."""
    print(f"\n  Query: {query}")
    print(f"  {'='*60}")

    query_vec = model.encode(query, normalize_embeddings=True).tolist()

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vec,
        limit=top_k,
        with_payload=True,
    )

    for i, point in enumerate(results.points, 1):
        payload = point.payload
        score = point.score
        company = payload.get("company_name", "?")
        doc_type = payload.get("doc_type", "?")
        source = payload.get("source_file", "?")
        text_preview = payload.get("text", "")[:300]

        print(f"\n  [{i}] Score: {score:.4f}  |  {company}  |  {doc_type}")
        print(f"      Source: {source}")
        print(f"      Text:   {text_preview}...")


# ── Main ─────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Ingest IPO markdowns into Qdrant for RAG")
    parser.add_argument(
        "--input", type=str, default=DEFAULT_INPUT,
        help=f"Path to ipo_data_md directory (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--recreate", action="store_true",
        help="Drop and recreate the Qdrant collection before ingesting",
    )
    parser.add_argument(
        "--query", type=str, default=None,
        help="Run a test query instead of ingesting",
    )
    parser.add_argument(
        "--qdrant-url", type=str, default=QDRANT_URL,
        help=f"Qdrant server URL (default: {QDRANT_URL})",
    )
    parser.add_argument(
        "--model", type=str, default=EMBED_MODEL,
        help=f"Sentence-transformer model name (default: {EMBED_MODEL})",
    )
    parser.add_argument(
        "--top-k", type=int, default=5,
        help="Number of results for --query (default: 5)",
    )
    args = parser.parse_args()

    # Init model
    print(f"Loading embedding model: {args.model}")
    model = SentenceTransformer(args.model)
    embed_dim = model.get_sentence_embedding_dimension()

    # Init Qdrant client
    print(f"Connecting to Qdrant: {args.qdrant_url}")
    client = QdrantClient(url=args.qdrant_url)

    if args.query:
        query_rag(client, model, args.query, top_k=args.top_k)
        return

    # Ingest flow
    init_qdrant(client, embed_dim, recreate=args.recreate)

    print(f"\nLoading documents from: {args.input}")
    documents = load_ipo_documents(args.input)
    print(f"  Loaded {len(documents)} documents")

    if not documents:
        print("  No documents found. Check the input path.")
        return

    ingest_documents(client, model, documents)

    # Print collection info
    info = client.get_collection(COLLECTION_NAME)
    print(f"\n  Collection '{COLLECTION_NAME}': {info.points_count} points")


if __name__ == "__main__":
    main()