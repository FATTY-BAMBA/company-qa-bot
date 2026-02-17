"""
Company Q&A Indexer.

Handles the full re-indexing pipeline:
1. Fetch data from Google Sheets
2. Build chunk texts with metadata
3. Generate embeddings via OpenAI
4. Clear and re-upsert to Pinecone namespace
"""

import hashlib
import logging
import time
from typing import List, Dict, Optional

from openai import OpenAI
from pinecone import Pinecone

from app.config import (
    OPENAI_API_KEY,
    EMBEDDING_MODEL,
    PINECONE_API_KEY,
    PINECONE_INDEX_NAME,
    COMPANY_QA_NAMESPACE,
)
from app.sheets_fetcher import fetch_sheet_data

logger = logging.getLogger(__name__)

BATCH_SIZE = 50

# Initialize clients
openai_client = OpenAI(api_key=OPENAI_API_KEY)


def build_chunk_text(record: dict) -> str:
    """
    Combine fields into a single embeddable text block.

    Handles both minimal (question + answer) and enhanced
    (+ category, keywords) structures. Links are stored in
    metadata only, not embedded — they don't add semantic value.
    """
    parts = []

    if record.get("category", "").strip():
        parts.append(f"Category: {record['category'].strip()}")

    parts.append(f"Question: {record['question'].strip()}")
    parts.append(f"Answer: {record['answer'].strip()}")

    if record.get("keywords", "").strip():
        parts.append(f"Keywords: {record['keywords'].strip()}")

    return "\n".join(parts)


def build_metadata(record: dict) -> dict:
    """
    Build Pinecone metadata dict from a sheet record.
    Only includes fields that have values — no empty strings stored.
    """
    metadata = {
        "row_number": record["_row_number"],
        "question": record["question"].strip(),
        "answer": record["answer"].strip(),
        "source": "company-qa-sheet",
    }

    # Optional fields — only include if present
    if record.get("id", "").strip():
        metadata["row_id"] = record["id"].strip()

    if record.get("category", "").strip():
        metadata["category"] = record["category"].strip()

    if record.get("keywords", "").strip():
        metadata["keywords"] = record["keywords"].strip()

    if record.get("link", "").strip():
        metadata["link"] = record["link"].strip()

    return metadata


def build_vector_id(record: dict) -> str:
    """Deterministic vector ID from content hash."""
    identifier = record.get("id", "").strip() or str(record["_row_number"])
    content = f"{identifier}-{record['question'].strip()}"
    return hashlib.md5(content.encode()).hexdigest()


def generate_embeddings(texts: List[str]) -> List[List[float]]:
    """Batch-embed texts using OpenAI."""
    response = openai_client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts,
    )
    return [item.embedding for item in response.data]


def reindex_company_qa(
    spreadsheet_id: str,
    sheet_name: str = "Sheet1",
) -> dict:
    """
    Full re-index pipeline.

    Steps:
    1. Fetch all active records from Google Sheets
    2. Build chunk text + metadata for each record
    3. Generate embeddings in batches
    4. Clear the Pinecone namespace
    5. Upsert all new vectors

    Returns a summary dict with counts and timing.
    """
    start_time = time.time()
    logger.info("=" * 60)
    logger.info("Starting Company Q&A re-indexing...")
    logger.info(f"  Spreadsheet: {spreadsheet_id}")
    logger.info(f"  Sheet: {sheet_name}")
    logger.info(f"  Namespace: {COMPANY_QA_NAMESPACE}")

    # ── Step 1: Fetch data ──
    records = fetch_sheet_data(spreadsheet_id, sheet_name)
    if not records:
        logger.warning("No active records found. Skipping re-index.")
        return {"status": "skipped", "reason": "no_active_records", "record_count": 0}

    # ── Step 2: Build chunks ──
    chunks = []
    for record in records:
        chunks.append({
            "text": build_chunk_text(record),
            "metadata": build_metadata(record),
            "vector_id": build_vector_id(record),
        })

    logger.info(f"Built {len(chunks)} chunks from sheet data")

    # ── Step 3: Generate embeddings ──
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(PINECONE_INDEX_NAME)

    all_vectors = []
    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i : i + BATCH_SIZE]
        texts = [c["text"] for c in batch]

        logger.info(f"Embedding batch {i // BATCH_SIZE + 1} ({len(batch)} chunks)...")
        embeddings = generate_embeddings(texts)

        for chunk, embedding in zip(batch, embeddings):
            all_vectors.append({
                "id": chunk["vector_id"],
                "values": embedding,
                "metadata": chunk["metadata"],
            })

    # ── Step 4: Clear namespace ──
    logger.info(f"Clearing existing '{COMPANY_QA_NAMESPACE}' namespace...")
    try:
        index.delete(delete_all=True, namespace=COMPANY_QA_NAMESPACE)
    except Exception as e:
        logger.warning(f"Namespace clear warning (may be empty): {e}")

    # ── Step 5: Upsert ──
    logger.info(f"Upserting {len(all_vectors)} vectors...")
    for i in range(0, len(all_vectors), BATCH_SIZE):
        batch = all_vectors[i : i + BATCH_SIZE]
        index.upsert(
            vectors=[(v["id"], v["values"], v["metadata"]) for v in batch],
            namespace=COMPANY_QA_NAMESPACE,
        )

    elapsed = round(time.time() - start_time, 2)
    logger.info(f"Re-indexing complete. {len(all_vectors)} vectors in '{COMPANY_QA_NAMESPACE}' ({elapsed}s)")
    logger.info("=" * 60)

    return {
        "status": "success",
        "record_count": len(records),
        "vector_count": len(all_vectors),
        "namespace": COMPANY_QA_NAMESPACE,
        "elapsed_seconds": elapsed,
    }
