"""Ingest Bangladesh legal acts into Qdrant.

Run from the repo root:

    uv run ingest            # console script
    uv run python -m ingestion

Reads configuration from the project ``.env``:
    QDRANT_VECTORESTORE, QDRANT_API_KEY, QDRANT_COLLECTION,
    EMBEDDING_MODEL, HF_TOKEN, EMBEDDING_MAX_TOKENS (optional),
    INGEST_MAX_RECORDS (optional, for smoke runs)

All embedding / chunking / Qdrant logic lives in the ``shared`` package so the
ingestion (passage) side and the API (query) side never drift apart.
"""
import os
import uuid
from pathlib import Path

from dotenv import load_dotenv

from shared import embedding, qdrant
from shared.chunking import collect_all_records

# apps/ingestion/src/ingestion/pipeline.py -> repo root
ROOT = Path(__file__).resolve().parents[4]
load_dotenv(ROOT / ".env")

QDRANT_URL = os.getenv("QDRANT_VECTORESTORE", "http://213.136.80.53:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY") or None
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "legal_acts_event_rag_full")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
HF_TOKEN = os.getenv("HF_TOKEN") or None
ACTS_DIR = ROOT / "data" / "acts"

UPSERT_BATCH_SIZE = 64


def _load_model():
    if not EMBEDDING_MODEL:
        raise ValueError("EMBEDDING_MODEL is required in .env")
    override = os.getenv("EMBEDDING_MAX_TOKENS", "").strip()
    max_tokens = int(override) if override.isdigit() and int(override) > 0 else None
    return embedding.load_embedding_model(EMBEDDING_MODEL, HF_TOKEN, max_tokens)


def ingest() -> dict:
    print(f"[ingest] Embedding model: {EMBEDDING_MODEL} (CPU)")
    print(f"[ingest] Qdrant: {QDRANT_URL} | collection: {COLLECTION_NAME}")

    model = _load_model()
    e5 = embedding.is_e5(EMBEDDING_MODEL)

    print("[ingest] Collecting records...")
    records, prep_stats = collect_all_records(ACTS_DIR, model, e5)
    print(
        f"[ingest] Files scanned: {prep_stats['files_seen']}, "
        f"chunks prepared: {len(records)}"
    )

    max_records_env = os.getenv("INGEST_MAX_RECORDS", "").strip()
    if max_records_env.isdigit() and int(max_records_env) > 0:
        limit = int(max_records_env)
        print(f"[ingest] Applying INGEST_MAX_RECORDS={limit}")
        records = records[:limit]

    if not records:
        print("[ingest] No records to ingest.")
        return {"points_indexed": 0, "vector_size": None}

    from qdrant_client.http import models

    client = qdrant.build_client(QDRANT_URL, QDRANT_API_KEY, timeout=300)
    points_indexed = 0
    vector_size = None
    collection_ready = False
    total_batches = (len(records) + UPSERT_BATCH_SIZE - 1) // UPSERT_BATCH_SIZE
    print(f"[ingest] Starting: {len(records)} records in {total_batches} batches")

    for batch_num, start in enumerate(range(0, len(records), UPSERT_BATCH_SIZE), start=1):
        batch = records[start : start + UPSERT_BATCH_SIZE]
        vectors = embedding.embed_passages(model, [r["embedding_text"] for r in batch], e5)
        if len(vectors) != len(batch):
            raise ValueError(
                f"Embedding count mismatch: {len(vectors)} vectors for {len(batch)} records"
            )

        if not collection_ready:
            vector_size = len(vectors[0])
            print(f"[ingest] Recreating collection '{COLLECTION_NAME}' (vector_size={vector_size})...")
            qdrant.recreate_collection(
                client,
                COLLECTION_NAME,
                vector_size,
                integer_indexes=("act_year",),
                keyword_indexes=("language", "section_uid"),
            )
            collection_ready = True

        points = [
            models.PointStruct(id=str(uuid.uuid4()), vector=vec, payload=rec["payload"])
            for rec, vec in zip(batch, vectors)
        ]
        qdrant.upsert_with_retry(client, COLLECTION_NAME, points)
        points_indexed += len(points)
        print(f"[ingest] Batch {batch_num}/{total_batches} -> {points_indexed}/{len(records)} points indexed")

    print("[ingest] Completed.")
    return {"points_indexed": points_indexed, "vector_size": vector_size}


def main() -> None:
    print(ingest())


if __name__ == "__main__":
    main()
