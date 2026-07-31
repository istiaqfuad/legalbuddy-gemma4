"""Ingest structured court judgments (data/cases_json) into Qdrant `legal_cases`.

Mirrors ``ingestion.pipeline`` (the acts ingester) but for the case-law corpus:
same e5 model and ``passage:`` prefix, a separate collection so retrieval can
route/weight precedent differently from statute, and payload indexes geared to
case filtering (type, year, court, disposition, parent-doc uid).

    uv run cases-ingest                       # whole cases_json corpus
    CASES_INGEST_LIMIT=200 uv run cases-ingest # subset (prove value first)
    EMBEDDING_DEVICE=cuda uv run cases-ingest  # embed on the Fedora GPU

Reuses the shared embedding/qdrant helpers so the ingestion (passage) side and
the API (query) side never drift.
"""
import os
import uuid
from pathlib import Path

from dotenv import load_dotenv

from shared import embedding, qdrant
from ingestion.cases_chunking import collect_case_records

ROOT = Path(__file__).resolve().parents[4]
load_dotenv(ROOT / ".env")

QDRANT_URL = os.getenv("QDRANT_VECTORESTORE", "http://213.136.80.53:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY") or None
COLLECTION_NAME = os.getenv("CASES_COLLECTION", "legal_cases")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE", "cpu")
HF_TOKEN = os.getenv("HF_TOKEN") or None
CASES_JSON_DIR = ROOT / "data" / "cases_json"

UPSERT_BATCH_SIZE = 64


def _load_model():
    if not EMBEDDING_MODEL:
        raise ValueError("EMBEDDING_MODEL is required in .env")
    override = os.getenv("EMBEDDING_MAX_TOKENS", "").strip()
    max_tokens = int(override) if override.isdigit() and int(override) > 0 else None
    return embedding.load_embedding_model(EMBEDDING_MODEL, HF_TOKEN, max_tokens, EMBEDDING_DEVICE)


def ingest() -> dict:
    print(f"[cases-ingest] Embedding model: {EMBEDDING_MODEL} (device={EMBEDDING_DEVICE})")
    print(f"[cases-ingest] Qdrant: {QDRANT_URL} | collection: {COLLECTION_NAME}")

    model = _load_model()
    e5 = embedding.is_e5(EMBEDDING_MODEL)

    limit_env = os.getenv("CASES_INGEST_LIMIT", "").strip()
    limit = int(limit_env) if limit_env.isdigit() and int(limit_env) > 0 else None

    # Append/incremental mode: CASES_INGEST_FILES points at a newline list of
    # case_ids (stems). Only those cases are (re-)embedded and upserted into the
    # existing collection -- no recreate -- so e.g. the OCR'd hard tail joins
    # legal_cases without re-embedding the 8k already indexed.
    only_ids = None
    append = False
    files_env = os.getenv("CASES_INGEST_FILES", "").strip()
    if files_env:
        raw = Path(files_env).read_text().splitlines()
        only_ids = {
            (s[:-4] if s.lower().endswith((".pdf", ".json")) else s)
            for s in (ln.strip() for ln in raw) if s
        }
        append = True
        print(f"[cases-ingest] APPEND mode: {len(only_ids)} case_ids from {files_env}")

    print("[cases-ingest] Chunking case records...")
    records, stats = collect_case_records(CASES_JSON_DIR, model, e5, limit=limit, only_ids=only_ids)
    print(f"[cases-ingest] Cases: {stats['files_seen']}, chunks: {len(records)}")
    if not records:
        print("[cases-ingest] No records to ingest.")
        return {"points_indexed": 0, "vector_size": None}

    from qdrant_client.http import models

    client = qdrant.build_client(QDRANT_URL, QDRANT_API_KEY, timeout=300)
    points_indexed = 0
    vector_size = None
    # In append mode the collection already exists; drop any prior points for
    # these cases first so a re-run is idempotent, then upsert (never recreate).
    collection_ready = append
    if append:
        client.delete(
            collection_name=COLLECTION_NAME,
            points_selector=models.FilterSelector(filter=models.Filter(must=[
                models.FieldCondition(key="case_uid",
                                      match=models.MatchAny(any=sorted(only_ids)))
            ])),
        )
        print(f"[cases-ingest] APPEND: cleared prior points for {len(only_ids)} cases")
    total_batches = (len(records) + UPSERT_BATCH_SIZE - 1) // UPSERT_BATCH_SIZE
    print(f"[cases-ingest] Starting: {len(records)} records in {total_batches} batches")

    for batch_num, start in enumerate(range(0, len(records), UPSERT_BATCH_SIZE), start=1):
        batch = records[start : start + UPSERT_BATCH_SIZE]
        vectors = embedding.embed_passages(model, [r["embedding_text"] for r in batch], e5)
        if len(vectors) != len(batch):
            raise ValueError(f"Embedding count mismatch: {len(vectors)} for {len(batch)}")

        if vector_size is None:
            vector_size = len(vectors[0])
        if not collection_ready:
            print(f"[cases-ingest] Recreating '{COLLECTION_NAME}' (vector_size={vector_size})...")
            qdrant.recreate_collection(
                client,
                COLLECTION_NAME,
                vector_size,
                integer_indexes=("case_year",),
                keyword_indexes=("case_type", "court", "disposition", "case_uid"),
            )
            collection_ready = True

        points = [
            models.PointStruct(id=str(uuid.uuid4()), vector=vec, payload=rec["payload"])
            for rec, vec in zip(batch, vectors)
        ]
        qdrant.upsert_with_retry(client, COLLECTION_NAME, points)
        points_indexed += len(points)
        if batch_num % 20 == 0 or batch_num == total_batches:
            print(f"[cases-ingest] Batch {batch_num}/{total_batches} -> {points_indexed} points")

    print("[cases-ingest] Completed.")
    return {"points_indexed": points_indexed, "vector_size": vector_size}


def main() -> None:
    print(ingest())


if __name__ == "__main__":
    main()
