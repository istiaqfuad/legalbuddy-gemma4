"""Ingest the eval subset into a named Qdrant collection using a chosen chunker.

    PYTHONPATH=eval uv run python eval/ingest_eval.py --mode baseline
    PYTHONPATH=eval uv run python eval/ingest_eval.py --mode improved

Both modes write the same chunking-independent identity tags into each point's
payload (act_file, section_ord, gold_key) so the gold set scores either collection.
"""
import argparse
import json
import uuid

from common import (
    ACTS_DIR,
    COLLECTION_BASELINE,
    COLLECTION_IMPROVED,
    SUBSET_PATH,
    build_qdrant_client,
    get_model,
    gold_key,
    passage_text,
)

ENCODE_BATCH = 32
UPSERT_BATCH = 64


def collect_records(mode: str):
    subset = json.loads(SUBSET_PATH.read_text(encoding="utf-8"))
    model = get_model()

    if mode == "baseline":
        from baseline_chunk import section_records
    else:
        # Improved chunker is the shared production chunker.
        from common import EMBEDDING_MODEL
        from shared.chunking import section_records as _shared_records
        from shared.embedding import is_e5

        e5 = is_e5(EMBEDDING_MODEL)

        def section_records(act_obj, section, model):
            return _shared_records(act_obj, section, model, e5)

    records = []
    stats = {"acts": 0, "sections": 0, "chunks": 0, "skipped_sections": 0}
    for stem in subset:
        fp = ACTS_DIR / f"{stem}.json"
        try:
            act_obj = json.loads(fp.read_text(encoding="utf-8"))
        except Exception:
            continue
        stats["acts"] += 1
        for section_ord, section in enumerate(act_obj.get("sections") or []):
            stats["sections"] += 1
            recs = section_records(act_obj, section, model)
            if not recs:
                stats["skipped_sections"] += 1
                continue
            n_parts = len(recs)
            for rec in recs:
                payload = {
                    "act_file": stem,
                    "section_ord": section_ord,
                    "gold_key": gold_key(stem, section_ord),
                    "n_parts": n_parts,
                    "act_title": act_obj.get("act_title"),
                    "act_no": act_obj.get("act_no"),
                    "act_year": (
                        int(act_obj["act_year"])
                        if str(act_obj.get("act_year", "")).isdigit()
                        else None
                    ),
                    "language": act_obj.get("language"),
                    "source_url": act_obj.get("source_url"),
                    **rec["payload_extra"],
                }
                records.append({"embedding_text": rec["embedding_text"], "payload": payload})
    stats["chunks"] = len(records)
    return records, stats


def embed_passages(model, texts):
    prefixed = [passage_text(t) for t in texts]
    vecs = model.encode(
        prefixed, batch_size=ENCODE_BATCH, normalize_embeddings=True, show_progress_bar=False
    )
    return [v.tolist() for v in vecs]


def recreate_collection(client, name, vector_size):
    from qdrant_client.http import models

    existing = {c.name for c in client.get_collections().collections}
    if name in existing:
        print(f"[ingest] dropping existing collection '{name}'")
        client.delete_collection(collection_name=name)
    client.create_collection(
        collection_name=name,
        vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE),
    )
    client.create_payload_index(
        collection_name=name, field_name="gold_key",
        field_schema=models.PayloadSchemaType.KEYWORD, wait=True,
    )
    client.create_payload_index(
        collection_name=name, field_name="act_file",
        field_schema=models.PayloadSchemaType.KEYWORD, wait=True,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["baseline", "improved"], required=True)
    ap.add_argument("--collection", default=None)
    args = ap.parse_args()
    collection = args.collection or (
        COLLECTION_BASELINE if args.mode == "baseline" else COLLECTION_IMPROVED
    )

    from qdrant_client.http import models

    print(f"[ingest] mode={args.mode} collection={collection}")
    records, stats = collect_records(args.mode)
    print(f"[ingest] {stats}")
    if not records:
        print("[ingest] no records")
        return

    client = build_qdrant_client()
    model = get_model()
    ready = False
    total = (len(records) + UPSERT_BATCH - 1) // UPSERT_BATCH
    indexed = 0
    for bn, start in enumerate(range(0, len(records), UPSERT_BATCH), start=1):
        batch = records[start : start + UPSERT_BATCH]
        vecs = embed_passages(model, [r["embedding_text"] for r in batch])
        if not ready:
            recreate_collection(client, collection, len(vecs[0]))
            ready = True
        points = [
            models.PointStruct(id=str(uuid.uuid4()), vector=v, payload=r["payload"])
            for r, v in zip(batch, vecs)
        ]
        client.upsert(collection_name=collection, points=points, wait=True)
        indexed += len(points)
        print(f"[ingest] batch {bn}/{total} -> {indexed}/{len(records)}")
    print(f"[ingest] done: {indexed} points in '{collection}'")


if __name__ == "__main__":
    main()
