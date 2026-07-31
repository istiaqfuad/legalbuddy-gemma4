# ingestion

Builds the Qdrant index from the legal acts in `data/acts/`. All embedding,
chunking, and Qdrant logic comes from the `shared` package.

```bash
uv run ingest                       # full re-ingest (drops + rebuilds the collection)
INGEST_MAX_RECORDS=200 uv run ingest   # smoke run
```

Config is read from the repo `.env` (`EMBEDDING_MODEL`, `QDRANT_*`, `HF_TOKEN`,
optional `EMBEDDING_MAX_TOKENS`). A re-ingest is destructive to the target
collection — run it deliberately.
