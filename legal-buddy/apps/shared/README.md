# shared

Core primitives shared across the legal RAG pipeline so the query side (API) and
the passage side (ingestion) can never drift apart.

- `shared.embedding` — model loading, e5 `passage:`/`query:` prefixes, encode helpers.
- `shared.qdrant` — client builder, collection (re)create, upsert-with-retry.
- `shared.chunking` — token-aware, structure-aware section chunker + record collection.
