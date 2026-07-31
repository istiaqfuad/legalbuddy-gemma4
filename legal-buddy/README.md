# Legal Acts RAG Monorepo

Production-style Retrieval Augmented Generation (RAG) project for Bangladesh legal acts.

This workspace contains:

- A FastAPI backend: Qdrant retrieval + answer generation (Gemini or Groq)
- A Next.js chat frontend (`apps/web`)
- A `shared` package (embedding, Qdrant, chunking) used by both the API and ingestion
- An `ingestion` package that builds the Qdrant index from `data/acts/`
- Workspace tooling via `uv`, Docker, and Docker Compose

## What This Project Does

For each user question:

1. The API creates an embedding with a local HuggingFace sentence-transformers model (CPU)
2. It retrieves top matching legal sections from Qdrant
3. It builds a grounded prompt with citation tags
4. It generates an answer with Gemini
5. It returns answer + source metadata to the UI

The UI displays the answer and expandable source cards with similarity scores.

## Repository Layout

```text
law_buddy/
├── apps/
│   ├── api/                # FastAPI RAG service (query side)
│   ├── web/                # Next.js chat frontend
│   ├── shared/             # embedding, Qdrant, chunking (one source of truth)
│   └── ingestion/          # builds the Qdrant index (passage side)
├── data/acts/              # legal act JSON corpus (gitignored)
├── eval/                   # chunking/retrieval A/B harness (see eval/README.md)
├── docs/                   # chunking_and_retrieval.md, …
├── notebooks/              # exploratory notebooks only
├── docker-compose.yml
├── Makefile
├── pyproject.toml          # uv workspace root
└── README.md
```

## Prerequisites

- Python 3.13+
- `uv` installed: https://docs.astral.sh/uv/
- Docker + Docker Compose (recommended for full stack)
- Access to:
  - A Gemini API key
  - A HuggingFace embedding model id (downloaded locally on first run)
  - Reachable Qdrant instance

## Quick Start (Recommended)

1. Create your local env file from the template:

```bash
cp .env.example .env
```

Then fill values in `.env`:

```env
# Gemini chat model
GEMINI_API_KEY=your_gemini_key
# Optional; defaults to gemini-2.5-flash
CHAT_MODEL=gemini-2.5-flash

# HuggingFace embedding model (run locally via sentence-transformers, CPU)
EMBEDDING_MODEL=org/your-embedding-model
HF_TOKEN=

# Qdrant
QDRANT_VECTORESTORE=http://your-qdrant-host:6333
QDRANT_API_KEY=
QDRANT_COLLECTION=legal_acts_event_rag_full

# Retrieval + generation defaults
RETRIEVAL_TOP_K=6
# Optional. Leave empty/unset to let model default behavior apply.
ANSWER_MAX_TOKENS=

# Frontend -> API URL (inside docker network default is fine)
API_URL=http://api:8000

# Optional LangSmith tracing
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=
# Optional overrides
# LANGSMITH_ENDPOINT=https://api.smith.langchain.com
# LANGSMITH_PROJECT=legal-buddy
```

2. Install workspace deps:

```bash
make sync
```

3. Start both services:

```bash
docker compose up --build
```

4. Open apps:

- UI: `http://localhost:3000`
- API docs: `http://localhost:8000/docs`
- Health: `http://localhost:8000/rag/health`

## Local Development Without Docker

Install all workspace packages once:

```bash
uv sync --all-packages --all-extras --all-groups
```

Run API:

```bash
uv run --package api uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload
```

Run the web UI in another terminal (Node 20+ / pnpm):

```bash
cd apps/web
pnpm install
API_URL=http://localhost:8000 pnpm dev   # http://localhost:3000
```

## API Contract Summary

`POST /rag/legal/chat`

Request body:

```json
{
  "question": "What is the punishment for theft?",
  "top_k": 6,
  "max_tokens": null
}
```

Notes:

- `question` is required
- `top_k` is optional and falls back to `RETRIEVAL_TOP_K`
- `max_tokens` is optional. If omitted/null and `ANSWER_MAX_TOKENS` is also unset, the model default token limit behavior is used.
- `provider` (`gemini`|`groq`), `model`, and `temperature` are optional per-request
  testing knobs; the web UI exposes them (plus `top_k`/`max_tokens`) in a dev settings panel.

## Observability

LangSmith instrumentation is integrated in the API for retrieval/generation spans. Set `LANGSMITH_TRACING=true` and `LANGSMITH_API_KEY` to enable ingestion. If tracing is disabled or the key is not configured, the pipeline still runs without tracing.

## Data and Ingestion

- Legal act JSON files live under `data/acts/`
- Reingest the corpus into Qdrant with the configured HuggingFace embedding model:

```bash
uv run ingest
```

  This recreates `QDRANT_COLLECTION` with the embedding model's vector size and upserts all
  non-repealed sections. Set `INGEST_MAX_RECORDS=N` to limit records for a quick smoke test.
  Orchestration lives in `apps/ingestion`; chunking/embedding/Qdrant come from `shared`.

## Useful Commands

```bash
make sync                # install/update workspace deps
make run-docker-compose  # sync + docker compose up --build
```

## Additional Docs

- API details: `apps/api/README.md`
- Web UI: `apps/web/README.md`
- Shared primitives: `apps/shared/README.md`
- Ingestion: `apps/ingestion/README.md`
- Chunking & retrieval strategy: `docs/chunking_and_retrieval.md`
- Eval harness: `eval/README.md`
