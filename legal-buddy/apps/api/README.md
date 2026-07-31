# API Service (FastAPI)

Backend service for legal RAG chat.

## Responsibilities

- Accept legal chat requests
- Embed question via AWS Bedrock
- Retrieve relevant sections from Qdrant
- Build grounded prompt with source citations
- Generate final answer with Gemini
- Return answer + structured sources
- Emit LangSmith traces when configured

## Main Endpoints

- `GET /rag/health`
- `POST /rag/legal/chat`

OpenAPI docs are available at `/docs` when running.

## Request and Response

### `POST /rag/legal/chat`

Request:

```json
{
  "question": "Can a contract be oral under Bangladesh law?",
  "top_k": 6,
  "max_tokens": null
}
```

- `question`: required, min length 3
- `top_k`: optional, falls back to `RETRIEVAL_TOP_K`
- `max_tokens`: optional. If `null` (or omitted), API uses `ANSWER_MAX_TOKENS`; if that is also unset, token limit is not forced in Gemini config.

Response shape:

```json
{
  "answer": "... grounded answer with [Source n] citations ...",
  "sources": [
    {
      "citation_id": 1,
      "act_title": "Penal Code",
      "act_year": 1860,
      "section_index": "378",
      "source_url": "https://...",
      "excerpt": "...",
      "score": 0.88
    }
  ]
}
```

## Configuration

The service reads env vars via `api.core.config.Config`.

Required:

```env
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
AWS_DEFAULT_REGION=...
EMBEDDING_MODEL=...
```

Common optional:

```env
GEMINI_API_KEY=...
DEFAULT_MODEL_NAME=gemini-2.5-flash
QDRANT_URL=http://your-qdrant:6333
QDRANT_COLLECTION=legal_acts_event_rag_full
RETRIEVAL_TOP_K=6
ANSWER_MAX_TOKENS=

LANGSMITH_TRACING=true
LANGSMITH_API_KEY=
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
LANGSMITH_PROJECT=legal-buddy
```

## Run With Docker

From repository root:

```bash
docker compose up --build api
```

Service listens on `http://localhost:8000`.

## Run Locally (No Docker)

From repository root:

```bash
uv sync --all-packages --all-extras --all-groups
uv run --package api uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload
```

## Code Map

- `apps/api/src/api/app.py`: FastAPI app + lifespan hooks
- `apps/api/src/api/api/endpoints.py`: health + chat endpoints
- `apps/api/src/api/api/models.py`: request/response models
- `apps/api/src/api/agents/retrieval_generation.py`: retrieval + generation pipeline
- `apps/api/src/api/core/config.py`: settings
- `apps/api/src/api/core/observability.py`: LangSmith client lifecycle

## Troubleshooting

- 500 error with model call: verify `GEMINI_API_KEY` and model name
- Retrieval returns no sources: check `QDRANT_URL`, collection name, and embedding compatibility
- Bedrock errors: validate AWS credentials, region, and `EMBEDDING_MODEL`
- No traces in LangSmith: confirm `LANGSMITH_TRACING=true`, the API key, and startup auth check logs
