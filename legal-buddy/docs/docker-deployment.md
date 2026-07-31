# Docker Deployment

Runs the whole LawBuddy stack on a single host with one command:

```
web (Next.js :3000)  →  api (FastAPI :8000)  →  qdrant (:6333)
                                         ↘   llm (llama.cpp :8080, 4-bit Gemma 4)
```

Compose file: [`docker-compose.full.yml`](../docker-compose.full.yml).
The development `docker-compose.yml` (web+api only, hot reload) and the
Dokploy-specific `docker-compose.prod.yml` remain available.

## Prerequisites

- Docker with the Compose plugin (`docker compose version`)
- The quantized model: `models/lawbuddy-q4.gguf` (~9 GB), produced by the
  `vm_quantize.sh` pipeline (HF → GGUF → Q4_K_M)
- Qdrant data: the API serves whatever collections exist in the vector store.
  The corpus is **not** in the repo — restore a snapshot, or run the ingestion
  pipeline (`apps/ingestion`) once against `localhost:6333` to rebuild
  `legal_acts_event_rag_full`.

## 1. Prepare

```bash
cd legal-buddy
cp .env.example .env        # then edit — see .env.example comments
mkdir -p models
cp /path/to/lawbuddy-q4.gguf models/
```

`.env` keys that matter for the stack (all pre-filled in `.env.example`):

| Key | Value for this stack |
|---|---|
| `QDRANT_VECTORESTORE` | `http://qdrant:6333` (compose DNS name) |
| `EMBEDDING_MODEL` | `intfloat/multilingual-e5-base` (+ `EMBEDDING_MAX_TOKENS=512`) |
| `DEFAULT_LLM_PROVIDER` | `groq` — routes to any OpenAI-compatible base URL |
| `GROQ_BASE_URL` | `http://llm:8080/v1` |
| `GROQ_MODEL` | must equal the llama.cpp `--alias` (`lawbuddy-gemma4`) |
| `GROQ_API_KEY` | any dummy value — llama.cpp does not validate keys |

## 2. Start

```bash
docker compose -f docker-compose.full.yml up -d --build
```

First boot downloads the embedding model into the `hf-cache` volume, so give the
api up to 3 minutes to become healthy.

## 3. Verify

```bash
docker compose -f docker-compose.full.yml ps        # all four services running
docker compose -f docker-compose.full.yml logs -f api

curl -s http://localhost:8000/rag/health            # 200 = model warm + Qdrant verified
curl -s -X POST http://localhost:8000/rag/legal/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the punishment for theft under the Penal Code?"}'
# → grounded answer with [Source N] citations + sources array

open http://localhost:3000                           # web UI
```

`/rag/health` is the real readiness gate: it returns 200 only after the e5
embedding model is loaded **and** Qdrant is reachable, so a dead vector store
fails the healthcheck at boot rather than on the first chat.

## GPU

Swap the `llm` image to `ghcr.io/ggml-org/llama.cpp:server-cuda-b4738` and
uncomment the `deploy.resources.reservations.devices` block in the compose file.
CPU inference on a 4-bit 31B model works (~1–3 tok/s) but GPU is strongly
recommended for a public demo.

## Production hardening

- Front **only** `web:3000` behind a reverse proxy (Traefik/nginx + TLS) and
  remove the `ports:` entries for `api`, `qdrant`, and `llm`.
- Set `QDRANT_API_KEY` and pass the same key to the Qdrant container
  (`QDRANT__SERVICE__API_KEY` env) so the vector store is not open on the host.
- Pin the `qdrant` image to a specific version instead of `latest`.
- If structured-output parsing misbehaves on llama.cpp (the api uses
  `instructor` JSON mode), serve the merged model with
  `vllm/vllm-openai:latest` instead — see [`deployment.yml`](../deployment.yml).

## Alternative: vLLM on a GPU VM / Azure

For scale, the same `.env` works with a vLLM OpenAI-compatible server. Point
`GROQ_BASE_URL` at the vLLM endpoint and set `GROQ_MODEL` to the served model
name. The Azure ML managed online deployment (`deployment.yml`/`endpoint.yml`)
runs the merged 16-bit model on an A100 via vLLM.

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| api stuck `unhealthy` | Qdrant down or collections missing — check `docker compose logs api`; restore/ingest data first |
| 404 on `/rag/health` | api not warm yet (embedding download) — wait up to 3 min |
| `404 model not found` from LLM | `GROQ_MODEL` ≠ the llama.cpp `--alias` |
| slow first answer | embedding model download on first boot; CPU inference |
| JSON parse errors on answers | llama.cpp structured-output edge case — try a newer `server-bXXXX` image or switch to vLLM |
