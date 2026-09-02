# LegalBuddy Technical Documentation

Blockchain Olympiad Bangladesh 2026 (BCOLBD), AI Category, Final Round
Team LegalBuddy (ID 6a739711a3937) — Istiaqur Rahman Fuad, University of Rajshahi

## 1. Repository Layout

```
legal-buddy/
  apps/api/        FastAPI service — RAG pipeline (query side)
  apps/web/        Next.js 15 chat UI (App Router, SSE)
  apps/shared/     embedding, chunking, Qdrant client (single source for both sides)
  apps/ingestion/  corpus -> Qdrant index builder (passage side)
  eval/            retrieval A/B harness, gold set, results
  docs/            deployment, chunking strategy, demo script
  docker-compose.full.yml   full stack: web + api + qdrant + llm
training/          SFT dataset, LoRA scripts, run logs
infra/azure/       HF -> GGUF conversion, Q4_K_M quantization, VM serving
```

## 2. Runtime Topology

`docker-compose.full.yml`:

| service | image / build | port | notes |
|---|---|---|---|
| llm | `ghcr.io/ggml-org/llama.cpp:server-b4738` | 8080 (internal) | `-m /models/lawbuddy-q4.gguf --alias lawbuddy-gemma4 -c 4096 -n 512`; CUDA variant available |
| qdrant | `qdrant/qdrant:latest` | 6333 | named volume `qdrant-storage` |
| api | `apps/api/Dockerfile` | 8000 | uvicorn, single process; depends_on: qdrant + llm healthy |
| web | `apps/web/Dockerfile` | 3000 | `API_URL=http://api:8000`; depends_on: api healthy |

Health gating: the API's `/rag/health` returns 200 only after (a) the sentence-transformers model is loaded (app lifespan warms it) and (b) `verify_qdrant()` has confirmed the configured collection exists. The compose healthcheck on api uses this endpoint (start_period 180 s for the first-boot model download), and web waits on it.

## 3. API Contract

`POST /rag/legal/chat` — JSON response.
`POST /rag/legal/chat/stream` — `text/event-stream`.
`GET /rag/health`.

Request schema (`api/api/models.py`):

| field | type | default | notes |
|---|---|---|---|
| `question` | str | required | min length 3 |
| `history` | `[{role, content}]` | `[]` | last 6 turns sent by the UI; roles `user`/`assistant` |
| `top_k` | int? | `RETRIEVAL_TOP_K` | unique sections returned |
| `max_tokens` | int? | `ANSWER_MAX_TOKENS` | passed through to the LLM |
| `provider` | `gemini\|groq\|openai`? | `DEFAULT_LLM_PROVIDER` | `groq` and `openai` are the same code path |
| `model`, `temperature` | str?, float? | provider default / 0.2 | |
| `clarify_score_floor`, `low_confidence_floor` | float? | 0.79 / 0.83 | per-request threshold overrides |

Response (`LegalChatResponse`): `{answer: str, sources: [SourceItem]}` with `SourceItem = {citation_id, act_title, act_year, section_index, source_url, excerpt, score}`. `excerpt` carries the full section text (`section_full` from the payload).

SSE event sequence on the stream endpoint: `sources` (once, before generation) → `delta` (per token chunk) → `done`. Errors after the stream starts are emitted as `error` events with a user-safe message; the HTTP status is already 200. Mid-stream exceptions are logged server-side with the traceback.

Error mapping (`_client_error`): timeouts → 504, provider rate limits / quota → 429, auth/config failures → 503, everything else → 500. Raw exception text never reaches the client.

## 4. Retrieval

Embedding: `intfloat/multilingual-e5-base`, 768 dims, run locally via sentence-transformers on CPU. e5 asymmetric prefixes: `query: ` at search, `passage: ` at ingestion (`shared/embedding.py`). Embeddings L2-normalized; Qdrant distance metric COSINE.

Query path (`agents/legal_chat/`):

1. `contextualize.condense_question` — skipped when history is empty. One LLM call at temperature 0 on the rewrite model; output stripped of quotes; any exception falls back to the raw question.
2. `retrieval.retrieve_sources` — `candidate_limit = max(top_k * 5, 20)` chunk hits via `query_points`, then parent-document collapse: dedupe by `section_uid` (fallback key `source_url#section_index`), keep first `top_k` unique sections, excerpt prefers `section_full` over the chunk. Citation ids renumbered 1..k after the score floor filter.
3. Two-tier gate (`pipeline.py`): top score < 0.79 → clarify path (no answer LLM call); top score < 0.83 → grounded prompt with a low-confidence instruction appended.

Score calibration data (e5 cosine, in `config.py` comments): off-topic "what time is it" 0.836; answerable "my neighbor keeps threatening me" top hit 0.818 (correct §503); "best pizza recipe" 0.781.

Chunking (`shared/chunking.py`): token-aware splitting at the model's runtime-read `max_seq_length` (512 with `EMBEDDING_MAX_TOKENS=512`), overlap 24 tokens; contextual header `Act | Title | Section (part k/n)` reserved out of the token budget; footnote markers stripped by regex; omitted/repealed sections skipped by pattern (`VOID_SECTION_RE`); subsection splits preserved on `(1)`/`(a)` boundaries.

Ingestion (`apps/ingestion/pipeline.py`): reads `data/acts/*.json`, recreates the collection (drop + create, COSINE, payload indexes), upserts in batches of 64 with retry ×4 (backoff 2s·attempt). `INGEST_MAX_RECORDS=N` for smoke runs. Corpus: 35,312 sections loaded; ~1,000 acts.

Case-law pipeline (`apps/ingestion/cases_*.py`): 2,081 judgment PDFs, hybrid text/OCR extraction, structured metadata (court, judges, parties, dates, disposition), garbled legacy-font rejection heuristic. Cases are not indexed as citable sources; they feed SFT dataset construction.

## 5. Generation

Provider resolution (`generation.py`): `groq`/`openai` → OpenAI-compatible client against `GROQ_BASE_URL`; `gemini` → google-genai SDK. Client construction: `httpx.Timeout(GROQ_TIMEOUT_SECONDS=300, connect=30)`, `max_retries=1`.

Structured output (non-streaming path): instructor wraps the client; response model

```
StructuredLegalAnswer { answer: str (Markdown, [Source N] inline),
                        citations: int[] (default []),
                        limitations: str | None }
```

Post-processing: citation ids filtered to 1..len(sources), sorted, deduped; appended as `[Source N] ...` only when the answer text contains no inline citation. Any structured-output failure falls back to a plain completion.

Sampling defaults: temperature 0.2, top_p 0.9.

Prompt structure (`prompting.py`): system prompt (grounding rules: cite only provided sources, one number per bracket, clarify-when-vague, follow-up question on first-person situations, no advice disclaimer in text) + user message assembled as `Conversation so far` (optional) → `Question / situation` → optional low-confidence instruction → `Statute sources (citable)` with one block per source:

```
[Source n]
Act / Year / Section
Text: <full section>
URL: <bdlaws link>
```

Streaming path: plain completion, no instructor wrapper.

## 6. Frontend (`apps/web`)

Routes: `/` (chat), `POST /api/chat` and `/api/chat/stream` (Next.js proxies to `API_URL`; connect timeout 120 s, provider allowlist `gemini|groq|openai`).

Client behavior: SSE frames parsed from the raw stream (`event:`/`data:` lines, `\n\n` frame boundary). Citation marks: `[Source N]` in the answer is regex-rewritten to per-number links; hover previews the matching source card, click pins it and scrolls it into view (`Message.tsx`). History: last 6 non-error turns resent per request. Empty-state examples cover English/Bengali/Banglish. Font stack: Geist + Noto Sans Bengali (shaping for conjuncts/matras) + system sans.

## 7. Fine-Tuning

Dataset (`training/style_sft_prod/`, builder `legal-buddy/training/build_style_sft.py`): 1,000 train / 50 eval rows, each a full RAG interaction (system prompt, retrieved sources, user turn, target response). Mix: 630 direct Q&A / 170 situational / 125 clarify / 75 abstain / 50 Banglish. Validation per row: every citation in the target must exist in that row's provided sources; banned phrases ("As an AI language model…") absent; garbled case text rejected. 16 rows rejected (`rejected.jsonl`). Seed 20260730.

LoRA config: r=16, α=32, dropout 0.05, targets q/k/v/o/gate/up/down projections.

SFT hyperparameters (both runs): 2 epochs, `per_device_train_batch_size=1`, `gradient_accumulation_steps=8` (effective 8), lr 2e-4 cosine, warmup ratio 0.05, weight decay 0.01, max_grad_norm 0.3, bf16, `adamw_8bit`, gradient checkpointing (unsloth), `max_seq_length=4096`, eval + save every 25/50 steps, `load_best_model_at_end` on eval_loss, seed 42. Gemma chat-template formatting with system folded into the first user turn; EOS appended.

Runs (identical data and hyperparameters, single RTX PRO 6000 96 GB):

| run | base model | steps | duration | train loss | eval loss |
|---|---|---|---|---|---|
| dev | `unsloth/gemma-2-27b-it` | 250 | 28.3 min | 1.0312 | 0.5106 |
| production | `unsloth/gemma-4-31B-it` | 250 | 40.4 min | 0.6546 | 0.3869 |

Production run eval-loss trajectory (every 25 steps): 0.759 → 0.527 → 0.464 → 0.428 → 0.402 → 0.397 → 0.390 → 0.387 → 0.386. Logs: `training/train_gemma4.log`, `.json`.

Outputs: LoRA adapter (`lawbuddy-gemma4-31b-16bit-final/`, 498 MB) and merged 16-bit checkpoint (`lawbuddy-gemma4-31b-merged/`, 62.5 GB), both archived on Google Drive under `LegalBuddy/`.

## 8. Serving Pipeline

`infra/azure/vm_quantize*.sh`, `finish_fast.sh`: `convert_hf_to_gguf.py --outtype f16` → `llama-quantize ... Q4_K_M`. Tiers: BF16 ~62 GB, Q8_0 ~32 GB, Q4_K_M ~18 GB. The API accepts any OpenAI-compatible endpoint; configuration is env-only:

| variable | default | purpose |
|---|---|---|
| `DEFAULT_LLM_PROVIDER` | `groq` | `groq`/`openai` = OpenAI-compatible, `gemini` = google-genai |
| `GROQ_BASE_URL` | `https://api.groq.com/openai/v1` | llama.cpp `http://llm:8080/v1` in the full stack |
| `GROQ_MODEL` | `llama-3.3-70b-versatile` | must equal llama.cpp `--alias` locally |
| `GROQ_API_KEY` | — | any non-empty value for llama.cpp |
| `GROQ_CONDENSE_MODEL` | unset → `GROQ_MODEL` | distinct rewrite model only on multi-model endpoints |
| `GROQ_TIMEOUT_SECONDS` | 300 | total request timeout (connect 30 s) |

## 9. Retrieval Evaluation

Harness: `legal-buddy/eval/`. Gold set 120 question→section pairs (mostly Bengali, 58 acts), over a deterministic 201-act / 5,357-section sample. Gold key `(act_file, section_ord)` — chunking-independent. Metric: embed query → top-100 chunks → parent-document collapse → section-level Recall@{1,3,5,10} + MRR.

| variant | R@1 | R@3 | R@5 | R@10 | MRR |
|---|---|---|---|---|---|
| baseline (1200-char chunks, triBne-e5-small 128-tok) | 0.258 | 0.400 | 0.508 | 0.575 | 0.364 |
| improved chunking (#1–#4) | 0.308 | 0.458 | 0.500 | 0.608 | 0.408 |
| triBne-e5-small forced to 512 tokens | 0.175 | 0.308 | 0.392 | 0.458 | 0.274 |
| improved + multilingual-e5-base @512 (production) | 0.567 | 0.733 | 0.825 | 0.883 | 0.675 |

Findings: the baseline model truncated at 128 tokens (verified on `SentenceTransformer.max_seq_length`), losing ~60% of 1200-char chunks at embed time; forcing it to 512 is worse because positions 128–512 are untrained. Reproduction commands in `eval/REPORT.md`; eval ingests into throwaway collections only.

## 10. Observability

LangSmith via env (`LANGSMITH_TRACING`, `LANGSMITH_API_KEY`, `LANGSMITH_ENDPOINT`, `LANGSMITH_PROJECT`, `LANGSMITH_WORKSPACE_ID`); pydantic settings are bridged to `os.environ` at import (`core/observability.py`) before any trace runs. Trace tree per request: `legal-chat-request` / `legal-chat-stream` (chain) → `embed-query` (embedding), `vector-search` (retriever), `answer-generation` (llm). Startup runs an auth check (`list_projects`); shutdown flushes. With tracing disabled the pipeline is identical.

## 11. Known Limitations

- e5 cosine scores overlap between off-topic and answerable queries; the two-tier floor is a calibration, not a classifier. A cross-encoder reranker is the identified fix.
- Corpus freshness ends at ingestion time; no amendment feed.
- Q4_K_M trades reasoning quality for the 18 GB footprint.
- End-to-end answer quality on labeled conversations is not yet measured (SFT eval loss covers style; request-time validation covers citation integrity).

## 12. Reproduction

```bash
# index (acts JSON under legal-buddy/data/acts/)
cd legal-buddy && uv sync --all-packages && uv run ingest

# fine-tune (one >=96 GB GPU)
cd ../training && python train_gemma4.py          # ~40 min

# quantize merged checkpoint -> lawbuddy-q4.gguf (~18 GB)
../infra/azure/finish_fast.sh

# serve
cp lawbuddy-q4.gguf legal-buddy/models/
cd legal-buddy && cp .env.example .env
docker compose -f docker-compose.full.yml up -d --build
curl -s localhost:8000/rag/health                 # 200 when ready
```
