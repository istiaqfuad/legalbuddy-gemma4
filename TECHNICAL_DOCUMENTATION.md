# LegalBuddy Technical Documentation

Blockchain Olympiad Bangladesh 2026 (BCOLBD), AI Category, Final Round
Team LegalBuddy (ID 6a739711a3937) — Istiaqur Rahman Fuad, University of Rajshahi

This document describes the built system: how the data is prepared, how retrieval works, how the model was fine-tuned and validated, and how everything is deployed. Each section points to the code and artifacts in this repository that back it. The white paper covers motivation and business context; this document covers the implementation.

## 1. System Overview

LegalBuddy is a retrieval-augmented generation (RAG) system for Bangladesh statutory law. A user asks a question in English, Bengali, or Banglish. The system finds the relevant statute sections in a vector index, hands them to a fine-tuned Gemma 4 model with the question, and streams back an answer in which legal claims carry inline citations. Clicking a citation opens the full text of the cited section, with a link to the source page on bdlaws.minlaw.gov.bd.

The design principle behind the whole pipeline is that the model is never asked to answer a legal question from memory. It only ever explains text that was retrieved for that specific question, and the fine-tune exists to enforce that behavior rather than to teach law.

There are four services, all defined in legal-buddy/docker-compose.full.yml:

The web frontend (legal-buddy/apps/web), a Next.js application that renders the chat interface and proxies requests to the API.

The API (legal-buddy/apps/api), a FastAPI service that owns the RAG pipeline.

Qdrant, the vector store holding the indexed statute sections.

The LLM server, llama.cpp serving the fine-tuned model as a 4-bit GGUF behind an OpenAI-compatible API.

A shared package (legal-buddy/apps/shared) contains the embedding, chunking, and Qdrant logic once, so the ingestion side and the query side cannot drift apart. An ingestion package (legal-buddy/apps/ingestion) builds the index from the corpus.

## 2. Request Flow

This section follows one question through the system. The code lives in legal-buddy/apps/api/src/api/agents/legal_chat/.

When the question arrives with conversation history (the frontend sends the last six turns), a query rewrite step first condenses it into a standalone search query. A follow-up like "শাস্তি কত বছর?" has no subject on its own, so the rewrite resolves references against the history before retrieval. On the first turn this is skipped. If the rewrite call fails for any reason, the original question is used; retrieval is never blocked by the rewrite.

The query is embedded with Multilingual-E5-Base running locally through sentence-transformers, using the "query:" prefix the model expects. Qdrant returns up to max(5k, 20) candidate chunks, where k is the number of sources requested.

The candidates are collapsed to unique statute sections. Each indexed chunk carries the section's identity (a section ID, the act title and year, the section number, the source URL) and the section's full text. The collapse step keeps the top k distinct sections and returns their full text rather than the chunks. This matters because a raw chunk can be a fragment without its section heading, and a model given fragments tends to misread scope.

A two-tier confidence gate runs on the retrieval scores before any LLM call. If the best score is below 0.79, the query is treated as off-topic and the system answers with a clarifying question without calling the model at all. Between 0.79 and 0.83, the question proceeds to the model but the prompt includes a note that the retrieved material may not fit, and the model is expected to ask a follow-up rather than push an answer. The thresholds came from measurement, not intuition: off-topic questions score around 0.78 to 0.84 on this embedding model, and answerable situational questions land in the same range, so a single cutoff cannot separate them and the borderline judgment is better left to the model with a hint.

The prompt lists the retrieved sections as [Source 1] through [Source k], each with its act, section number, and full text, followed by the question and the recent conversation. Two prompts exist: one for answering and one for the clarify path.

## 3. Answer Generation

The generation code is in generation.py. The API talks to the LLM through an OpenAI-compatible client, which is the same code path whether the endpoint is the local llama.cpp server, Groq's cloud API, or vLLM. The active endpoint is chosen by environment variables only (GROQ_BASE_URL, GROQ_MODEL, GROQ_API_KEY), so switching backends requires no code change and no rebuild.

On the non-streaming path, the request is wrapped with the instructor library, which asks the model for a structured answer: an answer field in Markdown, a citations list, and an optional limitations field. The citation numbers are validated against the number of sources actually provided; numbers outside that range are dropped. If the model already cited inline, the citation list is not appended again. If the structured request fails for any reason, the call falls back to a plain completion rather than returning an error to the user.

The streaming path, which the UI uses, requests a plain completion and streams it token by token. The response is sent to the browser as server-sent events: one "sources" event with the retrieved sections before generation starts, then "delta" events with text chunks, then a final "done" event. Source cards render while the answer is still generating.

The OpenAI-compatible client uses a 300-second total timeout with a 30-second connect timeout (configurable via GROQ_TIMEOUT_SECONDS). The defaults matter in practice: llama.cpp does not accept connections until the model has finished loading, and a CPU-only server can take minutes to produce its first token, which is well outside the OpenAI SDK's 5-second default connect timeout. Request timeouts surface to the user as a 504 with a plain explanation, rate limits as a 429, and configuration problems as a 503, with the underlying exception details kept in the server log.

## 4. Data Preprocessing

The corpus has two parts.

Statutes. Roughly 1,000 acts from the Ministry of Law portal, parsed to JSON, yielding 35,312 loaded sections. Each section becomes a record with act title, year, section number, source URL, and full text. Chunking is token-aware: sections are split to fit the embedding model's real 512-token window, with a 24-token overlap, and each part gets a contextual header (act, title, section, part number) so a fragment still knows what it belongs to. Footnote markers are stripped, and omitted or repealed sections are dropped by pattern. The chunker reads the model's actual max_seq_length at runtime instead of trusting configuration, which is how we found the original embedding model was silently truncating input at 128 tokens.

Case law. More than 2,000 High Court and Appellate Division judgment PDFs, extracted with a hybrid of direct text extraction for clean English pages and OCR for scanned or Bengali content, then parsed into structured metadata (court, judges, parties, dates, disposition). Case text with garbled legacy-font extraction is rejected by a heuristic. Case law is background: it informed the fine-tuning dataset and the system prompt's framing, but judgments are not indexed as citable sources, because a misattributed case citation is worse than none.

The ingestion entry point is legal-buddy/apps/ingestion (uv run ingest), which recreates the Qdrant collection with the configured embedding model and upserts all non-repealed sections. Ingestion and query sides share the same embedding and chunking code from apps/shared.

## 5. Retrieval Evaluation

We built a gold set of 120 question-to-section pairs over a deterministic sample of 201 acts (5,357 sections), with questions mostly in Bengali. The gold key is the section, not the chunk, so one set scores every chunking variant. The harness (legal-buddy/eval) ingests candidate configurations into throwaway collections and reports Recall@1/3/5/10 and MRR at section level, with parent-document collapse applied.

Four configurations were measured:

Baseline: the original 1200-character chunks and a small fine-tuned embedding model, scoring Recall@1 of 0.258 and MRR of 0.364.

Improved chunking alone: Recall@1 of 0.308, MRR of 0.408.

The small model forced to a 512-token window: worse across the board, because the checkpoint was trained at 128 tokens and its higher positions are untrained.

Improved chunking with Multilingual-E5-Base: Recall@1 of 0.567, Recall@10 of 0.883, MRR of 0.675.

The model swap contributed about 31 points on every metric; the chunking work contributed about 5. Production uses the combination that measured best. The full report with per-metric numbers and reproduction commands is legal-buddy/eval/REPORT.md.

## 6. Fine-Tuning

The dataset (training/style_sft_prod) contains 1,000 training and 50 evaluation examples, generated from the parsed case PDFs and real act sources. Each example is a complete RAG interaction: the system prompt, retrieved statute context, a user turn, and a target assistant turn. The mix is 60% direct statutory Q&A with citations, 16% situational questions, 5% Banglish, 12% clarification turns where the correct behavior is to ask for a detail that would change the answer, and 7% abstention cases where the correct behavior is to decline.

Every example passed automated validation before entering the set: each citation in the target response must reference a source present in that example's prompt, and phrases like "As an AI language model" must not appear. Sixteen rows failed and were logged in rejected.jsonl. The manifest, per-row metadata, and validation outcomes are all in the repository.

Training was 16-bit LoRA (rank 16, alpha 32, dropout 0.05) over all linear projections, 2 epochs, effective batch size 8, learning rate 2e-4 with cosine decay, bf16, gradient checkpointing. The script is training/train_gemma4.py. The fine-tune teaches style only; the statute text comes from retrieval at inference time.

The full pipeline was run twice with identical data and hyperparameters, differing only in the base model:

The development run on gemma-2-27b-it: 28.3 minutes, final training loss 1.0312, evaluation loss 0.5106 (training/train_prod.log).

The production run on gemma-4-31B-it: 40.4 minutes, final training loss 0.6546, evaluation loss 0.3869 (training/train_gemma4.log). Evaluation loss fell at every 25-step checkpoint, from 0.759 to 0.387, with no sign of overfitting within the two-epoch budget.

Both runs finished on a single RTX PRO 6000 (96 GB). Since the runs are directly comparable, the lower loss on the same holdout set is attributable to the base model. The production run saves both the LoRA adapter and a merged 16-bit checkpoint; the merged checkpoint is the input to the serving pipeline.

## 7. Serving and Deployment

The merged checkpoint is converted to GGUF with llama.cpp's conversion tooling and quantized to Q4_K_M (infra/azure holds the conversion scripts), producing an approximately 18 GB file. Q8_0 (~32 GB) and full BF16 (~62 GB) tiers are available for better hardware. llama.cpp serves the GGUF behind an OpenAI-compatible chat completions endpoint.

The full stack starts with one command (docker compose -f docker-compose.full.yml up -d --build), after placing the GGUF in legal-buddy/models and copying .env.example to .env. The compose file wires startup order through health checks: the API reports healthy only after the embedding model has loaded and the statute collection has been verified in Qdrant, and the web container waits on the API. A misconfigured vector store fails the container at startup instead of surfacing as a 500 on the first question.

The .env file documents the two backend configurations: the local llama.cpp server for the competition demo, and Groq's cloud API for quick testing without a GPU. Both use the same client code. The default provider, model names, timeouts, and the optional rewrite model are all environment variables.

## 8. Frontend

The chat interface (legal-buddy/apps/web) is a Next.js application. Answers render as Markdown. Inline [Source N] citations become small clickable marks; hovering one highlights the matching source card and clicking pins it and scrolls it into view. The source panel, collapsed by default, lists each cited section with its score and the external link to the official portal.

The frontend keeps conversation memory client-side and sends the last six turns with each request, which is what powers the follow-up rewrite on the backend. Error states are handled: backend errors arrive as events on the stream and render as inline notices, and a disclaimer sits under the composer at all times. Example prompts on the empty state include English, Bengali, and Banglish questions. The font stack includes Noto Sans Bengali so conjuncts and vowel signs render correctly regardless of the client's system fonts.

## 9. Observability

The API emits LangSmith traces when configured (LANGSMITH_TRACING, LANGSMITH_API_KEY). Both the JSON and streaming paths produce one trace per request, with nested spans for the query embedding, the Qdrant search, and answer generation, plus a span for the follow-up rewrite. Tracing is optional: with no key configured the pipeline runs identically without it.

## 10. Validation Summary

Retrieval was measured against the 120-pair gold set (Section 5). Training was measured with a held-out evaluation loss at every checkpoint (Section 6). The dataset itself was validated row by row at build time. At serving time, citation numbers are validated against the provided sources on every non-streaming request, and the two-tier gate bounds what the model is asked to do. What is not yet measured systematically is end-to-end answer quality on a labeled set of full conversations; the SFT evaluation set covers style, and the citation validation covers grounding, but a human-labeled answer-quality set is future work.

## 11. Limitations

The confidence floors are calibrated to the e5 score distribution and are a pragmatic fix, not a principled one; a cross-encoder reranker would separate answerable from off-topic queries more cleanly.

The corpus is current only as of the last ingestion. Amendments passed after ingestion are invisible until the pipeline runs again, and the UI says so.

The 4-bit tier trades reasoning quality for hardware accessibility, which matters for nuanced legal interpretation even if it is acceptable for grounded Q&A.

The system provides legal information, not legal advice. The system prompt, the training data, and the UI disclaimer all enforce that line, and it is a line the system should not cross.

## 12. Reproduction

```bash
# Build the index (acts JSON under legal-buddy/data/acts/)
cd legal-buddy && uv sync --all-packages && uv run ingest

# Fine-tune (needs one >=96 GB GPU)
cd ../training && python train_gemma4.py

# Quantize the merged checkpoint
../infra/azure/finish_fast.sh   # produces lawbuddy-q4.gguf

# Serve the stack
cp lawbuddy-q4.gguf legal-buddy/models/
cd legal-buddy && cp .env.example .env
docker compose -f docker-compose.full.yml up -d --build
```

The retrieval harness can be re-run with the commands in legal-buddy/eval/REPORT.md, against throwaway collections so production is never touched.
