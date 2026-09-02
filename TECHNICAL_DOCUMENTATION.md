# LegalBuddy — Technical Documentation

**Blockchain Olympiad Bangladesh 2026 (BCOLBD), AI Category — Final Round**
Team LegalBuddy (ID 6a739711a3937) · Istiaqur Rahman Fuad, University of Rajshahi

This document describes the developed system end-to-end: architecture,
algorithms, data preprocessing, model training, validation methods, and
deployment. Every number cited here is reproducible from artifacts in this
repository (training logs, eval reports, dataset manifests).

---

## 1. System Overview

LegalBuddy is a citation-grounded legal AI assistant for Bangladesh law. A user
asks a question in English, Bengali, or Banglish (code-mixed) and receives a
plain-language answer in which every legal claim cites a specific act and
section (`[Source N]`) that can be expanded in the UI to show the original
statute text.

The core design principle: **the model never answers law from memory.** The
only path to a legal answer runs through retrieval-augmented generation (RAG)
against an indexed statutory corpus, with the fine-tuned model responsible for
style — explaining, citing, asking clarifying questions, and declining when
the retrieved context is insufficient.

### Components

| component | tech | location |
|---|---|---|
| Web chat UI | Next.js 15 (App Router), SSE streaming | `legal-buddy/apps/web` |
| RAG API | FastAPI + instructor (structured output) | `legal-buddy/apps/api` |
| Vector store | Qdrant | `legal-buddy/docker-compose.full.yml` |
| Embeddings | `intfloat/multilingual-e5-base` (local, CPU) | `legal-buddy/apps/shared` |
| LLM | Fine-tuned Gemma (LoRA → merged → 4-bit GGUF) served by llama.cpp (OpenAI-compatible API) | `training/`, `infra/azure/` |
| Ingestion | act JSON corpus + case-law PDFs → Qdrant | `legal-buddy/apps/ingestion` |

### Request flow

```
user question (en / bn / banglish)
  │
  ├─ follow-up? → condense_question(): rewrite into standalone query
  │              (small model, history-aware)            contextualize.py
  │
  ├─ embed query ("query:" prefix, e5)
  │              → Qdrant top max(5·k, 20) chunks        retrieval.py
  │
  ├─ parent-document collapse → k unique full sections
  │
  ├─ two-tier confidence gate (0.79 hard / 0.83 soft)    pipeline.py
  │      below hard floor → deterministic clarify, no LLM call
  │      below soft floor → answer WITH low-confidence hint
  │
  ├─ grounded prompt: [Source 1..k] full section texts   prompting.py
  │
  ├─ fine-tuned Gemma → StructuredLegalAnswer            generation.py
  │      {answer, citations, limitations} via instructor
  │      citation ids validated ∈ 1..k, fallback to plain text
  │
  └─ SSE stream: sources → deltas → done                 endpoints.py
```

The streaming path (used by the UI) emits an initial `sources` event before
generation starts, so source cards render while the answer streams. The
non-streaming path enforces the structured schema and appends a validated
`[Source N]` citation list when the model did not cite inline.

## 2. Data Preprocessing

### 2.1 Statutory corpus

- ~1,000 Bangladesh acts from the Ministry of Law portal
  (bdlaws.minlaw.gov.bd), parsed to JSON; **35,312 sections loaded** in the
  production build (dataset manifest).
- Each section → record with `act_title`, `act_year`, `section_index`,
  `source_url`, `section_full`.
- **Chunking** (`apps/shared/src/shared/chunking.py`): token-aware splitting
  sized to the embedding model's *actual* `max_seq_length` (no silent
  truncation), 24-token overlap (~20% of a 128-token window), with a
  contextual header (`Act | Title | Section (part k/n)`) prepended to every
  part so fragments retain statutory context. Footnote markers are stripped;
  omitted/repealed sections are skipped by regex.
- Payload keeps both the chunk and `section_full` + `section_uid` so retrieval
  can return whole sections (below).

### 2.2 Case-law corpus

- 2,081 High Court / Appellate Division judgment PDFs parsed
  (`apps/ingestion/cases_*.py`: hybrid direct-text + OCR extraction, structured
  metadata — court, judges, parties, dates, disposition).
- Case law is **reasoning-only background**: it informed the SFT dataset
  construction (§4.1) but is not surfaced as a citable source, reducing the
  risk of misattributed legal claims. Garbled legacy-font extractions are
  rejected by heuristic.

## 3. Retrieval

### 3.1 Embedding model selection (measured, not assumed)

The retrieval eval harness (`legal-buddy/eval/`, gold set of 120
question→section pairs, mostly Bengali, over a 201-act / 5,357-section
deterministic sample) compared chunking strategies and embedding models.
Full report: `legal-buddy/eval/REPORT.md`.

| variant | R@1 | R@3 | R@5 | R@10 | MRR |
|---|---|---|---|---|---|
| baseline (1200-char chunks, triBne-e5-small) | 0.258 | 0.400 | 0.508 | 0.575 | 0.364 |
| improved chunking (#1–#4) | 0.308 | 0.458 | 0.500 | 0.608 | 0.408 |
| **improved + multilingual-e5-base @512** | **0.567** | **0.733** | **0.825** | **0.883** | **0.675** |

Two firm findings drove the production configuration:

1. The original embedding model truncated at 128 tokens — baseline's
   1200-character chunks lost ~60% of their text at embed time.
2. The model swap dwarfs every chunking change (+31 points on every metric vs
   +5 for chunking). Production therefore uses
   `intfloat/multilingual-e5-base` (768-dim, genuinely trained at 512 tokens,
   strong Bengali↔English alignment) with the improved chunking.

### 3.2 Parent-document retrieval

Retrieval fetches `max(5·k, 20)` candidate chunks, collapses them to unique
sections by `section_uid`, and returns the **full section text** — the model
never sees headless fragments, and duplicate parts of one section cannot crowd
out other relevant acts (`retrieval.py`).

### 3.3 Two-tier confidence gating

Measured e5 cosine scores are a *weak* separator — off-topic "what time is it"
scores 0.836 while an answerable "my neighbor keeps threatening me" tops out
at 0.818 (with the correct §503 as its top hit). No single cutoff cleanly
splits answerable from off-topic, so:

- **Hard floor 0.79** — only genuine garbage ("best pizza recipe", 0.781)
  routes to a deterministic clarify without an LLM call.
- **Soft floor 0.83** — borderline turns still go to the model, *with* their
  sources plus a low-confidence hint; the fine-tuned model decides whether to
  answer or ask a follow-up. Borderline judgment is the model's, not a brittle
  threshold.

## 4. Model Training

### 4.1 SFT dataset (`training/style_sft_prod/`)

1,000 training + 50 eval examples built by
`legal-buddy/training/build_style_sft.py` from the parsed case PDFs and real
act JSON sources. Each example is a full RAG interaction: system prompt,
retrieved statute context, user turn, target assistant turn.

| interaction type | share | n |
|---|---|---|
| direct statutory Q&A, citation-grounded | 60% | 630 |
| situational (user describes own problem) | 16% | 170 |
| Banglish code-mixed queries | 5% | 50 |
| clarification turns (model asks for details) | 12% | 125 |
| abstention (out of scope) | 7% | 75 |

**Automated validation** (16 rows rejected and logged in `rejected.jsonl`):
every citation in a target response must reference a source actually provided
in that example's prompt; banned phrases ("As an AI language model…") must be
absent; garbled legacy-font case text is dropped heuristically. Per-row
outcomes are in `validation.csv`; seed and counts in `manifest.json`.

### 4.2 Fine-tuning configuration

16-bit LoRA (r=16, α=32, dropout 0.05) on all linear projections
(q/k/v/o/gate/up/down), 2 epochs, effective batch 8 (1 × 8 accumulation),
lr 2e-4 cosine, warmup 5%, bf16, gradient checkpointing, adamw_8bit,
max sequence 4096. The fine-tune teaches response style only — the law itself
always comes from retrieval at inference time.

### 4.3 Completed run (gemma-2-27b-it development run)

The full pipeline was executed end-to-end on `unsloth/gemma-2-27b-it`
(`training/train_prod.py`, log: `training/train_prod.log`):

- 250 steps / 2 epochs over the 1,000-example set, 28.3 minutes on a 96 GB GPU
- final train loss **1.0312**, best/final eval loss **0.5106** (50-example holdout)
- adapter merged and quantized to the 4-bit GGUF currently served by the demo
  stack; re-quantizing from the Gemma 4 merged checkpoint (§4.4) replaces it

### 4.4 Production run (Gemma 4 31B)

`training/train_gemma4.py` ran the *identical* dataset and hyperparameters on
`gemma-4-31B-it` (whitepaper §4.4), executed on a single RTX PRO 6000
(102 GB) — log: `training/train_gemma4.log`:

- 250 steps / 2 epochs over the 1,000-example set, **42.0 minutes**
- final train loss **0.6543** (from an initial 7.30), best/final eval loss
  **0.3863** on the 50-example holdout
- eval loss fell monotonically every 25-step checkpoint: 0.759 → 0.527 → 0.464
  → 0.428 → 0.402 → 0.397 → 0.390 → 0.387 → 0.386 — no overfitting within
  the 2-epoch budget
- outputs: LoRA adapter (`lawbuddy-gemma4-31b-16bit-final/`) and merged 16-bit
  checkpoint (59 GB, `lawbuddy-gemma4-31b-merged/`), the direct input to the
  GGUF → Q4_K_M serving pipeline (`infra/azure/`)

**Comparison with the 27B development run** (same data, same hyperparameters):

| run | duration | final train loss | best eval loss |
|---|---|---|---|
| gemma-2-27b-it (dev) | 28.3 min | 1.0312 | 0.5106 |
| **gemma-4-31B-it (production)** | 42.0 min | **0.6543** | **0.3863** |

The production model halves holdout loss — the stronger base adapts to the
citation-grounded style task substantially better.

## 5. Serving & Deployment

### Quantization tiers (whitepaper §4.4)

| tier | memory | use |
|---|---|---|
| BF16 merged | ~62 GB | institutional, multi-GPU |
| Q8_0 | ~32 GB | single high-end GPU |
| **Q4_K_M (production demo)** | ~18 GB | consumer GPU / high-RAM CPU |

The merged checkpoint is converted with llama.cpp's `convert_hf_to_gguf.py`
(F16) then quantized to Q4_K_M (`infra/azure/vm_quantize*.sh`,
`finish_fast.sh`). llama.cpp serves it behind an OpenAI-compatible
`/v1/chat/completions` endpoint with `--alias lawbuddy-gemma4`.

### One-command full stack

`legal-buddy/docker-compose.full.yml` — web (Next.js :3000) → api
(FastAPI :8000) → qdrant (:6333) + llm (llama.cpp :8080). Health-check
dependencies: the API reports healthy only once the embedding model is warm
*and* the Qdrant collection is verified (`retrieval.py::verify_qdrant`), so
the stack never accepts traffic half-booted. Full instructions:
`legal-buddy/docs/docker-deployment.md`.

### Structured output

The API wraps the OpenAI-compatible endpoint with `instructor`
(`generation.py`), enforcing `StructuredLegalAnswer {answer, citations,
limitations}`. Citation ids are validated against the actual source count
before rendering; on any structured-output failure the call degrades to plain
text rather than erroring.

## 6. Validation Summary

| layer | method | result |
|---|---|---|
| retrieval | 120-pair gold set, R@k + MRR, 4 variants A/B | R@1 0.258 → 0.567, MRR 0.364 → 0.675 (`eval/REPORT.md`) |
| SFT data | per-row citation + phrase validation | 1,050 valid / 16 rejected (`style_sft_prod/`) |
| training | held-out eval loss each 25 steps | eval loss 0.3863 (Gemma 4 production run) |
| generation | structured schema + citation-id validation at serve time | invalid ids dropped, silent fallback to text |
| behavioral | clarify/abstain share of the SFT mix | 19% of training examples are clarify or abstain turns |

## 7. Reproducibility

```bash
# 1. Ingest the corpus into Qdrant (acts JSON in legal-buddy/data/acts/)
cd legal-buddy && uv sync --all-packages && uv run ingest

# 2. Train (>=96GB GPU)
cd ../training && python train_gemma4.py

# 3. Quantize the merged checkpoint (llama.cpp toolchain)
../infra/azure/finish_fast.sh          # → lawbuddy-q4.gguf

# 4. Serve the whole stack
cp lawbuddy-q4.gguf legal-buddy/models/
cd legal-buddy && cp .env.example .env && docker compose -f docker-compose.full.yml up -d --build

# 5. Retrieval eval harness (optional, throwaway collections)
cd legal-buddy && PYTHONPATH=eval uv run python eval/run_eval.py --collection ... --tag ...
```

## 8. Known Limitations

- Retrieval confidence floors are calibrated for the e5 cosine distribution;
  a cross-encoder reranker is the identified robust long-term fix
  (`config.py` documents the calibration).
- Corpus coverage ends at ingestion time — recent amendments are blind spots
  until re-ingested; the UI disclaimer states this.
- The 4-bit tier trades some reasoning quality for the ability to run on
  consumer hardware (whitepaper §3.1 quantization trade-off).
- Legal information, not legal advice: the system prompt, SFT data, and UI
  all enforce the information side of that line.
