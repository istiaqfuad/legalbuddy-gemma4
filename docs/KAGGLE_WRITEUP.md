# LawBuddy — A Legal Assistant for Bangladesh, Powered by a Fine-Tuned Gemma 4

## 🎯 TL;DR

**Live demo**: [lb.irflab.tech](https://lb.irflab.tech) · **Public code**: [github.com/istiaqfuad/legalbuddy-gemma4](https://github.com/istiaqfuad/legalbuddy-gemma4)

**LawBuddy** is a citation-grounded legal chat assistant for Bangladesh that combines **Retrieval-Augmented Generation (RAG) over the full body of Bangladesh statutes and case law** with a **LoRA fine-tune of Gemma 4 31B** — served as a **4-bit quantized GGUF** so a 31B-parameter legal model runs affordably on modest hardware. Ask in Bengali, Banglish, or English; LawBuddy answers with the exact act and section, citing every source it relies on.

## 🌍 The Problem: Law is written for lawyers, not for the 170 million people it governs

Bangladesh operates under **35,312 statutory sections** across ~1,000 acts, plus decades of case law — scattered across government PDF portals and unindexed archives. The law is written in formal English legalese, yet most citizens communicate in Bengali and Banglish. Legal help is expensive and concentrated in cities.

The result: ordinary people face police stations, courts, landlords, and contracts **without knowing their rights or the rules that protect them**. Paralegals and legal-aid clinics are stretched thin, and there is no free, trustworthy way to get a plain-language answer to "what does the law say about my situation?"

## 💡 The Solution

**A RAG pipeline over Bangladesh law + a Gemma 4 fine-tune that speaks it back in plain, cited language.**

The core design insight: *don't make the model memorize law — make it read the law and explain it.* Every answer is grounded in statute sections retrieved at query time, cited with numbered `[Source N]` tags that the user can expand and verify against the original act text. This keeps answers accurate, up-to-date with whatever corpus is ingested, and free of hallucinated legal claims.

## 🤖 Gemma 4 as the Core Model

- **Base**: `google/gemma-4-31B-it` — Google DeepMind's most capable open model — used as the **only generation model** in the final pipeline (no proprietary LLMs in the demo).
- **Fine-tune**: 16-bit LoRA (`r=16, α=32`, target all linear projections) on a **1,000-example citation-safe SFT dataset** we built from real Bangladesh judgments + statute sections — deliberately *not* raw memorization of acts.
- **Training**: 250 steps over 2 epochs; final train loss **1.03**, best held-out eval loss **0.51** — clean convergence without overfitting on a small set.
- **Serving**: LoRA merged → HF→GGUF conversion → **Q4_K_M 4-bit quantization** (~9 GB) → served via llama.cpp / vLLM as an OpenAI-compatible endpoint. A 31B legal model that fits on a single affordable GPU — or even CPU.
- **Also used for**: history-aware rewriting of multi-turn follow-ups into standalone search queries.

## 🏗️ Technical Implementation

### Data & ingestion
- **Statutes**: ~1,000 Bangladesh acts parsed into **35,312 structured sections** with act title, year, and section number.
- **Case law**: 2,000+ real judgment PDFs extracted and structured into facts, disposition, and court-reasoning excerpts.
- **Chunking**: engineered with contextual headers and 512-token windows sized to the embedding model's real context length (a measured 128-token truncation bug in the original model cost ~60% of chunk text at embed time).

### Retrieval
- **Embedding**: `intfloat/multilingual-e5-base` (768-dim, 512-token window) — chosen because queries and corpus are **bilingual (Bengali/English)**.
- **Vector store**: Qdrant, with **parent-document retrieval** — 5× candidate chunks are pulled per query, then collapsed to unique whole sections so the model never sees headless split fragments and duplicate parts of one source can't crowd out others.
- **Confidence gating**: two-tier score floors — a hard floor (0.79) routes genuine off-topic queries to a clarifying response; a soft floor (0.83) passes a low-confidence hint to the model, which decides whether to answer or ask.

### Generation
- Fine-tuned Gemma 4 receives the question, the conversation history, citable statute sources, and (when relevant) reasoning-only precedent background.
- **Structured output** (`{answer, citations[], limitations}`) via `instructor` — machine-parseable, citation-verified responses.
- **Anti-hallucination**: the prompt instructs the model to cite *only* retrieved sources, avoid generic court-process commentary not present in sources, and **abstain or ask a clarifying question** when the law is unclear.

### System
- FastAPI backend (single uvicorn process to bound RAM), Next.js standalone frontend, Qdrant, and the LLM server — all Dockerized with healthchecks (`/rag/health` only turns green once the embedding model is warm and the vector store is verified).
- LangSmith tracing on every retrieval and generation step for debugging.

## 📊 Evaluation (Measured, Not Vibes)

A **120-question gold set** across 58 unique acts (mostly Bengali queries) scores every retrieval change:

| Variant | R@1 | R@3 | R@5 | R@10 | MRR |
|---|---|---|---|---|---|
| Baseline | 0.258 | 0.400 | 0.508 | 0.575 | 0.364 |
| + chunking fixes | 0.308 | 0.458 | 0.500 | 0.608 | 0.408 |
| + **multilingual-e5-base @512** | **0.567** | **0.733** | **0.825** | **0.883** | **0.675** |

Every metric roughly **doubled** (+31 pts) from the embedding-model swap, with chunking contributing a further ~+5 pts. The SFT dataset itself teaches the model *when to ask instead of hallucinate*, mixing five real interaction types: direct statutory Q&A (630), personal-situation questions (170), Banglish queries (50), clarifying turns (125), and abstain cases (75).

## 🎨 User Experience

A polished Next.js chat app with a clean ink-and-emerald design:

- Ask in **plain Bengali or English** — "আমার প্রতিবেশী আমাকে হুমকি দিচ্ছে" ("my neighbor keeps threatening me") gets a real statutory answer
- **Streaming** responses with a "Thinking" indicator
- **Inline citation pins** (`[Source 2]`) that expand into source cards showing the act, section, text excerpt, and similarity score
- **Multi-turn conversation memory** (6-turn window) with follow-up rewriting
- A settings drawer exposing retrieval and generation knobs, plus example prompts for first-time users

## 🚀 Deployment

- **One-command Docker stack**: `web + api + qdrant + llama.cpp` via docker-compose, with persistent volumes for the vector store and embedding-model cache.
- **Azure ML alternative**: managed online endpoint running vLLM on an A100 (`deployment.yml`) for GPU-scale serving.
- **Reproducible quantization**: the full HF → GGUF → Q4_K_M pipeline is scripted (`vm_quantize.sh`), so the exact 4-bit artifact can be rebuilt from the merged adapter.

## 🌱 Real-World Impact

- **Access to justice**: free, plain-language explanations of the law in the language people actually speak.
- **Trust through citations**: every claim links to its source — verifiable by users, paralegals, and legal-aid professionals.
- **Cost-effective**: 4-bit quantization makes a frontier-class 31B model deployable on modest hardware — realistic for NGOs, university law clinics, and community legal-aid organizations across Bangladesh.
- **Bilingual by design**: Bengali, Banglish, and English all work natively.
- **Foundation for an ecosystem**: legal-education tools, contract review, and dispute-prevention advice all build on the same grounded-retrieval core.

## 🚀 Innovation Highlights

1. **Fine-tuned 31B open model in 4-bit** — frontier-grade legal reasoning at commodity serving cost.
2. **Citation-safe SFT dataset** — designed around *grounding behavior* (answer / clarify / abstain), not memorization.
3. **Bilingual + Banglish support** for how users actually speak.
4. **A/B-measured retrieval stack** — chunking, embeddings, and parent-document collapse each validated against a gold set.

## 🔗 Links

- **Live demo**: [lb.irflab.tech](https://lb.irflab.tech)
- **Public repo**: [github.com/istiaqfuad/legalbuddy-gemma4](https://github.com/istiaqfuad/legalbuddy-gemma4) — full fine-tuning code, SFT dataset, training logs, app source, Docker compose, and deployment scripts

*Built in a day with Google DeepMind's Gemma 4 — for the people who need the law explained, not just the lawyers who wrote it.*
