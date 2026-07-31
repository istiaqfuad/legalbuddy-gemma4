# Cases Phase-2 plan — quality layers, reprioritized for the Gemini free tier

This plan covers the work **after** Phase 1 (precedent dual-retrieve, done). It
reorders the quality backlog around one hard constraint: the project runs on the
**Gemini free API tier** (low requests-per-minute + daily quota). The governing
rule is therefore:

> **Prefer levers that cost nothing per user query** — one-time batch jobs and
> local models — over anything that adds an LLM call to the request path.

That single rule demotes HyDE and promotes summary cards + a local reranker.

Related docs: [`situational_rag_plan.md`](situational_rag_plan.md) (this plan
supersedes its build ordering), [`cases_ingestion_plan.md`](cases_ingestion_plan.md),
[`chunking_and_retrieval.md`](chunking_and_retrieval.md).

---

## Status (Phase 1 — done)

- `legal_cases` is populated (~79k points / ~8k cases). English chunks clean;
  ~1,417 Bengali/legacy-font cases are garbled mojibake in the collection (OCR
  tail — see (f)). Phase 0 (chunk+embed) is therefore already satisfied; **do not
  re-run `cases-ingest`** (it recreates the collection).
- Precedent retrieval is wired into the live RAG (`apps/api/src/api/agents/legal_chat/`):
  - `retrieval.py` — `retrieve_cases()` (Qdrant `query_points_groups` on
    `case_uid` → top_k distinct cases) + `retrieve_dual()` (embed once,
    query both collections, drop sub-floor, renumber). (The function shipped as
    `retrieve_dual`, returning statutes and precedents separately.)

    ```python
    # apps/api/src/api/agents/legal_chat/retrieval.py
    def _run():
        return _qdrant_client.query_points_groups(
            collection_name=config.CASES_COLLECTION,
            query=vector,
            group_by="case_uid",
            limit=top_k,
            group_size=2,
            with_payload=True,
        ).groups
    ```

    ```python
    # apps/api/src/api/agents/legal_chat/retrieval.py — retrieve_dual()
    traced = get_langsmith_client() is not None
    vector = _embed(question, traced)

    statutes = [
        s
        for s in retrieve_sources(question, statute_k, vector=vector)
        if s.score >= config.STATUTE_SCORE_FLOOR
    ]
    precedents = [
        c
        for c in retrieve_cases(question, case_k, vector=vector)
        if c.score >= config.CASE_SCORE_FLOOR
    ]

    for new_id, source in enumerate(statutes, start=1):
        source.citation_id = new_id
    return statutes, precedents
    ```
  - `api/api/models.py` — `SourceItem` += `source_type` + case fields.

    ```python
    # apps/api/src/api/api/models.py — SourceItem
    source_type: Literal["statute", "precedent"] = "statute"
    # Statute fields
    act_title: str | None = None
    act_year: int | None = None
    section_index: str | None = None
    source_url: str | None = None
    # Precedent (case-law) fields
    full_case_ref: str | None = None
    court: str | None = None
    disposition: str | None = None
    judgment_date: str | None = None
    case_year: int | None = None
    ```
  - `core/config.py` — `CASES_COLLECTION`, `CASES_TOP_K`, `CASE_SCORE_FLOOR=0.82`,
    `STATUTE_SCORE_FLOOR=0.0`.

    ```python
    # apps/api/src/api/core/config.py
    CASES_COLLECTION: str = "legal_cases"
    # ...
    CASES_TOP_K: int = 4
    # ...
    STATUTE_SCORE_FLOOR: float = 0.0
    CASE_SCORE_FLOOR: float = 0.82
    ```
  - `prompting.py` — statute (binding) vs precedent (persuasive) blocks + no-
    precedent rule + disclaimer. The system prompt makes the asymmetry explicit:

    ```python
    # apps/api/src/api/agents/legal_chat/prompting.py — build_grounded_prompt()
    "1. STATUTE SOURCES — the BINDING law and your ONLY citable sources. "
    "Identify the governing act and section number(s), state the rule, and "
    "cite each one you rely on as [Source N].\n"
    "2. PRECEDENT BACKGROUND — how courts have reasoned about similar facts. "
    "This is for YOUR REASONING ONLY. Use it to judge the likely outcome and to "
    "reason like a court would, but DO NOT cite it, DO NOT present it as a "
    "source, and DO NOT mention, quote, or refer to any specific case, case "
    "number, party name, judge, or date. ..."
    ```
  - `pipeline.py` — uses `retrieve_dual` (and abstains when both pools are empty):

    ```python
    # apps/api/src/api/agents/legal_chat/pipeline.py
    statutes, precedents = retrieve_dual(
        search_query, statute_k=resolved_top_k, case_k=config.CASES_TOP_K
    )
    if not statutes and not precedents:
        return LegalChatResponse(answer=ABSTENTION_TEXT, sources=[])
    ```
- Verified end-to-end: a dowry scenario's statute side missed the governing law
  (vocabulary gap), but the precedents supplied Nari-O-Shishu / Dowry Act and the
  answer cited them.

---

## Constraint: the Gemini free tier

- The user-facing **answer generation** is the one LLM call we must always spend
  per query — reserve the Gemini free quota for it.
- Any feature that adds a **second per-query LLM call** (e.g. HyDE) roughly halves
  throughput and invites HTTP 429s. Avoid on the free tier.
- Both `GEMINI_API_KEY` and `GROQ_API_KEY` exist (`.env.example`). Groq's free tier
  has higher RPM, so route **non-user-facing / bulk** LLM work to Groq and keep
  Gemini for the answer.
- Local models (cross-encoder reranker, embeddings) cost **no API quota at all** —
  these are the safest quality levers here.

---

## Reprioritized Phase-2 backlog (free-tier ROI order)

### (a) Summary cards — DO FIRST
Highest scenario-matching ROI and **zero per-query cost** (a one-time batch, not a
request-path call).

- New `apps/ingestion/src/ingestion/cases_summarize.py` →
  `summarize_case(rec, llm)` writes a structured headnote
  (`summary_card` + `card_issue`, `card_holding`, `statutes_cited`) into each
  `data/cases_json/<id>.json`. Reuse the `instructor` structured-output stack from
  `apps/api/.../legal_chat/generation.py` — the same `response_model` pattern
  already in use for the answer:

  ```python
  # apps/api/src/api/agents/legal_chat/generation.py — _run_gemini()
  structured_client = instructor.from_genai(
      client, model=model, mode=instructor.Mode.GENAI_STRUCTURED_OUTPUTS
  )
  structured_messages = _build_structured_messages(messages)
  max_source_id = len(sources)
  try:
      structured_answer = structured_client.create(
          response_model=StructuredLegalAnswer,
          messages=structured_messages,
          config=_gemini_config(temperature, max_tokens),
      )
  ```

  Truncate oversized bodies before the
  call (matters for the ~13MB outlier; `disposition` is null ~33% so do not rely
  on it alone).
- Cache-in-JSON + **resumable** (skip files that already have `summary_card`) so
  the ~8k calls can dribble through the Gemini free quota over days, **or** run
  faster on Groq. The chunker stays LLM-free and deterministic.
- `apps/ingestion/src/ingestion/cases_chunking.py` `case_records`: add `chunk_kind`
  to `base_payload` (default `"body"`); prepend a `chunk_kind="summary"` record
  built from `rec["summary_card"]` when present (`chunk_part=0`).

  The current `base_payload` (the one to extend) looks like:

  ```python
  # apps/ingestion/src/ingestion/cases_chunking.py — case_records()
  base_payload = {
      "case_id": rec.get("case_id"),
      "case_uid": case_uid,
      # ...
      "full_case_ref": rec.get("full_case_ref"),
      "judgment_date": rec.get("judgment_date"),
      "disposition": rec.get("disposition"),
      "district": rec.get("district"),
      "source_type": "precedent",
  }
  ```

  ```python
  # proposed — not yet implemented
  base_payload = {
      # ... existing fields ...
      "source_type": "precedent",
      "chunk_kind": "body",  # NEW; a summary record would set "summary"
  }
  ```
- Re-embed via the existing append path `CASES_INGEST_FILES` (idempotent per
  `case_uid`) — **never recreate** the collection. The same pass later absorbs the
  OCR'd Bengali tail (f).
- `retrieval.py` `_case_groups_to_sources`: prefer the `chunk_kind="summary"`
  chunk for the excerpt / boost summary-card hits. Today the excerpt is assembled
  from the top-scoring body chunks, re-ordered for readability — that selection is
  what the summary-card preference would override:

  ```python
  # apps/api/src/api/agents/legal_chat/retrieval.py — _case_groups_to_sources()
  chunks = sorted(
      (
          (
              (h.payload or {}).get("chunk_part") or 0,
              float(h.score or 0.0),
              str((h.payload or {}).get("chunk_text") or ""),
          )
          for h in hits
      ),
      key=lambda c: c[1],
      reverse=True,
  )[:2]
  chunks = sorted(chunks, key=lambda c: c[0])  # reading order by chunk_part
  ```

### (b) Cross-encoder rerank — DO SECOND
**Local model, no API** → ideal for the free tier, and the robust fix for the thin
`CASE_SCORE_FLOOR=0.82` margin (bi-encoder cosine barely separates on-topic ~0.83
from off-topic ~0.80). The floor's own comment in `core/config.py` already flags
this reranker as the intended robust fix:

```python
# apps/api/src/api/core/config.py
# STATUTE floor stays 0.0 (don't touch the working acts path). CASE floor 0.82
# is empirical for multilingual-e5-base: off-topic scenarios top out ~0.80-0.81,
# on-point precedents score 0.83-0.86 (thin margin — the Phase-2 reranker is the
# robust fix; this is a coarse off-topic cut for MV).
```

- New `apps/api/src/api/agents/legal_chat/rerank.py` → `rerank(question, sources)`
  using a local cross-encoder (e.g. `BAAI/bge-reranker-v2-m3`, multilingual, or
  `cross-encoder/ms-marco-MiniLM-L-6-v2` for English-only). Load via
  sentence-transformers `CrossEncoder`, CPU is fine for a top-N pool.
- Rerank the merged statute+precedent pool from `retrieve_all_sources`; config-
  flagged; insert in `pipeline.py` between retrieve and `build_grounded_prompt`.
- Use the cross-encoder score for abstention (a calibrated relevance signal, far
  better than the raw cosine floor).

### (c) Abstention + floor tuning — no API
- Tune `CASE_SCORE_FLOOR` against real score distributions; consider a small
  `STATUTE_SCORE_FLOOR` to drop the junk acts that lay queries surface.
- Extend the empty-sources branch in `pipeline.py` to also abstain when the best
  (post-rerank) score is below floor — return "insufficient sources / consult a
  lawyer" rather than a confident wrong answer. The branch to extend currently
  only fires when both pools are empty:

  ```python
  # apps/api/src/api/agents/legal_chat/pipeline.py
  if not statutes and not precedents:
      return LegalChatResponse(answer=ABSTENTION_TEXT, sources=[])
  ```

### (d) HyDE — DEFERRED (free-tier reason)
HyDE rewrites the lay query into a hypothetical holding/statute paragraph before
retrieval. It is the biggest *query-side* lever for the statute vocabulary gap —
**but it adds a second LLM call per query**, which the Gemini free tier cannot
sustain. Deferred. Fallbacks if/when needed:

- **Groq-only rewrite:** run the cheap HyDE rewrite on Groq (higher free RPM),
  keep Gemini for the answer — one Gemini call per query preserved.
- **Non-LLM query expansion:** a static legal synonym/keyword map (lay term →
  statutory term, e.g. "thrown out" → "ejectment / notice to quit") expands the
  query with **zero API cost**. Lower ceiling than LLM HyDE but free.
- Revisit full HyDE only on a paid/raised tier.

Note: summary cards (a) already close much of the gap from the *corpus* side, so
HyDE's marginal value drops once cards land.

### (e) Eval harness — mostly no API
- `eval/`: gold scenario→{section, case} pairs; report recall@k / MRR (no API).
  The metric helpers already exist in `eval/common.py` and can be reused as-is for
  case-level scoring:

  ```python
  # eval/common.py
  def recall_at_k(ranked_keys: list[str], gold: str, ks=(1, 3, 5, 10)) -> dict:
      out = {}
      for k in ks:
          out[k] = 1.0 if gold in ranked_keys[:k] else 0.0
      return out


  def reciprocal_rank(ranked_keys: list[str], gold: str) -> float:
      for i, key in enumerate(ranked_keys, start=1):
          if key == gold:
              return 1.0 / i
      return 0.0
  ```
- Add the 5 intrinsic chunking metrics from arXiv 2603.25333 (Size Compliance,
  Intrachunk Cohesion, Doc Contextual Coherence, Block Integrity, Filtered
  Missing-Reference) — no gold labels, no API — to compare summary-card vs body-
  chunk segmentation before committing the re-embed.

### (f) OCR Bengali tail — GPU, not LLM
- Re-OCR the ~1,417 garbled/legacy-font cases (`cases-build` with
  `CASES_HYBRID_RESUME=1` on the GPU box), then append via `CASES_INGEST_FILES`
  (replaces the mojibake points). No LLM quota involved; blocked only on GPU
  availability.

---

## Provider-routing guidance

| Step | Engine | Cost shape |
|---|---|---|
| (a) Summary cards | Groq (fast) or Gemini (slow, resumable) | one-time batch |
| (b) Cross-encoder rerank | local CrossEncoder (CPU) | none (per query, local) |
| (c) Abstention / floor | none | none |
| (d) HyDE *(deferred)* | Groq rewrite, or non-LLM map | per query (avoid on Gemini free) |
| (e) Eval | local (intrinsic metrics) | none |
| (f) OCR tail | GPU (EasyOCR) | none (GPU) |
| **User answer** | **Gemini** (reserved) | **per query** |

Rule of thumb: **Gemini = the answer only. Everything auxiliary goes to Groq,
local, or GPU.**

---

## Suggested order

1. (a) summary cards → biggest quality jump, no per-query cost.
2. (b) local reranker → precision + real abstention.
3. (c) floor/abstention tuning on top of the reranker.
4. (e) eval to quantify (a)/(b).
5. (f) OCR tail when the GPU frees.
6. (d) HyDE only if the API tier is upgraded (or via the Groq/non-LLM fallback).
