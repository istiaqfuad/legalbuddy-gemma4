# Chunking & Retrieval Strategy — Law Buddy RAG

How Bangladesh statutory law in `data/acts/*.json` becomes a searchable vector
index, and how a question is answered from it. This documents the pipeline as it
stands after the chunking/retrieval/model work validated in `eval/REPORT.md`.

---

## 1. Source data

Each file `data/acts/act-print-<id>.json` is one Act:

```jsonc
{
  "act_title": "The Penal Code, 1860",
  "act_no": "XLV",
  "act_year": "1860",
  "language": "...",
  "source_url": "http://bdlaws.minlaw.gov.bd/act-print-11.html",
  "csv_metadata": { "is_repealed": false },
  "sections": [
    { "section_title": "Punishment of offences committed within Bangladesh",
      "section_content": "2. Every person shall be liable to punishment ..." },
    ...
  ]
}
```

Corpus shape (measured): **1,281 acts with sections, ~35,633 sections.**
Section length is heavily skewed — median ~429 chars, p99 ~4,156, max ~45,807; the
section is the natural legal unit and is the atom the pipeline chunks around.

---

## 2. Embedding model

`intfloat/multilingual-e5-base` (set via `EMBEDDING_MODEL`):

- **768-dim**, **512-token** window, multilingual (handles the corpus's Bengali +
  English), run locally on CPU via `sentence-transformers`.
- **e5 is asymmetric**: passages are embedded with a `passage:` prefix at ingest,
  queries with a `query:` prefix at search. Both sides must use the same model.

  ```python
  # apps/shared/src/shared/embedding.py
  def is_e5(model_name: str | None) -> bool:
      return bool(model_name) and "e5" in model_name.lower()

  def passage_prefix(text: str, e5: bool) -> str:
      return f"passage: {text}" if e5 else text

  def query_prefix(text: str, e5: bool) -> str:
      return f"query: {text}" if e5 else text
  ```

- This choice is the single biggest quality lever — see §7. The previous model
  (`triBne-e5-small`, 384-dim) silently truncated at 128 tokens; swapping to a
  model genuinely trained at 512 tokens roughly doubled retrieval recall.

Optional `EMBEDDING_MAX_TOKENS` env var overrides the model's `max_seq_length`
(left unset for e5-base, whose native window is already 512):

```python
# apps/shared/src/shared/embedding.py — load_embedding_model(...)
model = SentenceTransformer(model_name, device=device, token=token)
if max_tokens:
    model.max_seq_length = int(max_tokens)
    model.tokenizer.model_max_length = int(max_tokens)
return model
```

---

## 3. Ingestion — `apps/ingestion` (chunking from `shared`)

Run: `uv run ingest`. The orchestration lives in `apps/ingestion`; the embedding,
Qdrant, and chunking logic comes from the `shared` package (one source of truth
with the API query side). Reads config from `.env` (`EMBEDDING_MODEL`,
`QDRANT_VECTORESTORE`, `QDRANT_COLLECTION`, `HF_TOKEN`).

### 3.1 Cleaning & filtering (per section)

1. **Skip repealed acts** — `csv_metadata.is_repealed == true`.
2. **Skip void sections** — `[Omitted by ...]`, `[Repealed by ...]`, `[Repeal.-]`
   etc. (`VOID_SECTION_RE`). These carry no legal content, only noise.
3. **Strip footnote markers** — `clean_section_content` rewrites inline
   `12[text]` reference markers down to `text` (`FOOTNOTE_MARKER_RE`).
4. **Extract the section index** — `extract_section_index` pulls the leading
   numbering (`"380."`, `"4A."`, Bengali digits `০-৯`) for citation/metadata.

The regexes and helpers (all in `apps/shared/src/shared/chunking.py`):

```python
# apps/shared/src/shared/chunking.py
SECTION_INDEX_RE = re.compile(r"^[\s\"'\[\]]*\[?([0-9০-৯]+[a-zA-Z]*)[.।৷\-\s]")
FOOTNOTE_MARKER_RE = re.compile(r"\d+\[(.*?)\]")
VOID_SECTION_RE = re.compile(
    r"\[\s*(Omitted|Repealed?|Rep\.)\s+by" r"|\[\s*Repeal\.\-" r"|\[\s*Omit\.\-",
    re.IGNORECASE,
)
SUBSECTION_SPLIT_RE = re.compile(r"(?=(?:\(\d+\)|\([a-zA-Z]+\)))")

def clean_section_content(section_content: str) -> str:
    if not section_content:
        return ""
    return FOOTNOTE_MARKER_RE.sub(r"\1", section_content).strip()
```

### 3.2 Token-aware chunking — sizing to the model window

`section_records()` + `chunk_section_tokens()`:

- The chunk budget is the model's **real** `max_seq_length` (read at runtime),
  minus tokens reserved for the contextual header and `passage:` prefix — so the
  *entire* chunk is actually embedded, never silently truncated.

  ```python
  # apps/shared/src/shared/chunking.py — section_records(...)
  max_tokens = model_max_tokens(model)
  # +12 covers the "passage:" prefix, the "(part k/n)" marker, and the newline
  # that sit alongside the header but aren't in `header` itself.
  reserved = _token_len(model, passage_prefix(header, e5)) + 12
  budget = max(32, min(max_tokens - reserved, max_tokens - 4))
  overlap = min(TOKEN_OVERLAP, max(4, budget // 4))
  parts = chunk_section_tokens(model, cleaned, budget, overlap)
  ```

- A section that fits the budget stays whole (one chunk).
- A longer section is **split on natural subsection boundaries** (`(1)`, `(2)`,
  `(a)` … via `SUBSECTION_SPLIT_RE`), greedily merging adjacent subsections up to
  the budget. Only a single subsection that is itself over-budget falls back to a
  token-window slice (`_token_slice`) with **24-token overlap** (`TOKEN_OVERLAP`)
  for continuity.

  ```python
  # apps/shared/src/shared/chunking.py — chunk_section_tokens(...)
  if _token_len(model, text) <= budget:
      return [text]
  parts = [p.strip() for p in SUBSECTION_SPLIT_RE.split(text) if p.strip()]
  if len(parts) > 1:
      merged, current = [], ""
      for part in parts:
          candidate = f"{current} {part}".strip() if current else part
          if _token_len(model, candidate) <= budget:
              current = candidate
              continue
          if current:
              merged.append(current)
              current = ""
          if _token_len(model, part) <= budget:
              current = part
              continue
          merged.extend(_token_slice(model, part, budget, overlap))
      # ...
      if merged:
          return merged
  return _token_slice(model, text, budget, overlap)
  ```

- This keeps legal sub-provisions intact instead of cutting mid-clause.

### 3.3 Contextual header (#1, #3)

Every chunk — including each part of a split section — is prefixed with:

```
Act: <act_title> | Title: <section_title> | Section <index> (part k/n)
<chunk text>
```

`section_title` (previously discarded) is the strongest retrieval signal in
statutes; repeating the header on every part means split parts 2..n keep their
context instead of being a headless tail.

```python
# apps/shared/src/shared/chunking.py
def context_header(act_title: str, section_title: str | None, section_index: str) -> str:
    bits = [f"Act: {act_title}"]
    if section_title:
        bits.append(f"Title: {section_title}")
    bits.append(f"Section {section_index}")
    return " | ".join(bits)

def build_embedding_text(header: str, part_no: int, n_parts: int, chunk_text: str) -> str:
    cont = f" (part {part_no}/{n_parts})" if n_parts > 1 else ""
    return f"{header}{cont}\n{chunk_text}"
```

### 3.4 Payload (#4 parent-document support)

Each Qdrant point stores:

| field | purpose |
|---|---|
| `act_file`, `section_ord` | stable identity of the source section |
| `section_uid` = `act_file#section_ord` | **dedupe key for parent-document retrieval** |
| `section_index`, `section_title`, `chunk_part`, `n_parts` | citation + display |
| `section_content_clean` | this chunk's text (excerpt) |
| `section_full` | the **whole** cleaned section (returned to the LLM) |
| `act_title`, `act_no`, `act_year`, `language`, `source_url`, `govt_system` | metadata / filtering |

Assembled per chunk in `collect_all_records`:

```python
# apps/shared/src/shared/chunking.py — collect_all_records(...)
payload = {
    "act_file": file_path.stem,
    "section_ord": section_ord,
    "section_uid": f"{file_path.stem}#{section_ord}",
    "act_title": act_obj.get("act_title"),
    "act_no": act_obj.get("act_no"),
    "act_year": (
        int(act_obj["act_year"])
        if str(act_obj.get("act_year", "")).isdigit()
        else None
    ),
    "language": act_obj.get("language"),
    "govt_system": act_obj.get("government_context", {}).get("govt_system"),
    "source_url": act_obj.get("source_url"),
    **rec["payload_extra"],  # section_title, section_index, chunk_part,
                             # n_parts, section_content_clean, section_full
}
```

### 3.5 Qdrant collection

`recreate_collection()` drops + recreates the target collection
(`QDRANT_COLLECTION`, default `legal_acts_event_rag_full`):

- vectors: **cosine distance**, size inferred from the model (768 for e5-base).
- payload indexes: `act_year` (int), `language` (keyword), `section_uid` (keyword).

The ingest pipeline calls it once, on the first batch, with the index fields:

```python
# apps/ingestion/src/ingestion/pipeline.py — ingest()
vector_size = len(vectors[0])
qdrant.recreate_collection(
    client,
    COLLECTION_NAME,
    vector_size,
    integer_indexes=("act_year",),
    keyword_indexes=("language", "section_uid"),
)
```

```python
# apps/shared/src/shared/qdrant.py — recreate_collection(...)
existing = {c.name for c in client.get_collections().collections}
if name in existing:
    client.delete_collection(collection_name=name)
client.create_collection(
    collection_name=name,
    vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE),
)
```

- passages embedded in batches with the `passage:` prefix and L2-normalised
  (`embed_passages` calls `model.encode(..., normalize_embeddings=True)`);
  upserted with retry/back-off (`upsert_with_retry`).

---

## 4. Retrieval — `apps/api/src/api/agents/legal_chat/retrieval.py`

`retrieve_sources(question, top_k)`:

1. Embed the question with the `query:` prefix (same model).
2. Pull a **candidate pool** of `max(top_k*5, 20)` chunks from Qdrant — wider than
   `top_k` because the next step collapses chunks to sections.

   ```python
   # apps/api/src/api/agents/legal_chat/retrieval.py
   CANDIDATE_MULTIPLIER = 5
   MIN_CANDIDATES = 20

   def retrieve_sources(
       question: str, top_k: int, *, vector: list[float] | None = None
   ) -> list[SourceItem]:
       """Retrieve statute sections from the acts collection."""
       candidate_limit = max(top_k * CANDIDATE_MULTIPLIER, MIN_CANDIDATES)
       # ... embed query into `vector` ...
       hits = _search(config.QDRANT_COLLECTION, vector, candidate_limit, traced)
       return _hits_to_sources(hits, top_k)
   ```

3. **Parent-document dedupe** (`_hits_to_sources`): walk hits in score order,
   keep the first chunk of each `section_uid`, and return its **full section**
   (`section_full`), stopping at `top_k` unique sections. So the LLM receives
   whole sections, and duplicate parts of one section never crowd out other
   relevant sections.

   ```python
   # apps/api/src/api/agents/legal_chat/retrieval.py — _hits_to_sources(...)
   for hit in hits:
       payload = hit.payload or {}
       uid = payload.get("section_uid") or (
           f"{payload.get('source_url')}#{payload.get('section_index')}"
       )
       if uid in seen:
           continue
       seen.add(uid)
       # Prefer the full section text; fall back to the chunk excerpt.
       excerpt = str(
           payload.get("section_full") or payload.get("section_content_clean") or ""
       ).strip() or "No excerpt available."
       # ... build SourceItem ...
       if len(sources) >= top_k:
           break
   ```

4. Each result becomes a `SourceItem` (`citation_id`, `act_title`, `act_year`,
   `section_index`, `source_url`, `excerpt`, `score`).

Backward-compatible: if a collection lacks `section_uid`/`section_full`, it falls
back to `source_url#section_index` and the chunk excerpt.

---

## 5. Generation — handoff

`legal_chat_pipeline` (`pipeline.py`) ties it together:

1. `retrieve_sources(question, top_k=RETRIEVAL_TOP_K)` (default `top_k=6`). The
   defaults and model live in `config`:

   ```python
   # apps/api/src/api/core/config.py
   CHAT_MODEL: str = "gemini-2.5-flash"
   EMBEDDING_MODEL: str
   QDRANT_COLLECTION: str = "legal_acts_event_rag_full"
   RETRIEVAL_TOP_K: int = 6
   ```

   (The live `legal_chat_pipeline` now calls `retrieve_dual(...)`, which wraps
   `retrieve_sources` for statutes and adds reasoning-only precedents — see §4's
   `retrieve_cases`/`retrieve_dual`.)
2. `build_grounded_prompt(question, sources)` — grounds the model in the retrieved
   sections.
3. `run_llm(...)` — Gemini (`CHAT_MODEL`, `gemini-2.5-flash`) via `instructor`,
   returning a **structured** `StructuredLegalAnswer` (`answer`, `citations`,
   `limitations`). Citations render as `[Source N]` tags tied to the SourceItems.

   ```python
   # apps/api/src/api/agents/legal_chat/structured_models.py
   class StructuredLegalAnswer(BaseModel):
       answer: str = Field(description="Final answer to the user")
       citations: list[int] = Field(
           default_factory=list,
           description="List of supporting source ids, e.g. [1, 2]",
       )
       limitations: str | None = Field(
           default=None,
           description="Optional uncertainty or missing context",
       )
   ```

   ```python
   # apps/api/src/api/agents/legal_chat/generation.py — _run_gemini(...)
   structured_client = instructor.from_genai(
       client, model=model, mode=instructor.Mode.GENAI_STRUCTURED_OUTPUTS
   )
   structured_answer = structured_client.create(
       response_model=StructuredLegalAnswer,
       messages=structured_messages,
       config=_gemini_config(temperature, max_tokens),
   )
   return _render_structured_answer(structured_answer, max_source_id)
   ```

4. If retrieval is empty, it returns an explicit "no relevant sources" answer
   rather than hallucinating.

LangSmith tracing wraps the whole chain when configured.

---

## 6. Request flow (end to end)

```
POST /rag/legal/chat {question, top_k?, max_tokens?}
   └─ legal_chat_pipeline
        ├─ embed query ("query: …")  ─ multilingual-e5-base
        ├─ Qdrant search  top_k*5 candidate chunks  (cosine)
        ├─ parent-doc dedupe → top_k unique full sections  (SourceItem[])
        ├─ build_grounded_prompt(question, sources)
        └─ run_llm → StructuredLegalAnswer → {answer + [Source N], sources[]}
```

---

## 7. Why these choices — evidence

Validated on an isolated A/B harness (`eval/`, n=120 LLM-generated
question→section pairs, recall@k + MRR). Full numbers in `eval/REPORT.md`.

| variant (same gold set, same chunking) | R@1 | R@5 | R@10 | MRR |
|---|---|---|---|---|
| old model, 1200-char chunks (baseline) | 0.258 | 0.508 | 0.575 | 0.364 |
| old model + improved chunking #1–#4 (128 tok) | 0.308 | 0.500 | 0.608 | 0.408 |
| old model forced to 512 tokens | 0.175 | 0.392 | 0.458 | 0.274 |
| **multilingual-e5-base @512 + improved chunking** | **0.567** | **0.825** | **0.883** | **0.675** |

Lessons baked into the pipeline:

- **Size chunks to the model's real token window.** The old model truncated at
  128 tokens, so 1200-char chunks lost ~60% of their text at embed time. Token-
  aware chunking fixed it (+5pp).
- **Don't fake a bigger window.** Forcing the 128-token checkpoint to 512 was the
  *worst* variant — its positions 128–512 were untrained. A larger window only
  helps with a model actually trained for it.
- **The model dominates.** Swapping to a 512-token, multilingual, 768-dim e5
  added **~31pp across every metric** — six times the chunking gain. Recall@10
  went 0.58 → 0.88.
- **Section-as-unit + parent-document retrieval** matches the literature for
  *structured* statutes (whole sections are the legal atom); the contextual
  header and `section_title` are cheap, high-signal additions.

---

## 8. Operations

- **Changing `EMBEDDING_MODEL` requires a full re-ingest** — the vector size
  changes (384 → 768), so the collection must be dropped and rebuilt. Query and
  ingest must use the same model.
- The API reads `EMBEDDING_MODEL` from `.env` at container start; after a model
  swap, **restart the API container** so it embeds queries with the new model and
  matches the rebuilt collection.
- Re-ingest is destructive to the target collection and CPU-bound on the full
  corpus; run it deliberately, not on every deploy.
- To experiment without touching prod, the `eval/` harness ingests into throwaway
  `legal_acts_eval_*` collections and cleans them up with `eval/cleanup.py`:

  ```python
  # eval/cleanup.py — only touches the eval prefix, never production
  PREFIX = "legal_acts_eval_"

  for name in names:
      if name.startswith(PREFIX):
          client.delete_collection(collection_name=name)
          print(f"dropped {name}")
  ```
