# Plan: Situational legal RAG — answer a user's real situation with statute + precedent

## The core reframe: retrieve, don't "learn"

The model does **not** learn the judgments/statutes by training on them. Training
(fine-tuning) bakes *style and format*, not *factual recall* — a fine-tuned model
invents case numbers and section text and cannot cite a source. Every fact and
citation in an answer must come from the **vector store at query time (RAG)**.
"Make the AI learn from the cases" therefore means: make the corpus retrievable,
and make the model reason over what is retrieved. Fine-tuning has only a narrow,
optional role (style / issue-spotting / reranking — see §6).

## What exists today (statute-only, single hop)

> Note: this section describes the original statute-only baseline. Per
> `docs/cases_retrieval_quality_plan.md`, the live pipeline has since moved past it
> and now dual-retrieves statutes + precedents — the per-stage snippets below show
> the *current* code where it differs from this baseline.

`apps/api/src/api/agents/legal_chat/`:

- `pipeline.py` — `legal_chat_pipeline(question)` → `retrieve_sources` →
  `build_grounded_prompt` → `run_llm` → `LegalChatResponse`.
- `retrieval.py` — embeds the **raw question** (e5), queries one collection
  (`config.QDRANT_COLLECTION` = acts), `_hits_to_sources` collapses chunks to
  unique sections by `section_uid` (parent-document retrieval).
- `prompting.py` — `build_grounded_prompt`: statute-only system prompt, cite
  `[Source N]`.

`retrieve_sources` is the single-collection path, and `_hits_to_sources` dedupes
chunks to unique sections by `section_uid`:

```python
# apps/api/src/api/agents/legal_chat/retrieval.py
def retrieve_sources(
    question: str, top_k: int, *, vector: list[float] | None = None
) -> list[SourceItem]:
    """Retrieve statute sections from the acts collection."""
    candidate_limit = max(top_k * CANDIDATE_MULTIPLIER, MIN_CANDIDATES)
    traced = get_langsmith_client() is not None
    if vector is None:
        vector = _embed(question, traced)
    hits = _search(config.QDRANT_COLLECTION, vector, candidate_limit, traced)
    return _hits_to_sources(hits, top_k)
```

```python
# apps/api/src/api/agents/legal_chat/retrieval.py  (_hits_to_sources)
uid = payload.get("section_uid") or (
    f"{payload.get('source_url')}#{payload.get('section_index')}"
)
if uid in seen:
    continue
seen.add(uid)
```

### Why it cannot yet handle a *situational* question
1. **Vocabulary gap.** It embeds the user's lay narrative directly. "My landlord
   evicted me without notice" shares almost no tokens with statutory language
   ("ejectment", "notice to quit", "section 106") → weak or empty retrieval.
2. **Statute only.** No precedent corpus in the loop. `_hits_to_sources` is
   hard-keyed to acts payload (`section_uid`, `act_title`); cases use
   `case_uid` / `full_case_ref` and need a parallel path.
3. **No issue-spotting, no rerank, no binding-vs-persuasive distinction.**

## Target pipeline (extend, don't replace)

```
question
  └─ Stage 0  Situation understanding (LLM)  -> issues, domain, relief, expanded queries (HyDE)
  └─ Stage 1  Dual retrieve                  -> legal_acts + legal_cases, tagged source_type
  └─ Stage 2  Rerank (cross-encoder)         -> best statute + precedent for THIS situation
  └─ Stage 3  Grounded generation            -> binding statute + persuasive precedent, cited
  └─ Stage 4  Guardrails                      -> abstain on low score, disclaimer, scope
```

### Stage 0 — Situation understanding (new; biggest quality lever)
An LLM pre-pass converts the messy narrative into structure **and better search
queries**:

```
user: "landlord threw me out, no notice, rent was paid"
  -> issues:  [unlawful eviction, tenancy notice]
  -> domain:  tenancy / property      relief: restoration, damages
  -> queries (legal vocabulary + HyDE):
       "ejectment of tenant without statutory notice to quit"
       "tenant's right to restoration of possession Bangladesh"
```

HyDE (Hypothetical Document Embeddings): have the LLM draft a *hypothetical
statute paragraph / how a court would frame it*, and embed **that** instead of
the lay text. This is what closes the vocabulary gap.

- New module `query_understanding.py`; runs before retrieval.
- Output is structured (reuse the project's structured-output approach).

Status: **not yet implemented** (deferred — see `docs/cases_retrieval_quality_plan.md`
§(d): HyDE adds a second per-query LLM call the Gemini free tier can't sustain).
There is no `query_understanding.py` and no HyDE in the repo today. The
"structured-output approach" it would reuse is the `instructor` stack already used
for the answer in `generation.py` against a Pydantic model:

```python
# apps/api/src/api/agents/legal_chat/generation.py  (Gemini path)
structured_client = instructor.from_genai(
    client, model=model, mode=instructor.Mode.GENAI_STRUCTURED_OUTPUTS
)
# ...
structured_answer = structured_client.create(
    response_model=StructuredLegalAnswer,
    messages=structured_messages,
    config=_gemini_config(temperature, max_tokens),
)
```

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

(There *is* a history-aware query rewrite — `condense_question` in
`contextualize.py` — but it only resolves follow-up references against the
conversation; it does not do issue-spotting or HyDE expansion.)

### Stage 1 — Dual retrieve (extend `retrieval.py`)
- Embed the expanded query; query **`legal_acts` AND `legal_cases`** in parallel.
- Tag every `SourceItem` with `source_type` = `statute | precedent`.
- Parent-document dedupe **per corpus**: acts by `section_uid`, cases by
  `case_uid`. Add `retrieve_cases()` mirroring `_hits_to_sources` against the
  cases payload (`full_case_ref`, `division`, `disposition`, `chunk_text`).
- Optional payload filters from Stage 0: cases by `case_type` / `case_year`,
  acts by domain. (Indexes already built on both collections.)
- Either merge both result sets, or add a light **router** (rules/LLM) that
  decides statute-only vs precedent vs both from the question.

Status: **built.** `retrieve_cases()` uses Qdrant `query_points_groups` on
`case_uid` so each result is a distinct case (a long judgment would otherwise flood
a flat top-N pool):

```python
# apps/api/src/api/agents/legal_chat/retrieval.py  (retrieve_cases)
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

`retrieve_dual()` embeds once, queries both collections, drops sub-floor hits, and
renumbers — but note it diverges from the plan: **only statutes are user-facing
citable sources; precedents are reasoning-only background** (see Stage 3):

```python
# apps/api/src/api/agents/legal_chat/retrieval.py
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

The optional Stage-0 payload filters and the LLM/rule router are **not yet
implemented** — `retrieve_dual` always queries both collections and merges by
score floor.

### Stage 2 — Rerank (new; high ROI)
Cross-encoder reranks the merged statute+precedent pool against the user's actual
situation text. Higher ROI than generator fine-tuning (cf. cases plan §7). New
`rerank.py`; gate behind a config flag so it can be A/B'd.

Status: **not yet implemented.** There is no `rerank.py` and no `CrossEncoder` in
the repo. It is only foreshadowed in the `CASE_SCORE_FLOOR` comment, which calls it
the robust fix for the thin off-topic margin:

```python
# apps/api/src/api/core/config.py
# STATUTE floor stays 0.0 (don't touch the working acts path). CASE floor 0.82
# is empirical for multilingual-e5-base: off-topic scenarios top out ~0.80-0.81,
# on-point precedents score 0.83-0.86 (thin margin — the Phase-2 reranker is the
# robust fix; this is a coarse off-topic cut for MV).
```

### Stage 3 — Grounded generation (extend `prompting.py`)
Prompt must separate **binding statute** from **persuasive precedent** and shape
the answer:
1. The situation, restated in legal terms.
2. Governing **statute + section number** (binding).
3. How courts have applied it — **case number + holding** (persuasive precedent).
4. Practical steps / relief available.
5. Caveats + "this is general information, not legal advice; consult a lawyer."

Cite section numbers **and** case numbers, only from retrieved sources. Keep the
existing "synthesize across ALL relevant sources / do not fabricate" discipline.

Status: **built, but stricter than this plan.** The implemented prompt separates
binding statute from precedent — but precedents are folded in as **reasoning-only
background that must NOT be cited or named** (no case number/name/date), not as a
citable "persuasive precedent". So statutes are the only `[Source N]`:

```python
# apps/api/src/api/agents/legal_chat/prompting.py  (system prompt)
"1. STATUTE SOURCES — the BINDING law and your ONLY citable sources. "
"Identify the governing act and section number(s), state the rule, and "
"cite each one you rely on as [Source N].\n"
"2. PRECEDENT BACKGROUND — how courts have reasoned about similar facts. "
"This is for YOUR REASONING ONLY. Use it to judge the likely outcome and to "
"reason like a court would, but DO NOT cite it, DO NOT present it as a "
"source, and DO NOT mention, quote, or refer to any specific case, case "
"number, party name, judge, or date. "
# ... (continues: never write 'in a previous case'; fold reasoning into your own)
```

Correspondingly, the precedent block is rendered without any case reference —
only outcome + reasoning:

```python
# apps/api/src/api/agents/legal_chat/prompting.py  (_format_precedent_background)
return "\n".join(
    [
        "- Court treatment of similar facts:",
        f"  Outcome: {source.disposition or 'Unknown'}",
        f"  Reasoning: {source.excerpt}",
    ]
)
```

### Stage 4 — Guardrails (extend `pipeline.py`)
- Jurisdiction = Bangladesh; flag when a question is outside the corpus.
- **Abstain when the top retrieval score is low** — return "insufficient
  sources" rather than a confident wrong answer (the empty-sources branch in
  `pipeline.py` is the seed of this; extend to a score threshold).
- Standard legal disclaimer on every answer.

Status: **partially built.** The empty-sources abstention exists, and the per-corpus
score floors (`STATUTE_SCORE_FLOOR` / `CASE_SCORE_FLOOR`) already drop sub-floor hits
inside `retrieve_dual` (above), so a corpus whose best hit is below floor contributes
nothing and can trigger abstention:

```python
# apps/api/src/api/agents/legal_chat/pipeline.py
statutes, precedents = retrieve_dual(
    search_query, statute_k=resolved_top_k, case_k=config.CASES_TOP_K
)
if not statutes and not precedents:
    return LegalChatResponse(answer=ABSTENTION_TEXT, sources=[])
```

The legal disclaimer is in the system prompt ("End with a brief note that this is
general legal information, not legal advice, and a lawyer should be consulted").

Not yet built: an explicit **post-rerank score-threshold** abstention (depends on
Stage 2) and an out-of-corpus / out-of-jurisdiction flag.

## File-by-file change map

| stage | file | change |
|---|---|---|
| 0 | `query_understanding.py` (new) | LLM issue-spot + HyDE query expansion |
| 1 | `retrieval.py` | `retrieve_cases()`, dual-retrieve + merge/router, `source_type` |
| 1 | `api/api/models.py` | add `source_type` (and case fields) to `SourceItem` |
| 2 | `rerank.py` (new) | cross-encoder rerank, config-flagged |
| 3 | `prompting.py` | statute-vs-precedent prompt, cite case numbers |
| 4 | `pipeline.py` | score-threshold abstention, disclaimer, wire stages |

The `SourceItem` change (row 1) is **built** — `source_type` defaults to `statute`
so the acts path is unchanged, and the case-law fields hang off the same model:

```python
# apps/api/src/api/api/models.py
class SourceItem(BaseModel):
    citation_id: int
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
    excerpt: str
    score: float
```

The new config knobs (row "1" of `config.py`) are also live:

```python
# apps/api/src/api/core/config.py
QDRANT_COLLECTION: str = "legal_acts_event_rag_full"
CASES_COLLECTION: str = "legal_cases"
# ...
RETRIEVAL_TOP_K: int = 6
CASES_TOP_K: int = 4
STATUTE_SCORE_FLOOR: float = 0.0
CASE_SCORE_FLOOR: float = 0.82
```

## Suggested build order
1. **Stage 1 dual-retrieve** — smallest change; precedent immediately shows up in
   answers alongside statutes. (Depends on `legal_cases` being populated.)
2. **Stage 0 situation understanding** — biggest jump for lay/situational questions.
3. **Stage 3 prompt** — make answers cite cases + separate binding/persuasive.
4. **Stage 2 rerank** + **Stage 4 guardrails** — precision + safety.
5. Eval at each step (reuse `eval/`: gold Q→{section, case} pairs, recall@k / MRR,
   plus a small situational-question set graded for correct statute + case).

The `eval/` harness exists today but is **statute-only**: it grades a gold
`gold_key` (a section key) at chunk- and section-level granularity. The gold
Q→{section, **case**} pairs and the situational set are not yet in the harness:

```python
# eval/run_eval.py  (per-query scoring against the gold section key)
chunk_keys = [str((h.payload or {}).get("gold_key")) for h in hits]
section_keys = dedupe_preserve(chunk_keys)
target = g["gold_key"]

c_rec = recall_at_k(chunk_keys, target, KS)
s_rec = recall_at_k(section_keys, target, KS)
```

## §6 Fine-tuning — the narrow role (not a fact store)
- Style / structure / citation format, or an **issue-spotting** model, trained on
  case Q→A pairs.
- A **cross-encoder reranker** is the highest-ROI training target.
- Never rely on a fine-tuned model to recall a specific section or holding — facts
  stay in Qdrant.

## Prerequisite (in progress)
Both corpora populated in Qdrant: `legal_acts_event_rag_full` (done) and
`legal_cases` (cases — see `docs/cases_ingestion_plan.md`; full clean-corpus embed
in progress, 444-file Docling hard tail pending).
