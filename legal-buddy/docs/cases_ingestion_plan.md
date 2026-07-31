# Plan: Ingesting `data/cases` (court judgments) into the RAG system

## Context

`data/cases/` holds **8,493 PDF court judgments** (~1.8 GB) from the Supreme Court
of Bangladesh (High Court Division) and lower courts — Civil Revisions, Writ
Petitions, Criminal Appeals, Death References, Tenancy (TN) matters, etc. They are
*precedent / case law*, complementary to the *statutes* already indexed from
`data/acts`. Goal: decide whether they belong in the product, and if so, get them
extracted, structured, embedded, and retrievable alongside the acts.

This plan is grounded in a data analysis (below), not assumptions.

## Data analysis (measured)

| property | finding | implication |
|---|---|---|
| count / size | 8,493 PDFs, 1.8 GB, avg 0.21 MB, 93% < 0.5 MB | manageable; mostly short docs |
| text layer | **~99% born-digital** (pdftotext yields clean text, ~1.5–3k chars/page) | OCR is the *exception* |
| scanned | ~1–3% are image-only (e.g. `303612_Crl_Appl_248_2011.pdf`: 32 pages / 32 chars) | OCR fallback only |
| Bangla | ~2.5% of files carry meaningful Bengali; **pdftotext garbles it** (broken conjuncts/spacing: `গভীর স ে মৎ`) | needs a Bengali-aware path |
| structure | very consistent header: District, Court, Division, Jurisdiction, `Present:` judge, `<Case Type> No. N of YYYY`, `With` (linked cases), Petitioner `-Versus-` Opposite Party, advocates, `Heard on:`, `Judgment on:`, body after `<Judge>, J:` | rich metadata is regex-extractable |
| filenames | encode case type + number + year + party + status (`_SUMMARILY_REJECTED`, `_RS_FIX`) | metadata cross-check |
| outliers | one 11,407-page mega-PDF (`769454_DeathRef58of2013_1.pdf`, 56 MB); multi-part `_1/_2` bundles | cap pages, merge parts |
| tooling | `pdftotext pdfinfo pdfimages tesseract ocrmypdf gs` present; **`tesseract` has only `eng`, no `ben`** | install `tesseract-ocr-ben` |
| existing code | none — greenfield | mirror the acts pipeline |

## Are cases necessary? — Yes, but scoped. And NOT a finetune target.

**Keep them, as a separate corpus.** Acts say *what the law is*; cases show *how
courts apply it* — precedent retrieval ("has this been decided?", "how have courts
ruled on X?", "find similar facts"). That is genuinely additive to a legal
assistant and not derivable from statutes alone.

**Do NOT replace retrieval with fine-tuning.** Fine-tuning bakes *style/format*,
not *factual recall* of 8,493 specific holdings — a fine-tuned model would
hallucinate case numbers and holdings and cannot cite sources. For factual,
citeable case lookup, **RAG is the correct tool**. Fine-tuning has a *narrow*
optional role (see §7), not as a substitute.

**Scope discipline:** ingest a high-value subset first (reported / recent /
non-rejected judgments), prove value, then scale — do not blindly embed all 8,493
on day one (scale note in §6).

---

## Pipeline (mirrors `notebooks/ingest_qdrant.py`, reuses the e5-base + parent-doc work)

### 1. Extraction — hybrid: fast text path + Docling (GPU) for the hard tail
Two tiers, routed by a cheap probe (chars-per-page + Bengali-garble heuristic).
The probe that actually shipped — now run *per page* (Docling was dropped, see
[Update 3](#update-3--page-level-hybrid-final-pipeline)) — is exactly this routing
function:

```python
# apps/ingestion/src/ingestion/cases_hybrid.py
def probe_page(text: str, has_image: bool) -> str:
    """Route a single page from its tier-1 text. See module docstring."""
    stripped = text.strip()
    if len(stripped) < SCANNED_PAGE_CHARS:
        return "scanned" if has_image else "empty"
    if _bengali_ratio(text) >= PAGE_BENGALI_RATIO:
        return "bengali"
    if _nonascii_ratio(text) >= PAGE_NONASCII_RATIO:
        return "garbled"
    return "ok"
```

> The `PdfPipelineOptions` / `EasyOcrOptions` / `AcceleratorDevice` Docling config
> below is the *original* plan and is **not implemented** — the shipped pipeline
> calls EasyOCR directly per page (`cases_hybrid._make_reader` / `_ocr_page`), no
> Docling dependency.

- **Tier 1 — fast path (~95%+, local CPU):** `pdftotext`/`pymupdf` for clean
  born-digital **English** PDFs. Instant, no models. Normalize to Unicode **NFC**.
- **Tier 2 — Docling (the right tool for the hard tail, run on the Fedora GPU):**
  the scanned ~1–3% **and** Bengali-bearing files whose text layer is garbled.
  Docling does layout-aware PDF→Markdown/JSON (reading order, headings, **tables**
  in cause-lists/schedules) and bundles OCR. Config that fits this corpus:
  - `PdfPipelineOptions(do_ocr=True, do_table_structure=True)`
  - `EasyOcrOptions(lang=["en","bn"], force_full_page_ocr=True, use_gpu=True)` —
    EasyOCR covers **Bengali natively**, fixing the garbled-conjunct problem by
    re-OCRing the bitmap instead of trusting the broken text layer (no
    `tesseract-ocr-ben` needed; Tesseract/RapidOCR are alternative backends).
  - `AcceleratorOptions(device=AcceleratorDevice.CUDA)` → runs on the RTX 2070 S.

  Why hybrid, not all-Docling: Docling runs ML layout+OCR per page (slow,
  GPU-bound). Restricting it to the ~5% that need it keeps cost sane while getting
  its quality where layouts are messy/scanned/Bengali. (All-Docling is viable if
  GPU time is cheap — simpler, uniform Markdown + `HybridChunker`, but much slower.)
- **Outliers:** skip/triage over a page cap (e.g. > 200 pages — catches the
  11,407-page mega-PDF); merge `<id>_1/_2` multipart files by base id; log all drops.

### 2. Structuring — header parser → one JSON per case
- Regex/anchor parser over the consistent header to emit a record like
  `data/acts/*.json` but for cases:
  `{ case_id, source_file, court, division, jurisdiction, district, judge(s),
     case_type, case_no, case_year, linked_cases[], petitioners[], respondents[],
     advocates[], heard_dates[], judgment_date, disposition, body_text }`.
- `case_type`/`case_no`/`year`/`status` cross-checked against the filename.
- `disposition` from filename suffixes (`SUMMARILY_REJECTED`, etc.) + body cues
  (allowed/dismissed/disposed of).
- Write to `data/cases_json/<case_id>.json` (intermediate, reviewable, re-runnable).

The shipped record (`parse_case`) is close to the sketch above, with provenance
fields added (`ref_source`, `fn_type`, `fn_part`) and parties kept as raw blocks:

```python
# apps/ingestion/src/ingestion/cases_structure.py  (parse_case return value)
return {
    "case_id": fn["case_id"],
    "seq": fn["seq"],
    "source_file": f"{stem}.pdf",
    "court": _first(_COURT_RE, head),
    "division": _first(_DIVISION_RE, head),
    "jurisdiction": _first(_JURISDICTION_RE, head),
    # ...
    "case_type": case_type,
    "case_no": case_no,
    "case_year": case_year,
    "full_case_ref": full_ref,
    "ref_source": "header" if ref["full_case_ref"] else "filename",
    # ...
    "petitioners_raw": pet,
    "respondents_raw": resp,
    "advocates": _advocates(header),
    "heard_dates": heard,
    "judgment_date": judgment_date,
    "disposition": _disposition(fn["fn_status"], text),
    "body_text": text,
}
```

`_disposition` reads filename status first, then falls back to body cues — exactly
the two sources described above:

```python
# apps/ingestion/src/ingestion/cases_structure.py
def _disposition(fn_status: str | None, body: str) -> str | None:
    if fn_status:
        return fn_status.replace("_", " ")
    tail = body[-1500:].lower()
    for cue, label in (
        ("made absolute", "rule made absolute"),
        # ...
        ("dismissed", "dismissed"),
        ("disposed of", "disposed of"),
        ("is allowed", "allowed"),
        # ...
    ):
        if cue in tail:
            return label
    return None
```

### 3. Chunking — reuse the improved token-aware chunker
- Reuse `improved_section_records` logic from `notebooks/ingest_qdrant.py`
  (token-aware, sized to the model window, contextual header per chunk).
- Judgments are long prose, not sectioned like statutes → split on paragraphs /
  page boundaries into ~model-window chunks with overlap.
- Prepend a **case header** to every chunk (`Case: <type> No N of YYYY | Court |
  Parties`) — the cross-document analogue of the act/section header. As shipped:

  ```python
  # apps/ingestion/src/ingestion/cases_chunking.py
  def case_header(rec: dict) -> str:
      """One-line context header prepended to every chunk of a case."""
      bits = [f"Case: {rec.get('full_case_ref') or rec.get('case_type') or 'Judgment'}"]
      court = rec.get("division") or rec.get("court")
      if court:
          bits.append(court.title() if court.isupper() else court)
      parties = _short_parties(rec)
      if parties:
          bits.append(parties)
      return " | ".join(bits)
  ```

- Add a **synthetic summary chunk** per case (LLM 2–3 line holding summary) for
  high-recall "what did this case decide" queries. _(proposed — not yet
  implemented.)_ `section_full` = the whole case (or whole reasoning section) for
  parent-document retrieval. **Correction (built):** `section_full` was **dropped**
  (it blew Qdrant's 32MB request limit — see [Update 2](#update-2--full-corpus-uncapped-with-bengali-ocr));
  the chunk payload stores only `chunk_text` + `(case_uid, chunk_part)`, and the
  full text lives in `data/cases_json/`:

  ```python
  # apps/ingestion/src/ingestion/cases_chunking.py  (case_records, per chunk)
  records.append({
      "embedding_text": f"{header}{cont}\n{chunk_text}",
      "payload": {
          **base_payload,
          "chunk_part": i,
          "n_parts": n_parts,
          "chunk_text": chunk_text,
      },
  })
  ```

### 4. Embedding + storage — separate collection
- Same model as acts: **`intfloat/multilingual-e5-base`** (768-dim, 512-tok,
  handles Bengali), `passage:`/`query:` prefixes. Reuse `embed_passages`.
- **New Qdrant collection `legal_cases`** (cosine, 768-dim) — keep cases separate
  from `legal_acts_event_rag_full` so retrieval can weight/route them differently.
- Payload: all metadata from §2 + `case_uid` (parent-doc dedupe), `chunk_part`,
  `section_full`. Payload indexes on `case_type`, `case_year`, `court`,
  `disposition`, `case_uid`.

The collection is created exactly with those indexes (`case_year` integer, the rest
keyword) — note `section_full` is *not* part of the payload as built (see §3):

```python
# apps/ingestion/src/ingestion/cases_ingest.py
qdrant.recreate_collection(
    client,
    COLLECTION_NAME,
    vector_size,
    integer_indexes=("case_year",),
    keyword_indexes=("case_type", "court", "disposition", "case_uid"),
)
```

The bulk embed reuses the shared `embed_passages` (same e5 `passage:` prefix as
acts), and `EMBEDDING_DEVICE=cuda` sends it to the Fedora GPU:

```python
# apps/shared/src/shared/embedding.py
def embed_passages(model, texts: list[str], e5: bool, batch_size: int = 32) -> list[list[float]]:
    if not texts:
        return []
    vectors = model.encode(
        [passage_prefix(t, e5) for t in texts],
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return [vec.tolist() for vec in vectors]
```

### 5. Retrieval integration — `apps/api/src/api/agents/legal_chat/retrieval.py`
- Add a parallel `retrieve_cases()` querying `legal_cases` with the same
  parent-document dedupe as `_hits_to_sources`.
- Either (a) **dual-retrieve** acts + cases and merge, tagging each `SourceItem`
  with `source_type` (`statute` | `precedent`), or (b) a light **router** (LLM/rules)
  that decides statute-only vs precedent vs both from the question.
- Update `build_grounded_prompt` so the model distinguishes binding statute text
  from persuasive/precedent case holdings, and cites case numbers.

**Built (option a, dual-retrieve).** `retrieve_cases()` shipped — it uses Qdrant
*grouping* on `case_uid` (instead of `_hits_to_sources`-style flat dedupe) so one
long judgment can't flood the candidate pool:

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

`retrieve_dual` embeds the query once, queries both collections, and applies a
per-collection score floor. The shipped policy differs from the plan: **only
statutes are user-facing citable sources**; precedents are reasoning-only
background (no citation ids):

```python
# apps/api/src/api/agents/legal_chat/retrieval.py
def retrieve_dual(
    question: str, *, statute_k: int, case_k: int
) -> tuple[list[SourceItem], list[SourceItem]]:
    # ...
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

`SourceItem.source_type` is the `Literal` tag the plan called for:

```python
# apps/api/src/api/api/models.py
source_type: Literal["statute", "precedent"] = "statute"
```

`build_grounded_prompt(question, statutes, precedents, history)` takes both lists
and, contra the plan's "cites case numbers", instructs the model to fold precedent
reasoning in **without** citing any case number/name/date (precedent is background
only).

---

## 6. Scale & cost (the real risk)

8,493 judgments × ~6 pages × ~3 chunks/page ≈ **150k–250k chunks** — 4–6× the acts
corpus (40k). CPU embedding at the observed e5-base rate is likely **several hours
to a day**. Mitigations, in order:
1. **Subset first** — reported/recent/non-rejected (cuts volume a lot), prove value.
2. **GPU or managed embeddings** for the full run (batch job).
3. **Blue/green collection** (`legal_cases_v2` → flip) so re-ingest never drops a
   live collection (the outage pattern noted in `eval/REPORT.md`).

Extraction itself is cheap (pdftotext/pymupdf on 99% text PDFs = minutes–an hour);
OCR on the scanned ~1–3% is the slow extraction tail.

## 7. Fine-tuning — the narrow, optional role (not a data store)

Cases can *generate training data*, but for **style/format, not facts**:
- Build instruction pairs (question → grounded answer-with-citation) from
  case+holding to fine-tune **answer style** (tone, structure, citation format).
- Or train a **small Bengali-aware extraction/rerank model**. A cross-encoder
  **reranker** over retrieved cases is higher ROI than generator fine-tuning.
- Keep facts in the vector store; never rely on a fine-tuned model to recall a
  specific case number or holding.

---

## Verification

1. **Extraction QA**: run on a 200-file stratified sample (text / scanned / Bangla);
   assert chars-per-page, Bengali NFC validity, and metadata-parse success rate
   (target > 95% header fields on born-digital English).
2. **Retrieval eval**: reuse the `eval/` harness pattern — generate ~100 Q→case
   gold pairs (Groq/Gemini), ingest the subset into a throwaway `legal_cases_eval`
   collection, measure recall@k / MRR, then clean up (`eval/cleanup.py` pattern).
3. **End-to-end**: a precedent question through `/rag/legal/chat` returns case
   sources with correct case numbers + a holding-grounded answer; a pure-statute
   question still returns acts (router/merge sanity).
4. **Bengali spot-check**: confirm a known Bengali judgment is extracted cleanly
   (NFC, no broken conjuncts) and retrievable by a Bengali query.

## Suggested phasing
1. Extraction + structuring → `data/cases_json/` (incl. OCR fallback, `ben` install).
2. Eval harness + ~100 gold pairs on a subset → measure before full ingest.
3. Subset ingest into `legal_cases`, wire `retrieve_cases()` + dual-retrieve.
4. Scale to full corpus (GPU/managed embeddings, blue/green) once the subset proves out.
5. Optional: reranker; optional: style fine-tune from case Q→A pairs.

## New deps / system
- `uv add pymupdf` (fast-path extraction); reuse `sentence-transformers`, `qdrant-client`.
- `uv add docling` for the hard tail (scanned + garbled-Bengali) — bundles EasyOCR
  (Bengali) and runs on the **Fedora RTX 2070 S** via `AcceleratorDevice.CUDA`.
  Optionally `docling-core`'s `HybridChunker` for structure-aware case chunking.
- Fedora is bare (Python 3.14, no torch/uv): install `uv`, pin **Python ≤ 3.13**
  for the embed/Docling env (no torch wheels for 3.14), then `torch` (CUDA) +
  `docling` + `sentence-transformers`.
- `tesseract-ocr-ben` only if using the Tesseract backend instead of EasyOCR.

---

## Implementation addendum (review + phase 1 built)

> **Historical — superseded by [Update 3](#update-3--page-level-hybrid-final-pipeline).**
> The two-tier split (separate tier-1 build + tier-2 OCR of a hard-tail file list)
> described here was replaced by a single **page-level hybrid** pass. Kept for the
> measurement/decision history; the module/command names below are out of date.

Phase 1 (extraction + structuring) is implemented and run on the full corpus.
Notes below correct the plan where measurement disagreed with assumption.

### Plan corrections
- **Stale code references.** The plan points at `notebooks/ingest_qdrant.py` /
  `improved_section_records`; that logic was refactored into `apps/shared`
  (`shared.chunking.section_records`, `collect_all_records`) and `apps/ingestion`.
  The cases pipeline mirrors *those*, not the notebook.
- **The Bengali problem is mojibake, not broken conjuncts.** The dominant Bengali
  tail (Tenancy / 561-A judgments) is typed in a legacy **non-Unicode font**, so
  PyMuPDF/pdftotext decode it as Latin-1 symbol soup (`q¡C−L¡VÑ ®g±Sc¡l£`) that
  carries **zero** Bengali codepoints — a Unicode-block ratio check misses it
  entirely and passes it as clean English. The reliable signal is a high
  **non-ASCII ratio** (≥10%); these route to tier-2 OCR like the scanned files.
- **`case_id` cannot be the filename's numeric prefix.** 2,523 files share the
  placeholder prefix `1` (e.g. `1_TN_18218_2024_R2W`), so keying on it collapsed
  ~2,400 distinct judgments into one. The unique id is the **full filename stem**;
  the prefix is kept as `seq` only.
- **Citations come from filename + header.** The header `Type No. N of YYYY` is
  authoritative when present (~73%); otherwise the citation is synthesized from
  the filename (which independently encodes type/no/year), lifting coverage to
  ~85%. `court`/`division`/`jurisdiction` are genuinely absent in ~half of
  judgments (many open straight at `Present:`), so they stay best-effort.

### Measured tier distribution (all 8,493 PDFs, ~150s CPU)
| tier | files | % | path |
|---|---|---|---|
| ok (born-digital English) | 8,049 | 94.8% | tier-1 fast (CPU, done) |
| garbled (legacy-font Bengali) | 343 | 4.0% | tier-2 Docling+OCR (Fedora) |
| bengali (Unicode) | 78 | 0.9% | tier-2 Docling (Fedora) |
| empty / scanned | 23 | 0.3% | tier-2 / triage |

Hard tail = **444 (5.2%)** — matches the plan's estimate; trivial Docling load on
the 2070. 12 files exceed the 200-page cap (mega death-reference bundles).

### What is built
- `ingestion/cases_extract.py` — tier-1 PyMuPDF extraction + NFC + the routing
  probe (chars/page, non-ASCII ratio, Unicode-Bengali ratio, page cap).
- `ingestion/cases_structure.py` — header parser → record (court, division,
  jurisdiction, judges, `case_type/no/year`, `full_case_ref`, parties, advocates,
  dates, disposition, lossless `body_text`), cross-checked against the filename.
- `ingestion/cases_build.py` → `uv run cases-build` → `data/cases_json/<stem>.json`
  (**8,049 records written**) + `_hard_tail.json` (444) + `_manifest.json`.
- `ingestion/cases_chunking.py` — paragraph/token chunker (reuses the shared token
  primitives), per-chunk case header, `section_full` parent doc, `source_type=
  precedent`. ~18 chunks/case → ~142k chunks corpus-wide.
- `ingestion/cases_ingest.py` → `uv run cases-ingest` → Qdrant **`legal_cases`**
  (cosine 768-d, indexes on `case_type/case_year/court/disposition/case_uid`).
- `shared/embedding.py` — `load_embedding_model(..., device=...)`; set
  `EMBEDDING_DEVICE=cuda` to embed on the Fedora RTX 2070.

### Remaining
1. **Retrieval wiring** — `retrieve_cases()` + dual-retrieve/router in
   `apps/api/.../legal_chat/retrieval.py`, `source_type` on `SourceItem`
   (see `docs/situational_rag_plan.md`).
2. **Eval** — ~100 Q→case gold pairs, throwaway `legal_cases_eval`, recall@k/MRR.
3. Optional synthetic holding-summary chunk (LLM) and cross-encoder reranker.

---

## Update 2 — full corpus, uncapped, with Bengali OCR

> **Historical — superseded by [Update 3](#update-3--page-level-hybrid-final-pipeline).**
> `section_full`-dropped and no-page-cap still hold; the separate `cases-ocr`
> hard-tail stage and the AdarshaLipi-specific framing are replaced by the unified
> page-level hybrid.

Three corrections after running at scale:

- **No page cap.** The long bundles (multi-thousand-page death-reference paper
  books) carry dense record/evidence content worth retrieving, so extraction now
  reads **all pages** (`DEFAULT_PAGE_CAP = None`). The 11,407-page judgment yields
  ~13.2M chars → ~6.6k chunks. OCR of the giant *garbled* bundle is bounded
  (`CASES_OCR_MAX_PAGES=600`: judgment is at the front, the tail is evidence).
- **`section_full` dropped from the chunk payload.** Storing the whole judgment in
  every chunk blew Qdrant's 32MB request limit (a 64-chunk batch of long judgments
  = 53MB) and bloated storage ~10×. The retrievable unit is the chunk + metadata;
  neighbours are fetched by `(case_uid, chunk_part)`; `data/cases_json` keeps full
  text. ~8,049 clean cases → ~79k chunks.
- **Bengali is OCR'd, not excluded.** The garbled tail is the legacy **AdarshaLipi**
  font (no reliable byte→Unicode map), but the pages *render* correctly, so
  `ingestion/cases_ocr.py` (`uv run cases-ocr`) rasterizes each page and runs
  **EasyOCR `["en","bn"]`** on the bitmap → real Unicode Bengali (verified: the TN
  mojibake `q¡C−L¡VÑ` recovers as `হাইকোট … হাইকোট বিভাগ …`). English judge/party
  lines survive OCR, so the parser still gets partial metadata; filename gives
  case_no/year. EasyOCR runs on the Fedora 2070 (`torchvision==0.21.0+cu124`).
- **Incremental append.** `cases-ingest` with `CASES_INGEST_FILES=<ids>` embeds and
  upserts only the listed cases (delete-by-`case_uid` then upsert, no recreate), so
  the OCR'd hard tail joins `legal_cases` without re-embedding the 8k clean cases:

  ```python
  # apps/ingestion/src/ingestion/cases_ingest.py  (ingest, append mode)
  if append:
      client.delete(
          collection_name=COLLECTION_NAME,
          points_selector=models.FilterSelector(filter=models.Filter(must=[
              models.FieldCondition(key="case_uid",
                                    match=models.MatchAny(any=sorted(only_ids)))
          ])),
      )
  ```

Run order on Fedora (chained, GPU): full embed → `cases-ocr` (444) →
`cases-ingest` append. End state: all 8,493 judgments in `legal_cases`.

---

## Update 3 — page-level hybrid (final pipeline)

The two-tier split (tier-1 build + a separate OCR pass over a hard-tail *file*
list) was replaced by **one page-level hybrid pass**. The earlier design judged a
whole file English-or-not; but many English judgments quote Bengali statute/
lower-court text on *some* pages, and whole-file OCR both degraded the clean
English (`Shaishir` → `Slaishir`) and burned GPU on English pages. Page-level
routing fixes both.

### Decision rule (per page), validated by ablation

```
tier-1 PyMuPDF text per page  (cheap, CPU)        →  probe the page
  clean English            → keep tier-1           (no GPU; ~most pages)
  Bengali (real or scrambled-codepoint)  → OCR     (EasyOCR en+bn on the bitmap)
  legacy-font mojibake (q¡C-L Latin soup) → OCR
  scanned / no text layer  → OCR
  blank                    → drop
```

That rule is the per-page dispatch inside `extract_document`'s page loop:

```python
# apps/ingestion/src/ingestion/cases_hybrid.py  (extract_document, per page)
route = probe_page(t, has_image)

if route == "ok":
    parts.append(tidy_text(unicodedata.normalize("NFC", t)))
    n_tier1 += 1
elif route == "empty":
    n_empty += 1
    # nothing to append
else:  # garbled | bengali | scanned -> OCR
    if max_ocr_pages is not None and ocr_used >= max_ocr_pages:
        capped = True
        # ...
    otext = _ocr_page(page, get_reader(), dpi)
    ocr_used += 1
    if otext.strip():
        parts.append(tidy_text(unicodedata.normalize("NFC", otext)))
        n_ocr += 1
```

Why OCR, not a font map: the corpus mixes **two** legacy failure modes — mojibake
Latin (`q¡C−L¡VÑ`) *and* scrambled Bengali codepoints (`আমভ স` for `আমি সৌ…`,
where pdftotext emits Bengali-block characters in the wrong order). A byte→Unicode
map fixes only the first and only per-font (AdarshaLipi ≠ LipiChameli ≠ Bijoy).
OCR reads the **rendered bitmap**, so it is font-agnostic and generalizes to any
future PDF. Cost is paid only on the pages that need it.

### Text quality

- **Paragraph reflow (tier-1).** PyMuPDF often emits one block per visual line.
  `cases_hybrid._page_text` re-wraps by vertical gap: consecutive lines join into a
  paragraph; a new paragraph starts only where the gap exceeds the page's typical
  line leading. Result: **a linebreak marks a new paragraph, never a wrap.**
- **OCR paragraphs.** EasyOCR `paragraph=True`; paragraphs joined with a blank line.
- **`tidy_text`** collapses intra-paragraph wraps to spaces and blank-line runs to
  one — no per-visual-line newlines:

  ```python
  # apps/ingestion/src/ingestion/cases_hybrid.py
  def tidy_text(text: str) -> str:
      paras = re.split(r"\n\s*\n+", text)            # blank line(s) = paragraph break
      cleaned = [re.sub(r"\s+", " ", p).strip() for p in paras]   # wraps -> spaces
      return "\n\n".join(p for p in cleaned if p)
  ```
- **Header metadata is parsed from line-structured text, not the reflowed body.**
  Reflow collapses the line anchors the header regexes need, so `extract_document`
  also returns `header_text` (raw line text of the head pages) and
  `parse_case(stem, body, header_text=…)` parses judges/parties/advocates/ref from
  it while storing the reflowed prose as `body_text`.
- **Case ref is filename-authoritative.** `_case_ref` trusts a header
  `Type No. N of YYYY` only when its *number* matches the filename's; otherwise it
  synthesizes from the filename (the court's catalog id). This stops Bengali/OCR'd
  judgments from picking up a *cited* lower-court reference as their own:

  ```python
  # apps/ingestion/src/ingestion/cases_structure.py  (_case_ref)
  if fn_no is not None:
      # Number is the discriminator: a header ref whose NUMBER matches the
      # filename is this case ... A number that doesn't match is a cited/garbage
      # ref -> reject and let the caller synthesize from the filename.
      for cand in matches:
          if cand[1] == fn_no:
              ctype, no, year = cand
              return {"case_type": ctype, "case_no": no, "case_year": year,
                      "full_case_ref": f"{ctype} No. {no} of {year}"}
      return none
  ```

### Modules (final)

| module | role | command |
|---|---|---|
| `ingestion/cases_hybrid.py` | page-level extraction (`extract_document`, `_page_text` reflow, `tidy_text`, lazy `_make_reader`) | — |
| `ingestion/cases_structure.py` | header parser → record (`parse_case`, filename-authoritative ref) | — |
| `ingestion/cases_build.py` | driver: `data/cases/*.pdf` → `data/cases_json/<stem>.json` | `uv run cases-build` |
| `ingestion/cases_chunking.py` | paragraph/token chunker, per-chunk case header, `source_type=precedent` | — |
| `ingestion/cases_ingest.py` | embed (e5-base) → Qdrant `legal_cases` | `uv run cases-ingest` |

`cases-build` env: `EMBEDDING_DEVICE=cuda` (GPU OCR), `CASES_HYBRID_RESUME=1`
(skip already-written, crash-safe), `CASES_HYBRID_MAX_OCR_PAGES` (per-file OCR
budget, default 400), `CASES_LIMIT`. Must run on the Fedora GPU box (OCR inline);
English-only files never load EasyOCR (reader built lazily on the first OCR page):

```python
# apps/ingestion/src/ingestion/cases_build.py  (build)
_reader = {}
def get_reader():
    if "r" not in _reader:
        from ingestion.cases_hybrid import _make_reader
        print("[cases-build] building EasyOCR reader (first OCR page)", flush=True)
        _reader["r"] = _make_reader(["en", "bn"], gpu)
    return _reader["r"]
```

`_make_reader` imports `easyocr` lazily (it is not a declared dependency — it is
only present on the GPU box), so a pure-English corpus slice never touches torch:

```python
# apps/ingestion/src/ingestion/cases_hybrid.py
def _make_reader(langs: list[str], gpu: bool):
    import easyocr
    return easyocr.Reader(langs, gpu=gpu)
```

Per-record provenance: `extraction_tier="hybrid"`, `lang` (english/mixed/bengali),
`route_counts` (per-page tally), `pages_tier1`, `pages_ocr`, `pages_empty`,
`bengali_ratio`, `ocr_capped`.

The old file-level modules — `cases_extract.py` (file-level routing), the tier-1
`cases_build.py`, and `cases_ocr.py` (whole-file OCR) — are **removed**; their only
still-needed piece, the EasyOCR reader factory, moved to `cases_hybrid._make_reader`,
and the page-level driver took the freed `cases_build.py` name. The ablation harness
(`cases_ablation.py`, `cases_hybrid_ablation.py`, with saved results under
`data/ablation/`) is preserved on the **`cases-ablation` branch**, out of the
production package.

### Status (2026-06-23 — complete)

- **8,493 / 8,493 PDFs** extracted into `data/cases_json/` (page-level hybrid;
  manifest `_build_manifest.json`): 1,417 used OCR (745 mixed + 375 bengali + 297
  english pages), 9,091 OCR pages, 0 failures, 2 OCR-capped. 158 files synthesize
  the case ref from the filename (no corroborating header ref).
- **`legal_cases` fully re-ingested** from the complete corpus (full recreate, not
  append) — the prior garbled-mojibake Bengali tail is gone, replaced by readable
  OCR text. e5-base, 768-dim, `source_type=precedent`.
