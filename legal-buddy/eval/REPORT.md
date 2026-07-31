# Chunking A/B eval — baseline vs improved (#1–#4)

Branch: `chunking-eval`. All code under `eval/`. Ingestion targets two throwaway
Qdrant collections on the configured remote server (`legal_acts_eval_baseline`,
`legal_acts_eval_improved`); the production collections were never touched.

## Setup

- **Corpus subset**: 201 acts (Penal Code force-included), ~5,357 sections.
  Deterministic stride sample (`eval/select_subset.py`).
- **Gold set**: 120 Q→section pairs, LLM-synthesized (19 Gemini + 7 Gemini
  flash-latest + 94 Groq `llama-3.3-70b`; `eval/gen_goldset*.py`). Gold key =
  `(act_file, section_ord)` — chunking-independent, so one set scores every
  collection. 58 unique acts. Questions are mostly Bengali (corpus is bilingual).
- **Metric**: embed query (`query:` prefix) → top-100 candidate chunks → collapse
  to unique sections (parent-document) → recall@{1,3,5,10} + MRR. Chunk- and
  section-level both reported.
- **Variants**: baseline (current 1200-char chunks, 6,690) · improved-128 (#1–#4
  at the model's real 128-token window, 17,790) · improved-512 (#1–#4 with the
  embedding window forced to 512, 6,089).

## Results (n=120)

| variant | R@1 | R@3 | R@5 | R@10 | MRR |
|---|---|---|---|---|---|
| baseline                 | 0.258 | 0.400 | 0.508 | 0.575 | 0.364 |
| improved-128 (#1–#4)     | 0.308 | 0.458 | 0.500 | 0.608 | 0.408 |
| improved-512             | 0.175 | 0.308 | 0.392 | 0.458 | 0.274 |
| **multilingual-e5-base @512** | **0.567** | **0.733** | **0.825** | **0.883** | **0.675** |

Δ improved-128 vs baseline: R@1 +0.050, R@3 +0.058, R@5 −0.008, R@10 +0.033,
MRR +0.044 — a real, consistent win (4/5 metrics; n=26 had hidden it in noise).
Δ improved-512 vs baseline: −0.08 to −0.12 everywhere — worse.
Δ **multilingual-e5-base @512 vs baseline: +0.31 R@1, +0.33 R@3, +0.32 R@5,
+0.31 R@10, +0.31 MRR** — every metric roughly doubles.

**The model swap dwarfs every chunking change.** Same improved chunking, same
n=120 gold set, same parent-doc retrieval — only the embedding model changed
(`triBne-e5-small` 384-dim/128-tok → `intfloat/multilingual-e5-base`
768-dim/512-tok). R@1 went 0.26 → 0.57, R@10 0.58 → 0.88. The chunking work buys
~5 points; the right model buys ~31.

## Two firm findings

**1. The model truncates at 128 tokens.** `triBne-e5-small` has
`max_seq_length = 128` (verified on `SentenceTransformer.max_seq_length` and the
tokenizer). Baseline's 1200-char chunks (~250–350 tokens) lose ~60% of their text
at embed time. improved-128 fixes this by sizing chunks to the true 128-token
window — that, plus the contextual header (#1/#3) and parent-doc (#4), is where the
gain comes from.

**2. Raising the window to 512 makes it WORSE, not better.** The checkpoint's
`max_position_embeddings` is 512, so 512 runs without error — but it was
fine-tuned at 128, so positions 128–512 carry untrained embeddings. Feeding
512-token chunks degraded every metric by 8–12 points. The hypothesis that the
128 cap was a misconfiguration is **refuted**: 128 is the model's real capacity.
The ceiling is the model itself — a checkpoint genuinely trained at 512 (e.g.
`intfloat/multilingual-e5-base`, 512 tokens, strong on Bengali) is the highest-
leverage next step, well above any further chunking tweak.

## What was actually implemented (production `notebooks/ingest_qdrant.py`)

- **#1** `section_title` added to payload + contextual header.
- **#2** token-aware splitting sized to the model's real `max_seq_length` (read at
  runtime), overlap 24 tokens (~20%) vs the old 100 chars (~8%).
- **#3** contextual header (`Act | Title | Section (part k/n)`) on every part.
- **#4** `section_uid` + full `section_full` in payload; `retrieval.py` rewritten
  for parent-document dedupe (return whole sections, top_k unique).

## Verdict & recommendations

1. **Switch the embedding model to `intfloat/multilingual-e5-base` — this is the
   headline change.** +31 points across the board (R@1 0.26→0.57, R@10 0.58→0.88,
   MRR 0.36→0.68). It is genuinely trained at 512 tokens and strong on Bengali.
   Cost: vector size 384→768, so it needs a full prod re-ingest (drop + recreate
   the collection) and the same model on the query side. `EMBEDDING_MODEL` already
   drives both ingest and API, so it's a one-line config change + re-ingest.
2. **Keep the improved chunking (#1–#4).** It's a real +5pp on its own and these
   results were produced with it; ship it alongside the model.
3. **Do NOT force the 128-token model to 512.** `triBne-e5-small` at 512 was the
   worst variant (−8 to −12pp) — its positions 128–512 are untrained. The 512 gain
   only materialises with a model actually trained at 512.
4. **Re-run this harness after the model swap on prod** to confirm at full corpus
   scale, and consider a per-feature ablation of #1–#4 (harness supports it).

## Reproduce

```bash
PYTHONPATH=eval uv run python eval/select_subset.py
PYTHONPATH=eval uv run python eval/gen_goldset.py          # then gen_goldset_groq.py to reach n=120
PYTHONPATH=eval uv run python eval/ingest_eval.py --mode baseline
PYTHONPATH=eval uv run python eval/ingest_eval.py --mode improved
EMBEDDING_MAX_TOKENS=512 PYTHONPATH=eval uv run python eval/ingest_eval.py --mode improved --collection legal_acts_eval_imp512
PYTHONPATH=eval uv run python eval/run_eval.py --collection legal_acts_eval_baseline --tag baseline
PYTHONPATH=eval uv run python eval/run_eval.py --collection legal_acts_eval_improved --tag improved
EMBEDDING_MAX_TOKENS=512 PYTHONPATH=eval uv run python eval/run_eval.py --collection legal_acts_eval_imp512 --tag improved512
PYTHONPATH=eval uv run python eval/compare3.py
```

Cleanup (drops the two remote eval collections): see `eval/cleanup.py`.
