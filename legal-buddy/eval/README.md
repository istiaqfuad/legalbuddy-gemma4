# Chunking / retrieval eval harness

Isolated A/B harness used to validate the chunking, retrieval, and embedding-model
choices in the RAG pipeline. **Results and analysis: [`REPORT.md`](REPORT.md).**
Strategy explanation: [`../docs/chunking_and_retrieval.md`](../docs/chunking_and_retrieval.md).

Ingests into throwaway `legal_acts_eval_*` Qdrant collections — never touches the
production collection.

## Layout

```
eval/
  *.py            pipeline scripts + shared lib (run with PYTHONPATH=eval)
  data/           curated inputs   — subset_acts.json, goldset.json
  results/        generated metrics — metrics_<tag>.json
  REPORT.md       results + analysis
```

Artifact paths are centralized in `common.py` (`SUBSET_PATH`, `GOLDSET_PATH`,
`metrics_path(tag)`); scripts import those rather than hard-coding filenames.

## Files

| file | role |
|---|---|
| `common.py` | env, Qdrant client, embedding model, artifact paths, metrics (recall@k, MRR) |
| `baseline_chunk.py` | frozen copy of the *old* chunking, so the baseline stays reproducible |
| `select_subset.py` | pick a deterministic ~200-act subset → `data/subset_acts.json` |
| `gen_goldset.py` | LLM-generate Q→section gold pairs (Gemini); base prompt + candidate sampler |
| `gen_goldset_groq.py` | expand the gold set via Groq (appends to `data/goldset.json`) |
| `ingest_eval.py` | ingest the subset into a collection with `--mode baseline\|improved` |
| `run_eval.py` | score a collection against the gold set → `results/metrics_<tag>.json` |
| `compare3.py` | side-by-side compare of the recorded metrics |
| `cleanup.py` | drop every `legal_acts_eval_*` collection |

## Run

```bash
PYTHONPATH=eval uv run python eval/select_subset.py
PYTHONPATH=eval uv run python eval/gen_goldset.py        # then gen_goldset_groq.py for more
PYTHONPATH=eval uv run python eval/ingest_eval.py --mode baseline
PYTHONPATH=eval uv run python eval/ingest_eval.py --mode improved
PYTHONPATH=eval uv run python eval/run_eval.py --collection legal_acts_eval_baseline --tag baseline
PYTHONPATH=eval uv run python eval/run_eval.py --collection legal_acts_eval_improved --tag improved
PYTHONPATH=eval uv run python eval/compare3.py
PYTHONPATH=eval uv run python eval/cleanup.py            # when done
```

To test a different embedding model / window, prefix with
`EMBEDDING_MODEL=... EMBEDDING_MAX_TOKENS=...` and use a distinct `--collection`/`--tag`.

`data/goldset.json` and `results/metrics_*.json` are the recorded artifacts behind `REPORT.md`.
