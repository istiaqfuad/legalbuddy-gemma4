# Training — LegalBuddy SFT

Style fine-tuning for the LegalBuddy answer model. The fine-tune teaches a
response *style* — answer plainly, cite only provided sources as `[Source N]`,
ask a clarifying question when facts are missing, decline when context is
insufficient — not legal content (law comes from RAG at inference time).

## Dataset

`style_sft_prod/` — built by `legal-buddy/training/build_style_sft.py` from
parsed Bangladesh case-law PDFs (2,081 cases with extracted statutory sources)
and the act JSON corpus (35,312 loaded sections).

| artifact | contents |
|---|---|
| `train.jsonl` / `eval.jsonl` | 1,000 / 50 RAG-style chat examples (system + sources + user + assistant) |
| `metadata.jsonl` | per-row kind + source case |
| `validation.csv` | per-row validation outcome |
| `rejected.jsonl` | 16 rows dropped by the validation heuristics |
| `manifest.json` | seed, counts, mix, system prompt |

Mix (matches the whitepaper §4.4): 60% direct statutory Q&A · 16% situational ·
5% Banglish · 12% clarification · 7% abstention. Every citation in a target
response is validated against the sources provided in that example's prompt;
rows with garbled legacy-font case text are rejected heuristically.

## Runs

| script | base model | status |
|---|---|---|
| `train_prod.py` | `unsloth/gemma-2-27b-it` | ✅ completed — `train_prod.log` |
| `train_gemma4.py` | `unsloth/gemma-4-31B-it` | ✅ completed — `train_gemma4.log` |

Both runs used identical data and hyperparameters, so they are directly
comparable:

| run | duration | final train loss | best eval loss |
|---|---|---|---|
| gemma-2-27b-it (dev) | 28.3 min | 1.0312 | 0.5106 |
| **gemma-4-31B-it (production)** | 40.4 min | **0.6546** | **0.3869** |

The Gemma 4 run halves eval loss on the same 50-example holdout (0.511 →
0.386) — the stronger base model adapts to the citation-grounded style task
much better. Both were 16-bit LoRA (r=16 α=32, 2 epochs, effective batch 8).

## Running the Gemma 4 production fine-tune

Hardware: one ≥96 GB VRAM GPU (the 27B 16-bit run fit on an RTX PRO 6000).
On Molab:

```bash
# upload/clone the repo, then from training/:
pip install unsloth trl xformers
python train_gemma4.py            # ~35–45 min for the 27B; expect ~1.3x for 31B
```

Override the base checkpoint with `BASE_MODEL=... python train_gemma4.py` if
the unsloth mirror lags Google's release.

Outputs:

- `lawbuddy-gemma4-31b-16bit-final/` — LoRA adapter
- `lawbuddy-gemma4-31b-merged/` — merged 16-bit checkpoint, the direct input to
  GGUF conversion + Q4_K_M quantization (`infra/azure/finish_fast.sh`), which
  produces `lawbuddy-q4.gguf` served by `legal-buddy/docker-compose.full.yml`
- `train_gemma4.log.json` — training log; commit it next to `train_prod.log` so
  the technical documentation can cite real numbers

## After training

1. Convert + quantize: `infra/azure/finish_fast.sh` (HF → GGUF F16 → Q4_K_M).
2. Serve: copy `lawbuddy-q4.gguf` to `legal-buddy/models/` and
   `docker compose -f docker-compose.full.yml up -d --build`.
3. Update the training table above and `TECHNICAL_DOCUMENTATION.md` with the
   real loss numbers. *(Done — 2026-09-02 runs: 27B train 1.0312 / eval 0.5106;
   Gemma 4 train 0.6546 / eval 0.3869.)*
