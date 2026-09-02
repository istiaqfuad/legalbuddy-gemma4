# LegalBuddy

A citation-grounded legal AI assistant for Bangladesh law. Users ask questions
in English, Bengali, or Banglish and get plain-language answers in which every
legal claim cites a specific act and section that can be expanded to show the
original statute text.

**Competition:** Blockchain Olympiad Bangladesh 2026 (BCOLBD), AI Category —
final-round entry. Team LegalBuddy (ID 6a739711a3937).

- 📄 Technical documentation (final-round deliverable):
  [TECHNICAL_DOCUMENTATION.md](TECHNICAL_DOCUMENTATION.md)
- 🚀 Run the full stack: [legal-buddy/docs/docker-deployment.md](legal-buddy/docs/docker-deployment.md)
- 🎬 Demo script: [legal-buddy/docs/DEMO_SCRIPT.md](legal-buddy/docs/DEMO_SCRIPT.md)

## What this is

A full-stack RAG system: a 1,000-act Bangladesh statutory corpus (~35,000
sections) vector-indexed in Qdrant, retrieved with a multilingual
(Bengali↔English) embedding model, and answered by a **fine-tuned Gemma model**
(16-bit LoRA → merged → 4-bit GGUF, served locally via llama.cpp). The
fine-tune teaches answer style — plain language, `[Source N]` citations only
from provided context, clarifying questions when facts are missing, declining
when context is insufficient. The model never answers law from memory.

## Repository layout

```
├── TECHNICAL_DOCUMENTATION.md   # final-round technical deliverable
├── training/                    # SFT dataset + fine-tuning
│   ├── train_gemma4.py          # production run: Gemma-4-31B 16-bit LoRA
│   ├── train_prod.py            # completed run: gemma-2-27b-it (see logs)
│   ├── style_sft_prod/          # 1000-row SFT dataset + manifest + validation
│   └── train_prod.log*          # real training logs (loss curves)
├── legal-buddy/                 # the RAG chat application
│   ├── apps/api/                # FastAPI backend (RAG pipeline)
│   ├── apps/web/                # Next.js chat UI (streaming, source cards)
│   ├── apps/shared/             # chunking / embedding / qdrant primitives
│   ├── apps/ingestion/          # corpus → Qdrant index builder
│   ├── eval/                    # retrieval eval harness + REPORT.md (metrics)
│   ├── docs/                    # deployment, chunking strategy, demo script
│   └── docker-compose.full.yml  # one-command full stack (web+api+qdrant+llm)
├── infra/azure/                 # GGUF conversion + quantization + VM serving
└── submission_materials/        # whitepaper + technical doc (not committed)
```

## Training results

Both fine-tunes ran end-to-end with identical data and hyperparameters
(16-bit LoRA, r=16 α=32, 2 epochs, 1,000-example dataset):

| run | duration | final train loss | best eval loss |
|---|---|---|---|
| gemma-2-27b-it (development) | 28.3 min | 1.0312 | 0.5106 |
| **gemma-4-31B-it (production)** | 42.0 min | **0.6543** | **0.3863** |

Logs are committed: [training/train_gemma4.log](training/train_gemma4.log),
[training/train_prod.log](training/train_prod.log). The Gemma 4 merged
checkpoint feeds the GGUF → Q4_K_M quantization pipeline
([infra/azure/](infra/azure/)) that produces the served `lawbuddy-q4.gguf`.

## Retrieval quality (measured)

The embedding model and chunking were chosen by an A/B eval harness over a
120-pair Bengali gold set ([legal-buddy/eval/REPORT.md](legal-buddy/eval/REPORT.md)):

| variant | R@1 | R@10 | MRR |
|---|---|---|---|
| baseline | 0.258 | 0.575 | 0.364 |
| production (e5-base + improved chunking) | **0.567** | **0.883** | **0.675** |

## Quick start

```bash
cd legal-buddy
cp .env.example .env          # fill in; defaults target the local llama.cpp stack
mkdir -p models && cp /path/to/lawbuddy-q4.gguf models/
docker compose -f docker-compose.full.yml up -d --build
# UI: http://localhost:3000 · API docs: http://localhost:8000/docs
```

Local development without Docker: see [legal-buddy/README.md](legal-buddy/README.md).
