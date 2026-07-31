# Deployment

How law_buddy is deployed, managed, and controlled in production. Production runs on the **tommy-vm** VM via **Dokploy** (a self-hosted Heroku/Vercel alternative built on Docker Swarm + Traefik). Continuous deployment is git-driven: a push to the `production` branch triggers a build; a GitHub Actions CI gate keeps broken builds from reaching it.

## Topology

```
                Internet
                   │  HTTPS (443)
                   ▼
        ┌────────────────────┐   tommy-vm  (public IPv4 213.136.80.53,
        │  Traefik (Dokploy) │             Tailscale 100.116.76.0)
        │  :80 / :443  + LE  │   Ubuntu 22.04 · 4 vCPU · 7.8 GB RAM · no GPU
        └─────────┬──────────┘   Docker 29 (Swarm) · Dokploy v0.26.5
                  │ routes Host(domain) → web:3000
                  ▼
   ┌──────────────────────────── dokploy-network (overlay) ───────────────┐
   │                                                                       │
   │   ┌──────────┐  internal net   ┌──────────┐      ┌──────────────┐     │
   │   │   web    │ ──────────────▶ │   api    │ ───▶ │   qdrant     │     │
   │   │ Next.js  │  API_URL=       │ FastAPI  │ 6333 │ (pre-existing│     │
   │   │  :3000   │  http://api:8000│  :8000   │      │  collections)│     │
   │   └──────────┘                 └────┬─────┘      └──────────────┘     │
   │                                     │ HTTPS                            │
   └─────────────────────────────────────┼────────────────────────────────┘
                                          ▼
                            Gemini / Groq  ·  LangSmith (optional)
```

- **web** (`apps/web`, Next.js 16 standalone) — only service Traefik exposes publicly. Server-side route handlers proxy to the api via `API_URL=http://api:8000`.
- **api** (`apps/api`, FastAPI) — internal only. Runs retrieval + LLM generation. Loads the e5 embedding model on CPU.
- **qdrant** — already running on the VM (deployed separately, on `dokploy-network`, API-key protected, collections already ingested). Reached in-cluster as `qdrant:6333`. Note: `213.136.80.53:6333` is this same VM but Qdrant binds only `127.0.0.1`, so it is **not** reachable on the public IP — always use `qdrant:6333` from inside the cluster.

## Production compose

`docker-compose.prod.yml` (repo root) is the compose Dokploy deploys. It differs from the dev `docker-compose.yml`:

| Aspect | dev `docker-compose.yml` | prod `docker-compose.prod.yml` |
|---|---|---|
| Host ports | `3000:3000`, `8000:8000` | none — web via Traefik, api internal |
| api command | `uvicorn … --reload` (Dockerfile CMD) | overridden to `uvicorn … ` (no reload), single process |
| Source mounts | binds `apps/api/src`, `apps/shared/src` | none (immutable image) |
| Networks | default | `internal` (web↔api) + external `dokploy-network` (Traefik + qdrant) |
| TLS / domain | n/a | Traefik labels on web (Let's Encrypt) |
| hf-cache volume | yes | yes (persists the model across redeploys) |

Why two networks: web and api talk over a project-scoped `internal` network so `api` resolves unambiguously even though `dokploy-network` is shared by every Dokploy app; both also join `dokploy-network` so Traefik can reach web and api can reach `qdrant`.

Single uvicorn process (no `--workers`) keeps exactly one e5 model resident (~2–4 GB) within the 7.8 GB host.

## Boot readiness gate

The api makes "deployed" mean "actually ready", so the first request never hits a cold backend:

- `apps/api/src/api/app.py` lifespan warms the embedding model (`get_embedding_model()`) and probes Qdrant (`verify_qdrant()` in `apps/api/src/api/agents/legal_chat/retrieval.py`) **before** serving. uvicorn doesn't accept requests until lifespan startup completes.
- `/rag/health` therefore returns 200 only once the model is loaded **and** the `legal_acts_event_rag_full` collection is confirmed present. If Qdrant is unreachable or the collection is missing, lifespan raises → the container never becomes healthy.
- The compose `healthcheck` polls `/rag/health` (180 s `start_period` for the first model load); `web` has `depends_on: api: condition: service_healthy`, so web only starts after api is genuinely ready.

## Environment variables

Set in the **Dokploy → Environment** tab (Dokploy writes them to a `.env` in the project dir, which the compose reads via `env_file`). Never commit secrets — `.env` is gitignored; `.env.example` is the template. Full reference: `apps/api/src/api/core/config.py`.

Required:
- `EMBEDDING_MODEL=intfloat/multilingual-e5-base` — no default; **must match the model used at ingestion** (768-dim) or retrieval silently degrades.
- `QDRANT_VECTORESTORE=http://qdrant:6333`
- `QDRANT_API_KEY=<local qdrant key>`
- `GEMINI_API_KEY=<…>` (or `GROQ_API_KEY` — at least one LLM key).

Optional / defaulted: `QDRANT_COLLECTION` (`legal_acts_event_rag_full`), `CASES_COLLECTION` (`legal_cases`), `HF_TOKEN`, `LANGSMITH_API_KEY`/`LANGSMITH_TRACING`/`LANGSMITH_PROJECT`, `CHAT_MODEL`, `RETRIEVAL_TOP_K`, `CASE_SCORE_FLOOR`, etc.

## CPU-only torch

The VM has no GPU; torch must stay CPU-only. This is locked, not best-effort:
- `pyproject.toml` pins torch/torchvision to the `pytorch-cpu` index with `explicit = true` (torch can come only from `download.pytorch.org/whl/cpu`, never PyPI's CUDA default).
- `uv.lock` resolves Linux to `torch 2.12.0+cpu` with **zero** `nvidia-*`/CUDA packages.
- Both the api `Dockerfile` and CI run `uv sync --frozen`, so the lock is honored exactly. Keep `--frozen` to prevent drift.

## How it's managed: git flow + CI/CD

```
 feature work ─▶ main ──(PR: main → production)──▶ production ──▶ Dokploy auto-deploy
                  │            │                       │
                  │      GitHub Actions CI        webhook on push
                  │   (web build + api compile)        │
                  └── CI also runs here          builds image, restarts stack
```

- **`main`** = development branch.
- **`production`** = deploy branch Dokploy watches. A push here triggers a build (Dokploy installs a GitHub webhook when *Auto Deploy* is on).
- **CI** = `.github/workflows/ci.yml`: runs on push to `main` and on PRs to `production`/`main`. Two jobs: `web · next build` (catches the most common deploy-breaker) and `api · syntax check` (`uv sync --frozen` + `compileall`, no env/model needed).
- **Gate:** protect `production` (GitHub → Settings → Branches → require the `web · next build` + `api · syntax check` checks). Then the only way into production is a green PR → Dokploy only ever deploys checked commits.

### Deploy a change
```bash
# develop on main
git checkout main && git commit -am "…" && git push        # CI runs on main

# release: open a PR main → production (CI gates it), merge when green.
# OR fast-forward directly:
git push origin main:production
# → Dokploy webhook fires → builds docker-compose.prod.yml → rolling restart
```

### One-time Dokploy setup
1. New Project → **Compose** service → GitHub provider → repo `istiaqfuad/legal-buddy` → **Branch `production`** → **Compose Path `docker-compose.prod.yml`** → enable **Auto Deploy**.
2. Environment tab → paste the required env vars (above).
3. Domains → DNS **A-record → `213.136.80.53`**, then set the domain for service `web`, port `3000`, Let's Encrypt. (Either use the Domains UI **or** edit `Host(\`…\`)` in the compose — not both; duplicate Traefik labels conflict.)
4. Deploy.

## How to control it

| Task | How |
|---|---|
| Deploy latest | Merge/push to `production` (auto-deploy), or Dokploy → **Deploy/Redeploy**. |
| Rollback | Dokploy → Deployments → redeploy a previous successful deployment. (Git: revert the bad commit on `production` and push.) |
| Change env / secret | Dokploy → Environment → edit → **Redeploy** (env changes need a restart). |
| Change domain / TLS | Dokploy → Domains (re-issues the Let's Encrypt cert). |
| View build + runtime logs | Dokploy → the service → Logs / Deployments. |
| Pause auto-deploy | Dokploy → disable Auto Deploy (deploy manually thereafter). |
| Inspect on the box | `ssh istiaqfuad@tommy-vm` then `docker ps`, `docker logs <api>`, `docker stats`. |
| Check api health directly | `docker exec <api> python -c "import urllib.request;print(urllib.request.urlopen('http://localhost:8000/rag/health').read())"` → `{"status":"ok"}`. |

## Verification after deploy

1. **Dokploy logs**: both images build; api logs show the e5 model loading then startup completing with no `verify_qdrant` RuntimeError (confirms `qdrant:6333` reachable + collection present).
2. **State**: api reports `healthy`; web starts only after.
3. **Public**: `https://<domain>` loads over valid TLS; submitting a legal question returns an answer with statute sources — exercises web → api → Qdrant → LLM end to end.

## Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| api never becomes `healthy` | Qdrant unreachable or collection missing → check `QDRANT_VECTORESTORE=http://qdrant:6333`, `QDRANT_API_KEY`, and that `legal_acts_event_rag_full` exists. Lifespan raises `verify_qdrant` RuntimeError in logs. |
| First request after deploy is slow once | Cold model load (≤180 s). The healthcheck gate prevents user-facing errors; the hf-cache volume makes subsequent boots fast. |
| Build OOM on the VM | `next build` + torch on 7.8 GB is tight. Mitigate: build images on the fedora GPU box / locally, push to a registry, switch compose to `image:` instead of `build:`. |
| 404 / no TLS at the domain | DNS A-record not pointing to `213.136.80.53`, or both compose labels **and** Dokploy Domains set (conflict) — use one. |
| web 502 to api | api unhealthy, or `API_URL` not `http://api:8000`, or services not sharing the `internal` network. |
| Retrieval quality regressed | `EMBEDDING_MODEL` differs from the ingested model — must be `intfloat/multilingual-e5-base`. |
| Accidental CUDA torch pulled | Lock drifted — restore `uv sync --frozen`; the `pytorch-cpu` `explicit = true` pin keeps torch CPU-only. |

## Key files
- `docker-compose.prod.yml` — production stack (Dokploy compose path).
- `docker-compose.yml` — local dev stack.
- `apps/api/Dockerfile`, `apps/web/Dockerfile` — image builds.
- `apps/api/src/api/app.py` — boot warmup + readiness gate.
- `apps/api/src/api/agents/legal_chat/retrieval.py` — `verify_qdrant()`.
- `apps/api/src/api/core/config.py` — env var reference.
- `.github/workflows/ci.yml` — CI gate.
- `.env.example` — env template.
