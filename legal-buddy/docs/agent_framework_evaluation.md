# Agent Framework Evaluation — LangChain / LangGraph / Pydantic AI / LlamaIndex

**Status:** Decision recorded 2026-06-23. Verdict: **do not adopt any framework now; adopt selectively when a named trigger fires.**

## Purpose

law_buddy is a from-scratch legal RAG system (Bangladesh statutes + case-law precedents). The question recurs: should we adopt an agent/RAG framework, and which one? This doc answers it concretely — what each framework is for, where it would (and would not) fit *this* codebase, and the exact conditions under which adopting it becomes worthwhile. It is meant to be re-read before any "let's add framework X" decision.

Sources: the LangChain/LangGraph skills bundled in `.claude/skills/` (current `create_agent` / `StateGraph` APIs) and Context7 live docs for Pydantic AI and LlamaIndex (queried 2026-06-23). Framework APIs move fast — re-verify against live docs before implementing.

## Where law_buddy stands today

What is already built (see `docs/chunking_and_retrieval.md`, `docs/situational_rag_plan.md`, `docs/chat_session_memory.md`):

- **Ingestion** (`apps/ingestion`): custom page-level hybrid PDF extraction (PyMuPDF + EasyOCR routing) → token/subsection-aware chunker (`apps/shared/src/shared/chunking.py`) → `sentence-transformers` (`intfloat/multilingual-e5-base`) → Qdrant upsert.
- **Retrieval** (`apps/api/src/api/agents/legal_chat/retrieval.py`): dual-collection (`legal_acts_event_rag_full` + `legal_cases`), parent-document collapse, per-collection score floors, Qdrant grouping.
- **Chat pipeline** (`.../legal_chat/pipeline.py`): trim history → condense query → `retrieve_dual` → `build_grounded_prompt` → `run_llm` (Gemini default / Groq, structured via `instructor`) → SSE token stream.
- **Already-present framework pieces:** `instructor` (structured output) and `langsmith` (tracing, `@trace`). These are the two things most teams add a framework *for*. We already have them.

The eval-validated parts — chunking (+31pp recall, `docs/chunking_and_retrieval.md`), dual-retrieve, asymmetric statute-vs-precedent semantics, OCR routing — are the parts a framework default would *degrade*, not improve. Keep them. The framework question is only about **control flow and provider plumbing**, never about these components.

What is *planned but unbuilt* (the real decision drivers, from `docs/situational_rag_plan.md` and `docs/cases_retrieval_quality_plan.md`):

- Stage 0 — situation understanding / HyDE query expansion (adds an LLM call).
- Stage 2 — cross-encoder rerank of the merged statute+precedent pool.
- Stage 4 — score-threshold abstention guardrails.
- Server-side conversation persistence (today: browser-only, lost on refresh — a stated gap in `docs/chat_session_memory.md`).

---

## 1. LangChain — agent framework (bottom layer)

**What it is.** Provider-agnostic abstractions for models, tools, and the agent loop. Current API is `create_agent(model, tools=[...])`; older `LLMChain` / `create_history_aware_retriever` / `ConversationBufferWindowMemory` patterns are superseded. Structured output via `response_format=` or `model.with_structured_output(PydanticModel)`. RAG is "retriever-as-tool": wrap a vector search in `@tool` and hand it to an agent.

**Best for.** Single-purpose agents with a fixed tool set; quick RAG prototypes; standardized multi-provider model calls; getting from zero to a working demo fast.

**Where it fits law_buddy.**
- *Now:* essentially nowhere that pays off. Our retrieval is more sophisticated than `vectorstore.as_retriever(k=...)`; our structured output is already handled by `instructor`; our memory window is a deliberate 6-turn trim, not a framework default. Wrapping current code in LangChain adds an abstraction layer over logic that is already clean.
- *Future trigger — provider sprawl:* `generation.py` hand-rolls Gemini-vs-Groq branching. If we reach **4+ providers** or swap models frequently, `init_chat_model("anthropic:claude-...")` / `ChatOpenAI` / `ChatGoogleGenerativeAI` behind one interface removes real boilerplate. At today's two providers, the hand-roll is fine.
- *Future trigger — agentic chat:* if the assistant must *call functions* mid-conversation (look up a case by number, fetch a specific act, compute a limitation deadline), `create_agent` with those as `@tool`s and our retrieval as a `search_statutes` tool is the clean path. Pure one-shot RAG (what we have) does not need it.

**Verdict.** Defer. Its strongest pitch — structured output + tracing + fast demo — we already cover or have outgrown.

---

## 2. LangGraph — agent runtime (middle layer)

**What it is.** Low-level orchestration: model a workflow as a directed graph of `StateGraph` nodes and edges. Shared typed `State` (with reducers like `Annotated[list, operator.add]`), `add_conditional_edges` for branching, `Send` for parallel fan-out, `Command` to update-state-and-route in one return. Compile with a **checkpointer** for durable state, `interrupt()` + `Command(resume=...)` for human-in-the-loop, and `stream_mode="messages"` for token streaming. LangChain agents actually run on top of LangGraph.

**Best for.** Custom control flow — deterministic loops, conditional branches, parallel fan-out; workflows mixing deterministic and LLM steps; human-in-the-loop with precise pause/resume; state that must survive failures or span sessions.

**Where it fits law_buddy.** This is the **most likely real adoption** — it maps directly onto the unbuilt roadmap:
- *Multi-stage pipeline:* the day `pipeline.py` grows Stage 0 (HyDE) → dual-retrieve → Stage 2 (rerank) → Stage 4 (abstention) → generate, with conditional branches (e.g. "skip rerank if <3 candidates", "abstain if all scores below floor"), a `StateGraph` with `add_conditional_edges` is cleaner and more debuggable than nested functions. Each stage is a node; routing is explicit.
- *Parallel dual-retrieve:* statute and precedent retrieval are independent — `Send` fans them out to parallel worker nodes, results merged via a reducer. (Today they run sequentially in `retrieve_dual`.)
- *Server-side conversation persistence (the stated gap):* compile with `PostgresSaver` + a `thread_id` per conversation → history survives refresh/restart without us hand-rolling a store. `InMemorySaver` for dev, `PostgresSaver` for prod.
- *Long-term user memory:* a LangGraph `Store` (cross-thread) holds per-user facts/preferences across separate conversations — beyond today's single-window trim.
- *Lawyer-in-the-loop review:* `interrupt()` pauses after answer generation to surface a draft for human approval/edit before it reaches the user, `Command(resume=...)` continues. Relevant for a legal product where review-before-send may be required. (Caveat from the HITL skill: a node re-runs from the top on resume, so side effects before `interrupt()` must be idempotent.)
- *Streaming:* `stream_mode="messages"` reproduces our current SSE token stream natively.

**Verdict.** Defer, but this is the front-runner. Adopt when control flow genuinely branches/loops — i.e. when the planned stages land — or when server-side persistence becomes a requirement. Adopting it *just* to wrap today's linear pipeline is premature.

---

## 3. Pydantic AI — type-safe agent framework

**What it is.** A FastAPI-style, type-safe agent framework from the Pydantic team. Core shape: `Agent(model, deps_type=Deps, output_type=PydanticModel, instructions=...)`, tools via `@agent.tool` receiving a typed `RunContext[Deps]` (dependency injection for db handles, config, etc.), and validated structured output by construction. Model-agnostic (OpenAI/Anthropic/Gemini/Groq/local via Outlines). Durable execution via integrations (Restate, Temporal, DBOS) so tool steps are saved and replayed across failures. First-class LangSmith/Logfire observability.

**Best for.** Teams already standardized on Pydantic who want typed, IDE-checkable agents; structured-output-heavy agents where the output schema *is* the contract; agentic apps wanting dependency injection and durable execution without LangGraph's graph-authoring overhead.

**Where it fits law_buddy.**
- *Natural fit on paper:* we are already a Pydantic + `instructor` shop. Pydantic AI is essentially "`instructor` plus an agent loop, tools, and DI." Our `StructuredLegalAnswer` model would become an `output_type`; our retrieval would become an injected dependency on `RunContext`.
- *Future trigger — typed agentic chat:* if chat goes agentic (tool-calling) *and* we want every turn to remain a validated Pydantic object end-to-end, Pydantic AI is a lighter, more type-safe alternative to `create_agent`. It competes with LangChain for the "single typed agent" slot, and arguably wins it for a Pydantic-native codebase.
- *Not a fit for:* the graph-shaped multi-stage pipeline (that is LangGraph's job) or our custom retrieval internals (it does not replace those).

**Verdict.** Defer, but it is the **strongest LangChain alternative** for the "if we build one typed agent" path, precisely because of our existing Pydantic/`instructor` investment. Watch this one if chat becomes tool-calling.

---

## 4. LlamaIndex — data framework for RAG

**What it is.** A RAG-first framework: `SimpleDirectoryReader` → `VectorStoreIndex.from_documents` → `index.as_query_engine()` for an end-to-end pipeline in a few lines. Rich retrieval toolkit (retrievers, `LLMRerank` and other node postprocessors, query engines, response synthesizers like `CompactAndRefine`), a `Workflow` API (`@step` event-driven graph) for custom control flow, LlamaParse for hard document parsing, and 100+ connectors via LlamaHub. Also supports agents over data.

**Best for.** Standing up a competent RAG system fast; teams that want batteries-included ingestion + retrieval + rerank + synthesis without wiring each piece; complex document parsing (LlamaParse).

**Where it fits law_buddy.**
- *Mostly redundant with what we built.* We already have a Qdrant store, an eval-validated custom chunker, e5 embeddings with asymmetric prefixes, parent-document retrieval, and dual-collection orchestration. LlamaIndex's value is providing exactly those — but its defaults are *less* tuned than ours (generic splitters vs. our subsection-aware chunker). Adopting it wholesale means regressing on the parts we measured.
- *Cherry-pick candidate — rerank:* the one genuinely useful, low-commitment borrow is its reranking layer (`LLMRerank` / node postprocessors) for the unbuilt Stage 2. But `docs/cases_retrieval_quality_plan.md` already prefers a *local* cross-encoder (`bge-reranker-v2-m3`, no API cost) over an LLM reranker — so even here we likely use `sentence-transformers` CrossEncoder directly, not LlamaIndex.
- *LlamaParse* could help only if we hit PDFs our PyMuPDF+EasyOCR hybrid can't handle (none so far per `docs/cases_ingestion_plan.md`).

**Verdict.** Lowest fit of the four. We are past the stage where LlamaIndex helps; it would replace tuned code with generic code. Borrow individual ideas (rerank pattern), not the framework.

---

## Side-by-side

| Framework | Layer / role | Core strength | law_buddy fit today | Adopt-when trigger |
|---|---|---|---|---|
| **LangChain** | Agent framework | Multi-provider model calls, single agent, fast demo | Low (already have structured output + tracing) | 4+ LLM providers, or chat becomes tool-calling |
| **LangGraph** | Agent runtime | Stateful branching/looping, HITL, durable persistence | Low now, **highest future** | Planned multi-stage pipeline (HyDE/rerank/abstention) lands, or server-side persistence required |
| **Pydantic AI** | Typed agent framework | Type-safe agents, DI, validated output, durable exec | Low now, **strong LangChain alt** | Typed agentic chat (we're already Pydantic/instructor-native) |
| **LlamaIndex** | RAG data framework | Batteries-included ingestion/retrieval/rerank | Lowest (redundant, less-tuned than ours) | Never wholesale; borrow rerank pattern only |

## Mapping to the roadmap

| Planned work (from docs) | Best-fit framework | Why |
|---|---|---|
| Stage 0 HyDE / issue-spotting | LangGraph node (or none) | Conditional extra LLM call; a graph node if pipeline is already a graph, else a plain function |
| Stage 2 cross-encoder rerank | **Local `sentence-transformers` CrossEncoder** (not a framework) | `cases_retrieval_quality_plan.md` mandates no per-query API cost |
| Stage 4 abstention guardrail | LangGraph conditional edge | Natural as a routing decision on score state |
| Parallel dual-retrieve | LangGraph `Send` | Independent statute/precedent fan-out |
| Server-side chat persistence | LangGraph `PostgresSaver` + `thread_id` | Closes the stated browser-only gap without hand-rolling a store |
| Lawyer review before send | LangGraph `interrupt()` / `Command(resume)` | Precise pause/resume with checkpointed state |
| Tool-calling assistant | Pydantic AI (typed) or LangChain `create_agent` | One typed agent with our retrieval as a tool |

## Decision and adopt-later triggers

**Now: adopt nothing.** Boundaries in `pipeline.py` / `retrieval.py` / `generation.py` are already clean, so deferring carries ~zero lock-in cost; dropping a framework in later is a contained refactor, not a rewrite.

Revisit only when a concrete trigger fires:

1. **LLM providers reach 4+ or model-swapping is frequent** → LangChain `init_chat_model` (or Pydantic AI) to collapse `generation.py` provider branching.
2. **The pipeline becomes 5+ conditional stages** (HyDE/rerank/abstention land with branches/retries) → **LangGraph `StateGraph`**. *Most likely trigger.*
3. **Server-side conversation persistence becomes a requirement** → LangGraph `PostgresSaver` + `thread_id` (+ `Store` for cross-session user memory).
4. **Chat must call tools / become agentic** → **Pydantic AI** (typed, Pydantic-native) or LangChain `create_agent`.
5. **Human review-before-send is required** → LangGraph `interrupt()` / `Command(resume=...)`.

**Never frameworkize:** chunking, OCR routing, dual-retrieve, parent-document collapse, asymmetric statute/precedent semantics. These are eval-validated; generic framework defaults regress them.

**Rule of thumb:** adopt a framework the day the *control flow* gets complex (branches, loops, tools, durable state) — never for the *components*, which we have already nailed. When that day comes, LangGraph (orchestration) and Pydantic AI (typed agent) are the two to reach for first, given our Pydantic/`instructor`/`langsmith` foundation; LlamaIndex offers the least.

## Verdict rationale

Each verdict above is a conclusion; this section gives the reasoning so a future reader can challenge it rather than take it on faith. All four rest on four shared principles, then diverge per framework.

**The four governing principles.**

1. **The framework value curve peaks at day one and decays.** A framework's biggest payoff is zero-to-demo speed: 20 lines to a working RAG chain. That value falls as a system matures, because maturity *is* the accumulation of decisions a framework made generically and you now want to make specifically. law_buddy is past day one — it has eval numbers, a tuned chunker, and asymmetric source semantics. We are on the decaying part of every one of these curves.
2. **Don't pay for what you already own.** The two capabilities teams most often reach for a framework to get are structured output and observability. We have both — `instructor` and `langsmith` (`@trace`). So the headline reason to adopt is already satisfied by lighter dependencies. Whatever a framework adds on top must justify itself on *other* grounds.
3. **Components vs. control flow.** Every verdict separates two questions. *Components* = chunking, embedding, retrieval, source semantics — these are eval-validated and domain-specific; a generic default regresses them, so frameworks lose here by construction. *Control flow* = how stages are wired, branched, retried, paused, persisted — this is where frameworks can genuinely beat hand-rolled code, but only once the flow is actually complex. Today our flow is linear, so even the control-flow case is not yet live.
4. **Deferral is close to free here.** Lock-in cost is the real risk of "just adopt it." But `pipeline.py`, `retrieval.py`, and `generation.py` already have clean seams. Dropping in a framework later is a contained refactor at one seam, not a rewrite. Because waiting costs ~nothing and adopting early costs abstraction tax + churn exposure (LangChain's repeated API breaks), the expected-value math favors waiting until a trigger removes the uncertainty.

**Why LangChain = defer.** Its three pitches are multi-provider model calls, fast single-agent assembly, and the structured-output/tracing bundle. The third we already own (principle 2). The second only matters once chat is agentic, which it isn't. The first is real but small at two providers — `generation.py`'s Gemini/Groq branch is a dozen lines, not a maintenance burden, and `init_chat_model` only earns its keep at 4+ providers. So none of the three pitches clears the bar *today*, while adoption would wrap already-clean code in an abstraction known for churn. Net negative now; revisit on the provider/agentic triggers.

**Why LangGraph = defer but front-runner.** This is the one framework whose core strength (stateful branching, durable persistence, HITL) lands exactly on our *unbuilt* roadmap, not our built code — so it doesn't fight the eval-validated components (principle 3 is satisfied: it touches control flow, not components). The reason it's still "defer" is timing: its value is proportional to control-flow complexity, and our flow is linear *right now*. Wrapping a straight-line pipeline in a `StateGraph` today buys graph overhead with no branch to justify it. The moment Stage 0/2/4 add real branches (skip-rerank-if-few-candidates, abstain-on-low-scores) or persistence becomes a requirement, the same strength becomes a clear win. So the verdict is not "no" — it's "not yet, and this is the one to reach for when the day comes."

**Why Pydantic AI = defer but strongest LangChain alternative.** The reasoning is fit-to-existing-stack. We are already a Pydantic + `instructor` codebase; Pydantic AI is close to "`instructor` with an agent loop, typed dependency injection, and durable execution stapled on." That means if we ever need *one typed agent* (the same slot LangChain's `create_agent` fills), Pydantic AI reuses our mental model and our `StructuredLegalAnswer` schema becomes the agent's `output_type` directly — less impedance than LangChain. It ranks above LangChain for that specific future, but it's still "defer" because that future (tool-calling chat) hasn't arrived, and it does nothing for the graph-shaped pipeline (LangGraph's job) or the retrieval internals.

**Why LlamaIndex = lowest fit.** This verdict is the cleanest application of principle 3. LlamaIndex's whole value proposition is supplying ingestion + retrieval + rerank + synthesis — i.e. exactly the *components* we have already built and measured. Adopting it doesn't add a missing capability; it *replaces tuned code with generic code*, and our chunker beats its default splitter on our own eval. The only non-redundant piece is reranking, and `cases_retrieval_quality_plan.md` already commits to a local cross-encoder (no per-query API cost) — which we'd implement with `sentence-transformers` directly, not pull LlamaIndex for. So there's no slot where LlamaIndex is the right tool; borrow the pattern, skip the dependency.

**Why the overall stance isn't "never."** The decision is "defer with triggers," not "build everything from scratch forever," because the cost balance flips once a trigger fires. Before a trigger, uncertainty is high (will we even build Stage 2? how many providers?) and adoption is speculative. After a trigger, the requirement is concrete, the framework's strength maps to a real need, and the clean seams make adoption cheap. Triggers are simply the points where waiting stops being the higher-EV move.

## References

- LangChain / LangGraph / Deep Agents docs: `https://docs.langchain.com/oss/python/` (LangChain overview, LangGraph overview, `llms.txt` index)
- Local skills: `.claude/skills/{langchain-fundamentals,langchain-rag,langgraph-fundamentals,langgraph-human-in-the-loop,langgraph-persistence,ecosystem-primer}`
- Pydantic AI: `https://pydantic.dev/docs/ai` (overview, durable execution, structured output via `output_type`)
- LlamaIndex: `https://developers.llamaindex.ai/python/` (use cases, Workflow RAG with `LLMRerank`, query engines)
- Related internal docs: `docs/situational_rag_plan.md`, `docs/chunking_and_retrieval.md`, `docs/chat_session_memory.md`, `docs/cases_retrieval_quality_plan.md`
