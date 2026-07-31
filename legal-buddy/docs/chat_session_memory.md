# Chat Session Memory

How Law Buddy keeps a conversation coherent across turns — so a follow-up like
*"what if a servant does it?"* is understood and retrieves the right statute,
instead of being treated as a brand-new, context-free question.

> Scope note: memory here is **in-session only**. The conversation lives in the
> browser tab. There is no database persistence and no chat-history sidebar — a
> refresh starts a fresh conversation. This was a deliberate decision for the
> current stage, not an oversight (see [Limitations](#limitations--what-to-add-next)).

---

## The problem

The original chatbot was **stateless**. Each request sent only the *current*
question. Two things broke as a result:

1. **Retrieval broke.** A bare follow-up like *"what if it's a servant?"* embeds
   to a meaningless vector. The vector search has no idea "it" means *theft*, so
   it retrieves garbage and the answer is grounded in the wrong law.
2. **The answer broke.** Even with the right sources, the model had no idea what
   "it" referred to, so the prose made no sense as a continuation.

Both are the *same* underlying problem — the model can't see the conversation —
but they need **two different fixes**, applied at two different points in the
pipeline. Conflating them is the most common mistake here.

---

## The two mechanisms (do not conflate them)

### 1. History-aware retrieval (query rewrite / "condensation")

Before we embed anything, we rewrite the follow-up into a **standalone search
query** that resolves all references against the conversation.

```
history + "what if a servant does it?"  ──►  "punishment for theft by a servant"
```

This is about **reference resolution, not context-window size.** A token
threshold is irrelevant here — even a 2-turn conversation needs it.

- **Where:** `agents/legal_chat/contextualize.py::condense_question`
- **Prompt:** `prompting.py::build_condense_messages` — "rewrite the follow-up
  into a standalone query; if already self-contained, return it unchanged;
  output only the query."
- **Model:** a deliberately *cheap/fast* model, independent of the answer model
  (`GEMINI_CONDENSE_MODEL = gemini-2.5-flash-lite`,
  `GROQ_CONDENSE_MODEL = llama-3.1-8b-instant`). The rewrite is a tiny,
  latency-sensitive call (~100–200 ms), so it never uses the big answer model.
- **When:** on **every** follow-up. Turn 1 (empty history) skips it and returns
  the question unchanged. There is no heuristic gate — robust reference
  resolution is worth one cheap call per turn.
- **Failure mode:** any exception in the rewrite falls back to the raw question.
  The rewrite can never *block* retrieval; worst case, retrieval is as good as
  the old stateless behavior.
- **temperature = 0** — the rewrite is a deterministic transformation, not a
  creative one.

The skip-on-turn-1 and never-block guarantees are the whole function —
`contextualize.py::condense_question`:

```python
# apps/api/src/api/agents/legal_chat/contextualize.py
def condense_question(question, history, *, provider=None):
    if not history:
        return question                       # turn 1: no-op
    messages = build_condense_messages(question, history)
    try:
        rewritten = run_llm_text(
            messages, provider=provider,
            model=_condense_model(provider), temperature=0.0,
        )
    except Exception:
        return question                       # rewrite can never block retrieval
    rewritten = (rewritten or "").strip().strip('"').strip()
    return rewritten or question
```

The rewritten query is used **only for vector search**. The answer prompt still
receives the user's *original* wording, so the reply reads naturally.

### 2. Conversation context in the answer prompt

Separately, the answer model is shown the recent conversation so its prose reads
as a continuation.

- **Where:** `prompting.py::build_grounded_prompt` inserts a
  `Conversation so far:` block (formatted `User: ... / Assistant: ...`) into the
  user message, *before* the `Question / situation:` line.
- **System-prompt rule:** "use the conversation to interpret the current question
  (resolve 'it', 'that', 'the punishment'), but ground every legal claim ONLY in
  the statute sources." This keeps memory from leaking into ungrounded legal
  claims — the conversation provides *reference*, the statutes provide *truth*.

### History window (how much memory)

Both mechanisms operate on a **hard last-N-turn window** — `_trim_history` keeps
`history[-HISTORY_WINDOW_TURNS:]` (`HISTORY_WINDOW_TURNS = 6`, ≈3 exchanges).
**No summarization.** Older turns are simply dropped.

The frontend mirrors the same window: `ChatApp.tsx` builds
`history = turns.filter(!error).map({role, content}).slice(-6)` **before**
appending the new user turn, so the backend trim is a defensive second guard.

---

## End-to-end flow

```
Browser (ChatApp.tsx)
  turns[]  ──build history (last 6, drop errors)──►  POST /api/chat/stream
                                                          │
Next.js route handler (app/api/chat/stream/route.ts)      │  proxy, pass body through
  validates + forwards  ──────────────────────────────────►  POST /rag/legal/chat/stream
                                                          │
FastAPI (endpoints.py::legal_chat_stream)                 │
  └─ legal_chat_pipeline_stream(question, history, ...)
        1. _trim_history(history)                  ← hard last-6 window
        2. condense_question(question, history)    ← rewrite → standalone query  [mechanism 1]
        3. retrieve_dual(standalone_query)          ← statutes + precedents
        4. yield  event: sources                    ← sources known before generation
        5. build_grounded_prompt(question, ..., history)   ← conversation block  [mechanism 2]
        6. run_llm_stream(...)  ─► yield event: delta (per chunk)
        7. yield  event: done
```

Memory and **streaming** are layered together but independent. Sources are known
*before* generation, so they are sent as the first SSE event; text deltas follow.
The streaming path uses plain-text generation with inline `[Source N]` citations
(not the structured-output path). The non-streaming `legal_chat_pipeline` /
`POST /rag/legal/chat` keeps the structured path for eval and backward
compatibility, and applies the *same* memory mechanisms.

### Transport: SSE

The browser ↔ backend memory loop rides on Server-Sent Events:

- Backend: `StreamingResponse(media_type="text/event-stream")`, frames are
  `event: <type>\ndata: <json>\n\n`. Events: `sources`, `delta`, `done`, `error`.
- Next.js route: passes `upstream.body` straight through; the 120 s timeout
  guards only the initial connect (cleared in `finally` so it can't abort a live
  stream).
- Browser: `res.body.getReader()` + `TextDecoder`, split on `\n\n`, parse
  `event:`/`data:` lines. `delta` appends to the open assistant turn; `sources`
  sets its source list; `error` marks it.

---

## Is this industry standard?

**Yes — the core is the textbook pattern.** "History-aware retrieval" via query
condensation is exactly what LangChain calls
`create_history_aware_retriever`, what LlamaIndex calls the
`CondenseQuestion`/`CondensePlusContext` chat engines, and what most production
RAG chatbots do. The key correct decisions already made here:

- **Rewrite the query before retrieval** rather than embedding the raw follow-up.
  This is the single most important RAG-memory decision, and it's done right.
- **Separate cheap rewrite model from the answer model.** Standard cost/latency
  practice.
- **Graceful fallback** to the raw question on rewrite failure. Production-grade.
- **Sources sent before deltas** over SSE. Matches how ChatGPT/Perplexity stream
  citations ahead of prose.
- **Window + drop-errors on the client, re-trimmed on the server.** Defensive,
  correct.

Where it is intentionally simpler than a mature product:

- **No persistence.** Mature chat apps store threads in a DB (Postgres is already
  wired into config) keyed by a conversation id. This is the biggest gap vs. a
  "real" product, and a deliberate scope choice here.
- **Hard window, no summarization.** Long conversations silently lose their
  oldest context. Mature systems summarize-and-compress or use a token-budgeted
  window.

Neither omission makes the current design *wrong* — they're the standard
"start simple" tradeoffs. The retrieval core is the part that's hard to get
right, and it's right.

---

## Limitations & what to add next

Roughly in priority order for making it more robust:

1. **Persistence (thread store).** Persist turns to Postgres (already in config)
   keyed by a `conversation_id`; load history server-side instead of trusting the
   client to send it. Unlocks: survive refresh, a real history sidebar, auditing,
   and not trusting client-supplied history (see #7).
2. **Token-budgeted window + rolling summary.** Replace the fixed 6-turn cut with
   a token budget, and when older turns fall off, fold them into a running summary
   instead of dropping them. This is the "History condense" that was deliberately
   deferred — the upgrade path for long sessions.
3. **Skip the rewrite when it's clearly unnecessary.** Turn 1 already skips. A
   tiny heuristic (or letting the rewrite return-unchanged signal be trusted)
   could skip the call for obviously self-contained questions, saving a hop. Minor
   — the call is already cheap.
4. **Cache embeddings / retrieval per standalone query.** Identical rewritten
   queries (common in clarification loops) can reuse retrieval results.
5. **Trace the streaming path.** `legal_chat_pipeline_stream` is currently
   untraced (the non-stream path has LangSmith spans). Aggregate streamed deltas
   into one generation span so streamed turns are observable too.
6. **Stream resilience.** Add a heartbeat/keepalive comment frame and client
   reconnect with last-event-id, so long generations survive flaky proxies. Today
   a dropped connection loses the in-flight turn.
7. **Don't trust client history.** Today the browser sends the conversation, so a
   client could forge or poison it. With server-side persistence (#1), derive
   history from the store and treat the client copy as a hint only. Important once
   accounts/persistence exist.
8. **Rewrite quality guardrails.** The rewrite can occasionally over-specify or
   drop intent. Worth an eval set of follow-up→standalone pairs, and possibly
   keeping the original query as a fallback retrieval alongside the rewritten one
   (retrieve on both, merge) for recall safety.
9. **Per-turn memory of retrieved sources.** Carrying which statutes were cited in
   prior turns can help the rewrite and the answer stay anchored to the same act
   across a multi-turn thread.

### Quick reference — files

| Concern | File |
| --- | --- |
| Request/response models, `history` field | `apps/api/src/api/api/models.py` |
| Window size, condense models | `apps/api/src/api/core/config.py` |
| Query rewrite (mechanism 1) | `apps/api/src/api/agents/legal_chat/contextualize.py` |
| Prompts (rewrite + conversation block) | `apps/api/src/api/agents/legal_chat/prompting.py` |
| Pipeline (stream + non-stream), history trim | `apps/api/src/api/agents/legal_chat/pipeline.py` |
| LLM dispatch (text / stream / structured) | `apps/api/src/api/agents/legal_chat/generation.py` |
| SSE endpoints | `apps/api/src/api/api/endpoints.py` |
| SSE proxy route | `apps/web/app/api/chat/stream/route.ts` |
| Client history build + SSE parse | `apps/web/components/chat/ChatApp.tsx` |
