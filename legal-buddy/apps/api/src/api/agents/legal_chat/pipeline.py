from collections.abc import Iterator

from langsmith import trace

from api.api.models import ChatMessage, LegalChatResponse
from api.core.config import config
from api.core.observability import get_langsmith_client

from api.agents.legal_chat.contextualize import condense_question
from api.agents.legal_chat.generation import run_llm, run_llm_stream, run_llm_text
from api.agents.legal_chat.prompting import build_clarify_prompt, build_grounded_prompt
from api.agents.legal_chat.retrieval import retrieve_sources


def _trim_history(history: list[ChatMessage] | None) -> list[ChatMessage]:
    """Hard last-N-turn window (no summarization)."""
    if not history:
        return []
    return history[-config.HISTORY_WINDOW_TURNS :]


def _is_no_match(statutes: list, floor: float) -> bool:
    """Genuine no-match: nothing retrieved, or the top hit is below the hard floor
    (off-topic garbage). Only this routes to a deterministic clarify — borderline
    cases go to the model, which decides whether to answer or ask. The e5 score is
    a weak separator, so this floor is set low and the real judgment is the model's."""
    return not statutes or statutes[0].score < floor


def _is_low_confidence(statutes: list, floor: float) -> bool:
    """Top-statute score shaky (below the low-confidence floor) — passed to the
    answer prompt as a soft hint so the model leans toward clarifying when the
    sources may not fit, without forcing a hard branch."""
    return not statutes or statutes[0].score < floor


def _sources_for_trace(statutes: list) -> list[dict]:
    """Compact source summary for trace outputs — act, section, score."""
    return [
        {
            "act": s.act_title,
            "section": s.section_index,
            "score": round(s.score, 4),
        }
        for s in statutes
    ]


def _request_inputs(
    question: str, search_query: str, history: list[ChatMessage], top_k: int
) -> dict:
    return {
        "question": question,
        "standalone_query": search_query,
        "history_turns": len(history),
        "top_k": top_k,
    }


def legal_chat_pipeline(
    question: str,
    *,
    history: list[ChatMessage] | None = None,
    top_k: int | None = None,
    max_tokens: int | None = None,
    provider: str | None = None,
    model: str | None = None,
    temperature: float | None = None,
    clarify_score_floor: float | None = None,
    low_confidence_floor: float | None = None,
) -> LegalChatResponse:
    resolved_top_k = top_k or config.RETRIEVAL_TOP_K
    resolved_max_tokens = (
        max_tokens if max_tokens is not None else config.ANSWER_MAX_TOKENS
    )
    clarify_floor = (
        clarify_score_floor
        if clarify_score_floor is not None
        else config.CLARIFY_SCORE_FLOOR
    )
    low_conf_floor = (
        low_confidence_floor
        if low_confidence_floor is not None
        else config.LOW_CONFIDENCE_FLOOR
    )
    history = _trim_history(history)
    llm_kwargs = {"provider": provider, "model": model, "temperature": temperature}

    # History-aware retrieval: rewrite a follow-up into a standalone search query
    # (no-op on the first turn). The answer prompt still gets the original question
    # plus the conversation so the reply reads naturally.
    client = get_langsmith_client()
    search_query = condense_question(
        question, history, provider=provider, traced=client is not None
    )

    if client is None:
        statutes = retrieve_sources(search_query, top_k=resolved_top_k)
        if _is_no_match(statutes, clarify_floor):
            # Nothing to ground an answer in — ask for specifics, don't dead-end.
            clarify = run_llm_text(build_clarify_prompt(question, history), **llm_kwargs)
            return LegalChatResponse(answer=clarify, sources=[])

        messages = build_grounded_prompt(
            question,
            statutes,
            history,
            low_confidence=_is_low_confidence(statutes, low_conf_floor),
        )
        answer = run_llm(
            messages=messages,
            sources=statutes,
            max_tokens=resolved_max_tokens,
            **llm_kwargs,
        )
        return LegalChatResponse(answer=answer, sources=statutes)

    with trace(
        name="legal-chat-request",
        run_type="chain",
        inputs=_request_inputs(question, search_query, history, resolved_top_k)
        | {"max_tokens": resolved_max_tokens},
        metadata={"endpoint": "/rag/legal/chat"},
    ) as request_span:
        statutes = retrieve_sources(search_query, top_k=resolved_top_k)
        if _is_no_match(statutes, clarify_floor):
            clarify = run_llm_text(build_clarify_prompt(question, history), **llm_kwargs)
            response = LegalChatResponse(answer=clarify, sources=[])
            request_span.end(
                outputs=response.model_dump() | {"path": "clarify"}
            )
            return response

        messages = build_grounded_prompt(
            question,
            statutes,
            history,
            low_confidence=_is_low_confidence(statutes, low_conf_floor),
        )
        answer = run_llm(
            messages=messages,
            sources=statutes,
            max_tokens=resolved_max_tokens,
            **llm_kwargs,
        )
        response = LegalChatResponse(answer=answer, sources=statutes)
        request_span.end(
            outputs={
                "answer": answer,
                "sources": _sources_for_trace(statutes),
                "path": "grounded",
            }
        )
        return response


def legal_chat_pipeline_stream(
    question: str,
    *,
    history: list[ChatMessage] | None = None,
    top_k: int | None = None,
    max_tokens: int | None = None,
    provider: str | None = None,
    model: str | None = None,
    temperature: float | None = None,
    clarify_score_floor: float | None = None,
    low_confidence_floor: float | None = None,
) -> Iterator[dict]:
    """Streaming variant for the chat UI.

    Yields event dicts: ``{"type": "sources", ...}`` once (known before
    generation), then ``{"type": "delta", "text": ...}`` per token chunk, then
    ``{"type": "done"}``. Uses plain-text generation with inline ``[Source N]``
    citations (no structured wrapper). Wrapped in a LangSmith request trace
    (retrieval/generation child spans nest inside it automatically).
    """
    client = get_langsmith_client()
    if client is None:
        yield from _legal_chat_stream_events(
            question,
            history=history,
            top_k=top_k,
            max_tokens=max_tokens,
            provider=provider,
            model=model,
            temperature=temperature,
            clarify_score_floor=clarify_score_floor,
            low_confidence_floor=low_confidence_floor,
            traced=False,
        )
        return

    answer_chunks: list[str] = []
    sources_summary: list[dict] = []
    with trace(
        name="legal-chat-stream",
        run_type="chain",
        inputs={
            "question": question,
            "history": [
                {"role": m.role, "content": m.content}
                for m in _trim_history(history)
            ],
            "top_k": top_k,
            "stream": True,
        },
        metadata={"endpoint": "/rag/legal/chat/stream"},
    ) as request_span:
        for event in _legal_chat_stream_events(
            question,
            history=history,
            top_k=top_k,
            max_tokens=max_tokens,
            provider=provider,
            model=model,
            temperature=temperature,
            clarify_score_floor=clarify_score_floor,
            low_confidence_floor=low_confidence_floor,
            traced=True,
        ):
            if event.get("type") == "sources":
                sources_summary = [
                    {
                        "act": s.get("act_title"),
                        "section": s.get("section_index"),
                        "score": round(s.get("score", 0.0), 4),
                    }
                    for s in event.get("sources") or []
                ]
            elif event.get("type") == "delta":
                answer_chunks.append(event.get("text") or "")
            yield event
        request_span.end(
            outputs={
                "answer": "".join(answer_chunks),
                "sources": sources_summary,
            }
        )


def _legal_chat_stream_events(
    question: str,
    *,
    history: list[ChatMessage] | None = None,
    top_k: int | None = None,
    max_tokens: int | None = None,
    provider: str | None = None,
    model: str | None = None,
    temperature: float | None = None,
    clarify_score_floor: float | None = None,
    low_confidence_floor: float | None = None,
    traced: bool = False,
) -> Iterator[dict]:
    """Event generator behind legal_chat_pipeline_stream.

    When ``traced`` is set, the query rewrite and the streaming generation each
    get their own LangSmith span (the embedding and vector-search spans are
    created inside their own modules), so the full question→answer path is
    visible in one trace tree.
    """
    resolved_top_k = top_k or config.RETRIEVAL_TOP_K
    resolved_max_tokens = (
        max_tokens if max_tokens is not None else config.ANSWER_MAX_TOKENS
    )
    clarify_floor = (
        clarify_score_floor
        if clarify_score_floor is not None
        else config.CLARIFY_SCORE_FLOOR
    )
    low_conf_floor = (
        low_confidence_floor
        if low_confidence_floor is not None
        else config.LOW_CONFIDENCE_FLOOR
    )
    history = _trim_history(history)

    search_query = condense_question(
        question, history, provider=provider, traced=traced
    )
    statutes = retrieve_sources(search_query, top_k=resolved_top_k)

    yield {"type": "sources", "sources": [s.model_dump() for s in statutes]}

    if _is_no_match(statutes, clarify_floor):
        # Nothing to ground an answer in — stream a clarifying question instead.
        yield from _stream_traced(
            build_clarify_prompt(question, history),
            provider=provider,
            model=model,
            temperature=temperature,
            max_tokens=resolved_max_tokens,
            traced=traced,
            span_name="clarify-generation",
            span_metadata={"path": "clarify"},
        )
        yield {"type": "done"}
        return

    messages = build_grounded_prompt(
        question,
        statutes,
        history,
        low_confidence=_is_low_confidence(statutes, low_conf_floor),
    )
    yield from _stream_traced(
        messages,
        provider=provider,
        model=model,
        temperature=temperature,
        max_tokens=resolved_max_tokens,
        traced=traced,
        span_name="answer-generation",
        span_metadata={
            "path": "grounded",
            "source_count": len(statutes),
            "low_confidence": _is_low_confidence(statutes, low_conf_floor),
        },
    )
    yield {"type": "done"}


def _stream_traced(
    messages: list[dict],
    *,
    provider: str | None,
    model: str | None,
    temperature: float | None,
    max_tokens: int | None,
    traced: bool,
    span_name: str,
    span_metadata: dict,
) -> Iterator[dict]:
    """Stream generation, optionally under an LLM span that captures the full
    prompt and the assembled answer (chunk-by-chunk deltas stay visible via the
    request span)."""
    if not traced:
        for chunk in run_llm_stream(
            messages,
            provider=provider,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        ):
            yield {"type": "delta", "text": chunk}
        return

    chunks: list[str] = []
    with trace(
        name=span_name,
        run_type="llm",
        inputs={"messages": messages},
        metadata=span_metadata,
    ) as gen_span:
        for chunk in run_llm_stream(
            messages,
            provider=provider,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        ):
            chunks.append(chunk)
            yield {"type": "delta", "text": chunk}
        gen_span.end(outputs={"output": "".join(chunks)})
