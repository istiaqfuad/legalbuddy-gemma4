from langsmith import trace

from api.api.models import ChatMessage
from api.core.config import config

from api.agents.legal_chat.generation import run_llm_text
from api.agents.legal_chat.prompting import build_condense_messages


def _condense_model(provider: str | None) -> str:
    resolved = (provider or config.DEFAULT_LLM_PROVIDER or "gemini").lower()
    if resolved in ("groq", "openai"):
        # None -> reuse the answer model. Single-model servers (llama.cpp) serve
        # one alias and reject requests for any other model name, so a distinct
        # condense model must be an explicit opt-in via GROQ_CONDENSE_MODEL.
        return config.GROQ_CONDENSE_MODEL or config.GROQ_MODEL
    return config.GEMINI_CONDENSE_MODEL


def condense_question(
    question: str,
    history: list[ChatMessage],
    *,
    provider: str | None = None,
    traced: bool = False,
) -> str:
    """Rewrite a follow-up into a standalone retrieval query (history-aware).

    The first turn (empty history) returns the question unchanged. Otherwise this
    runs a tiny call on the provider's fast condense model at temperature 0 to
    resolve references ("it", "that offence", "the punishment") so the vector
    search hits the right statute. Any failure falls back to the raw question, so
    retrieval is never blocked by the rewrite.
    """
    if not history:
        return question

    messages = build_condense_messages(question, history)

    def _run() -> str:
        return run_llm_text(
            messages,
            provider=provider,
            model=_condense_model(provider),
            temperature=0.0,
        )

    if not traced:
        try:
            rewritten = _run()
        except Exception:
            return question
    else:
        try:
            with trace(
                name="condense-question",
                run_type="chain",
                inputs={
                    "follow_up": question,
                    "history_turns": len(history),
                },
                metadata={"purpose": "history-aware retrieval rewrite"},
            ) as span:
                rewritten = _run()
                span.end(outputs={"standalone_query": (rewritten or "").strip()})
        except Exception:
            return question

    rewritten = (rewritten or "").strip().strip('"').strip()
    return rewritten or question
