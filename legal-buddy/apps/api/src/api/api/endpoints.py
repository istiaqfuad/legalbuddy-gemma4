import json
import logging

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from api.api.models import LegalChatRequest, LegalChatResponse
from api.agents.legal_chat.pipeline import (
    legal_chat_pipeline,
    legal_chat_pipeline_stream,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

rag_router = APIRouter()


def _client_error(exc: Exception) -> tuple[int, str]:
    """Map an internal exception to a (status, user-safe message) pair.

    The raw exception text (provider JSON, stack details, keys) must never reach
    the browser — it is logged server-side instead. Known failure classes get a
    helpful, plain-language message; everything else gets a generic one."""
    low = str(exc).lower()
    if any(
        k in low
        for k in ("resource_exhausted", "429", "quota", "rate limit", "rate-limit")
    ):
        return 429, (
            "The model provider is rate-limited right now (free-tier quota "
            "reached). Wait a few seconds and try again, or switch the provider "
            "or model in settings."
        )
    if any(
        k in low
        for k in ("api_key", "api key", "unauthenticated", "permission denied", "401", "403")
    ):
        return 503, (
            "The assistant is temporarily unavailable due to a configuration "
            "issue. Please try again later."
        )
    return 500, "Something went wrong while generating the answer. Please try again."


@rag_router.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@rag_router.post("/legal/chat", response_model=LegalChatResponse)
async def legal_chat(payload: LegalChatRequest) -> LegalChatResponse:
    try:
        return legal_chat_pipeline(
            payload.question,
            history=payload.history,
            top_k=payload.top_k,
            max_tokens=payload.max_tokens,
            provider=payload.provider,
            model=payload.model,
            temperature=payload.temperature,
            clarify_score_floor=payload.clarify_score_floor,
            low_confidence_floor=payload.low_confidence_floor,
        )
    except Exception as exc:
        logger.exception("Error in /legal/chat")
        status, message = _client_error(exc)
        raise HTTPException(status_code=status, detail=message) from exc


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@rag_router.post("/legal/chat/stream")
async def legal_chat_stream(payload: LegalChatRequest) -> StreamingResponse:
    """SSE stream: a `sources` event, then `delta` events, then `done`.

    Errors raised mid-stream are emitted as an `error` event (the HTTP status is
    already 200 once streaming starts).
    """

    def event_stream():
        try:
            for event in legal_chat_pipeline_stream(
                payload.question,
                history=payload.history,
                top_k=payload.top_k,
                max_tokens=payload.max_tokens,
                provider=payload.provider,
                model=payload.model,
                temperature=payload.temperature,
                clarify_score_floor=payload.clarify_score_floor,
                low_confidence_floor=payload.low_confidence_floor,
            ):
                etype = event.get("type")
                if etype == "sources":
                    yield _sse("sources", {"sources": event["sources"]})
                elif etype == "delta":
                    yield _sse("delta", {"text": event["text"]})
                elif etype == "done":
                    yield _sse("done", {})
        except Exception as exc:
            logger.exception("Error in /legal/chat/stream")
            _, message = _client_error(exc)
            yield _sse("error", {"error": message})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


api_router = APIRouter()
api_router.include_router(rag_router, prefix="/rag", tags=["RAG"])
