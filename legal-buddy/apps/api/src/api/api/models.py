from typing import Literal

from pydantic import BaseModel


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class LegalChatRequest(BaseModel):
    question: str
    # Prior turns (oldest->newest), excluding the current question. Powers
    # multi-turn memory: history-aware retrieval + conversation context in the
    # prompt. Empty for the first turn.
    history: list[ChatMessage] = []
    max_tokens: int | None = None
    top_k: int | None = None
    # Testing knobs (remove in production). provider/model/temperature let the
    # frontend switch LLM backend and tune generation per request.
    provider: Literal["gemini", "groq"] | None = None
    model: str | None = None
    temperature: float | None = None
    # Per-request overrides for the two clarify thresholds (sidebar sliders).
    # None -> fall back to the config defaults. clarify_score_floor is the hard
    # no-match floor (below it -> deterministic clarify); low_confidence_floor is
    # the soft hint floor (below it -> nudge the model toward clarifying).
    clarify_score_floor: float | None = None
    low_confidence_floor: float | None = None


class SourceItem(BaseModel):
    citation_id: int
    # Statute fields
    act_title: str | None = None
    act_year: int | None = None
    section_index: str | None = None
    source_url: str | None = None
    excerpt: str
    score: float


class LegalChatResponse(BaseModel):
    answer: str
    sources: list[SourceItem]
