from langsmith import trace

from api.api.models import SourceItem
from api.core.config import config
from api.core.observability import get_langsmith_client

from api.agents.legal_chat.embedding import embed_text_query_with_trace
from shared.qdrant import build_client

# Parent-document retrieval (#4): pull several chunks per query, then collapse to
# unique sections so the LLM receives whole sections (not headless split parts)
# and duplicate parts of one source don't crowd out other relevant ones.
CANDIDATE_MULTIPLIER = 5
MIN_CANDIDATES = 20

_qdrant_client = build_client(
    config.QDRANT_VECTORESTORE, config.QDRANT_API_KEY, timeout=30
)


def verify_qdrant() -> None:
    """Boot-time readiness probe for the vector store.

    Confirms Qdrant is reachable (a bad URL / down server / bad API key raises
    here) and that the required statute collection exists. Called from the app
    lifespan so a dead or misconfigured vector DB fails the container's
    healthcheck at startup instead of surfacing as a 500 on the first chat
    request.
    """
    names = {c.name for c in _qdrant_client.get_collections().collections}
    if config.QDRANT_COLLECTION not in names:
        raise RuntimeError(
            f"Qdrant reachable but required collection "
            f"{config.QDRANT_COLLECTION!r} not found (have: {sorted(names)})"
        )


def _embed(question: str, traced: bool) -> list[float]:
    return embed_text_query_with_trace(question, max_input_chars=2048, traced=traced)


def _search(collection: str, vector: list[float], candidate_limit: int, traced: bool):
    """Run one Qdrant query against `collection`; optionally trace it."""
    if not traced:
        return _qdrant_client.query_points(
            collection_name=collection,
            query=vector,
            limit=candidate_limit,
            with_payload=True,
        ).points
    with trace(
        name="vector-search",
        run_type="retriever",
        inputs={"collection": collection, "candidate_limit": candidate_limit},
        metadata={"provider": "qdrant"},
    ) as search_span:
        hits = _qdrant_client.query_points(
            collection_name=collection,
            query=vector,
            limit=candidate_limit,
            with_payload=True,
        ).points
        search_span.end(outputs={"hit_count": len(hits)})
        return hits


def _hits_to_sources(hits, top_k: int) -> list[SourceItem]:
    """Collapse raw chunk hits to the top_k unique statute sections (parent-doc)."""
    sources: list[SourceItem] = []
    seen: set = set()
    for hit in hits:
        payload = hit.payload or {}
        # section_uid is written by the improved ingest; fall back to a composite key
        # so this still behaves sanely against older collections without it.
        uid = payload.get("section_uid") or (
            f"{payload.get('source_url')}#{payload.get('section_index')}"
        )
        if uid in seen:
            continue
        seen.add(uid)
        # Prefer the full section text; fall back to the chunk excerpt.
        excerpt = str(
            payload.get("section_full") or payload.get("section_content_clean") or ""
        ).strip() or "No excerpt available."
        sources.append(
            SourceItem(
                citation_id=len(sources) + 1,
                act_title=payload.get("act_title"),
                act_year=payload.get("act_year"),
                section_index=(
                    str(payload.get("section_index"))
                    if payload.get("section_index") is not None
                    else None
                ),
                source_url=payload.get("source_url"),
                excerpt=excerpt,
                score=float(hit.score or 0.0),
            )
        )
        if len(sources) >= top_k:
            break
    return sources


def retrieve_sources(
    question: str, top_k: int, *, vector: list[float] | None = None
) -> list[SourceItem]:
    """Retrieve statute sections from the acts collection."""
    candidate_limit = max(top_k * CANDIDATE_MULTIPLIER, MIN_CANDIDATES)
    traced = get_langsmith_client() is not None
    if vector is None:
        vector = _embed(question, traced)
    hits = _search(config.QDRANT_COLLECTION, vector, candidate_limit, traced)
    sources = [
        s
        for s in _hits_to_sources(hits, top_k)
        if s.score >= config.STATUTE_SCORE_FLOOR
    ]
    for new_id, source in enumerate(sources, start=1):
        source.citation_id = new_id
    return sources
