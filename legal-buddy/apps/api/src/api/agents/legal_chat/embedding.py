from langsmith import trace

from api.core.config import config
from shared.embedding import embed_query, is_e5, load_embedding_model


def get_embedding_model():
    """The query-side embedding model — same loader the ingestion side uses."""
    return load_embedding_model(config.EMBEDDING_MODEL, config.HF_TOKEN)


def embed_text_query(text: str, *, max_input_chars: int = 2048) -> list[float]:
    model = get_embedding_model()
    return embed_query(model, text[:max_input_chars], is_e5(config.EMBEDDING_MODEL))


def embed_text_query_with_trace(
    text: str,
    *,
    max_input_chars: int,
    traced: bool,
) -> list[float]:
    if not traced:
        return embed_text_query(text, max_input_chars=max_input_chars)

    query_text = text[:max_input_chars]
    with trace(
        name="embed-query",
        run_type="embedding",
        inputs={
            "input_chars": len(query_text),
            "max_input_chars": max_input_chars,
        },
        metadata={
            "provider": "sentence-transformers",
            "ls_model_name": config.EMBEDDING_MODEL,
        },
    ) as embedding_span:
        vector = embed_text_query(text, max_input_chars=max_input_chars)
        embedding_span.end(outputs={"embedding_dimensions": len(vector)})
        return vector
