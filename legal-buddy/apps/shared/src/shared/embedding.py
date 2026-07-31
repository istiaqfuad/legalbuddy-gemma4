"""Embedding model loading + e5 prefix handling, shared by the API and ingestion.

e5-family models are asymmetric: passages get a "passage:" prefix at ingestion,
queries get a "query:" prefix at search. Both sides MUST use the same model and
prefixes or retrieval breaks — which is exactly why this lives in one place.
"""
from functools import lru_cache


def is_e5(model_name: str | None) -> bool:
    return bool(model_name) and "e5" in model_name.lower()


def passage_prefix(text: str, e5: bool) -> str:
    return f"passage: {text}" if e5 else text


def query_prefix(text: str, e5: bool) -> str:
    return f"query: {text}" if e5 else text


@lru_cache(maxsize=4)
def load_embedding_model(
    model_name: str,
    token: str | None = None,
    max_tokens: int | None = None,
    device: str = "cpu",
):
    """Load a sentence-transformers model, cached per (model, token, window, device).

    max_tokens overrides the model's max_seq_length — some e5 checkpoints ship a
    128-token window but support 512; set it to embed longer chunks without
    silent truncation. ``device`` is "cpu" by default; pass "cuda" to embed on a
    GPU (e.g. the Fedora RTX 2070 for the large cases corpus). The API/query side
    stays on CPU; only the bulk ingestion run needs the GPU.
    """
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name, device=device, token=token)
    if max_tokens:
        model.max_seq_length = int(max_tokens)
        model.tokenizer.model_max_length = int(max_tokens)
    return model


def model_max_tokens(model) -> int:
    return int(getattr(model, "max_seq_length", 0) or 128)


def embed_passages(model, texts: list[str], e5: bool, batch_size: int = 32) -> list[list[float]]:
    if not texts:
        return []
    vectors = model.encode(
        [passage_prefix(t, e5) for t in texts],
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return [vec.tolist() for vec in vectors]


def embed_query(model, text: str, e5: bool) -> list[float]:
    vector = model.encode(query_prefix(text, e5), normalize_embeddings=True)
    return vector.tolist()
