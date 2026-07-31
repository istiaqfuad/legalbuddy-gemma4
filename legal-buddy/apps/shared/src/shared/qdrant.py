"""Qdrant client + collection helpers shared by the API and ingestion."""
import time
from urllib.parse import urlparse


def build_client(url: str, api_key: str | None = None, timeout: int = 30):
    from qdrant_client import QdrantClient

    parsed = urlparse(url)
    kwargs: dict = {"url": url, "api_key": api_key, "timeout": timeout}
    # qdrant-client defaults to port 6333 when the URL omits a port; an https
    # endpoint behind a reverse proxy (e.g. Cloudflare) is served on 443.
    if parsed.scheme == "https" and parsed.port is None:
        kwargs["port"] = 443
    return QdrantClient(**kwargs)


def recreate_collection(
    client,
    name: str,
    vector_size: int,
    *,
    keyword_indexes: tuple[str, ...] = (),
    integer_indexes: tuple[str, ...] = (),
) -> None:
    """Drop + recreate a cosine collection, then build the requested payload indexes."""
    from qdrant_client.http import models

    existing = {c.name for c in client.get_collections().collections}
    if name in existing:
        client.delete_collection(collection_name=name)

    client.create_collection(
        collection_name=name,
        vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE),
    )
    for field in integer_indexes:
        client.create_payload_index(
            collection_name=name, field_name=field,
            field_schema=models.PayloadSchemaType.INTEGER, wait=True,
        )
    for field in keyword_indexes:
        client.create_payload_index(
            collection_name=name, field_name=field,
            field_schema=models.PayloadSchemaType.KEYWORD, wait=True,
        )


def upsert_with_retry(client, name: str, points: list, attempts: int = 4) -> None:
    for attempt in range(1, attempts + 1):
        try:
            client.upsert(collection_name=name, points=points, wait=True)
            return
        except Exception as exc:  # transient network/proxy timeouts
            if attempt == attempts:
                raise
            print(f"[qdrant] upsert retry {attempt}/{attempts} after error: {exc}")
            time.sleep(2 * attempt)
