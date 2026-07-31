import logging
import os

from langsmith import Client
from langsmith.utils import tracing_is_enabled

from api.core.config import config

logger = logging.getLogger(__name__)

_langsmith_client: Client | None = None


def configure_tracing() -> None:
    """Forward config values into the env vars the LangSmith SDK reads.

    The SDK is driven entirely by environment variables, but settings are loaded
    from ``.env`` via pydantic (which does not export them to the process
    environment), so we bridge them explicitly here.
    """
    if not config.LANGSMITH_TRACING or not config.LANGSMITH_API_KEY:
        os.environ["LANGSMITH_TRACING"] = "false"
        return

    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGSMITH_API_KEY"] = config.LANGSMITH_API_KEY
    if config.LANGSMITH_ENDPOINT:
        os.environ["LANGSMITH_ENDPOINT"] = config.LANGSMITH_ENDPOINT
    if config.LANGSMITH_PROJECT:
        os.environ["LANGSMITH_PROJECT"] = config.LANGSMITH_PROJECT


# Apply configuration on import, before any tracing runs.
configure_tracing()


def get_langsmith_client() -> Client | None:
    global _langsmith_client

    if not tracing_is_enabled():
        return None

    if _langsmith_client is None:
        _langsmith_client = Client()
    return _langsmith_client


def validate_langsmith_auth() -> None:
    client = get_langsmith_client()
    if client is None:
        logger.info(
            "LangSmith disabled or not configured. Set LANGSMITH_TRACING=true and "
            "LANGSMITH_API_KEY to enable tracing."
        )
        return

    try:
        next(iter(client.list_projects(limit=1)), None)
        logger.info("LangSmith auth check passed.")
    except Exception:
        logger.exception("LangSmith auth check failed. Traces may not be ingested.")


def flush_langsmith() -> None:
    client = get_langsmith_client()
    if client is None:
        return

    try:
        client.flush()
    except Exception:
        logger.exception("Failed to flush LangSmith events.")
