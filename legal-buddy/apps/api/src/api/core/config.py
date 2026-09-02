from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # Chat LLM. provider selects the backend; each has its own default model.
    # "groq" (and its alias "openai") is ANY OpenAI-compatible /v1/chat/completions
    # endpoint — real Groq cloud, llama.cpp, vLLM, LM Studio — switched purely via
    # the GROQ_* env vars below, no code changes.
    DEFAULT_LLM_PROVIDER: str = "groq"  # "groq" | "gemini" ("openai" == "groq")
    GEMINI_API_KEY: str | None = None
    CHAT_MODEL: str = "gemini-2.5-flash"
    # OpenAI-compatible endpoint (Groq cloud, llama.cpp, vLLM, ...).
    GROQ_API_KEY: str | None = None
    GROQ_BASE_URL: str = "https://api.groq.com/openai/v1"
    GROQ_MODEL: str = "llama-3.3-70b-versatile"
    # Total request timeout for the OpenAI-compatible client. Generous default:
    # a CPU-only llama.cpp server can take minutes to first token (prompt
    # processing), and llama.cpp only starts accepting connections after the
    # model finishes loading. The OpenAI SDK's own default connect timeout
    # (5s) is far too tight for that.
    GROQ_TIMEOUT_SECONDS: float = 300.0

    # Fast/cheap model for the multi-turn query rewrite (history-aware retrieval).
    # Independent of the answer model above — the rewrite is a tiny, latency-
    # sensitive call. Leave unset to reuse the answer model: REQUIRED for
    # single-model servers (llama.cpp serves one alias and rejects other names).
    GEMINI_CONDENSE_MODEL: str = "gemini-2.5-flash-lite"
    GROQ_CONDENSE_MODEL: str | None = None  # None -> GROQ_MODEL
    # Turns of conversation history kept for the rewrite and answer prompt.
    HISTORY_WINDOW_TURNS: int = 6

    # HuggingFace embedding model (run locally via sentence-transformers).
    # EMBEDDING_DEVICE: "cpu" (default) or "cuda" — the e5-base encoder is a
    # 278M-parameter transformer, so a CPU-only container can take seconds per
    # query while a GPU does it in ~10ms.
    HF_TOKEN: str | None = None
    EMBEDDING_MODEL: str
    EMBEDDING_DEVICE: str = "cpu"

    LANGSMITH_TRACING: bool = True
    LANGSMITH_API_KEY: str | None = None
    LANGSMITH_ENDPOINT: str | None = None
    LANGSMITH_PROJECT: str | None = "legal-buddy"
    # Required only for org-scoped API keys.
    LANGSMITH_WORKSPACE_ID: str | None = None

    # Qdrant vector store
    QDRANT_VECTORESTORE: str = "http://213.136.80.53:6333"
    QDRANT_API_KEY: str | None = None
    QDRANT_COLLECTION: str = "legal_acts_event_rag_full"
    POSTGRES_CONNECTION_STRING: str | None = None

    RETRIEVAL_TOP_K: int = 8
    # Minimum cosine score for a retrieved statute to count as relevant.
    STATUTE_SCORE_FLOOR: float = 0.0
    ANSWER_MAX_TOKENS: int | None = None
    # Two-tier clarify control. The e5 cosine is a WEAK separator — measured,
    # off-topic "what time is it" scores 0.836 while answerable "my neighbor keeps
    # threatening me" tops out at 0.818 (with the correct §503 as its top hit). So
    # no single floor cleanly splits answerable from off-topic. Instead:
    #   • CLARIFY_SCORE_FLOOR (hard, low): only genuine garbage below this routes
    #     to a deterministic no-source clarify (e.g. "best pizza recipe" 0.781).
    #   • LOW_CONFIDENCE_FLOOR (soft, higher): between the two, the turn still goes
    #     to the model WITH its sources plus a low-confidence hint, and the model
    #     decides whether to answer or ask. Borderline judgment is the model's, not
    #     a brittle cutoff. A cross-encoder reranker is the robust long-term fix.
    CLARIFY_SCORE_FLOOR: float = 0.79
    LOW_CONFIDENCE_FLOOR: float = 0.83


config = Config()
