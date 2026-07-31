from pydantic import BaseModel, Field


class StructuredLegalAnswer(BaseModel):
    answer: str = Field(
        description="The answer in Markdown, grounded in the sources and cited as "
        "[Source N]; structured however best fits the question (prose, bullets, or "
        "a single line). For a user's own situation, may end with one short "
        "follow-up question."
    )
    citations: list[int] = Field(
        default_factory=list,
        description="List of supporting source ids, e.g. [1, 2]",
    )
    limitations: str | None = Field(
        default=None,
        description="Optional uncertainty or missing context",
    )
