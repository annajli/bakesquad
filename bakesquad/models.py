from __future__ import annotations

from typing import Literal, Optional
from urllib.parse import urlparse

from pydantic import BaseModel, Field, model_validator


class SearchSnippet(BaseModel):
    url: str
    title: str
    excerpt: str
    domain: str = Field(default="")
    query_source: str = ""           # which diversified query produced this result
    relevance_score: Optional[float] = None
    skip_reason: Optional[str] = None  # "paywall" | "low_score" | "domain_cap"

    @model_validator(mode="after")
    def _derive_domain(self) -> SearchSnippet:
        if not self.domain:
            self.domain = urlparse(self.url).netloc.removeprefix("www.")
        return self


class FetchedPage(BaseModel):
    url: str
    title: str
    raw_text: str
    fetch_method: Literal["webbaseloader", "requests_bs4"]
    fetch_error: Optional[str] = None
    snippet: SearchSnippet           # provenance — back-reference to the originating snippet


# --- Structured output models for LLM calls ---

class QueryVariants(BaseModel):
    queries: list[str]


class SnippetScore(BaseModel):
    index: int
    score: float = Field(ge=0.0, le=1.0)
    reason: str


class SnippetScoreBatch(BaseModel):
    scores: list[SnippetScore]
