from __future__ import annotations

from typing import Literal, Optional
from urllib.parse import urlparse

from pydantic import BaseModel, Field, model_validator


# ---------------------------------------------------------------------------
# Search + Ingestion models (steps 1–6)
# ---------------------------------------------------------------------------

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
    raw_text: str                    # full stripped page text (for length gate)
    ingredients_excerpt: str = ""    # BS4-extracted ingredients section sent to LLM parser
    fetch_method: Literal["requests_bs4"]
    fetch_error: Optional[str] = None
    snippet: SearchSnippet           # provenance — back-reference to the originating snippet


# ---------------------------------------------------------------------------
# LLM structured output models (query plan, snippet scoring)
# ---------------------------------------------------------------------------

class QueryPlan(BaseModel):
    """Output of step 1 — query understanding + diversification in a single LLM call."""
    category: Literal["cookie", "quick_bread", "cake", "yeasted_bread", "pastry", "other"]
    flour_type: str = "ap"           # primary flour: ap, almond, oat, coconut, rice, gf_blend, other
    modifiers: list[str] = Field(default_factory=list)  # e.g. ["gluten_free", "vegan", "paleo"]
    hard_constraints: list[str]      # must be true ("has chocolate", "stays moist for days")
    soft_preferences: list[str]      # inform weighting ("not too sweet", "crispy edges")
    queries: list[str]               # diversified search query variants (exactly 2)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)  # category classification confidence
    clarification_question: str = ""  # populated when confidence < 0.7


class SnippetScore(BaseModel):
    index: int
    score: float = Field(ge=0.0, le=1.0)
    reason: str


class SnippetScoreBatch(BaseModel):
    scores: list[SnippetScore]


# ---------------------------------------------------------------------------
# Recipe parsing models (step 7)
# ---------------------------------------------------------------------------

class RecipeIngredient(BaseModel):
    name: str                        # normalized lowercase name, e.g. "all-purpose flour"
    quantity: float                  # numeric amount
    unit: str                        # "cups", "tbsp", "tsp", "g", "oz", "whole", "sticks"
    low_confidence: bool = False     # flagged when normalization is uncertain


class ParsedRecipe(BaseModel):
    title: str
    url: str
    category: Literal["cookie", "quick_bread", "cake", "yeasted_bread", "pastry", "other"]
    flour_type: str = "ap"           # primary flour detected from ingredients
    modifiers: list[str] = Field(default_factory=list)  # e.g. ["gluten_free", "vegan"]
    ingredients: list[RecipeIngredient]
    yield_description: str = ""      # "1 loaf", "24 cookies"
    instruction_count: int = 0       # proxy for recipe specificity
    has_chocolate: bool = False      # True if any chocolate ingredient present
    technique_signals: list[str] = Field(default_factory=list)  # structured vocab signals
    technique_notes: str = ""    # free-text description of novel/unusual techniques
    parse_error: Optional[str] = None


# ---------------------------------------------------------------------------
# Normalization + ratio models (steps 8–9)
# ---------------------------------------------------------------------------

class NormalizedIngredient(BaseModel):
    name: str
    grams: float
    low_confidence: bool = False


class RatioResult(BaseModel):
    url: str
    category: Literal["cookie", "quick_bread", "cake", "yeasted_bread", "pastry", "other"]
    flour_type: str = "ap"           # primary flour type (propagated from ParsedRecipe)
    modifiers: list[str] = Field(default_factory=list)  # e.g. ["gluten_free"]

    # Quick bread / cake ratios
    liquid_to_flour: Optional[float] = None
    fat_to_flour: Optional[float] = None
    sugar_to_flour: Optional[float] = None
    leavening_to_flour: Optional[float] = None
    fat_type: Optional[Literal["oil", "butter", "mixed", "none"]] = None
    has_banana: bool = False

    # Cookie-specific ratios
    brown_to_white_sugar: Optional[float] = None
    butter_to_flour: Optional[float] = None
    has_extra_yolks: bool = False
    leavening_type: Optional[Literal["soda", "powder", "both", "none"]] = None

    has_chocolate: bool = False
    has_binding_agent: bool = False  # True if xanthan gum, psyllium, etc. detected
    flour_grams: float = 0.0         # raw flour weight for reference
    from_cache: bool = False


# ---------------------------------------------------------------------------
# Scoring models (step 10)
# ---------------------------------------------------------------------------

class CriterionScore(BaseModel):
    name: str
    score: float                     # 0–100
    weight: float                    # relative weight for this query
    details: str = ""                # short human-readable detail (no LLM)


class ScoredRecipe(BaseModel):
    recipe: ParsedRecipe
    ratios: RatioResult
    criteria: list[CriterionScore]
    composite_score: float           # 0–100 weighted composite
    constraint_violations: list[str] = Field(default_factory=list)
    explanation: str = ""            # filled by batched LLM call after scoring math
    rank: int = 0
    technique_note_delta: Optional[float] = None  # LLM-applied delta for novel technique (None = not applied)
    accessibility_score: Optional[float] = None   # LLM-assessed ease-of-making score (None = not computed)
