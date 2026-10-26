"""Validated contracts between discovery, research, analysis, and the UI."""

import operator
from typing import Annotated, List

from langchain.agents import AgentState
from pydantic import BaseModel, ConfigDict, Field, field_validator


class ResearchState(AgentState):
    """Shared TypedDict; AgentState supplies LangChain's message channels."""

    company: str
    competitor_queue: List[str]
    current_target: str
    raw_data: str
    final_reports: Annotated[List[dict], operator.add]


class Schema(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")


class SearchResult(Schema):
    title: str
    url: str
    content: str
    published_date: str | None = None


class Competitor(Schema):
    name: str = Field(min_length=1)
    reason: str = Field(description="Why this company competes in the primary market")


class DiscoveryResult(Schema):
    company_name: str
    primary_market: str
    competitors: list[Competitor] = Field(min_length=3, max_length=3)

    @field_validator("competitors")
    @classmethod
    def unique_competitors(cls, value):
        if len({item.name.casefold() for item in value}) != 3:
            raise ValueError("Return three distinct competitors")
        return value


class NewsItem(Schema):
    title: str
    summary: str
    url: str
    published_date: str | None = None


class CompetitorReport(Schema):
    competitor_name: str
    pricing_model: str = Field(description="How they make money; exact prices, currency, and billing period only if supported. Say 'Not found' for unknowns.")
    core_features: list[str] = Field(min_length=3, max_length=5)
    market_positioning: str = Field(description="Target customers, market segment, and differentiation grounded in sources")
    recent_news: list[NewsItem] = Field(description="Recent launches or news from supplied news evidence; empty if none found")
    source_urls: list[str] = Field(min_length=1, description="URLs from supplied evidence supporting this report")


class ResearchBundle(Schema):
    competitor_name: str
    web_results: list[SearchResult]
    news_results: list[SearchResult]


class MarketReport(Schema):
    discovery: DiscoveryResult
    reports: list[CompetitorReport]
    research: list[ResearchBundle]
    errors: dict[str, str]
