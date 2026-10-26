"""Tavily tools that preserve normalized source data outside LLM messages."""

from langchain_core.tools import tool

from models import SearchResult


def normalize_results(response: dict) -> list[dict]:
    return [
        SearchResult(
            title=item.get("title") or "Untitled",
            url=item.get("url") or "",
            content=item.get("content") or "",
            published_date=item.get("published_date") or None,
        ).model_dump()
        for item in response.get("results", [])
        if item.get("url") and item.get("content")
    ]


class SearchTools:
    def __init__(self, client, news_days: int = 30):
        self.client = client
        self.news_days = news_days
        self.records = {"general": [], "news": []}
        self.calls = {"general": 0, "news": 0}
        self.cache = {}

        @tool
        def tavily_web_search(query: str) -> list[dict]:
            """Search the web for company competitors, official pricing, products, and features."""
            return self.search(query, "general")

        @tool
        def tavily_news_search(query: str) -> list[dict]:
            """Search recent news for company launches, announcements, and business developments."""
            return self.search(query, "news")

        self.web = tavily_web_search
        self.news = tavily_news_search

    def search(self, query: str, topic: str) -> list[dict]:
        key = (topic, query)
        if key in self.cache:
            return self.cache[key]
        # Bound provider usage even if an agent repeatedly requests searches.
        if sum(self.calls.values()) >= 6:
            raise RuntimeError("Search budget exceeded for this agent")
        options = {"days": self.news_days} if topic == "news" else {}
        self.calls[topic] += 1
        response = self.client.search(
            query=query, topic=topic, max_results=5,
            search_depth="advanced", include_answer=False, timeout=30, **options,
        )
        results = normalize_results(response)
        self.cache[key] = results
        seen = {item["url"] for item in self.records[topic]}
        for item in results:
            if item["url"] not in seen:
                self.records[topic].append(item)
                seen.add(item["url"])
        return results
