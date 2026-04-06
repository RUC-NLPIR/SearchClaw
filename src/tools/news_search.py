"""
News search tool — searches for recent news articles.

Uses NewsAPI or Google News RSS as a source. Specialized for
time-sensitive queries, current events, and recent developments.

Concurrency-safe: multiple news searches can run in parallel.
"""

from __future__ import annotations

import logging
import os
import re
from datetime import datetime, timedelta

import httpx

from src.core.tool import Tool, ToolUseContext
from src.core.types import Citation, SourceType, ToolResult, ValidationResult

logger = logging.getLogger(__name__)


class NewsSearchTool(Tool):
    name = "news_search"
    description = (
        "Search for recent news articles. Use this for current events, "
        "breaking news, recent developments, or time-sensitive topics. "
        "Returns articles from major news outlets."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query for news articles.",
            },
            "num_results": {
                "type": "integer",
                "description": "Number of articles to return (default: 5, max: 10).",
                "default": 5,
            },
            "days_back": {
                "type": "integer",
                "description": "How many days back to search (default: 7, max: 30).",
                "default": 7,
            },
        },
        "required": ["query"],
    }

    is_concurrency_safe = True
    is_read_only = True

    def __init__(
        self,
        default_results: int = 5,
        max_results: int = 10,
        default_days_back: int = 7,
        max_days_back: int = 30,
        max_result_size_chars: int = 15_000,
        http_timeout: int = 30,
    ):
        self.default_results = default_results
        self.max_results = max_results
        self.default_days_back = default_days_back
        self.max_days_back = max_days_back
        self.max_result_size_chars = max_result_size_chars
        self._client = httpx.AsyncClient(timeout=float(http_timeout))

    def prompt(self) -> str:
        return (
            "Use news_search for current events and recent developments. Tips:\n"
            "- Use for time-sensitive questions ('What happened with...', 'Latest on...')\n"
            "- Set days_back to narrow or widen the time window\n"
            "- Cross-reference news with web_search for more context\n"
            "- Note article dates when citing — news can become outdated quickly"
        )

    def validate_input(self, args: dict) -> ValidationResult:
        query = args.get("query", "")
        if not query or len(query.strip()) < 2:
            return ValidationResult(valid=False, message="Query must be at least 2 characters")
        return ValidationResult(valid=True)

    async def call(self, args: dict, context: ToolUseContext) -> ToolResult:
        query = args["query"]
        num_results = min(args.get("num_results", self.default_results), self.max_results)
        days_back = min(args.get("days_back", self.default_days_back), self.max_days_back)

        # Try NewsAPI first, fall back to Google News RSS
        articles = await self._search_newsapi(query, num_results, days_back)

        if not articles:
            articles = await self._search_google_news_rss(query, num_results)

        if not articles:
            return ToolResult(
                data="No recent news articles found. Try web_search for broader results.",
            )

        # Format results
        formatted_parts = [f"## News Results: {query}\n"]
        citations = []

        for i, article in enumerate(articles, 1):
            title = article.get("title", "Untitled")
            url = article.get("url", "")
            source = article.get("source", "Unknown")
            published = article.get("published", "Unknown date")
            description = article.get("description", "No description available")

            formatted_parts.append(
                f"### {i}. {title}\n"
                f"**Source**: {source} | **Published**: {published}\n"
                f"**URL**: {url}\n"
                f"**Summary**: {description}\n"
            )

            citations.append(Citation(
                url=url,
                title=title,
                snippet=description[:300],
                source_type=SourceType.NEWS,
            ))

        formatted = "\n".join(formatted_parts)
        return ToolResult(data=formatted, citations=citations)

    async def _search_newsapi(
        self, query: str, num_results: int, days_back: int
    ) -> list[dict]:
        """Search using NewsAPI (requires NEWSAPI_KEY)."""
        api_key = os.environ.get("NEWSAPI_KEY", "")
        if not api_key:
            return []

        from_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")

        try:
            response = await self._client.get(
                "https://newsapi.org/v2/everything",
                params={
                    "q": query,
                    "from": from_date,
                    "sortBy": "relevancy",
                    "pageSize": num_results,
                    "language": "en",
                    "apiKey": api_key,
                },
            )
            response.raise_for_status()
            data = response.json()

            articles = []
            for item in data.get("articles", []):
                articles.append({
                    "title": item.get("title", ""),
                    "url": item.get("url", ""),
                    "source": item.get("source", {}).get("name", "Unknown"),
                    "published": item.get("publishedAt", "")[:10],
                    "description": item.get("description", ""),
                })

            return articles

        except Exception as e:
            logger.error(f"NewsAPI search failed: {e}")
            return []

    async def _search_google_news_rss(
        self, query: str, num_results: int
    ) -> list[dict]:
        """Fallback: search Google News via RSS feed (no API key needed)."""
        try:
            import urllib.parse
            encoded_query = urllib.parse.quote(query)
            rss_url = f"https://news.google.com/rss/search?q={encoded_query}&hl=en-US&gl=US&ceid=US:en"

            response = await self._client.get(
                rss_url,
                headers={
                    "User-Agent": "Mozilla/5.0 (compatible; SearchAgent/1.0)",
                },
            )
            response.raise_for_status()
            xml_text = response.text

            # Simple XML parsing (no external dependency)
            articles = []
            items = re.findall(r"<item>(.*?)</item>", xml_text, re.DOTALL)

            for item in items[:num_results]:
                title_match = re.search(r"<title>(.*?)</title>", item)
                link_match = re.search(r"<link>(.*?)</link>", item)
                pub_date_match = re.search(r"<pubDate>(.*?)</pubDate>", item)
                source_match = re.search(r"<source[^>]*>(.*?)</source>", item)
                desc_match = re.search(r"<description>(.*?)</description>", item, re.DOTALL)

                title = title_match.group(1) if title_match else "Untitled"
                url = link_match.group(1) if link_match else ""
                pub_date = pub_date_match.group(1) if pub_date_match else ""
                source = source_match.group(1) if source_match else "Unknown"
                description = desc_match.group(1) if desc_match else ""

                # Clean HTML from description
                description = re.sub(r"<[^>]+>", "", description).strip()
                # Clean CDATA
                title = title.replace("<![CDATA[", "").replace("]]>", "").strip()

                if url:
                    articles.append({
                        "title": title,
                        "url": url,
                        "source": source,
                        "published": pub_date[:16] if pub_date else "",
                        "description": description[:500],
                    })

            return articles

        except Exception as e:
            logger.error(f"Google News RSS search failed: {e}")
            return []
