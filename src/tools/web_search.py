"""
Web search tool — searches the web via API (Serper, SerpAPI, or Google Custom Search).

This is the primary discovery tool. It returns search result snippets
which the agent uses to decide which pages to fetch in full.

Concurrency-safe: multiple web searches can run in parallel.
"""

from __future__ import annotations

import logging
import os

import httpx

from src.core.tool import Tool, ToolUseContext
from src.core.types import Citation, SourceType, ToolResult, ValidationResult

logger = logging.getLogger(__name__)


class WebSearchTool(Tool):
    name = "web_search"
    description = (
        "Search the web using a search engine API. Returns a list of results "
        "with titles, URLs, and snippets. Use this to find relevant pages, "
        "then use web_fetch to read the most promising results."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query. Be specific and use relevant keywords.",
            },
            "num_results": {
                "type": "integer",
                "description": "Number of results to return (default: 10, max: 20).",
                "default": 10,
            },
        },
        "required": ["query"],
    }

    is_concurrency_safe = True
    is_read_only = True

    def __init__(
        self,
        default_results: int = 10,
        max_results: int = 20,
        max_result_size_chars: int = 15000,
        http_timeout: int = 30,
    ):
        self.default_results = default_results
        self.max_results = max_results
        self.max_result_size_chars = max_result_size_chars
        self._client = httpx.AsyncClient(timeout=float(http_timeout))
        # Browser integration (set externally by build_default_registry
        # when browser.enabled and browser.use_for_search are true)
        self._browser_manager = None
        self._search_engine = "google"

    def prompt(self) -> str:
        return (
            "Use web_search to find relevant pages for a topic. Tips:\n"
            "- Use specific, targeted queries (not vague ones)\n"
            "- Try multiple queries with different phrasing for thorough research\n"
            "- Add date qualifiers for time-sensitive topics (e.g., '2024' or 'latest')\n"
            "- After searching, use web_fetch to read the most relevant results"
        )

    def validate_input(self, args: dict) -> ValidationResult:
        query = args.get("query", "")
        if not query or len(query.strip()) < 2:
            return ValidationResult(valid=False, message="Query must be at least 2 characters")
        if len(query) > 500:
            return ValidationResult(valid=False, message="Query too long (max 500 chars)")
        return ValidationResult(valid=True)

    async def call(self, args: dict, context: ToolUseContext) -> ToolResult:
        query = args["query"]
        num_results = min(args.get("num_results", self.default_results), self.max_results)

        # Try providers in order
        results = await self._search_serper(query, num_results)

        if not results:
            return ToolResult(
                data="No search results found. Try a different query.",
                is_error=False,
            )

        # Format results
        formatted_parts = [f"## Search Results for: {query}\n"]
        citations = []

        for i, r in enumerate(results, 1):
            title = r.get("title", "Untitled")
            url = r.get("link", r.get("url", ""))
            snippet = r.get("snippet", "No description available")

            formatted_parts.append(
                f"### {i}. {title}\n"
                f"**URL**: {url}\n"
                f"**Snippet**: {snippet}\n"
            )

            citations.append(Citation(
                url=url,
                title=title,
                snippet=snippet,
                source_type=SourceType.WEB,
            ))

        formatted = "\n".join(formatted_parts)
        formatted, truncated, cached_path = await self._maybe_truncate(
            formatted, query, context
        )

        return ToolResult(
            data=formatted,
            citations=citations,
            truncated=truncated,
            cached_path=cached_path,
        )

    async def _search_serper(
        self, query: str, num_results: int
    ) -> list[dict]:
        """Search using Serper.dev API (Google Search API)."""
        api_key = os.environ.get("SERPER_API_KEY", "")
        if not api_key:
            logger.warning("SERPER_API_KEY not set, trying fallback search")
            return await self._search_browser_or_fallback(query, num_results)

        try:
            response = await self._client.post(
                "https://google.serper.dev/search",
                headers={
                    "X-API-KEY": api_key,
                    "Content-Type": "application/json",
                },
                json={
                    "q": query,
                    "num": num_results,
                },
            )
            response.raise_for_status()
            data = response.json()

            results = []
            for item in data.get("organic", []):
                results.append({
                    "title": item.get("title", ""),
                    "link": item.get("link", ""),
                    "snippet": item.get("snippet", ""),
                })

            # Also include knowledge graph if available
            kg = data.get("knowledgeGraph")
            if kg and kg.get("description"):
                results.insert(0, {
                    "title": kg.get("title", "Knowledge Graph"),
                    "link": kg.get("website", kg.get("descriptionLink", "")),
                    "snippet": kg.get("description", ""),
                })

            return results[:num_results]

        except Exception as e:
            logger.error(f"Serper search failed: {e}")
            return await self._search_browser_or_fallback(query, num_results)

    async def _search_browser_or_fallback(
        self, query: str, num_results: int
    ) -> list[dict]:
        """
        Try browser search (if enabled), then fall back to DDG HTML scraping.

        Fallback chain: Browser → DuckDuckGo HTML scrape.
        """
        # Try browser search first (if configured)
        if self._browser_manager:
            results = await self._search_via_browser(query, num_results)
            if results:
                return results

        # Last resort: DuckDuckGo HTML scraping
        return await self._search_fallback(query, num_results)

    async def _search_via_browser(
        self, query: str, num_results: int
    ) -> list[dict]:
        """Search via Playwright browser — fallback when Serper unavailable."""
        try:
            from src.utils.browser_search import browser_search
            results = await browser_search(
                query=query,
                num_results=num_results,
                search_engine=self._search_engine,
                manager=self._browser_manager,
            )
            if results:
                logger.info(
                    f"Browser search returned {len(results)} results "
                    f"for '{query[:60]}'"
                )
            return results
        except ImportError:
            logger.debug("Browser search not available (playwright not installed)")
            return []
        except Exception as e:
            logger.warning(f"Browser search failed: {e}")
            return []

    async def _search_fallback(
        self, query: str, num_results: int
    ) -> list[dict]:
        """
        Fallback search using DuckDuckGo HTML (no API key needed).

        This is a best-effort fallback — it scrapes DDG's HTML results
        page, which may break if DDG changes their HTML structure.
        """
        try:
            response = await self._client.get(
                "https://html.duckduckgo.com/html/",
                params={"q": query},
                headers={
                    "User-Agent": "Mozilla/5.0 (compatible; SearchAgent/1.0)",
                },
                follow_redirects=True,
            )
            response.raise_for_status()
            html = response.text

            # Simple regex parsing of DDG HTML results
            import re
            results = []

            # Find result blocks
            result_blocks = re.findall(
                r'<a[^>]*class="result__a"[^>]*href="([^"]*)"[^>]*>(.*?)</a>.*?'
                r'<a[^>]*class="result__snippet"[^>]*>(.*?)</a>',
                html,
                re.DOTALL,
            )

            for url, title, snippet in result_blocks[:num_results]:
                # Clean HTML tags from title and snippet
                title = re.sub(r"<[^>]+>", "", title).strip()
                snippet = re.sub(r"<[^>]+>", "", snippet).strip()
                # DDG wraps URLs in a redirect
                if "uddg=" in url:
                    import urllib.parse
                    parsed = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
                    url = parsed.get("uddg", [url])[0]

                if url and title:
                    results.append({
                        "title": title,
                        "link": url,
                        "snippet": snippet,
                    })

            return results

        except Exception as e:
            logger.error(f"Fallback search also failed: {e}")
            return []
