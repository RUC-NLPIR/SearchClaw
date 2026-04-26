"""
WeChat article search tool -- searches and fetches WeChat public account articles.

Uses Sogou's WeChat search engine (weixin.sogou.com) as the entry point,
then resolves Sogou proxy links to real mp.weixin.qq.com URLs and
extracts article content.

This is a standalone tool (NOT a browser-based tool). It uses plain
HTTP requests with proper headers and referrer chains to avoid triggering
Sogou's anti-spider verification. No browser or API key required.

Pipeline:
1. Search Sogou WeChat -> get proxy links + titles
2. For each proxy link, parse the JS redirect page to extract the real URL
3. Fetch the real WeChat article and extract text from #js_content

Concurrency-safe: multiple searches can run in parallel.
"""

from __future__ import annotations

import logging
import re
from urllib.parse import quote

import httpx
from lxml import html as lxml_html

from src.core.tool import Tool, ToolUseContext
from src.core.types import Citation, SourceType, ToolResult, ValidationResult

logger = logging.getLogger(__name__)

# Sogou WeChat search base URL
_SOGOU_BASE = "https://weixin.sogou.com"

# Common headers mimicking a real browser
_BROWSER_HEADERS = {
    "Accept": (
        "text/html,application/xhtml+xml,application/xml;q=0.9,"
        "image/avif,image/webp,image/apng,*/*;q=0.8,"
        "application/signed-exchange;v=b3;q=0.7"
    ),
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "Pragma": "no-cache",
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/131.0.0.0 Safari/537.36"
    ),
}

# Request timeout (seconds)
_TIMEOUT = 15


def _is_antispider(response: httpx.Response) -> bool:
    """Detect Sogou's anti-spider/captcha redirect page."""
    url_lower = str(response.url).lower()
    body_lower = response.text.lower()
    return (
        "antispider" in url_lower
        or "seccoderight" in body_lower
        or "anti.min.css" in body_lower
    )


class WeChatSearchTool(Tool):
    name = "wechat_search"
    description = (
        "Search for WeChat public account (微信公众号) articles via Sogou. "
        "Returns article titles, real URLs, and optionally their full content. "
        "Use this when the user asks about WeChat articles or Chinese social media content."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query (Chinese or English).",
            },
            "num_results": {
                "type": "integer",
                "description": "Number of articles to return (default: 5, max: 10).",
                "default": 5,
            },
            "fetch_content": {
                "type": "boolean",
                "description": (
                    "Whether to fetch full article content for each result. "
                    "Default: true. Set to false to only get titles and URLs."
                ),
                "default": True,
            },
        },
        "required": ["query"],
    }

    is_concurrency_safe = True
    is_read_only = True

    def __init__(self, http_timeout: int = 15, max_result_size_chars: int = 30000):
        self.max_result_size_chars = max_result_size_chars
        self._client = httpx.AsyncClient(
            timeout=float(http_timeout),
            follow_redirects=True,
            headers=_BROWSER_HEADERS,
        )

    def prompt(self) -> str:
        return (
            "Use wechat_search to find WeChat public account articles. Tips:\n"
            "- Best for Chinese-language queries about WeChat/微信公众号 content\n"
            "- Set fetch_content=true (default) to get full article text\n"
            "- Set fetch_content=false if you only need titles and URLs\n"
            "- Results come from Sogou's WeChat search engine"
        )

    def validate_input(self, args: dict) -> ValidationResult:
        query = args.get("query", "")
        if not query or len(query.strip()) < 2:
            return ValidationResult(valid=False, message="Query must be at least 2 characters")
        if len(query) > 200:
            return ValidationResult(valid=False, message="Query too long (max 200 chars)")
        return ValidationResult(valid=True)

    async def call(self, args: dict, context: ToolUseContext) -> ToolResult:
        query = args["query"]
        num_results = min(args.get("num_results", 5), 10)
        fetch_content = args.get("fetch_content", True)

        # Rate limiting
        if context.rate_limiter:
            await context.rate_limiter.acquire("weixin.sogou.com")

        # Step 1: Search Sogou WeChat
        search_results = await self._sogou_search(query)

        if not search_results:
            return ToolResult(
                data=(
                    f"No WeChat articles found for: {query}\n"
                    "Sogou may have triggered anti-spider verification. "
                    "Try again later or use web_search as an alternative."
                ),
                is_error=False,
            )

        # Limit results
        search_results = search_results[:num_results]

        # Step 2: Resolve real URLs
        for result in search_results:
            sogou_link = result.get("sogou_link", "")
            if sogou_link:
                real_url = await self._resolve_real_url(sogou_link)
                result["real_url"] = real_url

        # Step 3: Optionally fetch article content
        if fetch_content:
            for result in search_results:
                real_url = result.get("real_url", "")
                sogou_link = result.get("sogou_link", "")
                if real_url:
                    content = await self._fetch_article(real_url, referer=sogou_link)
                    result["content"] = content

        # Format output
        formatted_parts = [f"## WeChat Article Search Results: {query}\n"]
        citations = []

        for i, r in enumerate(search_results, 1):
            title = r.get("title", "Untitled")
            real_url = r.get("real_url", "")
            publish_time = r.get("publish_time", "")
            content = r.get("content", "")

            formatted_parts.append(f"### {i}. {title}")
            if real_url:
                formatted_parts.append(f"**URL**: {real_url}")
            if publish_time:
                formatted_parts.append(f"**Published**: {publish_time}")
            if content:
                # Truncate very long articles
                if len(content) > 3000:
                    content = content[:3000] + "\n\n[Article content truncated]"
                formatted_parts.append(f"\n{content}")
            formatted_parts.append("")  # blank line

            if real_url:
                citations.append(Citation(
                    url=real_url,
                    title=title,
                    snippet=content[:200] if content else "",
                    source_type=SourceType.WEB,
                ))

        formatted = "\n".join(formatted_parts)

        # Truncate if needed
        formatted, truncated, cached_path = await self._maybe_truncate(
            formatted, query, context
        )

        return ToolResult(
            data=formatted,
            citations=citations,
            truncated=truncated,
            cached_path=cached_path,
        )

    # ------------------------------------------------------------------
    # Step 1: Sogou WeChat search
    # ------------------------------------------------------------------

    async def _sogou_search(self, query: str) -> list[dict]:
        """
        Search Sogou WeChat for articles matching the query.

        Returns a list of dicts with keys: title, sogou_link, publish_time.
        """
        params = {
            "type": "2",           # Article search
            "s_from": "input",
            "query": query,
            "ie": "utf8",
        }
        headers = {
            **_BROWSER_HEADERS,
            "Referer": f"{_SOGOU_BASE}/weixin?query={quote(query)}",
        }

        try:
            response = await self._client.get(
                f"{_SOGOU_BASE}/weixin",
                params=params,
                headers=headers,
            )
            response.raise_for_status()

            if _is_antispider(response):
                logger.warning("Sogou WeChat search: anti-spider page detected")
                return []

            return self._parse_search_results(response.text)

        except httpx.TimeoutException:
            logger.warning("Sogou WeChat search timed out")
            return []
        except Exception as e:
            logger.warning(f"Sogou WeChat search failed: {e}")
            return []

    def _parse_search_results(self, html_text: str) -> list[dict]:
        """Parse Sogou search results HTML into structured results."""
        try:
            tree = lxml_html.fromstring(html_text)
        except Exception as e:
            logger.warning(f"Failed to parse Sogou HTML: {e}")
            return []

        results = []

        # Extract article title elements
        title_elements = tree.xpath(
            "//a[contains(@id, 'sogou_vr_11002601_title_')]"
        )
        # Extract publish time elements
        time_elements = tree.xpath(
            "//li[contains(@id, 'sogou_vr_11002601_box_')]"
            "/div[@class='txt-box']/div[@class='s-p']"
            "/span[@class='s2']"
        )

        for i, title_el in enumerate(title_elements):
            title = title_el.text_content().strip()
            link = title_el.get("href", "")
            if link and not link.startswith("http"):
                link = _SOGOU_BASE + link

            publish_time = ""
            if i < len(time_elements):
                publish_time = time_elements[i].text_content().strip()

            if title and link:
                results.append({
                    "title": title,
                    "sogou_link": link,
                    "publish_time": publish_time,
                })

        logger.info(f"Sogou WeChat search: found {len(results)} articles")
        return results

    # ------------------------------------------------------------------
    # Step 2: Resolve real URL from Sogou proxy page
    # ------------------------------------------------------------------

    async def _resolve_real_url(self, sogou_url: str) -> str:
        """
        Follow a Sogou proxy link and extract the real mp.weixin.qq.com URL.

        Sogou serves an intermediate page with JavaScript that constructs
        the real URL via string concatenation: url += '...'; url += '...';
        We parse these fragments without executing JS.
        """
        headers = {
            **_BROWSER_HEADERS,
            "Referer": f"{_SOGOU_BASE}/weixin",
        }

        try:
            response = await self._client.get(
                sogou_url,
                headers=headers,
            )

            if _is_antispider(response):
                logger.warning("Sogou URL resolution: anti-spider detected")
                return ""

            page_text = response.text

            # Extract URL fragments from JS: url += '...';
            # The pattern matches lines like: url += 'weixin.qq.com/s?__biz=...';
            fragments = re.findall(r"url\s*\+=\s*'([^']*)'", page_text)

            if not fragments:
                logger.debug(f"No URL fragments found in Sogou proxy page: {sogou_url[:80]}")
                return ""

            # Concatenate fragments and remove obfuscation characters
            raw_url = "".join(fragments).replace("@", "")

            if not raw_url:
                return ""

            # Build full URL — Sogou sometimes strips the protocol+subdomain,
            # sometimes includes it. Handle both cases.
            if raw_url.startswith("http"):
                real_url = raw_url
            elif raw_url.startswith("weixin.qq.com"):
                real_url = "https://mp." + raw_url
            else:
                real_url = "https://mp.weixin.qq.com/" + raw_url

            logger.debug(f"Resolved Sogou URL -> {real_url[:80]}")
            return real_url

        except httpx.TimeoutException:
            logger.warning(f"Sogou URL resolution timed out: {sogou_url[:80]}")
            return ""
        except Exception as e:
            logger.warning(f"Sogou URL resolution failed: {e}")
            return ""

    # ------------------------------------------------------------------
    # Step 3: Fetch article content
    # ------------------------------------------------------------------

    async def _fetch_article(self, real_url: str, referer: str = "") -> str:
        """
        Fetch a WeChat article and extract its text content.

        Uses the Sogou link as referrer to appear as a natural click-through.
        Extracts text from the #js_content div (WeChat's article body).
        """
        if not real_url or real_url == "https://mp.":
            return ""

        headers = {
            **_BROWSER_HEADERS,
            "sec-ch-ua": '"Chromium";v="131", "Not_A Brand";v="24"',
            "sec-ch-ua-mobile": "?0",
            "sec-ch-ua-platform": '"macOS"',
            "sec-fetch-dest": "document",
            "sec-fetch-mode": "navigate",
            "sec-fetch-site": "cross-site",
            "sec-fetch-user": "?1",
            "upgrade-insecure-requests": "1",
        }
        if referer:
            headers["Referer"] = referer

        try:
            response = await self._client.get(
                real_url,
                headers=headers,
            )
            response.raise_for_status()

            tree = lxml_html.fromstring(response.text)

            # Extract text from #js_content (WeChat's article body container)
            content_elements = tree.xpath("//div[@id='js_content']//text()")
            if not content_elements:
                # Fallback: try .rich_media_content
                content_elements = tree.xpath(
                    "//div[@class='rich_media_content']//text()"
                )

            if not content_elements:
                logger.info(f"No article content found at {real_url[:80]}")
                return ""

            # Clean and join text
            cleaned = [text.strip() for text in content_elements if text.strip()]
            content = "\n".join(cleaned)

            logger.info(f"Fetched WeChat article: {len(content)} chars from {real_url[:80]}")
            return content

        except httpx.TimeoutException:
            logger.warning(f"WeChat article fetch timed out: {real_url[:80]}")
            return ""
        except Exception as e:
            logger.warning(f"WeChat article fetch failed: {e}")
            return ""
