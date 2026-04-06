"""
Web fetch tool — fetches a URL and converts to readable markdown.

The primary content reading tool. Given a URL, it fetches the page
and converts it to clean markdown for the LLM.

Fetch strategies (tried in order):
1. Jina Reader API (r.jina.ai) — returns markdown directly, handles
   JS-rendered pages, PDFs, and paywalled content. Set JINA_API_KEY
   for higher rate limits (200 RPM vs 20 RPM without key).
2. Direct fetch + trafilatura — fetches raw HTML and extracts content
   locally. Works without any API key but struggles with JS-heavy pages.

Large pages are truncated and cached to disk (the deep_read tool
can retrieve specific sections from cached content).

Concurrency-safe: multiple pages can be fetched in parallel.
"""

from __future__ import annotations

import logging
import os
import re

import httpx

from src.core.tool import Tool, ToolUseContext
from src.core.types import Citation, SourceType, ToolResult, ValidationResult
from src.utils.html_to_markdown import html_to_markdown
from src.utils.url_validator import validate_url_for_ssrf

logger = logging.getLogger(__name__)

# Common headers to avoid bot detection (direct fetch fallback)
DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

# Jina Reader API base URL
JINA_READER_URL = "https://r.jina.ai/"


class WebFetchTool(Tool):
    name = "web_fetch"
    description = (
        "Fetch a web page and convert it to readable markdown. "
        "Use this after web_search to read the full content of a promising result. "
        "If the page is too long, it will be truncated and cached — use deep_read "
        "to access specific sections."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "The URL of the web page to fetch.",
            },
            "target_selector": {
                "type": "string",
                "description": (
                    "Optional CSS selector to extract only specific content "
                    "(e.g., 'article', '.main-content'). Only works with Jina backend."
                ),
            },
        },
        "required": ["url"],
    }

    is_concurrency_safe = True
    is_read_only = True

    def __init__(
        self,
        max_result_size_chars: int = 50000,
        http_timeout: int = 30,
        jina_timeout: int = 60,
        extraction_threshold: int = 15000,
    ):
        self.max_result_size_chars = max_result_size_chars
        self._extraction_threshold = extraction_threshold
        self._client = httpx.AsyncClient(
            timeout=float(http_timeout),
            follow_redirects=True,
            headers=DEFAULT_HEADERS,
        )
        self._jina_client = httpx.AsyncClient(timeout=float(jina_timeout))
        # Browser integration (set externally by build_default_registry
        # when browser.enabled and browser.use_for_fetch are true)
        self._browser_manager = None

    def prompt(self) -> str:
        return (
            "Use web_fetch to read the full content of a web page. Tips:\n"
            "- Always web_search first, then fetch the most relevant results\n"
            "- If the result says '[Content truncated]', use deep_read with the cached path\n"
            "- Don't fetch pages that are clearly irrelevant based on their title/snippet\n"
            "- Prefer pages from authoritative sources\n"
            "- Use target_selector (CSS selector) to extract only the relevant section "
            "of a page (e.g., 'article', '.post-content') — reduces noise"
        )

    def validate_input(self, args: dict) -> ValidationResult:
        url = args.get("url", "")
        if not url:
            return ValidationResult(valid=False, message="URL is required")
        if not url.startswith(("http://", "https://")):
            return ValidationResult(
                valid=False,
                message="URL must start with http:// or https://",
            )
        # SSRF prevention: block internal/private IPs, cloud metadata, etc.
        is_safe, reason = validate_url_for_ssrf(url)
        if not is_safe:
            return ValidationResult(valid=False, message=f"URL blocked (SSRF protection): {reason}")
        return ValidationResult(valid=True)

    async def call(self, args: dict, context: ToolUseContext) -> ToolResult:
        url = args["url"]
        target_selector = args.get("target_selector")

        # Rate limiting
        if context.rate_limiter:
            await context.rate_limiter.acquire(url)

        # --- Strategy 1: Jina Reader API ---
        jina_result = await self._fetch_via_jina(url, target_selector, context)
        if jina_result is not None:
            return jina_result

        # --- Strategy 2: Browser fetch (if enabled) ---
        if self._browser_manager:
            browser_result = await self._fetch_via_browser(url, context)
            if browser_result is not None:
                return browser_result

        # --- Strategy 3: Direct fetch + local conversion ---
        logger.info(f"Jina unavailable, falling back to direct fetch for {url}")
        return await self._fetch_direct(url, context)

    # ------------------------------------------------------------------
    # Strategy 1: Jina Reader API
    # ------------------------------------------------------------------

    async def _fetch_via_jina(
        self,
        url: str,
        target_selector: str | None,
        context: ToolUseContext,
    ) -> ToolResult | None:
        """
        Fetch via Jina Reader API (r.jina.ai).

        Returns None if Jina is unavailable (no key configured and rate
        limited, or network error), signaling the caller to try the
        direct-fetch fallback.

        Jina returns markdown directly — no local HTML-to-markdown needed.
        Handles JS-rendered pages, PDFs, and complex layouts better than
        local extraction.
        """
        # Build headers
        headers: dict[str, str] = {
            "Accept": "application/json",
            "X-Return-Format": "markdown",
            "X-No-Cache": "true",
            "X-Retain-Images": "none",   # Strip images to save tokens
        }

        # API key (optional but recommended — 200 RPM vs 20 RPM)
        jina_key = os.environ.get("JINA_API_KEY", "")
        if jina_key:
            headers["Authorization"] = f"Bearer {jina_key}"

        # Target selector — extract only matching elements
        if target_selector:
            headers["X-Target-Selector"] = target_selector

        # Remove common boilerplate
        headers["X-Remove-Selector"] = "nav, footer, .sidebar, .ads, .cookie-banner"

        try:
            response = await self._jina_client.get(
                f"{JINA_READER_URL}{url}",
                headers=headers,
            )
            response.raise_for_status()
        except httpx.TimeoutException:
            logger.warning(f"Jina timeout for {url}")
            return None  # Fall back to direct
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 402:
                logger.warning("Jina API token budget exceeded")
            elif e.response.status_code == 429:
                logger.warning("Jina API rate limited")
            else:
                logger.warning(f"Jina HTTP error {e.response.status_code} for {url}")
            return None  # Fall back to direct
        except Exception as e:
            logger.warning(f"Jina request failed for {url}: {e}")
            return None  # Fall back to direct

        # Parse response
        try:
            data = response.json()
            jina_data = data.get("data", {})
            markdown = jina_data.get("content", "")
            title = jina_data.get("title", "") or "Untitled Page"
        except Exception:
            # If Accept: application/json wasn't honored, body is plain markdown
            markdown = response.text
            title = self._extract_title_from_markdown(markdown)

        if not markdown or len(markdown.strip()) < 50:
            logger.info(f"Jina returned insufficient content for {url}, falling back")
            return None  # Fall back to direct

        logger.info(f"Jina fetched {url}: {len(markdown)} chars, title='{title[:60]}'")

        # Build full content
        full_content = f"## {title}\n**Source**: {url}\n\n{markdown}"

        # For large content: cache to disk, then extract key facts
        result = await self._maybe_extract_or_truncate(
            full_content, url, title, context
        )
        if result is not None:
            # Add citation to the extracted/truncated result
            result.citations = [Citation(
                url=url,
                title=title,
                snippet=markdown[:300],
                source_type=SourceType.WEB,
            )]
            return result

        # Small content — return raw (no extraction needed)
        return ToolResult(
            data=full_content,
            citations=[Citation(
                url=url,
                title=title,
                snippet=markdown[:300],
                source_type=SourceType.WEB,
            )],
        )

    # ------------------------------------------------------------------
    # Strategy 2: Browser fetch (Playwright)
    # ------------------------------------------------------------------

    async def _fetch_via_browser(
        self,
        url: str,
        context: ToolUseContext,
    ) -> ToolResult | None:
        """
        Fetch via Playwright browser — handles JS-rendered and auth-walled pages.

        Returns None if browser fetch is unavailable or fails, signaling
        the caller to try the direct-fetch fallback.

        Uses the user's real browser profile (if configured via CDP or
        persistent context) to access auth-walled pages like Twitter,
        LinkedIn, etc.
        """
        try:
            from src.utils.browser_fetch import browser_fetch
        except ImportError:
            logger.debug("Browser fetch not available (playwright not installed)")
            return None

        try:
            result = await browser_fetch(
                url=url,
                manager=self._browser_manager,
            )
        except Exception as e:
            logger.warning(f"Browser fetch failed for {url}: {e}")
            return None

        if not result or len(result.get("content", "")) < 50:
            logger.info(f"Browser fetch returned insufficient content for {url}")
            return None

        title = result.get("title") or "Untitled Page"
        content = result["content"]
        full_content = f"## {title}\n**Source**: {url}\n\n{content}"

        logger.info(
            f"Browser fetched {url}: {len(content)} chars, title='{title[:60]}'"
        )

        # Apply same extraction/truncation pipeline as other strategies
        extracted = await self._maybe_extract_or_truncate(
            full_content, url, title, context
        )
        if extracted is not None:
            extracted.citations = [Citation(
                url=url,
                title=title,
                snippet=content[:300],
                source_type=SourceType.WEB,
            )]
            return extracted

        # Small content — return raw
        return ToolResult(
            data=full_content,
            citations=[Citation(
                url=url,
                title=title,
                snippet=content[:300],
                source_type=SourceType.WEB,
            )],
        )

    # ------------------------------------------------------------------
    # Strategy 3: Direct fetch + local HTML-to-markdown
    # ------------------------------------------------------------------

    async def _fetch_direct(
        self, url: str, context: ToolUseContext
    ) -> ToolResult:
        """
        Direct HTTP fetch + local trafilatura/markdownify conversion.

        This is the last-resort fallback when both Jina and browser
        are unavailable. Works without any API key or browser but
        can't handle JS-rendered pages.
        """
        try:
            response = await self._client.get(url)
            response.raise_for_status()
        except httpx.TimeoutException:
            return ToolResult(
                data=f"Timeout fetching {url} — the page took too long to respond.",
                is_error=True,
            )
        except httpx.HTTPStatusError as e:
            return ToolResult(
                data=f"HTTP error fetching {url}: {e.response.status_code} {e.response.reason_phrase}",
                is_error=True,
            )
        except Exception as e:
            return ToolResult(
                data=f"Failed to fetch {url}: {str(e)}",
                is_error=True,
            )

        # Detect content type
        content_type = response.headers.get("content-type", "")

        if "application/pdf" in content_type:
            return ToolResult(
                data=f"This URL points to a PDF file. PDF parsing is not supported in direct mode. "
                     f"Set JINA_API_KEY to enable PDF reading via Jina. URL: {url}",
                is_error=False,
            )

        if "text/html" not in content_type and "application/xhtml" not in content_type:
            # Non-HTML content — return raw text (truncated)
            text = response.text[:self.max_result_size_chars]
            return ToolResult(
                data=f"## Content from {url}\n\n(Content-Type: {content_type})\n\n{text}",
                citations=[Citation(
                    url=url,
                    title=url,
                    snippet=text[:200],
                    source_type=SourceType.WEB,
                )],
            )

        # Convert HTML to markdown
        html = response.text
        markdown = html_to_markdown(html, url=url)

        if not markdown or len(markdown) < 50:
            return ToolResult(
                data=f"Could not extract meaningful content from {url}. "
                     f"The page may be dynamically rendered (JavaScript-heavy). "
                     f"Set JINA_API_KEY to handle JS-rendered pages via Jina Reader.",
                is_error=False,
            )

        # Extract title from the markdown or HTML
        title = self._extract_title(html, markdown)

        # Truncation handling (large pages → extract or cache to disk)
        full_content = f"## {title}\n**Source**: {url}\n\n{markdown}"

        # For large content: cache to disk, then extract key facts
        result = await self._maybe_extract_or_truncate(
            full_content, url, title, context
        )
        if result is not None:
            result.citations = [Citation(
                url=url,
                title=title,
                snippet=markdown[:300],
                source_type=SourceType.WEB,
            )]
            return result

        # Small content — return raw
        return ToolResult(
            data=full_content,
            citations=[Citation(
                url=url,
                title=title,
                snippet=markdown[:300],
                source_type=SourceType.WEB,
            )],
        )

    # ------------------------------------------------------------------
    # Content extraction / truncation for large pages
    # ------------------------------------------------------------------

    async def _maybe_extract_or_truncate(
        self,
        full_content: str,
        url: str,
        title: str,
        context: ToolUseContext,
    ) -> ToolResult | None:
        """
        Handle large content: extract key facts or fall back to truncation.

        For content exceeding the extraction threshold:
        1. Cache full content to disk (for deep_read)
        2. Try LLM-based extraction (key facts + relevant URLs)
        3. Fall back to raw truncation if extraction fails

        Returns None if content is small enough to return raw.
        """
        if len(full_content) <= self._extraction_threshold:
            return None  # Small enough — caller returns raw

        # Always cache full content to disk first (for deep_read)
        cached_path = await self._cache_full_content(full_content, url, context)

        # Try LLM-based extraction if threshold > 0 (0 disables extraction)
        if self._extraction_threshold > 0:
            from src.utils.content_extractor import extract_content

            research_query = context.extra.get("research_query", "")
            extracted = await extract_content(
                raw_content=full_content,
                research_query=research_query,
                source_url=url,
                source_title=title,
            )

            if extracted:
                # Append reference to cached full content
                extracted += (
                    f"\n\n---\n[Full content ({len(full_content):,} chars) "
                    f"cached at: {cached_path}. Use deep_read to access "
                    f"specific sections.]"
                )
                return ToolResult(
                    data=extracted,
                    truncated=True,
                    cached_path=str(cached_path),
                )

        # Fallback: raw truncation (same as _maybe_truncate)
        preview = full_content[: self.max_result_size_chars]
        preview += (
            f"\n\n---\n[Content truncated. Full content "
            f"({len(full_content):,} chars) saved to: {cached_path}]"
        )
        return ToolResult(
            data=preview,
            truncated=True,
            cached_path=str(cached_path),
        )

    async def _cache_full_content(
        self,
        content: str,
        url: str,
        context: ToolUseContext,
    ) -> str:
        """
        Cache full content to disk and return the file path.

        Extracted from _maybe_truncate so we can cache BEFORE extraction
        (the extraction uses a compressed version, but deep_read needs
        the full content on disk).
        """
        import hashlib

        url_hash = hashlib.sha256(url.encode()).hexdigest()[:16]
        cached_path = context.cache_dir / f"{self.name}_{url_hash}.md"
        cached_path.write_text(content, encoding="utf-8")
        logger.info(
            f"Cached {len(content):,} chars to {cached_path}"
        )
        return str(cached_path)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _extract_title(self, html: str, markdown: str) -> str:
        """Extract page title from HTML or markdown."""
        # Try HTML <title> tag
        match = re.search(r"<title[^>]*>(.*?)</title>", html, re.IGNORECASE | re.DOTALL)
        if match:
            title = match.group(1).strip()
            # Clean up common title suffixes
            title = re.split(r"\s*[|–—-]\s*(?=[A-Z])", title)[0].strip()
            if title:
                return title

        return self._extract_title_from_markdown(markdown)

    def _extract_title_from_markdown(self, markdown: str) -> str:
        """Extract title from the first markdown heading."""
        match = re.match(r"^#\s+(.+)$", markdown, re.MULTILINE)
        if match:
            return match.group(1).strip()
        return "Untitled Page"
